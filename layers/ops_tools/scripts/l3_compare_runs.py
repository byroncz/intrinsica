"""Compara dos corridas de L3 por sus hallazgos y mide la nueva (ITSC-364).

Cierra la validación del núcleo nuevo: por cada (θ, mes) compara el `content_hash` y las
`rows` del hallazgo `summaries_summary` de una corrida **base** (la referencia, ITSC-335) con
los de una **nueva**. Dos archivos con el mismo `content_hash` tienen el mismo contenido lógico
(RNF-L3-03), así que 5 500 pares iguales (110 meses × 50 θ) prueban que el núcleo nuevo produce
lo mismo que el anterior sin releer los Parquet de L3. Con `--timing` resume además los
`unit_timing` de la corrida nueva.

Cada lado se elige por fecha de detección (`detected_date=`) y/o por `run_id`, porque un
backfill se lanza por rangos y cada ejecución tiene su `run_id`:

  --base-dates 2026-10-01:2026-10-17     (AAAA-MM-DD o AAAA-MM-DD:AAAA-MM-DD; el rango es
  --new-dates  2026-10-22:2026-10-23      cerrado y se puede dejar abierto: `2026-10-22:`)
  --base-run <run_id> --new-run <run_id>  (repetibles; se combinan con las fechas)

Cada lado necesita al menos uno de los dos. Con solo `--*-run` se leen los hallazgos de todos los
días del lago: usa fechas siempre que puedas. Si un (θ, mes) aparece en varias corridas de un
lado, se toma la de mayor `detected_at`; si esas corridas discrepan entre sí (`content_hash` o
`rows` distintos) es una diferencia aparte, `INCONSISTENTE`: acota el lado con fechas o `run_id`.

Args (todos opcionales salvo el par de lados):
  --from AAAA-MM --to AAAA-MM        (solo esos meses, en los dos lados)
  --dq-root gs://<proyecto>-dq-findings/l3   (por defecto, el del proyecto según
                                              OPS_RESULTS_URI; también una ruta local)
  --provider binance --market spot --asset BTCUSDT
  --family summaries                 (la familia de descriptores que se compara)
  --timing                           (resume los `unit_timing` de la corrida nueva)
  --baseline-vcpu-s 165000           (con --timing: los vCPU-s del backfill original)
  --max-diffs 100                    (cuántas diferencias se imprimen; las cuentas las cuentan todas)

Salida: una línea por diferencia (`DIFERENCIA`, `SOLO EN BASE`, `SOLO EN NUEVA`,
`INCONSISTENTE`) y, como última línea de la comparación, `compare runs: pares=… coinciden=…
difieren=… solo_base=… solo_nueva=… inconsistentes=… veredicto=ok|diferencias`. Con `--timing`
imprime después la tabla por mes y los percentiles; con `OPS_RESULTS_URI` deja además
`l3_compare_runs_timing_<fecha>.csv`. La medición es informativa: no cambia el código de salida.

Códigos: 0 si todos los pares coinciden; 1 si hay alguna diferencia, un lado sin hallazgos o
no se pudo leer el lago; 2 por uso.

Solo lee. Lista el lago con un selector sobre `l3/` y abre cada archivo por su ruta: nunca
`pyarrow.dataset` sobre el prefijo, que pide el objeto `l3` sin barra y la condición IAM por
prefijo de `ops-script` lo niega con 403 (ITSC-370). Dónde corre: con `ops-script` o en local con
`uv run`.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from pyutils import resolve_fs

SUMMARY = "summaries_summary"
TIMING = "unit_timing"
FAMILY = "summaries"  # la única familia de L3 hoy; un `summaries_summary` sin `family` es de ella
FILE = re.compile(r"/detected_date=(\d{4}-\d{2}-\d{2})/[^/]+\.parquet$")
MONTH = re.compile(r"(\d{4})-(0[1-9]|1[0-2])")
UTILIZATION_P50_GOAL = 0.90
UTILIZATION_MIN_GOAL = 0.80

Key = tuple[str, int, int]  # (θ como en la partición, año, mes)


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class Side:
    """Qué hallazgos son de un lado: los de sus fechas y, si trae `runs`, solo esas corridas."""

    label: str
    first: str | None
    last: str | None
    runs: frozenset[str]

    def has_date(self, day: str) -> bool:
        return (self.first is None or day >= self.first) and (
            self.last is None or day <= self.last
        )


def parse_dates(text: str | None) -> tuple[str | None, str | None]:
    if text is None:
        return None, None
    first, colon, last = text.partition(":")
    if not colon:
        last = first
    for part in (first, last):
        if part:
            try:
                date.fromisoformat(part)
            except ValueError:
                raise ValueError(
                    f"{text!r}: se espera AAAA-MM-DD[:AAAA-MM-DD]"
                ) from None
    return first or None, last or None


def make_side(label: str, dates: str | None, runs: list[str] | None) -> Side:
    first, last = parse_dates(dates)
    if dates is None and not runs:
        raise ValueError(f"falta --{label}-dates o --{label}-run")
    return Side(label, first, last, frozenset(runs or ()))


def default_root() -> str:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        raise ValueError(
            "sin --dq-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    return f"gs://{match[1]}-dq-findings/l3"


def list_findings(fs: pafs.FileSystem, base: str) -> list[tuple[str, str]]:
    """Los archivos de hallazgos del lago y su `detected_date`, en un solo listado recursivo."""
    selector = pafs.FileSelector(f"{base}/", recursive=True, allow_not_found=True)
    found = []
    for info in fs.get_file_info(selector):
        match = FILE.search(info.path) if info.type == pafs.FileType.File else None
        if match:
            found.append((info.path, match[1]))
    return sorted(found)


def read_side(
    fs: pafs.FileSystem,
    files: list[tuple[str, str]],
    side: Side,
    series: tuple[str, str, str],
    checks: list[str],
) -> list[dict]:
    """Las filas de `checks` de la serie que son del lado, con su `details` ya decodificado."""
    paths = [path for path, day in files if side.has_date(day)]
    if not paths:
        return []
    provider, market, asset = series
    keep = (
        pc.field("check_type").isin(checks)
        & (pc.field("provider") == provider)
        & (pc.field("market") == market)
        & (pc.field("asset") == asset)
    )
    if side.runs:
        keep &= pc.field("run_id").isin(sorted(side.runs))
    table = ds.dataset(paths, filesystem=fs, format="parquet").to_table(
        columns=["check_type", "detected_at", "run_id", "year", "month", "details"],
        filter=keep,
    )
    rows = table.to_pylist()
    for row in rows:
        row["details"] = json.loads(row["details"])
    return rows


@dataclass
class Cell:
    """Lo que un lado dice de un (θ, mes): lo de su última corrida y si las corridas discrepan."""

    content_hash: str
    rows: int
    variants: int


def cells(
    rows: list[dict],
    months: tuple[tuple[int, int], tuple[int, int]],
    family: str = FAMILY,
) -> dict[Key, Cell]:
    first, last = months
    latest: dict[Key, tuple[tuple, dict]] = {}
    seen: dict[Key, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        if (
            row["check_type"] != SUMMARY
            or not first <= (row["year"], row["month"]) <= last
        ):
            continue
        details = row["details"]
        if details.get("family", FAMILY) != family:
            continue
        key = (details["theta"], row["year"], row["month"])
        seen[key].add((details["content_hash"], details["rows"]))
        order = (row["detected_at"], row["run_id"])
        if key not in latest or order > latest[key][0]:
            latest[key] = (order, details)
    return {
        key: Cell(details["content_hash"], details["rows"], len(seen[key]))
        for key, (_, details) in latest.items()
    }


def month_text(key: Key) -> str:
    return f"{key[1]:04d}-{key[2]:02d}"


@dataclass
class Comparison:
    pairs: int = 0
    equal: int = 0
    differ: int = 0
    only_base: int = 0
    only_new: int = 0
    inconsistent: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.pairs > 0
            and self.equal == self.pairs
            and not self.inconsistent
            and not self.only_base
            and not self.only_new
        )

    def line(self) -> str:
        return (
            f"compare runs: pares={self.pairs} coinciden={self.equal} difieren={self.differ} "
            f"solo_base={self.only_base} solo_nueva={self.only_new} "
            f"inconsistentes={self.inconsistent} veredicto={'ok' if self.ok else 'diferencias'}"
        )


def compare(
    base: dict[Key, Cell], new: dict[Key, Cell]
) -> tuple[Comparison, list[str]]:
    """Cuenta los pares y devuelve las líneas de lo que no coincide (todas; el tope lo pone quien imprime)."""
    result = Comparison(pairs=len(base.keys() | new.keys()))
    lines: list[str] = []
    for side, label in ((base, "base"), (new, "nueva")):
        for key in sorted(k for k, cell in side.items() if cell.variants > 1):
            result.inconsistent += 1
            lines.append(
                f"INCONSISTENTE {label} θ={key[0]} {month_text(key)}: sus corridas dejaron "
                f"{side[key].variants} pares (content_hash, rows) distintos"
            )
    inconsistent = {
        key for side in (base, new) for key, cell in side.items() if cell.variants > 1
    }
    for key in sorted(base.keys() | new.keys()):
        if key not in new:
            result.only_base += 1
            lines.append(f"SOLO EN BASE θ={key[0]} {month_text(key)}")
        elif key not in base:
            result.only_new += 1
            lines.append(f"SOLO EN NUEVA θ={key[0]} {month_text(key)}")
        elif (base[key].content_hash, base[key].rows) == (
            new[key].content_hash,
            new[key].rows,
        ):
            if key not in inconsistent:
                result.equal += 1
        else:
            result.differ += 1
            what = []
            if base[key].content_hash != new[key].content_hash:
                what.append(
                    f"content_hash base={base[key].content_hash} nueva={new[key].content_hash}"
                )
            if base[key].rows != new[key].rows:
                what.append(f"rows base={base[key].rows} nueva={new[key].rows}")
            lines.append(f"DIFERENCIA θ={key[0]} {month_text(key)}: {'; '.join(what)}")
    return result, lines


def percentile(values: list[float], q: float) -> float:
    """Percentil `q` (0 a 1) con interpolación lineal; `values` no vacío."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def unit_timings(rows: list[dict], months) -> dict[tuple[int, int], dict]:
    """El `unit_timing` de la última corrida de cada mes."""
    first, last = months
    latest: dict[tuple[int, int], tuple[tuple, dict]] = {}
    for row in rows:
        month = (row["year"], row["month"])
        if row["check_type"] != TIMING or not first <= month <= last:
            continue
        order = (row["detected_at"], row["run_id"])
        if month not in latest or order > latest[month][0]:
            latest[month] = (order, row["details"])
    return {month: details for month, (_, details) in sorted(latest.items())}


def number(details: dict, name: str) -> float | None:
    value = details.get(name)
    return float(value) if isinstance(value, int | float) else None


def timing_lines(
    timings: dict[tuple[int, int], dict], baseline: float | None
) -> list[str]:
    """La tabla por mes y los percentiles que pide la validación del núcleo (ITSC-364)."""
    out = [
        f"{'mes':>8} {'wall_s':>9} {'cpu_util':>9} {'rss_mib':>9} {'busy':>6} {'cores':>6}"
    ]
    utilization: list[tuple[float, str]] = []
    walls, busy, vcpu_s = [], [], 0.0
    rss: list[tuple[float, str]] = []
    for (year, month), details in timings.items():
        name = f"{year:04d}-{month:02d}"
        wall, util = number(details, "wall_s"), number(details, "cpu_utilization")
        peak, engine = number(details, "rss_peak_mib"), number(details, "engine_busy")
        cores = number(details, "cores")
        shown = [
            f"{value:{spec}}" if value is not None else "-"
            for value, spec in (
                (wall, ".1f"),
                (util, ".3f"),
                (peak, ".1f"),
                (engine, ".3f"),
            )
        ]
        out.append(
            f"{name:>8} {shown[0]:>9} {shown[1]:>9} {shown[2]:>9} {shown[3]:>6} "
            f"{'-' if cores is None else f'{cores:g}':>6}"
        )
        if wall is not None:
            walls.append(wall)
            vcpu_s += wall * (cores or 0.0)
        if util is not None:
            utilization.append((util, name))
        if peak is not None:
            rss.append((peak, name))
        if engine is not None:
            busy.append(engine)
    out.append(f"medición: {len(timings)} meses con unit_timing en la corrida nueva")
    if not timings:
        return out
    if utilization:
        values = [value for value, _ in utilization]
        low = min(utilization)
        p50 = percentile(values, 0.5)
        out.append(
            f"cpu_utilization: p10={percentile(values, 0.1):.3f} p50={p50:.3f} "
            f"mínimo={low[0]:.3f} ({low[1]})"
        )
        below = [
            f"{name} ({value:.3f})"
            for value, name in sorted(utilization)
            if value < UTILIZATION_MIN_GOAL
        ]
        out.append(
            f"meses bajo {UTILIZATION_MIN_GOAL:.2f}: {len(below)}"
            + (f": {', '.join(below)}" if below else "")
        )
        met_p50 = "sí" if p50 >= UTILIZATION_P50_GOAL else "no"
        met_min = "sí" if not below else "no"
        out.append(
            f"metas: p50 ≥ {UTILIZATION_P50_GOAL:.2f}: {met_p50}; "
            f"ningún mes bajo {UTILIZATION_MIN_GOAL:.2f}: {met_min}"
        )
    if walls:
        out.append(
            f"wall_s: suma={sum(walls):.0f} p50={percentile(walls, 0.5):.1f} máximo={max(walls):.1f}"
        )
        versus = (
            f" contra {baseline:.0f} ({100 * vcpu_s / baseline:.1f} %)"
            if baseline
            else ""
        )
        out.append(f"vCPU-s (Σ wall_s × cores): {vcpu_s:.0f}{versus}")
    if rss:
        top = max(rss)
        out.append(f"rss_peak_mib: máximo={top[0]:.1f} ({top[1]})")
    if busy:
        out.append(f"engine_busy: p50={percentile(busy, 0.5):.3f}")
    return out


def save_timing_csv(timings: dict[tuple[int, int], dict]) -> None:
    """Deja la tabla por mes en `OPS_RESULTS_URI`; si no se puede, la salida estándar basta."""
    uri = os.environ.get("OPS_RESULTS_URI")
    if not uri:
        return
    try:
        from google.cloud import storage

        bucket, _, prefix = uri.removeprefix("gs://").partition("/")
        name = f"{prefix}l3_compare_runs_timing_{time.strftime('%Y%m%dT%H%M%S')}.csv"
        columns = (
            "wall_s",
            "cpu_utilization",
            "rss_peak_mib",
            "engine_busy",
            "cores",
            "cpu_s",
        )
        rows = ["mes," + ",".join(columns)]
        for (year, month), details in timings.items():
            values = [
                "" if details.get(c) is None else str(details[c]) for c in columns
            ]
            rows.append(f"{year:04d}-{month:02d}," + ",".join(values))
        storage.Client().bucket(bucket).blob(name).upload_from_string(
            "\n".join(rows) + "\n"
        )
        log(f"tabla por mes en gs://{bucket}/{name}")
    except Exception as exc:  # noqa: BLE001 - la medición no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en results/: {type(exc).__name__}: {exc}")


def parse_month(
    parser: argparse.ArgumentParser, text: str | None, default: tuple[int, int]
):
    if text is None:
        return default
    match = MONTH.fullmatch(text)
    if match is None:
        parser.error(f"{text!r} inválido: se espera AAAA-MM")
    return int(match[1]), int(match[2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dates")
    parser.add_argument("--base-run", action="append")
    parser.add_argument("--new-dates")
    parser.add_argument("--new-run", action="append")
    parser.add_argument("--from", dest="first")
    parser.add_argument("--to", dest="last")
    parser.add_argument("--dq-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--family", default=FAMILY)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--baseline-vcpu-s", type=float)
    parser.add_argument("--max-diffs", type=int, default=100)
    ns = parser.parse_args(argv)
    try:
        base_side = make_side("base", ns.base_dates, ns.base_run)
        new_side = make_side("new", ns.new_dates, ns.new_run)
        root = (ns.dq_root or default_root()).rstrip("/")
    except ValueError as exc:
        parser.error(str(exc))
    months = (
        parse_month(parser, ns.first, (0, 1)),
        parse_month(parser, ns.last, (9999, 12)),
    )
    series = (ns.provider, ns.market, ns.asset)
    checks = [SUMMARY, TIMING] if ns.timing else [SUMMARY]

    log(f"compare runs: dq={root} serie={'/'.join(series)}")
    try:
        fs, base = resolve_fs(root)
        files = list_findings(fs, base)
        base_rows = read_side(fs, files, base_side, series, [SUMMARY])
        new_rows = read_side(fs, files, new_side, series, checks)
    except (OSError, ValueError) as exc:
        log(f"FALLO lectura: {type(exc).__name__}: {exc}")
        return 1

    base_cells = cells(base_rows, months, ns.family)
    new_cells = cells(new_rows, months, ns.family)
    log(f"base: {len(base_cells)} pares; nueva: {len(new_cells)} pares")
    result, lines = compare(base_cells, new_cells)
    for line in lines[: ns.max_diffs]:
        log(line)
    if len(lines) > ns.max_diffs:
        log(f"… y {len(lines) - ns.max_diffs} más (--max-diffs {ns.max_diffs})")
    if not base_cells or not new_cells:
        log(
            f"FALLO sin_hallazgos: el lado {'base' if not base_cells else 'nueva'} no tiene {SUMMARY}"
        )
    log(result.line())

    if ns.timing:
        timings = unit_timings(new_rows, months)
        for line in timing_lines(timings, ns.baseline_vcpu_s):
            log(line)
        save_timing_csv(timings)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
