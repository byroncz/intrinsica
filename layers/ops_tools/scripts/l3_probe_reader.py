"""Sonda del lector de tramas (`shared/dc_frames`): pared, memoria y bytes sobre un mes real.

Recorre `read_frames` para uno o varios θ y mide lo que decide el criterio de la Épica
E5 (TRD-L3 §10.3, ADR-L3-09): pared por fase, RSS pico, bytes leídos de
`consolidated.parquet` y `events.parquet`, decodificaciones de row group, ticks por
segundo y, por θ, eventos y ticks por fase. Al final compara con los umbrales y sale con
1 si alguno no se cumple o si el lector falla.

Args:
  --theta 0.00100000 | a,b,c | all   (obligatorio; `all` toma los θ que tienen L2 en
                                      todo el rango)
  --from YYYY-MM --to YYYY-MM        (meses de L2, inclusivos; `--to` por defecto = `--from`)
  --l1-root gs://<proyecto>-landing/l1    (por defecto, los del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2   OPS_RESULTS_URI; también rutas locales)
  --provider binance --market spot --asset BTCUSDT
  --max-wall-s N                     (umbral de pared; por defecto 300 con un θ, 1200 con más)
  --max-bytes-ratio 1.1              (con más de un θ, los bytes de L1 leídos no pasan de esto ×
                                      el tamaño de los archivos del mes; ver más abajo)
  --big-event-ticks 100000          (desde cuántos ticks un evento entra a la suma de
                                      eventos abiertos a la vez)
  --sin-contar-io                    (no envuelve los archivos: la pared es la del lector
                                      tal cual, pero sin bytes leídos ni fase de lectura)

Row groups: TRD-L3 §10.3 pide tantas decodificaciones como row groups del mes. El lector salta
por estadísticas los que ningún evento necesita (el tramo tras el último evento cerrado), así
que la sonda exige que ninguno se decodifique más de una vez y reporta cuántos se saltaron.
Los bytes incluyen los pies de Parquet, que el lector abre más de una vez por mes: en un
archivo real pesan nada, en uno de fixtures dominan (por eso `--max-bytes-ratio`).

Las fases se miden así. Lectura: tiempo dentro de las lecturas de archivo (la sonda envuelve
cada archivo para contarlas; serializa las lecturas de un row group, así que la pared
puede ser algo más alta que la del lector sin envolver: `--sin-contar-io` da esa pared).
Decodificación: lo que tarda `read_row_group` sin su lectura. Selección: el resto
(`searchsorted`, slices, validaciones y el conteo de la propia sonda).

Última línea: `sonda lector: thetas=… meses=… eventos=… ticks_l1=… pared_s=… lectura_s=…
decodificación_s=… selección_s=… rss_pico_mib=… bytes_l1=… ticks_s=… veredicto=ok|excedido|fallo`.
Con `OPS_RESULTS_URI` deja además `l3_probe_reader_<fecha>.csv` con la tabla por θ.

Solo lee. Dónde corre: con `ops-script` (la service account lee `l1/` y `l2/`) o en local con
`uv run`.
"""

import argparse
import os
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from dc_frames import FramesError, FramesInputError, lake, read_frames
from pyutils import resolve_fs

HISTORY_TICKS = 4_051_000_000  # TRD-L3 §10.3, criterio 6
GIB = 1 << 30
MIB = 1 << 20
# 3 columnas de 8 y 16 B más un bit: lo que pesa un tick en una trama.
BYTES_PER_TICK = 8 + 16 + 16 + 1 / 8
PROGRESS_EVERY_S = 30


@dataclass
class Stats:
    """Lo que miden los archivos envueltos; lo llenan `_CountedFile` y `_ProbeFile`."""

    count_io: bool = True
    io_s: float = 0.0
    io_in_rg_s: float = 0.0
    rg_s: float = 0.0
    bytes: dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0})
    expected: dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0})
    decodes: dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0})
    l1_ticks: int = 0
    # ruta de L1 -> (bytes del archivo, row groups, decodificaciones)
    l1_files: dict[str, list[int]] = field(default_factory=dict)
    l1_seen: set[tuple[str, int]] = field(default_factory=set)
    l1_repeated: int = 0
    started: float = 0.0
    last_progress: float = 0.0
    events: int = 0


STATS = Stats()


def rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def log(message: str) -> None:
    print(message, flush=True)


class _CountedFile:
    """Un archivo de solo lectura que cuenta los bytes y el tiempo de cada lectura."""

    def __init__(self, inner, kind: str) -> None:
        self._inner = inner
        self._kind = kind

    def read(self, nbytes: int = -1) -> bytes:
        start = time.perf_counter()
        data = (
            self._inner.read()
            if nbytes is None or nbytes < 0
            else self._inner.read(nbytes)
        )
        STATS.io_s += time.perf_counter() - start
        STATS.bytes[self._kind] += len(data)
        return data

    def seek(self, position: int, whence: int = 0) -> int:
        return self._inner.seek(position, whence)

    def tell(self) -> int:
        return self._inner.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def close(self) -> None:
        self._inner.close()


class _ProbeFile(pq.ParquetFile):
    """`ParquetFile` que mide cada `read_row_group`: tiempo, filas y bytes esperados."""

    def __init__(self, source, kind: str, path: str, **kwargs) -> None:
        super().__init__(source, **kwargs)
        self._kind = kind
        self._probe_path = path

    def close(self, force: bool = False) -> None:
        super().close(force=True)

    def read_row_group(self, i, columns=None, **kwargs):
        io_before = STATS.io_s
        start = time.perf_counter()
        table = super().read_row_group(i, columns=columns, **kwargs)
        STATS.rg_s += time.perf_counter() - start
        STATS.io_in_rg_s += STATS.io_s - io_before
        STATS.decodes[self._kind] += 1
        group = self.metadata.row_group(i)
        names = self.schema_arrow.names
        for name in columns or names:
            STATS.expected[self._kind] += group.column(
                names.index(name)
            ).total_compressed_size
        if self._kind == "l1":
            STATS.l1_ticks += table.num_rows
            STATS.l1_files[self._probe_path][2] += 1
            if (self._probe_path, i) in STATS.l1_seen:
                STATS.l1_repeated += 1
            STATS.l1_seen.add((self._probe_path, i))
        progress()
        return table


def probe_open_parquet(path: str, what: str) -> pq.ParquetFile:
    """Sustituto de `dc_frames.lake.open_parquet` que mide; hace lo mismo que el original."""
    fs, resolved = resolve_fs(path)
    info = fs.get_file_info(resolved)
    if info.type != pafs.FileType.File:
        raise FramesInputError(f"falta {what}: {path}")
    kind = "l1" if path.endswith(lake.CONSOLIDATED) else "l2"
    if STATS.count_io:
        source = pa.PythonFile(
            _CountedFile(fs.open_input_file(resolved), kind), mode="r"
        )
        parquet = _ProbeFile(source, kind, path)
    else:
        parquet = _ProbeFile(resolved, kind, path, filesystem=fs)
    if kind == "l1" and path not in STATS.l1_files:
        STATS.l1_files[path] = [info.size, parquet.num_row_groups, 0]
    return parquet


def progress() -> None:
    now = time.perf_counter()
    if now - STATS.last_progress < PROGRESS_EVERY_S:
        return
    STATS.last_progress = now
    log(
        f"  … {now - STATS.started:.0f} s: row groups de L1={STATS.decodes['l1']} "
        f"ticks_l1={STATS.l1_ticks} eventos={STATS.events} rss={rss_bytes() / MIB:.0f} MiB"
    )


@dataclass
class ThetaStats:
    events: int = 0
    confirmation: int = 0
    overshoot: int = 0
    biggest_ticks: int = 0
    biggest_bytes: int = 0


def frame_bytes(*frames) -> int:
    return sum(
        array.nbytes
        for frame in frames
        for array in (
            frame.transact_time,
            frame.price,
            frame.quantity,
            frame.is_buyer_maker,
        )
    )


def default_roots() -> tuple[str, str]:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit(
            "sin --l1-root o --l2-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    project = match[1]
    return f"gs://{project}-landing/l1", f"gs://{project}-dc-events/l2"


def parse_month(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if match is None or not 1 <= int(match[2]) <= 12:
        sys.exit(f"mes inválido {text!r}: se espera YYYY-MM")
    return int(match[1]), int(match[2])


def lake_thetas(l2_root: str, key: tuple[str, str, str], first, last) -> list[str]:
    """Los θ de L2 que tienen `events.parquet` en todos los meses de `first`..`last`."""
    fs, base = resolve_fs(l2_root)
    provider, market, asset = key
    prefix = f"{base.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
    found = []
    for info in fs.get_file_info(pafs.FileSelector(prefix)):
        match = re.fullmatch(r"theta=(0\.\d{8})", info.base_name)
        if match is not None and info.type == pafs.FileType.Directory:
            found.append(match[1])
    complete = [
        theta
        for theta in sorted(found)
        if all(
            fs.get_file_info(lake.l2_path(base, key, theta, month)).type
            == pafs.FileType.File
            for month in lake.months(first, last)
        )
    ]
    if len(complete) != len(found):
        skipped = sorted(set(found) - set(complete))
        log(f"θ sin L2 en todo el rango, omitidos: {', '.join(skipped)}")
    return complete


def parse_thetas(text: str, l2_root, key, first, last) -> list[str]:
    if text == "all":
        thetas = lake_thetas(l2_root, key, first, last)
        if not thetas:
            sys.exit("--theta all: ningún θ tiene L2 en todo el rango")
        return thetas
    thetas = []
    for part in text.split(","):
        theta = Decimal(part)
        scaled = theta.quantize(Decimal("1e-8"))
        if scaled != theta:
            sys.exit(f"θ {part} no cabe en 8 decimales, que es la partición de L2")
        thetas.append(format(scaled, "f"))
    return thetas


def open_sweep(bigs: list[tuple[int, int, int]]) -> int:
    """La mayor suma de ticks de eventos grandes abiertos a la vez (cota; ver el informe)."""
    if not bigs:
        return 0
    starts, ends, ticks = (
        np.array(column, dtype=np.int64) for column in zip(*bigs, strict=True)
    )
    position = np.concatenate([starts, ends])
    delta = np.concatenate([ticks, -ticks])
    # A igual posición, primero los cierres: un evento que termina donde otro empieza no se suma.
    order = np.lexsort((delta, position))
    return int(np.cumsum(delta[order]).max())


def run(ns: argparse.Namespace) -> int:
    default_l1, default_l2 = (
        (None, None) if ns.l1_root and ns.l2_root else default_roots()
    )
    l1_root = ns.l1_root or default_l1
    l2_root = ns.l2_root or default_l2
    key = (ns.provider, ns.market, ns.asset)
    first = parse_month(ns.start)
    last = parse_month(ns.end or ns.start)
    thetas = parse_thetas(ns.theta, l2_root, key, first, last)
    multi = len(thetas) > 1
    max_wall = ns.max_wall_s if ns.max_wall_s is not None else (1200 if multi else 300)
    months = [f"{y:04d}-{m:02d}" for y, m in lake.months(first, last)]
    log(
        f"sonda lector: {len(thetas)} θ, meses {months[0]}..{months[-1]}, l1={l1_root}, l2={l2_root}"
    )

    STATS.count_io = not ns.sin_contar_io
    lake.open_parquet = probe_open_parquet
    per_theta: dict[Decimal, ThetaStats] = {}
    bigs: list[tuple[int, int, int]] = []
    error = None
    STATS.started = STATS.last_progress = time.perf_counter()
    try:
        for event in read_frames(thetas, l1_root, l2_root, first, last, *key):
            confirmation = len(event.confirmation.transact_time)
            overshoot = len(event.overshoot.transact_time)
            ticks = confirmation + overshoot
            stats = per_theta.get(event.theta)
            if stats is None:
                stats = per_theta[event.theta] = ThetaStats()
            stats.events += 1
            stats.confirmation += confirmation
            stats.overshoot += overshoot
            if ticks > stats.biggest_ticks:
                stats.biggest_ticks = ticks
                stats.biggest_bytes = frame_bytes(event.confirmation, event.overshoot)
            if ticks >= ns.big_event_ticks:
                row = event.event
                bigs.append(
                    (
                        row.column("reference_agg_trade_id")[0].as_py(),
                        row.column("extreme_agg_trade_id")[0].as_py(),
                        ticks,
                    )
                )
            STATS.events += 1
    except FramesError as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - STATS.started

    decode = STATS.rg_s - STATS.io_in_rg_s
    reading = STATS.io_s if STATS.count_io else None
    selection = wall - STATS.rg_s - (STATS.io_s - STATS.io_in_rg_s)
    if STATS.count_io:
        phases = f"lectura={STATS.io_s:.1f} s, decodificación={decode:.1f} s, selección={selection:.1f} s"
    else:
        phases = f"lectura+decodificación={STATS.rg_s:.1f} s, selección={wall - STATS.rg_s:.1f} s"

    header = f"{'θ':>12} {'eventos':>10} {'ticks_confirmación':>19} {'ticks_overshoot':>16} {'evento_máx_ticks':>17} {'evento_máx_MiB':>15}"
    lines = [
        f"{theta:>12} {s.events:>10} {s.confirmation:>19} {s.overshoot:>16} "
        f"{s.biggest_ticks:>17} {s.biggest_bytes / MIB:>15.1f}"
        for theta, s in sorted(per_theta.items())
    ]
    log(header)
    for line in lines:
        log(line)
    save_csv(header, lines)

    target = {lake.l1_path(l1_root, key, m) for m in lake.months(first, last)}
    month_files = [v for path, v in STATS.l1_files.items() if path in target]
    month_bytes = sum(v[0] for v in month_files)
    month_groups = sum(v[1] for v in month_files)
    month_decodes = sum(v[2] for v in month_files)
    earlier_decodes = sum(
        v[2] for path, v in STATS.l1_files.items() if path not in target
    )
    delivered = sum(s.confirmation + s.overshoot for s in per_theta.values())
    events = sum(s.events for s in per_theta.values())
    rate = STATS.l1_ticks / wall if wall else 0.0
    biggest = max((s.biggest_bytes for s in per_theta.values()), default=0)
    big_open = open_sweep(bigs)
    small_open = len(thetas) * (ns.big_event_ticks - 1)
    rss = rss_bytes()

    log(f"pared por fase: {phases}; total {wall:.1f} s")
    log(
        f"ticks de L1 decodificados={STATS.l1_ticks} ({rate / 1e6:.2f} M/s); "
        f"ticks entregados={delivered} ({delivered / wall / 1e6 if wall else 0:.2f} M/s); eventos={events}"
    )
    log(
        f"row groups de L1: decodificados del mes={month_decodes} de {month_groups} "
        f"(saltados por estadísticas={month_groups - month_decodes}), "
        f"de meses anteriores={earlier_decodes}, repetidos={STATS.l1_repeated}; "
        f"de L2={STATS.decodes['l2']}"
    )
    if STATS.count_io:
        log(
            f"bytes leídos: L1={STATS.bytes['l1']} ({STATS.bytes['l1'] / max(month_bytes, 1):.3f} × "
            f"los {month_bytes} del mes), L2={STATS.bytes['l2']}; por metadatos L1={STATS.expected['l1']}"
        )
    else:
        log(
            f"bytes por metadatos (no medidos): L1={STATS.expected['l1']}, L2={STATS.expected['l2']}"
        )
    log(
        f"RSS pico={rss / MIB:.0f} MiB; evento más grande={biggest / MIB:.1f} MiB; "
        f"suma máxima de eventos abiertos ≥{ns.big_event_ticks} ticks={big_open} "
        f"({big_open * BYTES_PER_TICK / MIB:.0f} MiB), los menores aportan a lo sumo {small_open}"
    )
    if rate:
        log(
            f"extrapolación al histórico ({HISTORY_TICKS // 1_000_000} M de ticks) a esa tasa: "
            f"{HISTORY_TICKS / rate / 3600:.1f} h"
        )

    checks: list[tuple[str, bool]] = []
    if error is None:
        checks.append((f"pared {wall:.0f} s ≤ {max_wall:.0f} s", wall <= max_wall))
        if not multi:
            limit = GIB + biggest
            checks.append(
                (
                    f"RSS pico {rss / MIB:.0f} MiB ≤ 1 GiB + evento más grande ({limit / MIB:.0f} MiB)",
                    rss <= limit,
                )
            )
        else:
            checks.append(
                (
                    f"row groups de L1 decodificados más de una vez: {STATS.l1_repeated}",
                    STATS.l1_repeated == 0,
                )
            )
            if STATS.count_io:
                checks.append(
                    (
                        f"bytes de L1 {STATS.bytes['l1']} ≤ {ns.max_bytes_ratio} × {month_bytes}",
                        STATS.bytes["l1"] <= ns.max_bytes_ratio * month_bytes,
                    )
                )
    failed = error is not None
    if error is not None:
        log(f"FALLO lector: {error}")
    for text, ok in checks:
        log(f"{'ok' if ok else 'FALLO'} {text}")
        failed |= not ok
    verdict = "fallo" if error is not None else ("excedido" if failed else "ok")
    log(
        f"sonda lector: thetas={len(thetas)} meses={months[0]}..{months[-1]} eventos={events} "
        f"ticks_l1={STATS.l1_ticks} pared_s={wall:.1f} lectura_s={'n/d' if reading is None else f'{reading:.1f}'} "
        f"decodificación_s={'n/d' if reading is None else f'{decode:.1f}'} selección_s={selection:.1f} "
        f"rss_pico_mib={rss / MIB:.0f} bytes_l1={STATS.bytes['l1'] if STATS.count_io else 'n/d'} "
        f"ticks_s={rate:.0f} veredicto={verdict}"
    )
    return 1 if failed else 0


def save_csv(header: str, lines: list[str]) -> None:
    """Deja la tabla por θ en `OPS_RESULTS_URI`; si no se puede, la salida estándar basta."""
    uri = os.environ.get("OPS_RESULTS_URI")
    if not uri:
        return
    try:
        from google.cloud import storage

        bucket, _, prefix = uri.removeprefix("gs://").partition("/")
        name = f"{prefix}l3_probe_reader_{time.strftime('%Y%m%dT%H%M%S')}.csv"
        body = "\n".join(",".join(row.split()) for row in [header, *lines]) + "\n"
        storage.Client().bucket(bucket).blob(name).upload_from_string(body)
        log(f"tabla por θ en gs://{bucket}/{name}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en results/: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end")
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--max-bytes-ratio", type=float, default=1.1)
    parser.add_argument("--big-event-ticks", type=int, default=100_000)
    parser.add_argument("--sin-contar-io", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
