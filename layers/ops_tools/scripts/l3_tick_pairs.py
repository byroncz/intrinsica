"""Sonda previa de la familia J de L3: pares de ticks seguidos de L1 (ITSC-433).

La familia J relaciona dos series del mismo tick (precio, cantidad, lado, tiempo). Dos mecanismos
las ligan por construcción: el **barrido** (una orden que queda como varios ticks a la misma hora)
y el **rebote** (el precio salta entre las dos puntas). Esta sonda mide cuánto pesa cada uno. No mira
eventos ni fases: recorre el mes entero. Es una medición, no una prueba contra un nulo: no lleva
veredicto.

Para cada mes de `--months` lee del `consolidated.parquet` de L1 `agg_trade_id`, `price`,
`is_buyer_maker` y `transact_time`, un row group a la vez y en una sola pasada, y trabaja sobre los
**pares de ticks seguidos**, incluido el par que une el último tick de un row group con el primero
del siguiente. De cada par toma tres cosas:

  - el lado de cada tick: `is_buyer_maker` verdadero es que el agresor vendió (`venta`); falso, que
    compró (`compra`). La transición es `compra→compra`, `compra→venta`, `venta→compra` o
    `venta→venta`;
  - el cambio de precio, en enteros exactos (el `decimal128(18, 8)` sin escalar): `baja`, `igual`
    o `sube`;
  - la espera entre los dos ticks: cero si tienen la misma `transact_time`; «mismo ms» si caen en el
    mismo milisegundo (`transact_time // 1000`). Hasta 2023 L1 viene de milisegundos convertidos a µs:
    allí las dos coinciden; en 2026 no.

Con eso arma, por mes, tres tablas de 4 × 3 (transición de lado por cambio de precio): `todos` los
pares, los de espera cero (`cero`) y los del mismo milisegundo (`ms`), y estas medidas:
  - `ticks`, `pares` (ticks − 1), `espera_cero` y `mismo_ms`;
  - `frac_cero` y `frac_mismo_ms`: `espera_cero` y `mismo_ms` sobre `pares`;
  - `frac_escalera_cero` y `frac_escalera_ms`: de los pares de su conjunto, los del mismo lado con el
    precio en el sentido del lado (`compra→compra` y `sube`, `venta→venta` y `baja`);
  - `reversion`: de los cambios de precio distintos de cero, tomados en orden y saltando los ceros, la
    fracción cuyo signo es contrario al del anterior (el signo se arrastra entre row groups);
  - `autocorr_1`: autocorrelación de un rezago del cambio de precio sobre todos los pares:
    Σ (dᵢ − m)(dᵢ₊₁ − m) / Σ (dᵢ − m)², con m la media de todos los cambios. Sale en float64 (suma por
    row group); los conteos y las tablas son exactos.
Una fracción sin denominador (sin pares, o un conjunto vacío) sale `-`; la `autocorr_1` también sale
`-` si todos los cambios son iguales (varianza cero).

Args:
  --months 2023-03,2026-08                         (obligatorio; una lista de meses YYYY-MM)
  --landing-root gs://<proyecto>-landing/l1        (por defecto, el del proyecto según
                                                    OPS_RESULTS_URI; también una ruta local con
                                                    la misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_tick_pairs.csv` bajo OPS_RESULTS_URI o, en
                                       local, en el directorio actual; una ruta que termina en `/`
                                       es un directorio)

No importa ninguna capa: una importación cambiaría `uv.lock`. Abre el `consolidated.parquet` de cada
mes por su ruta completa, nunca un dataset sobre un prefijo, que pide un objeto sin barra y la
condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3, «La trampa
del 403 por prefijo»).

Memoria: lee un row group a la vez y de un tick solo conserva, entre row groups, el último. Las
tablas son 3 × 12 enteros y los acumuladores, unos pocos números. El pico es O(row group), no O(mes).

Salida: una línea por mes en cuanto lo cierra, con todas las medidas, y su tabla de pares; el CSV (una
fila por `mes`, `medida` y `valor`; las celdas de las tablas se llaman `<conjunto>.<transición>.<cambio>`,
por ejemplo `todos.compra_venta.sube`); la ruta del CSV y, como última línea, `tick pairs: meses=…
ticks=… pared_s=…`, con los meses medidos y los ticks de todos ellos. Un mes que no se puede medir
imprime `FALLO …` y la corrida sigue con el siguiente.

Códigos: 0 si midió todos los meses; 1 si falta un mes, la lectura falla, `agg_trade_id` no crece o
`transact_time` decrece; 2 por uso. Solo lee; lo único que escribe es el CSV de `--out`.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

COLUMNS = ["agg_trade_id", "price", "is_buyer_maker", "transact_time"]
SETS = ("todos", "cero", "ms")
TRANSITIONS = ("compra_compra", "compra_venta", "venta_compra", "venta_venta")
CHANGES = ("baja", "igual", "sube")
SCALARS = (
    "ticks",
    "pares",
    "espera_cero",
    "mismo_ms",
    "frac_cero",
    "frac_mismo_ms",
    "frac_escalera_cero",
    "frac_escalera_ms",
    "reversion",
    "autocorr_1",
)
CSV_HEADER = ["mes", "medida", "valor"]
MONTH = re.compile(r"(\d{4})-(0[1-9]|1[0-2])")


def log(message: str) -> None:
    print(message, flush=True)


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        return pafs.FileSystem.from_uri(uri)
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def parse_months(text: str) -> list[str]:
    months = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        if MONTH.fullmatch(part) is None:
            raise ValueError(f"--months: {part!r} inválido: se espera YYYY-MM")
        months.append(part)
    if not months:
        raise ValueError("--months: falta al menos un mes YYYY-MM")
    if len(set(months)) != len(months):
        raise ValueError("--months: hay meses repetidos")
    return sorted(months)


def single(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    if isinstance(column, pa.Array):
        return column
    return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()


def int64s(table: pa.Table, name: str) -> np.ndarray:
    """Una columna `int64` sin nulos, vista sin copia."""
    array = single(table.column(name))
    if not pa.types.is_int64(array.type) or array.null_count:
        raise ValueError(f"{name} debe ser int64 sin nulos")
    words = np.frombuffer(array.buffers()[1], dtype="<i8")
    return words[array.offset : array.offset + len(array)]


def prices(table: pa.Table) -> np.ndarray:
    """El `decimal128(·, 8)` sin nulos como su entero sin escalar (la palabra baja), sin copia."""
    array = single(table.column("price"))
    if (
        array.null_count
        or not pa.types.is_decimal128(array.type)
        or array.type.scale != 8
        or array.type.precision > 18
    ):
        raise ValueError("price debe ser decimal128(18, 8) sin nulos")
    words = np.frombuffer(array.buffers()[1], dtype="<i8")
    return words[2 * array.offset : 2 * (array.offset + len(array)) : 2]


def sides(table: pa.Table) -> np.ndarray:
    """`is_buyer_maker` como booleanos: verdadero es que el agresor vendió."""
    array = single(table.column("is_buyer_maker"))
    if not pa.types.is_boolean(array.type) or array.null_count:
        raise ValueError("is_buyer_maker debe ser boolean sin nulos")
    return array.to_numpy(zero_copy_only=False)


class Pairs:
    """Los pares de ticks seguidos del mes, acumulados un row group a la vez."""

    def __init__(self) -> None:
        self.ticks = 0
        self.group = -1
        # El último tick visto: (agg_trade_id, price, transact_time, vendió).
        self.last: tuple[int, int, int, bool] | None = None
        # Una fila por conjunto (todos, cero, ms); 12 celdas = lado previo × lado actual × cambio.
        self.table = np.zeros((len(SETS), 12), dtype=np.int64)
        # Reversión: signo del último cambio distinto de cero (0 = ninguno) y cuentas.
        self.sign = 0
        self.flips = 0
        self.moves = 0
        # Autocorrelación: sumas de los cambios (float64), sus extremos y su rango.
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.sx = 0.0
        self.first = 0.0
        self.final = 0.0
        self.low = 0.0
        self.high = 0.0

    def add(
        self,
        group: int,
        ids: np.ndarray,
        price: np.ndarray,
        times: np.ndarray,
        sold: np.ndarray,
    ) -> None:
        """Suma un row group: sus ticks, ya ordenados, y el par que lo une al anterior."""
        count = len(ids)
        if not count:
            return
        self.group = group
        if (count > 1 and int(np.diff(ids).min()) <= 0) or (
            self.last is not None and int(ids[0]) <= self.last[0]
        ):
            raise ValueError(
                f"row group {group}: agg_trade_id no crece; L1 debe venir ordenado"
            )
        if self.last is not None:
            _, last_price, last_time, last_sold = self.last
            self._pairs(
                np.array([last_price]),
                price[:1],
                np.array([last_time]),
                times[:1],
                np.array([last_sold]),
                sold[:1],
            )
        if count > 1:
            self._pairs(
                price[:-1], price[1:], times[:-1], times[1:], sold[:-1], sold[1:]
            )
        self.last = (int(ids[-1]), int(price[-1]), int(times[-1]), bool(sold[-1]))
        self.ticks += count

    def _pairs(self, p0, p1, t0, t1, s0, s1) -> None:
        """Suma los pares `(0 → 1)`; cada argumento trae el mismo largo y el orden del mes."""
        wait = t1 - t0
        if int(wait.min()) < 0:
            raise ValueError(
                f"row group {self.group}: transact_time decrece; L1 debe venir ordenado"
            )
        step = p1 - p0
        sign = np.sign(step)
        # Celda: (lado previo × 2 + lado actual) × 3 + (cambio: 0 baja, 1 igual, 2 sube).
        cell = (s0.astype(np.int64) * 2 + s1) * 3 + sign + 1
        self.table[0] += np.bincount(cell, minlength=12)
        self.table[1] += np.bincount(cell[wait == 0], minlength=12)
        self.table[2] += np.bincount(cell[(t1 // 1000) == (t0 // 1000)], minlength=12)
        self._reversion(sign)
        self._autocorr(step)

    def _reversion(self, sign: np.ndarray) -> None:
        moves = sign[sign != 0]
        if not len(moves):
            return
        if self.sign:
            self.moves += 1
            self.flips += int(moves[0] != self.sign)
        if len(moves) > 1:
            self.moves += len(moves) - 1
            self.flips += int(np.count_nonzero(moves[1:] != moves[:-1]))
        self.sign = int(moves[-1])

    def _autocorr(self, step: np.ndarray) -> None:
        change = step.astype(np.float64)
        if not self.n:
            self.first = float(change[0])
            self.low = float(change.min())
            self.high = float(change.max())
        else:
            self.sx += self.final * float(change[0])
            self.low = min(self.low, float(change.min()))
            self.high = max(self.high, float(change.max()))
        if len(change) > 1:
            self.sx += float(np.dot(change[:-1], change[1:]))
        self.s1 += float(change.sum())
        self.s2 += float(np.dot(change, change))
        self.final = float(change[-1])
        self.n += len(change)

    def autocorr(self) -> float | None:
        if self.n < 2 or self.low == self.high:
            return None
        mean = self.s1 / self.n
        numerator = (
            self.sx
            - mean * ((self.s1 - self.final) + (self.s1 - self.first))
            + (self.n - 1) * mean * mean
        )
        denominator = self.s2 - self.n * mean * mean
        return numerator / denominator if denominator > 0 else None

    def summary(self) -> dict:
        """Conteos exactos, tablas y derivados del mes."""
        tables = {
            name: [
                [int(v) for v in row.reshape(4, 3)[i]]
                for i in range(4)  # una fila por transición de lado
            ]
            for name, row in zip(SETS, self.table, strict=True)
        }
        pairs = int(self.table[0].sum())
        zero = int(self.table[1].sum())
        same_ms = int(self.table[2].sum())

        def ratio(part: int, whole: int) -> float | None:
            return part / whole if whole else None

        def ladder(name: str) -> int:
            cells = tables[name]
            return cells[0][2] + cells[3][0]  # compra→compra y sube; venta→venta y baja

        return {
            "ticks": self.ticks,
            "pares": pairs,
            "espera_cero": zero,
            "mismo_ms": same_ms,
            "frac_cero": ratio(zero, pairs),
            "frac_mismo_ms": ratio(same_ms, pairs),
            "frac_escalera_cero": ratio(ladder("cero"), zero),
            "frac_escalera_ms": ratio(ladder("ms"), same_ms),
            "reversion": ratio(self.flips, self.moves),
            "autocorr_1": self.autocorr(),
            "tablas": tables,
        }


def measure(path: str) -> dict:
    """Recorre el archivo de un mes, un row group a la vez; devuelve el resumen."""
    fs, resolved = open_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        raise FileNotFoundError(path)
    pairs = Pairs()
    with fs.open_input_file(resolved) as source:
        parquet = pq.ParquetFile(source)
        for group in range(parquet.num_row_groups):
            if not parquet.metadata.row_group(group).num_rows:
                continue
            table = parquet.read_row_group(group, columns=COLUMNS)
            pairs.add(
                group,
                int64s(table, "agg_trade_id"),
                prices(table),
                int64s(table, "transact_time"),
                sides(table),
            )
            del table
    return pairs.summary()


def cell_text(value, digits: int = 6) -> str:
    if value is None:
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def month_line(month: str, summary: dict, wall: float) -> str:
    cells = " ".join(f"{n}={cell_text(summary[n])}" for n in SCALARS)
    return f"mes {month}: {cells} pared_s={wall:.1f}"


def table_lines(summary: dict) -> list[str]:
    lines = []
    for name in SETS:
        for transition, row in zip(TRANSITIONS, summary["tablas"][name], strict=True):
            counts = " ".join(f"{c}={v}" for c, v in zip(CHANGES, row, strict=True))
            lines.append(f"  {name:<5} {transition.replace('_', '→'):<14} {counts}")
    return lines


def csv_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for month, summary in results.items():
        for name in SCALARS:
            value = summary[name]
            if value is None:
                value = ""
            elif not isinstance(value, int):
                value = format(value, ".10g")
            rows.append({"mes": month, "medida": name, "valor": value})
        for name in SETS:
            for transition, row in zip(
                TRANSITIONS, summary["tablas"][name], strict=True
            ):
                for change, count in zip(CHANGES, row, strict=True):
                    rows.append(
                        {
                            "mes": month,
                            "medida": f"{name}.{transition}.{change}",
                            "valor": count,
                        }
                    )
    return rows


def write_csv(rows: list[dict], out: str) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if out.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(out)
        with fs.open_output_stream(path) as stream:
            stream.write(data)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    try:
        months = parse_months(ns.months)
    except ValueError as exc:
        parser.error(str(exc))
    name_project = project()
    if not ns.landing_root and name_project is None:
        parser.error(
            "sin --landing-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    root = ns.landing_root or f"gs://{name_project}-landing/l1"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}l3_tick_pairs.csv"
    if out.endswith("/"):
        out += "l3_tick_pairs.csv"
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))

    started = time.perf_counter()
    results: dict[str, dict] = {}
    failed = 0
    for month in months:
        year, number = month.split("-")
        path = (
            f"{root.rstrip('/')}/provider={ns.provider}/market={ns.market}/asset={ns.asset}"
            f"/year={year}/month={number}/consolidated.parquet"
        )
        begun = time.perf_counter()
        try:
            summary = measure(path)
        except FileNotFoundError:
            log(f"FALLO mes {month}: falta el archivo {path}")
            failed += 1
        except (OSError, ValueError, pa.ArrowException) as exc:
            log(
                f"FALLO mes {month}: no se pudo leer {path}: {type(exc).__name__}: {exc}"
            )
            failed += 1
        else:
            results[month] = summary
            log(month_line(month, summary, time.perf_counter() - begun))
            for line in table_lines(summary):
                log(line)

    if results:
        try:
            write_csv(csv_rows(results), out)
            log(f"tabla en {out}")
        except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
            log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    ticks = sum(summary["ticks"] for summary in results.values())
    log(
        f"tick pairs: meses={len(results)} ticks={ticks} "
        f"pared_s={time.perf_counter() - started:.1f}"
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--months", required=True)
    parser.add_argument("--landing-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
