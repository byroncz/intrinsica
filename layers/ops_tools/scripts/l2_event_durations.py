"""Sonda de duración y tamaño de los eventos cerrados de L2, por θ (ITSC-338).

Mide cuánto duran y cuántos ticks tienen los eventos de cada θ, para trazar la línea
entre θ de operación (eventos de horas a días, con análisis de microestructura por
ticks) y θ de régimen (semanas o meses, solo contexto de mercado), y para dimensionar
el lector de tramas y L3: el evento más grande en ticks es la cota de RAM del lector
y la duración dice cuántos meses hacia atrás releería un monthly sin carry-over.

Por θ reporta:
  - eventos cerrados;
  - duración en días (`extreme_time - reference_time`): mediana, p99 y máximo;
  - tamaño en ticks (`extreme_agg_trade_id - reference_agg_trade_id`): mediana, p99 y
    máximo. Es una cota superior: incluye los huecos de id del proveedor;
  - eventos con duración >= 1, 7, 30, 60 y 90 días (exactos);
  - mes y fecha de inicio del evento más largo y del más grande en ticks.

Imprime la tabla por θ; el θ mínimo con al menos un evento >= 90, 60 y 30 días; y el
evento más grande del histórico en ticks y en MiB a 48 B por tick (las cinco columnas
que lee el lector, TRD-L3 §7.1).

Máximos y conteos son exactos. Mediana y p99 son aproximados: salen de un histograma
logarítmico de pasos de 0,5 % (error relativo <= 0,25 %), porque el θ más fino tiene
millones de eventos y retenerlos todos rompería el pico O(lote) de AGENTS.md.

Solo cuenta eventos cerrados: no lee `carry_over.parquet` ni el evento pendiente.
Lee de cada `events.parquet` cuatro columnas (`reference_time`, `extreme_time`,
`reference_agg_trade_id`, `extreme_agg_trade_id`), un archivo a la vez: acumula y lo
suelta antes de abrir el siguiente.

Args (todos opcionales):
  --events-root gs://<proyecto>-dc-events/l2   (por defecto, el del proyecto según
                                                OPS_RESULTS_URI; también una ruta local
                                                con la misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>            (por defecto, `l2_event_durations.csv`
                                                bajo OPS_RESULTS_URI o, en local, en el
                                                directorio actual; una ruta que termina
                                                en `/` es un directorio)

Con `ops-script` el CSV queda en `gs://<proyecto>-ops/results/<ejecución>/`
(OPS_RESULTS_URI): la cuenta del job solo crea objetos bajo `results/`, y `scripts/` es
de solo lectura por diseño. Un `--out gs://` fuera de OPS_RESULTS_URI falla al arrancar,
no después de leer los ~5.450 archivos. El tiempo de L2 está en microsegundos.

"""

import argparse
import csv
import io
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

COLUMNS = [
    "reference_time",
    "extreme_time",
    "reference_agg_trade_id",
    "extreme_agg_trade_id",
]
DAY_US = 86_400_000_000
THRESHOLDS = (1, 7, 30, 60, 90)
BYTES_PER_TICK = 48
LOG_STEP = math.log(1.005)
EVENTS_FILE = re.compile(
    r"/theta=(0\.\d{8})/year=(\d{4})/month=(\d{2})/events\.parquet$"
)
CSV_NAME = "l2_event_durations.csv"
CSV_HEADER = [
    "theta",
    "eventos",
    "dias_mediana",
    "dias_p99",
    "dias_max",
    "ticks_mediana",
    "ticks_p99",
    "ticks_max",
    *[f"ge_{d}d" for d in THRESHOLDS],
    "mes_dias_max",
    "inicio_dias_max",
    "mes_ticks_max",
    "inicio_ticks_max",
]


class Hist:
    """Histograma logarítmico de enteros >= 0: el bin 0 guarda los ceros."""

    def __init__(self) -> None:
        self.bins: dict[int, int] = {}
        self.n = 0
        self.low = 0
        self.high = 0

    def add(self, values: pa.ChunkedArray) -> None:
        # max(v, 1) evita ln(0); los ceros se reasignan al bin 0 y el resto va desde 1.
        floor = pc.floor(
            pc.divide(
                pc.ln(pc.cast(pc.max_element_wise(values, 1), pa.float64())), LOG_STEP
            )
        )
        bins = pc.if_else(pc.equal(values, 0), 0, pc.add(pc.cast(floor, pa.int64()), 1))
        for item in pc.value_counts(bins).to_pylist():
            self.bins[item["values"]] = (
                self.bins.get(item["values"], 0) + item["counts"]
            )
        low, high = pc.min(values).as_py(), pc.max(values).as_py()
        self.low = low if self.n == 0 else min(self.low, low)
        self.high = high if self.n == 0 else max(self.high, high)
        self.n += len(values)

    def quantile(self, q: float) -> float:
        """Valor medio del bin que contiene el cuantil `q`, acotado por mínimo y máximo."""
        target = max(1, math.ceil(q * self.n))
        seen = 0
        for bin_ in sorted(self.bins):
            seen += self.bins[bin_]
            if seen >= target:
                if bin_ == 0:
                    return 0.0
                value = math.exp((bin_ - 1 + 0.5) * LOG_STEP)
                return min(max(value, self.low), self.high)
        raise ValueError("histograma vacío")


class Peak:
    """El valor máximo visto y dónde: mes de la partición y fecha de inicio."""

    def __init__(self) -> None:
        self.value = -1
        self.month = ""
        self.start = ""

    def update(self, values: pa.ChunkedArray, table: pa.Table, month: str) -> None:
        top = pc.max(values).as_py()
        if top > self.value:
            at = pc.index(values, top).as_py()
            micros = table["reference_time"][at].as_py()
            self.value = top
            self.month = month
            self.start = datetime.fromtimestamp(micros / 1e6, UTC).strftime("%Y-%m-%d")


class ThetaStats:
    def __init__(self) -> None:
        self.days = Hist()  # la duración se guarda en µs
        self.ticks = Hist()
        self.ge = dict.fromkeys(THRESHOLDS, 0)
        self.longest = Peak()
        self.largest = Peak()

    def add(self, table: pa.Table, month: str) -> None:
        duration = pc.subtract(table["extreme_time"], table["reference_time"])
        size = pc.subtract(
            table["extreme_agg_trade_id"], table["reference_agg_trade_id"]
        )
        self.days.add(duration)
        self.ticks.add(size)
        for d in THRESHOLDS:
            self.ge[d] += pc.sum(
                pc.cast(pc.greater_equal(duration, d * DAY_US), pa.int64())
            ).as_py()
        self.longest.update(duration, table, month)
        self.largest.update(size, table, month)


def open_fs(root: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta base de una raíz `gs://` o local."""
    if root.startswith("gs://"):
        fs, base = pafs.FileSystem.from_uri(root.rstrip("/"))
        return fs, base
    return pafs.LocalFileSystem(), str(Path(root).resolve())


def default_events_root() -> str:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit("sin --events-root y sin OPS_RESULTS_URI: no sé de qué proyecto es")
    return f"gs://{match[1]}-dc-events/l2"


def default_out() -> str:
    return f"{os.environ.get('OPS_RESULTS_URI', '')}{CSV_NAME}"


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        sys.exit(
            f"FALLO out_fuera_de_results: {out} no está bajo {results}; "
            "el job solo crea objetos ahí"
        )


def event_files(fs: pafs.FileSystem, series: str) -> list[tuple[str, str, str]]:
    """`(θ, año-mes, ruta)` de cada `events.parquet` de la serie, en orden de θ y mes."""
    selector = pafs.FileSelector(series, recursive=True, allow_not_found=True)
    found = []
    for info in fs.get_file_info(selector):
        match = EVENTS_FILE.search(info.path)
        if match and info.type == pafs.FileType.File:
            found.append((match[1], f"{match[2]}-{match[3]}", info.path))
    return sorted(found)


def collect(fs: pafs.FileSystem, series: str) -> tuple[dict[str, ThetaStats], int]:
    stats: dict[str, ThetaStats] = {}
    files = event_files(fs, series)
    for theta, month, path in files:
        table = pq.read_table(path, columns=COLUMNS, filesystem=fs).drop_null()
        if len(table):
            if (
                pc.min(
                    pc.subtract(table["extreme_time"], table["reference_time"])
                ).as_py()
                < 0
                or pc.min(
                    pc.subtract(
                        table["extreme_agg_trade_id"], table["reference_agg_trade_id"]
                    )
                ).as_py()
                < 0
            ):
                sys.exit(f"FALLO extremo_antes_de_la_referencia: θ={theta} {month}")
            stats.setdefault(theta, ThetaStats()).add(table, month)
        del table
    return stats, len(files)


def rows(stats: dict[str, ThetaStats]) -> list[dict]:
    out = []
    for theta, s in sorted(stats.items()):
        out.append(
            {
                "theta": theta,
                "eventos": s.days.n,
                "dias_mediana": s.days.quantile(0.5) / DAY_US,
                "dias_p99": s.days.quantile(0.99) / DAY_US,
                "dias_max": s.days.high / DAY_US,
                "ticks_mediana": s.ticks.quantile(0.5),
                "ticks_p99": s.ticks.quantile(0.99),
                "ticks_max": s.ticks.high,
                **{f"ge_{d}d": s.ge[d] for d in THRESHOLDS},
                "mes_dias_max": s.longest.month,
                "inicio_dias_max": s.longest.start,
                "mes_ticks_max": s.largest.month,
                "inicio_ticks_max": s.largest.start,
            }
        )
    return out


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if value < 100 else f"{value:,.0f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def print_table(table: list[dict]) -> None:
    cells = [[fmt(r[k]) for k in CSV_HEADER] for r in table]
    widths = [
        max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(CSV_HEADER)
    ]
    print("  ".join(h.rjust(w) for h, w in zip(CSV_HEADER, widths, strict=True)))
    for line in cells:
        print("  ".join(c.rjust(w) for c, w in zip(line, widths, strict=True)))


def min_theta(table: list[dict], days: int) -> str:
    """El θ más chico con al menos un evento de `days` días o más (el catálogo sube de θ)."""
    for r in table:
        if r[f"ge_{days}d"] > 0:
            return f"{r['theta']} ({r[f'ge_{days}d']} eventos)"
    return "ninguno"


def print_summary(table: list[dict], stats: dict[str, ThetaStats]) -> None:
    print()
    for days in (90, 60, 30):
        print(f"θ mínimo con eventos >= {days} días: {min_theta(table, days)}")
    theta, best = max(stats.items(), key=lambda item: item[1].largest.value)
    ticks = best.largest.value
    print(
        f"evento más grande en ticks: θ={theta} mes {best.largest.month} "
        f"(inicia {best.largest.start}): {ticks:,} ticks = "
        f"{ticks * BYTES_PER_TICK / 2**20:,.1f} MiB a {BYTES_PER_TICK} B por tick"
    )


def write_csv(table: list[dict], out: str) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(table)
    data = buffer.getvalue().encode()
    if out.endswith("/"):
        out += CSV_NAME
    if out.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(out)
        with fs.open_output_stream(path) as stream:
            stream.write(data)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    ns = parser.parse_args(argv)

    out = ns.out or default_out()
    check_out(out)
    fs, base = open_fs(ns.events_root or default_events_root())
    series = f"{base}/provider={ns.provider}/market={ns.market}/asset={ns.asset}"
    stats, files = collect(fs, series)
    if not stats:
        print(f"FALLO sin_eventos: ningún events.parquet con filas bajo {series}")
        return 1

    table = rows(stats)
    print_table(table)
    print_summary(table, stats)
    events = sum(r["eventos"] for r in table)
    print(f"\nθ={len(table)}; archivos={files}; eventos cerrados={events:,}")
    print(f"tabla: {write_csv(table, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
