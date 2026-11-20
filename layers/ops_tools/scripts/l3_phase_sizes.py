"""Cobertura por tamaño de fase de `summaries.parquet`, por θ (ITSC-431).

Para cada θ del catálogo y cada mes con `summaries.parquet` en L3, lee **solo** `c_ticks` y
`os_ticks`, un row group a la vez, y cuenta cuántas fases (eventos) alcanzan cada tamaño en ticks.
No toca ticks ni calcula ninguna medida: sirve para saber sobre cuántas fases se podría calcular
una medida que necesita series largas, y cuánto trabajo sería.

Por θ y por fase (`c` y `os`) cuenta, exacto:
  - `fases`: en `c`, todas las filas; en `os`, las de `os_ticks > 0`. Aparte, `os_vacias`
    (las de `os_ticks = 0`, overshoot vacío);
  - `ge_<N>`: fases con ticks >= N, para N = 16, 64, 256, 1024, 4096, 10000, 16384 y 65536;
  - `puntos` = Σ min(ticks, --sample) y `pares` = Σ min(ticks, --sample)²: la base del costo de
    una medida lineal y de una cuadrática por fase, con `--sample` en 10000 por defecto.

Args (todos opcionales):
  --catalog gs://<proyecto>-manifest/l2/thetas.yaml   (el catálogo único de θ; también local)
  --l3-root gs://<proyecto>-dc-descriptors/l3         (también una ruta local, disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --from YYYY-MM --to YYYY-MM   (meses, ambos incluidos; por defecto todo el histórico)
  --sample 10000   (tope de ticks por fase para `puntos` y `pares`, de 1 a 1 000 000)
  --readers 8      (archivos leídos a la vez, uno por lector)
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_phase_sizes.csv` bajo OPS_RESULTS_URI
                                       o, en local, en el directorio actual; una ruta que termina
                                       en `/` es un directorio)
  Sin argumentos, con OPS_RESULTS_URI, deduce el proyecto y usa los dos de arriba.

El histórico va de `--from` (o el primer mes con archivo) a `--to` (o el último). Un `(θ, mes)` de
ese rango sin `summaries.parquet`, o cuyo archivo no se puede leer, es un faltante.

No importa ninguna capa: una importación cambiaría `uv.lock`. Lee cada archivo por su ruta
completa (un listado de la raíz y una apertura por archivo), nunca un dataset sobre un prefijo,
que pide un objeto sin barra y la condición IAM por prefijo de `ops-script` lo niega con 403
(runbook de operación de L3, «La trampa del 403 por prefijo»).

Lectores: un archivo por lector. Cada lector devuelve los conteos de su archivo y la suma se hace
en el hilo principal, sin estado compartido entre hilos. Memoria: a lo sumo un row group de dos
columnas por lector; ningún dato sobrevive a su archivo, solo sus conteos.

Salida: una línea por θ en cuanto lo cierra (el job vence a los 3.600 s: si no alcanza, se lanza
por tramos con `--from` y `--to` y queda lo impreso), una fila de totales, la ruta del CSV y, como
última línea, `phase sizes: thetas=… archivos=… faltantes=… fases_c=… fases_os=…`. `thetas` son los
θ con al menos un archivo leído; `archivos`, los leídos; `faltantes`, los `(θ, mes)` sin archivo o
ilegibles.

Códigos: 0 si leyó al menos un archivo; 1 si ninguno, o si no se puede listar o leer el catálogo;
2 por uso. Solo lee; lo único que escribe es el CSV de `--out`.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

THRESHOLDS = (16, 64, 256, 1024, 4096, 10_000, 16_384, 65_536)
PHASES = ("c", "os")
COLUMNS = ("c_ticks", "os_ticks")
# El orden de los conteos de una fase: `fases`, `vacias`, un `ge_<N>` por umbral, `puntos`, `pares`.
METRICS = ("fases", "vacias", *(f"ge_{n}" for n in THRESHOLDS), "puntos", "pares")
SAMPLE_MAX = (
    1_000_000  # con `min(ticks, sample)²` y un row group, la suma cabe en int64
)
CSV_HEADER = [
    "theta",
    "fase",
    "archivos",
    "faltantes",
    "fases",
    "os_vacias",
    *(f"ge_{n}" for n in THRESHOLDS),
    "puntos",
    "pares",
]
FILE = re.compile(r"/theta=(0\.\d{8})/year=(\d{4})/month=(\d{2})/summaries\.parquet$")
MONTH = re.compile(r"(\d{4})-(0[1-9]|1[0-2])$")

Counts = dict[str, list[int]]


class InputError(Exception):
    """No se pudo listar o leer una raíz o el catálogo: sale con código 1."""


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


def series_root(root: str, key: tuple[str, str, str]) -> str:
    provider, market, asset = key
    return f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"


def gcs_names(bucket: str, prefix: str) -> list[str]:
    """Los nombres de los objetos de `gs://bucket/prefix`, listados por la API de GCS."""
    from google.cloud import storage

    return [b.name for b in storage.Client().list_blobs(bucket, prefix=prefix)]


def list_files(root: str) -> list[str]:
    """Todos los archivos bajo `root`, de forma recursiva: objetos por prefijo en GCS."""
    if not root.startswith("gs://"):
        return [str(p) for p in Path(root).rglob("*") if p.is_file()]
    bucket, _, prefix = root.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    try:
        return [f"gs://{bucket}/{name}" for name in gcs_names(bucket, prefix)]
    except Exception as exc:
        raise InputError(
            f"no se pudo listar {root}: {type(exc).__name__}: {exc}"
        ) from exc


def listing(files: list[str], thetas: set[str]) -> dict[tuple[str, str], str]:
    """`{(θ, YYYY-MM): ruta}` de los `summaries.parquet` de los θ dados, entre los `files`."""
    found = {}
    for path in files:
        match = FILE.search(path)
        if match and match[1] in thetas:
            found[match[1], f"{match[2]}-{match[3]}"] = path
    return found


def read_catalog(source: str) -> list[str]:
    """Los θ del catálogo de L2 (`scale: 100000000` y `- <round(θ × 10⁸)>`) como texto."""
    try:
        fs, resolved = open_fs(source)
        with fs.open_input_stream(resolved) as stream:
            text = stream.read().decode()
    except OSError as exc:
        raise InputError(f"no se pudo leer el catálogo {source}: {exc}") from exc
    if not re.search(r"^scale:\s*100000000\s*(?:#.*)?$", text, re.MULTILINE):
        raise InputError(f"{source}: `scale` debe ser 100000000")
    integers = [
        int(m[1]) for m in re.finditer(r"^\s*-\s*(\d+)\s*(?:#.*)?$", text, re.MULTILINE)
    ]
    if not integers:
        raise InputError(f"{source}: sin θ")
    return [f"{Decimal(i).scaleb(-8):.8f}" for i in sorted(integers)]


def month_number(month: str) -> int:
    year, number = month.split("-")
    return int(year) * 12 + int(number) - 1


def month_range(first: str | None, last: str | None) -> list[str]:
    """Los meses `YYYY-MM` de `first` a `last`, ambos incluidos; vacío si falta un extremo."""
    if first is None or last is None:
        return []
    return [
        f"{n // 12:04d}-{n % 12 + 1:02d}"
        for n in range(month_number(first), month_number(last) + 1)
    ]


def count_phase(ticks: pa.ChunkedArray, sample: int, overshoot: bool) -> list[int]:
    """Los conteos de una fase de un row group, en el orden de `METRICS`.

    En `os`, las fases son las de `ticks > 0` y las demás son `vacias`; en `c`, todas son fases.
    """
    if ticks.null_count:
        raise ValueError("ticks nulos")
    phases = pc.filter(ticks, pc.greater(ticks, 0)) if overshoot else ticks
    capped = pc.min_element_wise(phases, sample)
    return [
        len(phases),
        len(ticks) - len(phases),
        *(pc.sum(pc.greater_equal(phases, n)).as_py() or 0 for n in THRESHOLDS),
        pc.sum(capped).as_py() or 0,
        pc.sum(pc.multiply_checked(capped, capped)).as_py() or 0,
    ]


def add(total: list[int], other: list[int]) -> None:
    for i, value in enumerate(other):
        total[i] += value


def zeros() -> Counts:
    return {phase: [0] * len(METRICS) for phase in PHASES}


def read_file(fs: pafs.FileSystem, path: str, sample: int) -> Counts:
    """Los conteos de un archivo: dos columnas, un row group a la vez."""
    totals = zeros()
    with fs.open_input_file(path) as handle:
        file = pq.ParquetFile(handle)
        for index in range(file.num_row_groups):
            table = file.read_row_group(index, columns=list(COLUMNS))
            for phase, column in zip(PHASES, COLUMNS, strict=True):
                add(totals[phase], count_phase(table[column], sample, phase == "os"))
            del table
    return totals


def collect(
    fs: pafs.FileSystem,
    catalog: list[str],
    months: list[str],
    present: dict[tuple[str, str], str],
    readers: int,
    sample: int,
):
    """Genera `(θ, leídos, faltantes, conteos)` de cada θ del catálogo en cuanto lo cierra.

    Un archivo por lector. Todos los archivos se piden de entrada y se consumen en orden de θ y
    mes: el hilo principal suma los conteos de cada archivo y suelta el futuro.
    """
    pool = ThreadPoolExecutor(max_workers=readers)
    try:
        wanted = [(t, m) for t in catalog for m in months if (t, m) in present]
        futures = {
            key: pool.submit(read_file, fs, present[key].removeprefix("gs://"), sample)
            for key in wanted
        }
        for theta in catalog:
            counts, read, missing = zeros(), 0, 0
            for month in months:
                future = futures.pop((theta, month), None)
                if future is None:
                    missing += 1
                    continue
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - un archivo ilegible es un faltante
                    log(f"FALLO θ={theta} {month}: {type(exc).__name__}: {exc}")
                    missing += 1
                    continue
                for phase in PHASES:
                    add(counts[phase], result[phase])
                read += 1
            yield theta, read, missing, counts
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def metric(counts: list[int], name: str) -> int:
    return counts[METRICS.index(name)]


def summary_line(label: str, read: int, missing: int, counts: Counts) -> str:
    c, os_ = counts["c"], counts["os"]
    return (
        f"{label} archivos={read} faltantes={missing} "
        f"c: fases={metric(c, 'fases'):,} ge_1024={metric(c, 'ge_1024'):,} "
        f"puntos={metric(c, 'puntos'):,} pares={metric(c, 'pares'):,} | "
        f"os: fases={metric(os_, 'fases'):,} vacias={metric(os_, 'vacias'):,} "
        f"ge_1024={metric(os_, 'ge_1024'):,}"
    )


def csv_rows(label: str, read: int, missing: int, counts: Counts) -> list[dict]:
    rows = []
    for phase in PHASES:
        row = {"theta": label, "fase": phase, "archivos": read, "faltantes": missing}
        for name, value in zip(METRICS, counts[phase], strict=True):
            if name != "vacias":
                row[name] = value
            elif phase == "os":
                row["os_vacias"] = value
        rows.append(row)
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
    key = (ns.provider, ns.market, ns.asset)
    name = "l3_phase_sizes.csv"
    name_project = project()
    if not (ns.catalog and ns.l3_root) and name_project is None:
        parser.error(
            "sin --catalog y --l3-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    catalog_uri = ns.catalog or f"gs://{name_project}-manifest/l2/thetas.yaml"
    l3_root = ns.l3_root or f"gs://{name_project}-dc-descriptors/l3"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))
    for option, value in (("--from", ns.from_month), ("--to", ns.to_month)):
        if value is not None and not MONTH.match(value):
            parser.error(f"{option} {value}: se esperaba YYYY-MM")
    if ns.from_month and ns.to_month and ns.from_month > ns.to_month:
        parser.error(f"--from {ns.from_month} es posterior a --to {ns.to_month}")
    if ns.readers < 1:
        parser.error("--readers debe ser al menos 1")
    if not 1 <= ns.sample <= SAMPLE_MAX:
        parser.error(f"--sample debe estar entre 1 y {SAMPLE_MAX:,}")

    started = time.perf_counter()
    try:
        catalog = read_catalog(catalog_uri)
        files = list_files(series_root(l3_root, key))
    except InputError as exc:
        log(f"FALLO {exc}")
        return 1
    present = listing(files, set(catalog))
    del files
    seen = sorted({month for _, month in present})
    months = month_range(
        ns.from_month or (seen[0] if seen else None),
        ns.to_month or (seen[-1] if seen else None),
    )
    fs, _ = open_fs(l3_root)
    log(
        f"inicio: catálogo={catalog_uri} l3={l3_root} θ={len(catalog)} "
        f"meses={months[0] if months else '-'}..{months[-1] if months else '-'} "
        f"readers={ns.readers} sample={ns.sample}"
    )

    total, rows = zeros(), []
    thetas = files_read = files_missing = 0
    for theta, read, missing, counts in collect(
        fs, catalog, months, present, ns.readers, ns.sample
    ):
        log(summary_line(f"θ={theta}", read, missing, counts))
        rows += csv_rows(theta, read, missing, counts)
        for phase in PHASES:
            add(total[phase], counts[phase])
        thetas += read > 0
        files_read += read
        files_missing += missing
    log(summary_line("total", files_read, files_missing, total))
    rows += csv_rows("total", files_read, files_missing, total)
    try:
        write_csv(rows, out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    log(f"pared_s={time.perf_counter() - started:.1f}")
    log(
        f"phase sizes: thetas={thetas} archivos={files_read} faltantes={files_missing} "
        f"fases_c={metric(total['c'], 'fases')} fases_os={metric(total['os'], 'fases')}"
    )
    return 0 if files_read else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog")
    parser.add_argument("--l3-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--from", dest="from_month", metavar="YYYY-MM")
    parser.add_argument("--to", dest="to_month", metavar="YYYY-MM")
    parser.add_argument("--sample", type=int, default=10_000)
    parser.add_argument("--readers", type=int, default=8)
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
