"""Sonda de E/S de L3 (ITSC-425): ¿el tope de pyarrow contra GCS también frena leer datos y escribir?

ITSC-421 midió que el acceso de pyarrow a GCS tiene un tope por proceso al leer pies. Falta saber si
el mismo tope frena lo caro de una unidad: leer los datos y escribir los resultados. El indicio
viejo (TRD-L3 §10.2: con 8 vCPU el motor trabajó el 13,6 % del tiempo y `write_s` y `read_wall_s`
fueron casi toda la pared) es un indicio, no una prueba: nadie midió el transporte aislado. Esta
sonda lo mide. No arregla nada: no cambia ninguna capa ni decide la migración.

Mide solo transporte: baja y sube bytes, no decodifica ni comprime. Es una cota superior: en
producción la escritura también comprime (ZSTD) y la lectura decodifica. Si el transporte no limita
ni siquiera aislado, migrarlo no sirve.

Lecturas. Del `consolidated.parquet` de un mes de L1 (`--month`, 2023-03) toma del pie los rangos de
bytes de una muestra fija de row groups (`--groups`, 24, sorteados con `--seed`) y los baja con
`--read-threads` hilos (1, 2, 8 y 32), de tres formas:

  a   pyarrow con un sistema de archivos compartido por todos los hilos (`open_input_file` por
      rango y `read_at`)
  a2  pyarrow con un sistema de archivos por hilo
  b   google-cloud-storage: una petición de rango por row group, con tiempo límite de 30 s

Escrituras. Sube `--objects` objetos (12) de `--object-mib` MiB (4) de bytes aleatorios bajo
`<OPS_RESULTS_URI>io_probe/<corrida>/` con `--write-threads` escritores (1, 4 y 12), de dos formas:

  a   pyarrow `open_output_stream`, escribiendo por trozos de 1 MiB
  b   google-cloud-storage (`upload_from_string`, 30 s de tiempo límite)

Los nombres no se repiten (la corrida lleva un sufijo único y el cliente de Google sube con
`if_generation_match=0`): el job solo puede crear objetos, no pisarlos. Tampoco puede borrarlos: el
humano borra el prefijo `io_probe/` al terminar (≈ 300 MiB con los valores por defecto).

Qué reporta, por corrida: la pared, los MiB por segundo (solo lo que bajó o subió sin error), la CPU
del proceso (`time.process_time`) ÷ pared, la ocupación de los hilos (suma de las paredes de las
tareas ÷ pared × hilos efectivos, que son `min(hilos, tareas)`: con 32 hilos y 24 rangos solo hay 24)
y los errores. Del entorno: cores del contenedor (el menor entre las CPU permitidas y la cuota de
cgroup), versión de pyarrow y tamaño de la muestra en MiB.

Regla del veredicto (escrita antes de medir). Sea `R(x, n)` los MiB por segundo de la lectura de
forma `x` con `n` hilos y `W(x, n)` los de la escritura. La concurrencia de producción es 2 lectores
y 12 escritores. Una corrida con más del 5 % de tareas fallidas, o que no corrió, no vale: su cifra
es nula y cada veredicto que la necesita sale `fallo`.

  lecturas=limita     si `R(b, 2) ≥ 2 × R(a, 2)`; si no, `no_limita`
  escrituras=limita   si `W(b, 12) ≥ 2 × W(a, 12)`; si no, `no_limita`
  candado=sistema_de_archivos  si `R(a2, 8) ≥ 2 × R(a, 8)`: cada sistema de archivos de pyarrow
                               trae su propio límite, el arreglo no necesita otra librería
  candado=proceso     si no, y ni (a) ni (a2) escalan (`R(x, 8) < 2 × R(x, 1)`) y (b) sí
                      (`R(b, 8) ≥ 2 × R(b, 1)`): el límite es del proceso o de pyarrow
  candado=ninguno     en cualquier otro caso

«Limita» quiere decir que migrar el transporte de esa dirección al cliente de Google rendiría al
menos el doble con la concurrencia de producción. La decisión la toma el arquitecto con este
resultado.

Args:
  --month 2023-03   --groups 24   --seed 425
  --read-threads 1,2,8,32       (siempre corren 1, 2 y 8: la regla los necesita)
  --write-threads 1,4,12        (siempre corre 12)
  --objects 12   --object-mib 4
  --budget-s 2700   (pasado ese tiempo no empieza más corridas; `ops-script` corta a los 3 600 s)
  --max-inflight-mib 2048   (una corrida de lectura cuyo peor caso, el rango mayor × sus hilos,
                             pasa de esto no corre: la memoria manda)
  --l1-root gs://<proyecto>-landing/l1   (por defecto el del proyecto según OPS_RESULTS_URI;
                                          también rutas locales)
  --provider binance --market spot --asset BTCUSDT
  --write-root <raíz>   (por defecto OPS_RESULTS_URI; una ruta local sirve para probar)
  --out <ruta o gs://…/archivo.csv>   (por defecto `l3_io_probe.csv` bajo OPS_RESULTS_URI o, en
                                       local, en el directorio actual; una ruta que termina en `/`
                                       es un directorio)

Las corridas de la regla van primero (lecturas con 1, 2 y 8 hilos y escrituras con 12), para que si
se acaba el tiempo no se pierda lo que decide el veredicto. Cada forma hace una lectura de un byte
antes de cronometrar, para que la creación del cliente y la credencial no cuenten; el resto del
costo de conexión sí cuenta, igual para todas. Todas las corridas de lectura bajan los mismos rangos.

Memoria. Cada hilo retiene a lo más un rango o un objeto a la vez y lo suelta al terminar: las
lecturas cuentan los bytes y los sueltan; las escrituras comparten un único búfer de `--object-mib`
MiB de solo lectura (nadie lo copia) y lo envían por trozos. Nada se acumula entre corridas.

No importa ninguna capa. Solo lee el pie y los rangos de L1; lo único que escribe, aparte del CSV,
son los objetos de prueba.

Salida: la tabla de corridas, el CSV (`seccion,direccion,forma,hilos,medida,valor`) y, como última
línea, `io probe: mes=… lecturas_mib=… a_2=… b_2=… escrituras_mib=… a_12=… b_12=… lecturas=…
escrituras=… candado=…` (los `a_*` y `b_*`, en MiB por segundo; `-` si no se midió).

Códigos: 0 si midió y ningún veredicto es `fallo`; 1 si el pie no se lee, el mes no tiene row
groups o algún veredicto sale `fallo`; 2 por uso. Si el CSV no se puede escribir, la sonda lo avisa
y no cambia el código.
"""

import argparse
import csv
import io
import math
import os
import queue
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

DEFAULT_MONTH = "2023-03"
DEFAULT_GROUPS = 24
DEFAULT_SEED = 425
DEFAULT_READ_THREADS = (1, 2, 8, 32)
DEFAULT_WRITE_THREADS = (1, 4, 12)
DEFAULT_OBJECTS = 12
DEFAULT_OBJECT_MIB = 4
# `ops-script` muere a los 3 600 s: pasado este tiempo, las corridas que faltan se omiten.
DEFAULT_BUDGET_S = 2700
DEFAULT_MAX_INFLIGHT_MIB = 2048
MIB = 1024 * 1024
WRITE_CHUNK = MIB
REQUEST_TIMEOUT_S = 30
# La concurrencia de producción y las corridas que la regla necesita siempre.
PROD_READERS = 2
PROD_WRITERS = 12
SCALE_CHECK = 8
REQUIRED_READ_THREADS = (1, PROD_READERS, SCALE_CHECK)
REQUIRED_WRITE_THREADS = (PROD_WRITERS,)
# Umbrales de la regla del veredicto, fijados antes de medir.
LIMIT_GAIN = 2.0
SCALE_GAIN = 2.0
MAX_ERROR_SHARE = 0.05
READ_FORMS = ("a", "a2", "b")
WRITE_FORMS = ("a", "b")
CSV_HEADER = ["seccion", "direccion", "forma", "hilos", "medida", "valor"]
READ, WRITE = "lecturas", "escrituras"
Rates = dict[tuple[str, int], float | None]


def log(message: str) -> None:
    print(message, flush=True)


class InputError(Exception):
    """El pie no se lee o el mes no tiene row groups: sale con código 1."""


# ----------------------------------------------------------------------------------------------
# Rutas y muestra
# ----------------------------------------------------------------------------------------------


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def l1_file(root: str, key: tuple[str, str, str], month: str) -> str:
    provider, market, asset = key
    year, number = month.split("-")
    return (
        f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/year={year}/month={number}/consolidated.parquet"
    )


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(uri)
        return fs, path.rstrip("/")
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def split_gs(uri: str) -> tuple[str, str]:
    bucket, _, name = uri.removeprefix("gs://").partition("/")
    return bucket, name


def row_group_ranges(meta: pq.FileMetaData) -> list[tuple[int, int]]:
    """`(inicio, bytes)` de cada row group con datos: del primer byte de sus columnas al último."""
    ranges = []
    for index in range(meta.num_row_groups):
        group = meta.row_group(index)
        starts, ends = [], []
        for number in range(group.num_columns):
            column = group.column(number)
            start = column.data_page_offset
            if column.has_dictionary_page and column.dictionary_page_offset:
                start = min(start, column.dictionary_page_offset)
            starts.append(start)
            ends.append(start + column.total_compressed_size)
        if starts and max(ends) > min(starts):
            ranges.append((min(starts), max(ends) - min(starts)))
    return ranges


def sample_ranges(
    uri: str, groups: int, seed: int
) -> tuple[list[tuple[int, int]], int]:
    """Los rangos de `groups` row groups sorteados con `seed` y cuántos tiene el archivo."""
    try:
        fs, path = open_fs(uri)
        with fs.open_input_file(path) as handle:
            ranges = row_group_ranges(pq.read_metadata(handle))
    except (OSError, pa.ArrowException) as exc:
        raise InputError(
            f"no se pudo leer el pie de {uri}: {type(exc).__name__}: {exc}"
        ) from exc
    if not ranges:
        raise InputError(f"{uri} no tiene row groups con datos")
    chosen = sorted(
        random.Random(seed).sample(range(len(ranges)), min(groups, len(ranges)))
    )
    return [ranges[i] for i in chosen], len(ranges)


# ----------------------------------------------------------------------------------------------
# Transportes: cada uno baja un rango o sube un objeto y devuelve cuántos bytes movió
# ----------------------------------------------------------------------------------------------


class GoogleClient:
    """Un cliente de google-cloud-storage con tantas conexiones como hilos (el grupo de `requests`
    trae 10)."""

    def __init__(self, threads: int) -> None:
        from google.cloud import storage

        self.client = storage.Client()
        try:
            from requests.adapters import HTTPAdapter

            adapter = HTTPAdapter(pool_connections=threads, pool_maxsize=threads)
            self.client._http.mount("https://", adapter)
        except AttributeError:
            pass

    def blob(self, uri: str):
        bucket, name = split_gs(uri)
        return self.client.bucket(bucket).blob(name)


class ReadShared:
    """(a) pyarrow con un sistema de archivos compartido: un `open_input_file` por rango."""

    def __init__(self, uri: str, threads: int) -> None:
        self.fs, self.path = open_fs(uri)

    def warm(self) -> None:
        self.read(0, 1)

    def read(self, start: int, size: int) -> int:
        with self.fs.open_input_file(self.path) as handle:
            return len(handle.read_at(size, start))


class ReadPerThread:
    """(a2) pyarrow con un sistema de archivos por hilo, creados antes de cronometrar."""

    def __init__(self, uri: str, threads: int) -> None:
        self.path = open_fs(uri)[1]
        self.spare = [open_fs(uri)[0] for _ in range(threads)]
        self.free: queue.SimpleQueue = queue.SimpleQueue()
        self.local = threading.local()

    def warm(self) -> None:
        for fs in self.spare:
            with fs.open_input_file(self.path) as handle:
                handle.read_at(1, 0)
            self.free.put(fs)
        self.spare = []

    def read(self, start: int, size: int) -> int:
        fs = getattr(self.local, "fs", None)
        if fs is None:
            fs = self.local.fs = self.free.get_nowait()
        with fs.open_input_file(self.path) as handle:
            return len(handle.read_at(size, start))


class ReadGoogle:
    """(b) google-cloud-storage: una petición de rango por row group (`end` es inclusivo)."""

    def __init__(self, uri: str, threads: int) -> None:
        self.uri = uri
        self.client = GoogleClient(threads) if uri.startswith("gs://") else None

    def warm(self) -> None:
        self.read(0, 1)

    def read(self, start: int, size: int) -> int:
        if self.client is None:
            with open(self.uri, "rb") as handle:
                handle.seek(start)
                return len(handle.read(size))
        blob = self.client.blob(self.uri)
        data = blob.download_as_bytes(
            start=start, end=start + size - 1, timeout=REQUEST_TIMEOUT_S
        )
        return len(data)


READERS = {"a": ReadShared, "a2": ReadPerThread, "b": ReadGoogle}


class WriteArrow:
    """(a) pyarrow `open_output_stream`, por trozos de 1 MiB de un búfer compartido."""

    def __init__(self, threads: int, root: str) -> None:
        self.gs = root.startswith("gs://")
        self.fs = open_fs(root)[0]

    def put(self, uri: str, payload: bytes) -> int:
        path = uri.removeprefix("gs://") if self.gs else str(Path(uri).resolve())
        if not self.gs:
            self.fs.create_dir(str(Path(path).parent), recursive=True)
        view = memoryview(payload)
        with self.fs.open_output_stream(path) as stream:
            for offset in range(0, len(view), WRITE_CHUNK):
                stream.write(view[offset : offset + WRITE_CHUNK])
        return len(payload)


class WriteGoogle:
    """(b) google-cloud-storage: `upload_from_string`, sin pisar objetos."""

    def __init__(self, threads: int, root: str) -> None:
        self.client = GoogleClient(threads) if root.startswith("gs://") else None

    def put(self, uri: str, payload: bytes) -> int:
        if self.client is None:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            Path(uri).write_bytes(payload)
            return len(payload)
        self.client.blob(uri).upload_from_string(
            payload,
            content_type="application/octet-stream",
            if_generation_match=0,
            timeout=REQUEST_TIMEOUT_S,
        )
        return len(payload)


WRITERS = {"a": WriteArrow, "b": WriteGoogle}


# ----------------------------------------------------------------------------------------------
# Corridas
# ----------------------------------------------------------------------------------------------


class Run(NamedTuple):
    direction: str
    form: str
    threads: int
    workers: int
    wall: float
    cpu: float
    task_wall: float
    tasks: int
    moved: int
    errors: int

    @property
    def label(self) -> str:
        return f"{self.form}_{self.threads}"

    @property
    def mib_per_s(self) -> float:
        return self.moved / MIB / self.wall if self.wall else math.nan

    @property
    def cpu_ratio(self) -> float:
        return self.cpu / self.wall if self.wall else math.nan

    @property
    def occupancy(self) -> float:
        return self.task_wall / (self.wall * self.workers) if self.wall else math.nan

    @property
    def reliable(self) -> bool:
        return self.tasks > 0 and self.errors <= MAX_ERROR_SHARE * self.tasks


def execute(direction: str, form: str, threads: int, jobs: list, work) -> Run:
    """Corre `work(job)` (devuelve los bytes que movió) con `threads` hilos y cronometra la corrida.

    Cada hilo retiene a lo más un trabajo; los resultados son tres números por tarea. El primer
    error de la corrida se imprime."""
    workers = max(1, min(threads, len(jobs)))
    failures: list[str] = []

    def timed(job) -> tuple[int, float]:
        began = time.perf_counter()
        try:
            moved = work(job)
        except Exception as exc:  # noqa: BLE001 - la sonda cuenta la tarea fallida y sigue
            failures.append(f"{type(exc).__name__}: {exc}")
            moved = 0
        return moved, time.perf_counter() - began

    wall0, cpu0 = time.perf_counter(), time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(timed, jobs))
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    if failures:
        log(f"  primer error de {form}_{threads}: {failures[0]}")
    return Run(
        direction,
        form,
        threads,
        workers,
        wall,
        cpu,
        sum(seconds for _, seconds in results),
        len(jobs),
        sum(moved for moved, _ in results),
        len(failures),
    )


def read_run(form: str, threads: int, uri: str, ranges: list[tuple[int, int]]) -> Run:
    reader = READERS[form](uri, max(1, min(threads, len(ranges))))
    reader.warm()

    def work(job: tuple[int, int]) -> int:
        start, size = job
        got = reader.read(start, size)
        if got != size:
            raise OSError(f"lectura corta: {got} de {size} bytes en {start}")
        return got

    return execute(READ, form, threads, ranges, work)


def write_run(
    form: str, threads: int, prefix: str, payload: bytes, objects: int
) -> Run:
    writer = WRITERS[form](max(1, min(threads, objects)), prefix)
    names = [f"{prefix}/{form}_{threads:02d}w_{i:03d}.bin" for i in range(objects)]
    return execute(WRITE, form, threads, names, lambda name: writer.put(name, payload))


# ----------------------------------------------------------------------------------------------
# Regla, salida
# ----------------------------------------------------------------------------------------------


def rates(runs: list[Run], direction: str) -> Rates:
    """MiB por segundo por `(forma, hilos)`; `None` si la corrida falla en más del 5 %."""
    return {
        (r.form, r.threads): r.mib_per_s if r.reliable else None
        for r in runs
        if r.direction == direction
    }


def limits(a: float | None, b: float | None) -> str:
    """`limita` si (b) rinde al menos el doble que (a); `fallo` si falta una de las dos."""
    if not a or b is None:
        return "fallo"
    return "limita" if b >= LIMIT_GAIN * a else "no_limita"


def lock(reads: Rates) -> str:
    """El candado de las lecturas: `sistema_de_archivos`, `proceso` o `ninguno`."""
    needed = [reads.get((form, n)) for form in READ_FORMS for n in (1, SCALE_CHECK)]
    if any(not value for value in needed):
        return "fallo"
    a1, a8, a2_1, a2_8, b1, b8 = needed
    if a2_8 >= LIMIT_GAIN * a8:
        return "sistema_de_archivos"
    if a8 < SCALE_GAIN * a1 and a2_8 < SCALE_GAIN * a2_1 and b8 >= SCALE_GAIN * b1:
        return "proceso"
    return "ninguno"


def verdicts(reads: Rates, writes: Rates) -> dict[str, str]:
    return {
        "lecturas": limits(
            reads.get(("a", PROD_READERS)), reads.get(("b", PROD_READERS))
        ),
        "escrituras": limits(
            writes.get(("a", PROD_WRITERS)), writes.get(("b", PROD_WRITERS))
        ),
        "candado": lock(reads),
    }


def cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def final_line(
    month: str,
    reads_mib: float | None,
    writes_mib: float | None,
    reads: Rates,
    writes: Rates,
    result: dict[str, str],
) -> str:
    return (
        f"io probe: mes={month} lecturas_mib={cell(reads_mib)} "
        f"a_{PROD_READERS}={cell(reads.get(('a', PROD_READERS)))} "
        f"b_{PROD_READERS}={cell(reads.get(('b', PROD_READERS)))} "
        f"escrituras_mib={cell(writes_mib)} "
        f"a_{PROD_WRITERS}={cell(writes.get(('a', PROD_WRITERS)))} "
        f"b_{PROD_WRITERS}={cell(writes.get(('b', PROD_WRITERS)))} "
        f"lecturas={result['lecturas']} escrituras={result['escrituras']} "
        f"candado={result['candado']}"
    )


def container_cores() -> int:
    """Las CPU del contenedor: las permitidas al proceso, o la cuota de cgroup si es menor."""
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            cores = min(cores, max(1, math.ceil(int(quota) / int(period))))
    except OSError, ValueError:
        pass
    return cores


def csv_rows(runs: list[Run], environment: dict, result: dict[str, str]) -> list[dict]:
    rows = []

    def add(section, direction, form, threads, measure, value):
        rows.append(
            {
                "seccion": section,
                "direccion": direction,
                "forma": form,
                "hilos": threads,
                "medida": measure,
                "valor": value if isinstance(value, str | int) else f"{value:.6g}",
            }
        )

    for name, value in environment.items():
        add("entorno", "", "", "", name, value)
    for run in runs:
        for measure, value in (
            ("pared_s", run.wall),
            ("mib_movidos", run.moved / MIB),
            ("mib_por_s", run.mib_per_s),
            ("cpu_s", run.cpu),
            ("cpu_ratio", run.cpu_ratio),
            ("ocupacion", run.occupancy),
            ("hilos_efectivos", run.workers),
            ("tareas", run.tasks),
            ("errores", run.errors),
        ):
            add("corrida", run.direction, run.form, run.threads, measure, value)
    for name, value in result.items():
        add("veredicto", "", "", "", name, value)
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


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(f"{out} no está bajo {results}; el job solo crea objetos ahí")


def print_table(runs: list[Run]) -> None:
    log(
        f"{'corrida':<10} {'dir':<11} {'pared_s':>8} {'MiB/s':>9} {'cpu/pared':>10} "
        f"{'ocupación':>10} {'hilos':>6} {'errores':>8}"
    )
    for r in runs:
        log(
            f"{r.label:<10} {r.direction:<11} {r.wall:>8.2f} {r.mib_per_s:>9.1f} "
            f"{r.cpu_ratio:>10.2f} {r.occupancy:>10.2f} {r.workers:>6} {r.errors:>8}"
        )


def parse_threads(text: str, name: str, required: tuple[int, ...]) -> tuple[int, ...]:
    try:
        threads = {int(p) for p in text.split(",") if p.strip()}
    except ValueError:
        raise ValueError(
            f"{name} {text!r}: se esperan enteros separados por comas"
        ) from None
    if not threads or min(threads) < 1:
        raise ValueError(f"{name} {text!r}: cada número debe ser al menos 1")
    return tuple(sorted(threads | set(required)))


def plan_runs(
    read_threads: tuple[int, ...], write_threads: tuple[int, ...]
) -> list[tuple[str, str, int]]:
    """Primero las corridas que decide la regla; el resto, de menos a más hilos."""
    plan = [(READ, f, n) for n in read_threads for f in READ_FORMS]
    plan += [(WRITE, f, n) for n in write_threads for f in WRITE_FORMS]

    def key(run: tuple[str, str, int]) -> tuple[int, int, str, str]:
        direction, form, threads = run
        needed = threads in (
            REQUIRED_READ_THREADS if direction == READ else REQUIRED_WRITE_THREADS
        )
        return (0 if needed else 1, threads, direction, form)

    return sorted(plan, key=key)


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    try:
        read_threads = parse_threads(
            ns.read_threads, "--read-threads", REQUIRED_READ_THREADS
        )
        write_threads = parse_threads(
            ns.write_threads, "--write-threads", REQUIRED_WRITE_THREADS
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", ns.month):
        parser.error(f"--month {ns.month!r}: se espera AAAA-MM")
    if min(ns.groups, ns.objects, ns.object_mib, ns.max_inflight_mib) < 1:
        parser.error(
            "--groups, --objects, --object-mib y --max-inflight-mib deben ser al menos 1"
        )
    name_project = project()
    if not ns.l1_root and name_project is None:
        parser.error("sin --l1-root y sin OPS_RESULTS_URI: no sé de qué proyecto es")
    write_root = ns.write_root or os.environ.get("OPS_RESULTS_URI", "")
    if not write_root:
        parser.error("sin --write-root y sin OPS_RESULTS_URI: no sé dónde escribir")
    name = "l3_io_probe.csv"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
        check_out(write_root)
    except ValueError as exc:
        parser.error(f"--out o --write-root: {exc}")
    key = (ns.provider, ns.market, ns.asset)
    uri = l1_file(ns.l1_root or f"gs://{name_project}-landing/l1", key, ns.month)
    cores = container_cores()

    try:
        ranges, total_groups = sample_ranges(uri, ns.groups, ns.seed)
    except InputError as exc:
        log(f"FALLO {exc}")
        none: Rates = {}
        log(final_line(ns.month, None, None, none, none, verdicts(none, none)))
        return 1
    reads_mib = sum(size for _, size in ranges) / MIB
    writes_mib = ns.objects * ns.object_mib
    largest = max(size for _, size in ranges)
    log(
        f"muestra: {len(ranges)} de {total_groups} row groups de {uri} = {reads_mib:.1f} MiB "
        f"(mayor {largest / MIB:.1f} MiB); cores={cores}; pyarrow={pa.__version__}"
    )
    # Un único búfer de solo lectura: ningún hilo lo copia y se suelta al salir de la función.
    payload = random.Random(ns.seed).randbytes(ns.object_mib * MIB)
    prefix = f"{write_root.rstrip('/')}/io_probe/{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    log(f"objetos de prueba bajo {prefix}/ (el humano los borra al terminar)")

    runs: list[Run] = []
    started = time.perf_counter()
    for direction, form, threads in plan_runs(read_threads, write_threads):
        label = f"{direction}:{form}_{threads}"
        if time.perf_counter() - started > ns.budget_s:
            log(f"{label}: omitida, pasó el presupuesto de {ns.budget_s} s")
            continue
        if (
            direction == READ
            and largest * min(threads, len(ranges)) > ns.max_inflight_mib * MIB
        ):
            log(
                f"{label}: omitida, su peor caso en vuelo pasa de {ns.max_inflight_mib} MiB "
                "(--max-inflight-mib)"
            )
            continue
        try:
            if direction == READ:
                done = read_run(form, threads, uri, ranges)
            else:
                done = write_run(form, threads, prefix, payload, ns.objects)
        except Exception as exc:  # noqa: BLE001 - una forma que no arranca no tumba la medición
            log(f"{label}: no corrió ({type(exc).__name__}: {exc})")
            continue
        runs.append(done)
        log(
            f"{label}: pared={done.wall:.2f}s MiB/s={done.mib_per_s:.1f} "
            f"cpu/pared={done.cpu_ratio:.2f} errores={done.errors}"
        )
    del payload
    print_table(runs)

    read_rates, write_rates = rates(runs, READ), rates(runs, WRITE)
    result = verdicts(read_rates, write_rates)
    environment = {
        "cores": cores,
        "pyarrow": pa.__version__,
        "google_cloud_storage": _version("google-cloud-storage"),
        "python": sys.version.split()[0],
        "mes": ns.month,
        "row_groups_archivo": total_groups,
        "row_groups_muestra": len(ranges),
        "muestra_mib": f"{reads_mib:.1f}",
        "semilla": ns.seed,
        "objetos": ns.objects,
        "objeto_mib": ns.object_mib,
        "escrituras_mib": writes_mib,
        "prefijo_objetos": prefix,
    }
    try:
        write_csv(csv_rows(runs, environment, result), out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    for r in runs:
        if not r.reliable:
            log(
                f"AVISO {r.direction}:{r.label}: más del {MAX_ERROR_SHARE:.0%} de sus tareas falla; no vale"
            )
    log(final_line(ns.month, reads_mib, writes_mib, read_rates, write_rates, result))
    return 1 if "fallo" in result.values() else 0


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", default=DEFAULT_MONTH)
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--read-threads", default=",".join(map(str, DEFAULT_READ_THREADS))
    )
    parser.add_argument(
        "--write-threads", default=",".join(map(str, DEFAULT_WRITE_THREADS))
    )
    parser.add_argument("--objects", type=int, default=DEFAULT_OBJECTS)
    parser.add_argument("--object-mib", type=int, default=DEFAULT_OBJECT_MIB)
    parser.add_argument("--budget-s", type=int, default=DEFAULT_BUDGET_S)
    parser.add_argument(
        "--max-inflight-mib", type=int, default=DEFAULT_MAX_INFLIGHT_MIB
    )
    parser.add_argument("--l1-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--write-root")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
