"""Sonda de la verificación de la frontera de L3 (ITSC-421): dónde se va el tiempo y qué la limita.

La frontera verifica cada mes de la serie de cada θ (`invalid_reason`, ITSC-419) y a 64 lectores
tarda 1 316 s, casi lo mismo que a 8: ≈ 12 pies por segundo sin importar los lectores, también con
8 vCPU. Esta sonda mide, no arregla: no cambia `frontier.py` ni ningún comportamiento de L3.

Qué mide. Sobre una muestra fija de `(θ, mes)` reales (`--sample`, 600 por defecto) repite la
secuencia de `invalid_reason` para el archivo de la familia (`path.parquet`) y el `events.parquet`
de L2 del mismo mes, y cronometra cada tarea por etapa (cada etapa, en pared y en CPU del hilo con
`time.thread_time`; las etapas suman la pared de la tarea):

  abrir         `fs.open_input_file`: el viaje de metadatos al objeto (los dos archivos)
  pie           `pq.ParquetFile(handle)`: leer y decodificar el pie (los dos archivos)
  estadisticas  `schema_arrow`, `state_version` y las estadísticas de `theta` de cada row group
  filas         filas por row group de `events.parquet` y de la familia, y su comparación
  cierre        cerrar los dos archivos

En la variante `b` las etapas no son las mismas: `abrir` es la descarga de la cola de cada objeto
(una petición de rango, o dos si el pie es mayor) y `pie` es `pq.read_metadata` sobre ese búfer; no
hay `estadisticas`, `filas` ni `cierre`.

`invalid_reason` abre con `pq.ParquetFile(ruta, filesystem=fs)`, que hace lo mismo que abrir el
archivo y luego el pie, pero en una sola llamada; la sonda las separa para verlas. Un sistema de
archivos por raíz y por corrida, como `FooterReader`. No compara el esquema con el de la imagen
(sería importar una capa): el costo de `schema_arrow` sí está.

Corridas (cada una sobre la misma muestra; `--readers`, 1, 8 y 64 por defecto; corren en el orden
base, c, a, b, para que lo que decide el veredicto no se pierda si se acaba el tiempo):
  base  la secuencia completa, con hilos
  a     solo abrir y leer el pie de los dos archivos, sin el recorrido por row group
  b     leer la cola del objeto con el cliente de google-cloud-storage (una petición de rango
        `bytes=-65536`, otra si el pie es mayor) y decodificar el pie con `pq.read_metadata` sobre
        un búfer en memoria, sin `GcsFileSystem`; el grupo de conexiones del cliente se ajusta a
        los lectores (el de `requests` trae 10)
  c     la base con los mismos lectores que `base` con `N` (el mayor de `--readers`), repartidos en
        `--processes` procesos (4): cada proceso corre un trozo de la muestra con `N ÷ procesos`
        hilos y su propio sistema de archivos. Compara `N` contra `N` y solo cambia que haya un
        GIL o varios

Cada corrida reporta la pared, los pies por segundo (dos por tarea), la CPU del proceso
(`time.process_time`; en `c`, la suma de los procesos hijos, medida al terminar su trozo y desde la
línea base que cada hijo toma al arrancar) y su razón con la pared, la ocupación
de los lectores (suma de las paredes de las tareas ÷ pared × lectores) y, por etapa, mediana y p95
por tarea. De la muestra: row groups y tamaño del pie por archivo (mediana, p95, máximo), cores
del contenedor (el menor entre las CPU permitidas y la cuota de cgroup) y versión de pyarrow.

Regla del veredicto (escrita antes de medir). Sea `N` el mayor número de lectores, `r` la razón
CPU ÷ pared de `base` con `N` (se espera ≈ 1 con un hilo bajo el GIL, ≈ cores con la CPU
saturada, mucho menos si se espera), `A` los pies por segundo de `base` con `N`, `escala` el
mejor de `base` ÷ el de `base` con 1 lector y `C` los pies por segundo de `c`:

  1. cpu     si `r ≥ 0,7 × cores`: la CPU está saturada.
  2. mixto   si `escala ≥ 4`: la base sí escala, el problema no se reproduce.
  3. mixto   si `c` no corrió.
  4. gil     si `C ≥ 2 × A`: el límite está dentro de un proceso, el GIL o un candado de pyarrow o
             del cliente, y repartir en procesos lo levanta.
  5. espera  si `r < 0,7` y `C < 2 × A`: ni un núcleo ocupado y los procesos no ayudan; el límite
             está fuera del proceso (red, servidor, cuotas).
  6. mixto   en cualquier otro caso.

`mejor_variante` es la corrida con más pies por segundo (`base_8`, `a_64`, `b_64`, `c_64`…): dice
cuál variante conviene al arreglo, no cambia el veredicto. Las etapas por mediana y por CPU del
hilo dicen cuál etapa carga el tiempo.

Args:
  --sample 600   --seed 421
  --readers 1,8,64
  --variants base,a,b,c      (`base` siempre corre)
  --processes 4              (procesos de `c`, que corre con el mayor de `--readers` repartido entre ellos)
  --budget-s 2700            (pasado ese tiempo no empieza más corridas; `ops-script` corta a los 3 600 s)
  --l3-root gs://<proyecto>-dc-descriptors/l3   (por defecto, los del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2         OPS_RESULTS_URI; también rutas locales)
  --provider binance --market spot --asset BTCUSDT
  --family path
  --out <ruta o gs://…/archivo.csv>   (por defecto `l3_frontier_probe.csv` bajo OPS_RESULTS_URI o,
                                       en local, en el directorio actual; una ruta que termina en
                                       `/` es un directorio)

Muestra. Los `(θ, mes)` con archivo de la familia y `events.parquet`, en cuatro estratos: θ chicos
o grandes (la mitad de los θ por valor) × meses viejos o recientes (la mitad de los meses por
fecha), un cuarto de la muestra de cada uno, sorteados con `--seed`; el estrato que no llena su
cuota la cede a los demás. Con la misma semilla y el mismo lago, la misma muestra.

No importa ninguna capa. Solo lee pies, un archivo a la vez por lector: la memoria la acota la
muestra, no el lago.

Salida: la tabla de corridas, la de etapas, el CSV (`seccion,variante,lectores,etapa,medida,valor`)
y, como última línea, `frontier probe: muestra=… base_1=…s base_8=…s base_64=…s cpu_ratio_64=…
mejor_variante=… pies_por_s_mejor=… veredicto=cpu|gil|espera|mixto|fallo`.

Códigos: 0 si midió; 1 si el listado falla, la muestra queda vacía, falta alguna corrida `base` o más
del 5 % de las tareas de `base` o de `c` falla (las que deciden el veredicto); 2 por uso. Una `a` o
una `b` con más del 5 % de fallas se avisa en el log y queda fuera de `mejor_variante`: no tumba la
medición. Solo lee; lo único que escribe es el CSV.
"""

import argparse
import csv
import io
import math
import multiprocessing
import os
import random
import re
import statistics
import sys
import threading
import time
from array import array
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

DEFAULT_SAMPLE = 600
DEFAULT_SEED = 421
DEFAULT_READERS = (1, 8, 64)
DEFAULT_PROCESSES = 4
# `ops-script` muere a los 3 600 s: pasado este tiempo, las corridas que faltan se omiten.
DEFAULT_BUDGET_S = 2700
STAGES = ("abrir", "pie", "estadisticas", "filas", "cierre")
OPEN, FOOTER, STATS, ROWS, CLOSE = range(len(STAGES))
THETA_DIGITS = 8
# Lo que pide `bytes=-N` en la primera petición de la variante b (lo mismo que adivina pyarrow).
TAIL_BYTES = 64 * 1024
# Umbrales de la regla del veredicto, fijados antes de medir.
CPU_SHARE = 0.7
SCALE_OK = 4.0
PROCESS_GAIN = 2.0
MAX_ERROR_SHARE = 0.05
DECIDING = ("base", "c")
CSV_HEADER = ["seccion", "variante", "lectores", "etapa", "medida", "valor"]
FILE_PATTERN = r"/theta=0\.(\d{{{digits}}})/year=(\d{{4}})/month=(\d{{2}})/{name}$"


def log(message: str) -> None:
    print(message, flush=True)


class InputError(Exception):
    """Un listado que falla o una muestra vacía: sale con código 1."""


class Task(NamedTuple):
    theta: int
    year: int
    month: int


class Result(NamedTuple):
    """Lo que devuelve una tarea: sus tiempos por etapa y lo que vio de los dos archivos."""

    wall: float
    stage_wall: tuple[float, ...]
    stage_cpu: tuple[float, ...]
    # (row groups, bytes del pie) del archivo de la familia y del de L2; vacío si falló antes.
    files: tuple[tuple[int, int], ...]
    error: str | None


# ----------------------------------------------------------------------------------------------
# Rutas, listado y muestra
# ----------------------------------------------------------------------------------------------


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def series_prefix(root: str, key: tuple[str, str, str]) -> str:
    provider, market, asset = key
    return f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"


def partition_rel(key: tuple[str, str, str], task: Task, name: str) -> str:
    """La ruta de `name` dentro de la raíz: `provider=…/theta=…/year=…/month=…/<name>`."""
    provider, market, asset = key
    return (
        f"provider={provider}/market={market}/asset={asset}/theta=0.{task.theta:0{THETA_DIGITS}d}"
        f"/year={task.year:04d}/month={task.month:02d}/{name}"
    )


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(uri)
        return fs, path.rstrip("/")
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def list_months(root: str, key: tuple[str, str, str], name: str) -> set[Task]:
    """Los `(θ, mes)` con archivo `name` bajo `root`: un listado recursivo, como la frontera."""
    fs, base = open_fs(series_prefix(root, key))
    selector = pafs.FileSelector(base, recursive=True, allow_not_found=True)
    pattern = re.compile(FILE_PATTERN.format(digits=THETA_DIGITS, name=re.escape(name)))
    found = set()
    try:
        for info in fs.get_file_info(selector):
            match = pattern.search(info.path)
            if match:
                found.add(Task(*(int(g) for g in match.groups())))
    except OSError as exc:
        raise InputError(
            f"no se pudo listar {root}: {type(exc).__name__}: {exc}"
        ) from exc
    return found


def stratified_sample(
    available: set[Task], size: int, seed: int
) -> tuple[list[Task], dict]:
    """`size` tareas en cuatro estratos (θ chicos o grandes × meses viejos o recientes).

    Cada estrato aporta un cuarto; el que no lo llena cede lo que le falta a los demás. Los
    estratos parten por la mediana de los θ (por valor) y de los meses (por fecha). Devuelve la
    muestra, ordenada, y cuántas tareas dio cada estrato.
    """
    ordered = sorted(available)
    if not ordered:
        return [], {}
    thetas = sorted({t.theta for t in ordered})
    months = sorted({(t.year, t.month) for t in ordered})
    theta_cut = thetas[len(thetas) // 2]
    month_cut = months[len(months) // 2]
    strata: dict[str, list[Task]] = defaultdict(list)
    for task in ordered:
        size_name = "grande" if task.theta >= theta_cut else "chico"
        age_name = "reciente" if (task.year, task.month) >= month_cut else "viejo"
        strata[f"theta_{size_name}_{age_name}"].append(task)
    rng = random.Random(seed)
    for pool in strata.values():
        rng.shuffle(pool)
    size = min(size, len(ordered))
    names = sorted(strata)
    taken: dict[str, list[Task]] = {name: [] for name in names}
    # Reparto por rondas de un cuarto; lo que sobra de un estrato vuelve a repartirse.
    while sum(map(len, taken.values())) < size:
        open_names = [n for n in names if len(taken[n]) < len(strata[n])]
        missing = size - sum(map(len, taken.values()))
        share = max(1, missing // len(open_names))
        for name in open_names:
            step = min(
                share,
                len(strata[name]) - len(taken[name]),
                size - sum(map(len, taken.values())),
            )
            taken[name] += strata[name][len(taken[name]) : len(taken[name]) + step]
    sample = sorted(task for chosen in taken.values() for task in chosen)
    return sample, {name: len(chosen) for name, chosen in taken.items()}


# ----------------------------------------------------------------------------------------------
# Lectura de un par de pies, por etapa
# ----------------------------------------------------------------------------------------------


class Config(NamedTuple):
    l3_root: str
    l2_root: str
    key: tuple[str, str, str]
    family: str
    # Conexiones del cliente de GCS de la variante b (los lectores de la corrida).
    pool: int = 1


class Reader:
    """Un sistema de archivos por raíz y por corrida, y la cola de un objeto con el cliente de GCS."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._roots: dict[str, tuple[pafs.FileSystem, str]] = {}
        self._client = None

    def locate(self, root: str, rel: str) -> tuple[pafs.FileSystem, str]:
        with self._lock:
            found = self._roots.get(root)
            if found is None:
                found = self._roots[root] = open_fs(root)
        fs, base = found
        return fs, f"{base}/{rel}"

    def _storage(self):
        with self._lock:
            if self._client is None:
                from google.cloud import storage

                client = storage.Client()
                try:
                    # El grupo de conexiones de `requests` es de 10: con más lectores, esperarían.
                    from requests.adapters import HTTPAdapter

                    adapter = HTTPAdapter(
                        pool_connections=self.config.pool, pool_maxsize=self.config.pool
                    )
                    client._http.mount("https://", adapter)
                except AttributeError:
                    pass
                self._client = client
        return self._client

    def tail(self, root: str, rel: str, size: int) -> bytes:
        """Los últimos `size` bytes del objeto (o el objeto entero si es menor)."""
        if root.startswith("gs://"):
            bucket, _, prefix = root.removeprefix("gs://").partition("/")
            blob = (
                self._storage()
                .bucket(bucket)
                .blob(f"{prefix.strip('/')}/{rel}".lstrip("/"))
            )
            return blob.download_as_bytes(start=-size)
        with open(f"{Path(root).resolve()}/{rel}", "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - size))
            return handle.read()


class Lap:
    """Cronómetro de una tarea: cada `mark` carga a una etapa lo que pasó desde la marca anterior."""

    def __init__(self) -> None:
        self.start = self.wall = time.perf_counter()
        self.cpu = time.thread_time()
        self.stage_wall = [0.0] * len(STAGES)
        self.stage_cpu = [0.0] * len(STAGES)

    def mark(self, stage: int) -> None:
        wall, cpu = time.perf_counter(), time.thread_time()
        self.stage_wall[stage] += wall - self.wall
        self.stage_cpu[stage] += cpu - self.cpu
        self.wall, self.cpu = wall, cpu

    def result(self, files: list[tuple[int, int]], error: str | None) -> Result:
        return Result(
            self.wall - self.start,
            tuple(self.stage_wall),
            tuple(self.stage_cpu),
            tuple(files),
            error,
        )


def row_group_sizes(meta: pq.FileMetaData) -> list[int]:
    sizes = (meta.row_group(i).num_rows for i in range(meta.num_row_groups))
    return [rows for rows in sizes if rows]


def open_footer(reader: Reader, root: str, rel: str, lap: Lap, handles: list):
    """Abre `rel` y lee su pie: dos etapas, como las dos mitades de `pq.ParquetFile(ruta)`."""
    fs, path = reader.locate(root, rel)
    handle = fs.open_input_file(path)
    handles.append(handle)
    lap.mark(OPEN)
    parquet = pq.ParquetFile(handle)
    lap.mark(FOOTER)
    return parquet


def check_base(reader: Reader, task: Task, deep: bool = True) -> Result:
    """La secuencia de `invalid_reason`; con `deep=False` (variante a) solo abre y lee los pies."""
    config = reader.config
    lap = Lap()
    handles: list = []
    files: list[tuple[int, int]] = []
    error = None
    try:
        rel = partition_rel(config.key, task, f"{config.family}.parquet")
        table = open_footer(reader, config.l3_root, rel, lap, handles)
        events = open_footer(
            reader,
            config.l2_root,
            partition_rel(config.key, task, "events.parquet"),
            lap,
            handles,
        )
        files = [(table.metadata.num_row_groups, table.metadata.serialized_size)]
        files.append((events.metadata.num_row_groups, events.metadata.serialized_size))
        if deep:
            skeleton = array("q", row_group_sizes(events.metadata))
            lap.mark(ROWS)
            schema = table.schema_arrow
            version = (schema.metadata or {}).get(b"state_version")
            index = schema.get_field_index("theta")
            wanted = Decimal(task.theta).scaleb(-THETA_DIGITS)
            meta = table.metadata
            wrong = 0
            for i in range(meta.num_row_groups):
                group = meta.row_group(i)
                if not group.num_rows:
                    continue
                stats = group.column(index).statistics if index >= 0 else None
                if (
                    stats is None
                    or not stats.has_min_max
                    or stats.min != wanted
                    or stats.max != wanted
                ):
                    wrong += 1
            lap.mark(STATS)
            same = array("q", row_group_sizes(meta)) == skeleton
            del (
                version,
                wrong,
                same,
            )  # la sonda no juzga: solo paga el costo de comparar
            lap.mark(ROWS)
    except Exception as exc:  # noqa: BLE001 - la sonda registra la tarea fallida y sigue
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
        lap.mark(CLOSE)
    return lap.result(files, error)


def check_tail(reader: Reader, task: Task) -> Result:
    """Variante b: la cola del objeto con el cliente de GCS y el pie decodificado en memoria."""
    config = reader.config
    lap = Lap()
    files: list[tuple[int, int]] = []
    error = None
    try:
        for root, name in (
            (config.l3_root, f"{config.family}.parquet"),
            (config.l2_root, "events.parquet"),
        ):
            rel = partition_rel(config.key, task, name)
            buffer = reader.tail(root, rel, TAIL_BYTES)
            if buffer[-4:] != b"PAR1":
                raise ValueError(f"{rel} no termina en PAR1")
            needed = int.from_bytes(buffer[-8:-4], "little") + 8
            if needed > len(buffer):
                buffer = reader.tail(root, rel, needed)
            lap.mark(OPEN)
            meta = pq.read_metadata(pa.BufferReader(buffer))
            files.append((meta.num_row_groups, meta.serialized_size))
            lap.mark(FOOTER)
    except Exception as exc:  # noqa: BLE001 - la sonda registra la tarea fallida y sigue
        error = f"{type(exc).__name__}: {exc}"
        lap.mark(OPEN)
    return lap.result(files, error)


RUNNERS = {
    "base": partial(check_base, deep=True),
    "a": partial(check_base, deep=False),
    "b": check_tail,
    "c": partial(check_base, deep=True),
}

# ----------------------------------------------------------------------------------------------
# Corridas
# ----------------------------------------------------------------------------------------------

_CHILD: Reader | None = None
_CHILD_BASE_CPU = 0.0


def _child_init(config: Config) -> None:
    """Línea base de CPU del proceso (importaciones hechas, sin tareas) y su lector."""
    global _CHILD, _CHILD_BASE_CPU
    _CHILD = Reader(config)
    _CHILD_BASE_CPU = time.process_time()


def _child_warm(_: int) -> int:
    """Calienta un proceso para que su arranque quede fuera de la pared; duerme para repartirse."""
    time.sleep(0.5)
    return os.getpid()


def _child_slice(
    job: tuple[list[Task], int],
) -> tuple[int, float, float, list[Result]]:
    """Un trozo de la muestra con `threads` hilos dentro del proceso: la base, en pequeño.

    La CPU se lee al terminar el trozo y se devuelve con la línea base del proceso."""
    tasks, threads = job
    runner = partial(check_base, _CHILD, deep=True)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(runner, tasks))
    return os.getpid(), time.process_time(), _CHILD_BASE_CPU, results


class Run(NamedTuple):
    variant: str
    workers: int
    wall: float
    cpu: float
    results: list[Result]

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def pies_per_s(self) -> float:
        return 2 * len(self.results) / self.wall if self.wall else math.nan

    @property
    def cpu_ratio(self) -> float:
        return self.cpu / self.wall if self.wall else math.nan

    @property
    def occupancy(self) -> float:
        return (
            sum(r.wall for r in self.results) / (self.wall * self.workers)
            if self.wall
            else math.nan
        )

    @property
    def label(self) -> str:
        return f"{self.variant}_{self.workers}"

    @property
    def over_limit(self) -> bool:
        """Más del 5 % de las tareas falla: sus pies por segundo no se pueden creer."""
        return self.errors > MAX_ERROR_SHARE * len(self.results)

    @property
    def decides(self) -> bool:
        """`base` y `c` deciden el veredicto; `a` y `b` solo orientan el arreglo."""
        return self.variant in DECIDING


def execute(
    variant: str,
    workers: int,
    config: Config,
    tasks: list[Task],
    processes: int = DEFAULT_PROCESSES,
) -> Run:
    """Corre `tasks` con `workers` lectores (hilos; en la variante c, repartidos en procesos)."""
    if variant == "c":
        return execute_processes(processes, workers, config, tasks)
    reader = Reader(config._replace(pool=workers))
    runner = partial(RUNNERS[variant], reader)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(runner, tasks))
    return Run(
        variant,
        workers,
        time.perf_counter() - wall0,
        time.process_time() - cpu0,
        results,
    )


def execute_processes(
    processes: int, workers: int, config: Config, tasks: list[Task]
) -> Run:
    """Variante c: los mismos `workers` lectores de la base, repartidos en `processes` procesos.

    Cada proceso corre un trozo de la muestra con sus hilos, así `c` y `base` con `workers`
    comparan la misma concurrencia y solo cambia que haya un GIL o varios. El arranque de los
    procesos queda fuera de la pared; la CPU es la de los hijos, desde su línea base."""
    processes = max(1, min(processes, workers, len(tasks)))
    threads = [
        workers // processes + (i < workers % processes) for i in range(processes)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=context,
        initializer=_child_init,
        initargs=(config,),
    ) as pool:
        list(pool.map(_child_warm, range(processes), chunksize=1))
        wall0, cpu0 = time.perf_counter(), time.process_time()
        jobs = [(tasks[i::processes], threads[i]) for i in range(processes)]
        slices = list(pool.map(_child_slice, jobs, chunksize=1))
        wall = time.perf_counter() - wall0
        parent_cpu = time.process_time() - cpu0
    latest: dict[int, tuple[float, float]] = {}
    for pid, cpu, base, _ in slices:
        latest[pid] = (max(cpu, latest.get(pid, (0.0, base))[0]), base)
    children = sum(cpu - base for cpu, base in latest.values())
    results = [r for *_, chunk in slices for r in chunk]
    return Run("c", workers, wall, parent_cpu + children, results)


# ----------------------------------------------------------------------------------------------
# Estadística, regla y salida
# ----------------------------------------------------------------------------------------------


def quantile(values: list[float], percent: int) -> float:
    """Rango más cercano: el lugar ⌈N·q⌉ de los valores ordenados; `nan` si no hay."""
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, -(-len(ordered) * percent // 100) - 1)]


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


def verdict(
    cores: int, ratio: float, scale: float, base_pps: float, process_pps: float | None
) -> str:
    """La regla del docstring. `ratio` es CPU ÷ pared de `base` con el mayor número de lectores."""
    if ratio >= CPU_SHARE * cores:
        return "cpu"
    if scale >= SCALE_OK or process_pps is None:
        return "mixto"
    if process_pps >= PROCESS_GAIN * base_pps:
        return "gil"
    if ratio < CPU_SHARE:
        return "espera"
    return "mixto"


def summarize(runs: list[Run], cores: int) -> dict:
    """Lo que entra a la última línea: `base_<n>`, la razón, la mejor corrida y el veredicto."""
    base = {r.workers: r for r in runs if r.variant == "base"}
    biggest, smallest = max(base), min(base)
    best = max((r for r in runs if not r.over_limit), key=lambda r: r.pies_per_s)
    process = next((r for r in runs if r.variant == "c"), None)
    scale = max(r.pies_per_s for r in base.values()) / base[smallest].pies_per_s
    return {
        "base": base,
        "biggest": biggest,
        "ratio": base[biggest].cpu_ratio,
        "best": best,
        "verdict": verdict(
            cores,
            base[biggest].cpu_ratio,
            scale,
            base[biggest].pies_per_s,
            process.pies_per_s if process else None,
        ),
    }


def final_line(
    sample: int | str, summary: dict | None, readers: tuple[int, ...]
) -> str:
    if summary is None:
        walls = " ".join(f"base_{n}=-" for n in readers)
        return (
            f"frontier probe: muestra={sample} {walls} cpu_ratio_{max(readers)}=- "
            "mejor_variante=- pies_por_s_mejor=- veredicto=fallo"
        )
    walls = " ".join(f"base_{n}={summary['base'][n].wall:.1f}s" for n in readers)
    best = summary["best"]
    return (
        f"frontier probe: muestra={sample} {walls} "
        f"cpu_ratio_{summary['biggest']}={summary['ratio']:.2f} mejor_variante={best.label} "
        f"pies_por_s_mejor={best.pies_per_s:.1f} veredicto={summary['verdict']}"
    )


def file_stats(results: list[Result]) -> dict[str, dict[str, list[float]]]:
    """Row groups y bytes del pie por archivo, de los resultados sin error; claves `path` y `events`."""
    columns: dict[str, dict[str, list[float]]] = {
        "familia": {"row_groups": [], "pie_bytes": []},
        "events": {"row_groups": [], "pie_bytes": []},
    }
    for result in results:
        if result.error or len(result.files) != 2:
            continue
        for name, (groups, size) in zip(columns, result.files, strict=True):
            columns[name]["row_groups"].append(groups)
            columns[name]["pie_bytes"].append(size)
    return columns


def csv_rows(runs: list[Run], environment: dict, strata: dict) -> list[dict]:
    rows: list[dict] = []

    def add(section, variant, workers, stage, measure, value):
        rows.append(
            {
                "seccion": section,
                "variante": variant,
                "lectores": workers,
                "etapa": stage,
                "medida": measure,
                "valor": value if isinstance(value, str | int) else f"{value:.6g}",
            }
        )

    for name, value in environment.items():
        add("entorno", "", "", "", name, value)
    for name, count in strata.items():
        add("muestra", "", "", name, "tareas", count)
    for name, measures in file_stats(runs[0].results).items():
        for measure, values in measures.items():
            for label, value in (
                ("mediana", statistics.median(values) if values else math.nan),
                ("p95", quantile(values, 95)),
                ("max", max(values, default=math.nan)),
            ):
                add("archivo", "", "", name, f"{measure}_{label}", value)
    for run in runs:
        for measure, value in (
            ("pared_s", run.wall),
            ("pies_por_s", run.pies_per_s),
            ("cpu_s", run.cpu),
            ("cpu_ratio", run.cpu_ratio),
            ("ocupacion", run.occupancy),
            ("errores", run.errors),
        ):
            add("corrida", run.variant, run.workers, "", measure, value)
        good = [r for r in run.results if not r.error] or run.results
        for index, stage in enumerate(STAGES):
            walls = [r.stage_wall[index] * 1000 for r in good]
            add(
                "etapa",
                run.variant,
                run.workers,
                stage,
                "mediana_ms",
                statistics.median(walls),
            )
            add("etapa", run.variant, run.workers, stage, "p95_ms", quantile(walls, 95))
            add(
                "etapa",
                run.variant,
                run.workers,
                stage,
                "total_s",
                sum(r.stage_wall[index] for r in good),
            )
            add(
                "etapa",
                run.variant,
                run.workers,
                stage,
                "cpu_hilo_s",
                sum(r.stage_cpu[index] for r in good),
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


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def print_tables(runs: list[Run]) -> None:
    log(
        f"{'corrida':<8} {'pared_s':>9} {'pies/s':>8} {'cpu_s':>8} {'cpu/pared':>10} {'ocupación':>10} {'errores':>8}"
    )
    for run in runs:
        log(
            f"{run.label:<8} {run.wall:>9.1f} {run.pies_per_s:>8.1f} {run.cpu:>8.1f} "
            f"{run.cpu_ratio:>10.2f} {run.occupancy:>10.2f} {run.errors:>8}"
        )
    log("etapas por tarea (ms; mediana / p95 / CPU del hilo ÷ pared de la etapa)")
    log(f"{'corrida':<8} " + " ".join(f"{s:>26}" for s in STAGES))
    for run in runs:
        good = [r for r in run.results if not r.error] or run.results
        cells = []
        for index in range(len(STAGES)):
            walls = [r.stage_wall[index] * 1000 for r in good]
            wall_total = sum(r.stage_wall[index] for r in good)
            cpu_total = sum(r.stage_cpu[index] for r in good)
            share = cpu_total / wall_total if wall_total else math.nan
            cells.append(
                f"{statistics.median(walls):>7.2f}/{quantile(walls, 95):>7.2f}/{share:>6.2f}"
            )
        log(f"{run.label:<8} " + " ".join(f"{c:>26}" for c in cells))


def parse_readers(text: str) -> tuple[int, ...]:
    try:
        readers = tuple(sorted({int(p) for p in text.split(",") if p.strip()}))
    except ValueError:
        raise ValueError(
            f"--readers {text!r}: se esperan enteros separados por comas"
        ) from None
    if not readers or readers[0] < 1:
        raise ValueError(f"--readers {text!r}: cada número debe ser al menos 1")
    return readers


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    try:
        readers = parse_readers(ns.readers)
    except ValueError as exc:
        parser.error(str(exc))
    variants = [v for v in ns.variants.split(",") if v]
    if "base" not in variants or set(variants) - set(RUNNERS):
        parser.error(
            f"--variants {ns.variants!r}: se esperan base, a, b y c, y base siempre"
        )
    if ns.sample < 1 or ns.processes < 1:
        parser.error("--sample y --processes deben ser al menos 1")
    name_project = project()
    if not (ns.l3_root and ns.l2_root) and name_project is None:
        parser.error(
            "sin --l3-root o --l2-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    l3_root = ns.l3_root or f"gs://{name_project}-dc-descriptors/l3"
    l2_root = ns.l2_root or f"gs://{name_project}-dc-events/l2"
    name = "l3_frontier_probe.csv"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))
    key = (ns.provider, ns.market, ns.asset)
    config = Config(l3_root, l2_root, key, ns.family)
    cores = container_cores()

    try:
        available = list_months(l3_root, key, f"{ns.family}.parquet") & list_months(
            l2_root, key, "events.parquet"
        )
        tasks, strata = stratified_sample(available, ns.sample, ns.seed)
        if not tasks:
            raise InputError(
                f"no hay (θ, mes) con {ns.family}.parquet y events.parquet en {l3_root} y {l2_root}"
            )
    except InputError as exc:
        log(f"FALLO {exc}")
        log(final_line("-", None, readers))
        return 1
    log(
        f"muestra: {len(tasks)} de {len(available)} (θ, mes); estratos {strata}; cores={cores}; pyarrow={pa.__version__}"
    )

    runs: list[Run] = []
    # Primero las que deciden el veredicto (base y c); si el presupuesto de tiempo se acaba, las
    # variantes a y b son las que se pierden.
    plan = [("base", n) for n in readers]
    plan += [("c", readers[-1])] if "c" in variants else []
    plan += [(v, n) for v in ("a", "b") if v in variants for n in readers]
    started = time.perf_counter()
    for variant, workers in plan:
        if time.perf_counter() - started > ns.budget_s:
            log(f"{variant}_{workers}: omitida, pasó el presupuesto de {ns.budget_s} s")
            continue
        try:
            done = execute(variant, workers, config, tasks, ns.processes)
        except Exception as exc:  # noqa: BLE001 - una variante que no arranca no tumba la medición
            log(f"{variant}_{workers}: no corrió ({type(exc).__name__}: {exc})")
            continue
        runs.append(done)
        log(
            f"{done.label}: pared={done.wall:.1f}s pies/s={done.pies_per_s:.1f} cpu/pared={done.cpu_ratio:.2f} errores={done.errors}"
        )
        if done.errors:
            first = next(r.error for r in done.results if r.error)
            log(f"  primer error de {done.label}: {first}")
    print_tables(runs)

    environment = {
        "cores": cores,
        "pyarrow": pa.__version__,
        "python": sys.version.split()[0],
        "muestra": len(tasks),
        "disponibles": len(available),
        "semilla": ns.seed,
        "familia": ns.family,
    }
    try:
        write_csv(csv_rows(runs, environment, strata), out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    for r in runs:
        if r.over_limit and not r.decides:
            log(
                f"AVISO {r.label}: más del {MAX_ERROR_SHARE:.0%} de sus tareas falla; "
                "no entra en mejor_variante y no cambia el veredicto"
            )
    if any(r.over_limit for r in runs if r.decides):
        log(f"FALLO más del {MAX_ERROR_SHARE:.0%} de las tareas de base o c falla")
        log(final_line(len(tasks), None, readers))
        return 1
    if {r.workers for r in runs if r.variant == "base"} != set(readers):
        log("FALLO alguna corrida base no corrió: sin ella no hay veredicto")
        log(final_line(len(tasks), None, readers))
        return 1
    log(final_line(len(tasks), summarize(runs, cores), readers))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--readers", default=",".join(map(str, DEFAULT_READERS)))
    parser.add_argument("--variants", default="base,a,b,c")
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--budget-s", type=int, default=DEFAULT_BUDGET_S)
    parser.add_argument("--l3-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--family", default="path")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
