"""Lo que mide una unidad de L3 (un mes) para el hallazgo `unit_timing` (TRD-L3 §9.3).

Es lo que ITSC-335 necesita para el umbral 3 de §10.3: pared total y por fase, RSS pico,
row groups de L1 decodificados (del mes y de meses anteriores), eventos y ticks por θ y
el límite efectivo de CPU. Todo se acumula en objetos que crea la unidad (`Phases`,
`RowGroupCount`) y que mueren con ella: ningún módulo guarda cuentas, así que dos
unidades en el mismo proceso, en hilos o tareas distintas, no se pisan.

Todas las fases son **pared**, no suma sobre hilos (ITSC-363): `gate_s`, `setup_s`,
`reconcile_s` y `publish_s` son tramos seguidos del hilo principal; `gate_s` es el primer
tramo de la entrada de los θ (abrir `events.parquet` y arrancar la lectura de sus fronteras,
ITSC-365) y `setup_s`, lo que queda hasta el primer row group (motor, pie de L1 y compuerta
de carry-over; los escritores se abren en ese tramo, pero nadie los espera); el linaje ya no
es una fase: sale de los
bytes que la lectura trae (`lineage_s` desapareció en ITSC-365);
`aggregate_s` es la pared del motor, del primer `push` a que el último ítem termina;
`write_s`, la pared con algún escritor ocupado (la unión de sus intervalos), y
`read_wall_s`, la pared con alguna lectura de L1 o L2 en vuelo. `write_s` y `read_wall_s`
corren dentro de `aggregate_s`, no a su lado: `other_s` es lo que falta hasta `wall_s` de
las fases del hilo principal. `read_s` es lo que el hilo principal esperó a las lecturas.

**El gancho de row groups.** Contar las decodificaciones de L1 exige mirar los
`read_row_group` del lector, que abre sus archivos con `dc_frames.lake.open_parquet`
(como hace la sonda de `ops_tools`). `count_row_groups` instala una sola vez un
sustituto que delega en el original y, si el contexto actual trae un contador, envuelve
el archivo de L1 que abre. El contador viaja en una `ContextVar`: es del hilo o la tarea
que corre la unidad. No cuenta las lecturas de la conciliación (`reconcile.L1Month`,
solo la columna `agg_trade_id` de los row groups de las fronteras), que no pasan por el
lector.
"""

import resource
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import pyarrow.parquet as pq
from dc_frames import lake

from l3_dc_descriptors.cpu import CpuLimit, KernelCounts

MIB = 1 << 20


class BusyClock:
    """Segundos con al menos una operación en curso: la pared de una fase concurrente.

    `busy()` marca una operación; con varias a la vez los intervalos se unen, así que el
    total no pasa de la pared aunque haya muchos hilos. Es seguro entre hilos.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._since = 0.0
        self._seconds = 0.0

    @contextmanager
    def busy(self) -> Iterator[None]:
        with self._lock:
            if self._active == 0:
                self._since = time.perf_counter()
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                if self._active == 0:
                    self._seconds += time.perf_counter() - self._since

    @property
    def seconds(self) -> float:
        with self._lock:
            open_for = time.perf_counter() - self._since if self._active else 0.0
            return self._seconds + open_for


class ReadMeter:
    """Lo que la unidad leyó de L1 y L2: bytes comprimidos y la pared de las lecturas.

    Cada lectura (`reading()`) corre dentro de su hilo, también de L1 por el gancho de
    `count_row_groups`; la pared es la unión de sus intervalos (`BusyClock`), lectura y
    decodificación juntas, que es lo que Arrow hace en una llamada.
    """

    def __init__(self) -> None:
        self.clock = BusyClock()
        self._lock = threading.Lock()
        self._bytes = 0

    def reading(self):
        return self.clock.busy()

    def add(self, size: int) -> None:
        with self._lock:
            self._bytes += size

    @property
    def bytes(self) -> int:
        with self._lock:
            return self._bytes


class Retained:
    """Bytes de fronteras de L2 que el motor tiene en mano, con su máximo.

    `hold(size, anchor)` anota un lote que se entregó al motor; cuando `anchor` (el buffer
    de Arrow que el motor ancla) se libera, el lote deja de contar. Mide lo que el motor de
    verdad retiene, que es de lotes enteros, no una cuenta de eventos (ITSC-363). Es seguro
    entre hilos: el motor suelta los lotes desde sus propios hilos.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.peak = 0

    def hold(self, size: int, anchor: object) -> None:
        with self._lock:
            self._current += size
            self.peak = max(self.peak, self._current)
        weakref.finalize(anchor, self._release, size)

    def _release(self, size: int) -> None:
        with self._lock:
            self._current -= size

    @property
    def current(self) -> int:
        with self._lock:
            return self._current


class Phases:
    """Segundos de cada fase de la unidad y la configuración con que corrió. El hilo
    principal llena las fases; los relojes (`read`, `write`) los comparten los hilos de
    lectura y de escritura.
    """

    __slots__ = (
        "aggregate_s",
        "engine_busy",
        "engine_threads",
        "gate_s",
        "inflight",
        "inflight_alone_bytes_max",
        "inflight_bytes",
        "inflight_bytes_max",
        "inflight_events_max",
        "inflight_max",
        "items",
        "publish_s",
        "read",
        "read_s",
        "readers",
        "reconcile_s",
        "retained",
        "setup_s",
        "threads",
        "write",
        "write_s",
        "writers",
    )

    def __init__(self) -> None:
        self.threads = 1
        self.readers = 1
        self.inflight = 1
        self.writers = 1
        self.engine_threads = 1
        self.inflight_max = 0
        self.inflight_bytes = 0
        self.inflight_events_max = 0
        self.inflight_bytes_max = 0
        self.inflight_alone_bytes_max = 0
        self.items = 0
        self.engine_busy = 0.0
        self.gate_s = 0.0
        self.setup_s = 0.0
        self.read_s = 0.0
        self.aggregate_s = 0.0
        self.write_s = 0.0
        self.publish_s = 0.0
        self.reconcile_s = 0.0
        self.read = ReadMeter()
        self.write = BusyClock()
        self.retained = Retained()


@dataclass(frozen=True, slots=True)
class ThetaStats:
    """Lo que la unidad hizo con un θ: eventos escritos y sus ticks."""

    theta: str
    events: int
    ticks: int
    ticks_before_month: int

    def details(self) -> dict:
        return {
            "theta": self.theta,
            "events": self.events,
            "ticks": self.ticks,
            "ticks_before_month": self.ticks_before_month,
        }


@dataclass
class RowGroupCount:
    """Decodificaciones de row group de L1 de una unidad.

    `month` y `previous` cuentan las de `consolidated.parquet` del mes y las de meses
    anteriores (los eventos que cruzan el borde de mes). `repeated` son las que
    repiten un row group ya decodificado: con el lector por row group deben ser 0.
    `bytes_in` son los bytes comprimidos de las columnas decodificadas y `month_total`,
    los row groups del archivo del mes.
    """

    month_path: str
    month: int = 0
    previous: int = 0
    repeated: int = 0
    bytes_in: int = 0
    month_total: int = 0
    meter: ReadMeter | None = None
    _seen: set[tuple[str, int]] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def opened(self, path: str, row_groups: int) -> None:
        if path == self.month_path:
            self.month_total = row_groups

    def decoded(self, path: str, index: int, size: int) -> None:
        # Los lectores de L1 corren en hilos: las cuentas se actualizan bajo un candado.
        with self._lock:
            if path == self.month_path:
                self.month += 1
            else:
                self.previous += 1
            if (path, index) in self._seen:
                self.repeated += 1
            self._seen.add((path, index))
            self.bytes_in += size
        if self.meter is not None:
            self.meter.add(size)


_COUNT: ContextVar[RowGroupCount | None] = ContextVar("l3_row_groups", default=None)
# El `open_parquet` que el lector usaría sin el gancho (constante: se fija al importar).
_OPEN = lake.open_parquet


def _compressed(parquet: pq.ParquetFile, index: int, columns) -> int:
    group = parquet.metadata.row_group(index)
    wanted = set(columns) if columns is not None else None
    chunks = (group.column(i) for i in range(group.num_columns))
    return sum(
        c.total_compressed_size
        for c in chunks
        if wanted is None or c.path_in_schema in wanted
    )


def _open_counting(
    path: str,
    what: str,
    metadata: pq.FileMetaData | None = None,
    pre_buffer: bool = False,
) -> pq.ParquetFile:
    """Sustituto de `dc_frames.lake.open_parquet`: igual al original, y cuenta los
    row groups de `consolidated.parquet` si la unidad en curso trae un contador.
    """
    parquet = _OPEN(path, what, metadata, pre_buffer)
    count = _COUNT.get()
    if count is None or not path.endswith(lake.CONSOLIDATED):
        return parquet
    count.opened(path, parquet.num_row_groups)
    read = parquet.read_row_group

    def read_row_group(i, columns=None, **kwargs):
        if count.meter is None:
            table = read(i, columns=columns, **kwargs)
        else:
            with count.meter.reading():
                table = read(i, columns=columns, **kwargs)
        count.decoded(path, i, _compressed(parquet, i, columns))
        return table

    parquet.read_row_group = read_row_group
    return parquet


@contextmanager
def count_row_groups(count: RowGroupCount) -> Iterator[RowGroupCount]:
    """Cuenta en `count` los row groups de L1 que el lector decodifique en este contexto."""
    if lake.open_parquet is not _open_counting:
        lake.open_parquet = _open_counting
    token = _COUNT.set(count)
    try:
        yield count
    finally:
        _COUNT.reset(token)


def reset_rss_peak() -> bool:
    """Pone el pico de RSS del proceso (`VmHWM`) en el RSS actual; False si no se puede.

    Sin esto `VmHWM` es el pico de todo el proceso y un `backfill` de varios meses
    repetiría el del mes más pesado en los siguientes. Linux ≥ 4.0 (`clear_refs` = 5).
    """
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


def rss_peak_bytes() -> int:
    """El RSS pico del proceso. En Linux, `VmHWM`: `ru_maxrss` hereda el pico del padre al
    hacer exec.
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _mib(size: int | None) -> float | None:
    return None if size is None else round(size / MIB, 1)


@dataclass(frozen=True)
class Timing:
    """Lo que midió la unidad. `wall_s` corre desde que `process_unit` empieza hasta
    justo antes de emitir los hallazgos.
    """

    wall_s: float
    phases: Phases
    rss_peak_bytes: int
    rss_scope: str
    row_groups: RowGroupCount
    thetas: tuple[ThetaStats, ...]
    cpu: CpuLimit
    cpu_throttled_s: float | None
    cpu_user_s: float
    cpu_system_s: float
    kernel: KernelCounts
    mmap_threshold: int | None = None
    trim_threshold: int | None = None
    rss_before_trim_bytes: int | None = None
    rss_after_trim_bytes: int | None = None
    spawn_s: float | None = None

    @property
    def cpu_s(self) -> float:
        return self.cpu_user_s + self.cpu_system_s

    @property
    def cpu_utilization(self) -> float | None:
        """`cpu_s / (wall_s × cores)`: la fracción de sus vCPU que usó la unidad."""
        capacity = self.wall_s * self.cpu.cores
        return self.cpu_s / capacity if capacity > 0 else None

    @property
    def events(self) -> int:
        return sum(t.events for t in self.thetas)

    @property
    def ticks(self) -> int:
        return sum(t.ticks for t in self.thetas)

    @property
    def other_s(self) -> float:
        p = self.phases
        explained = p.gate_s + p.setup_s + p.aggregate_s + p.publish_s + p.reconcile_s
        return max(0.0, self.wall_s - explained)

    def details(self) -> dict:
        """El payload del hallazgo `unit_timing` (segundos con ms de resolución).

        Todas las fases son pared (ver el módulo). `gate_s`, el primer tramo de la entrada
        de los θ: abrir `events.parquet` y arrancar la lectura de sus fronteras, una tarea
        por θ; `setup_s`, el motor, el pie de L1 y la compuerta de carry-over (los
        escritores se abren en el pool durante ese tramo, sin que nadie los espere);
        `aggregate_s`, la pared del motor, del primer row group a que termina el último
        ítem (incluye lo que espera por las lecturas y por la contrapresión); `write_s`, la
        pared con algún escritor ocupado; `publish_s`, publicar los archivos; `reconcile_s`,
        §9.1. `read_s` es lo que el hilo principal esperó a las lecturas y `read_wall_s`, la
        pared con alguna lectura de L1 o L2 en vuelo (lectura, decodificación y el hash del
        linaje de L2); `read_bytes`, los bytes comprimidos que leyó de las dos. `threads` es
        el pool de E/S (pasada de entrada, L2, conciliación y publicación); `engine_threads`,
        los hilos del motor; `readers`, los row groups de L1 que se leen a la vez; `inflight`,
        el tope de row groups empujados al motor que aún anclan sus ticks e `inflight_max`,
        el máximo que hubo; `boundaries_peak_mib`, el máximo de fronteras de L2 que el motor
        tuvo en mano; `writers`, los escritores; `engine_busy`, la fracción del tiempo del
        bucle con ítems en la cola del motor; `items`, los ítems `(row group, θ)` que corrió.
        `cpu_user_s` y `cpu_system_s` son la CPU del proceso (todos sus hilos) durante la
        unidad y `cpu_utilization`, `cpu_s / (wall_s × cores)`. `minor_faults`,
        `major_faults`, `voluntary_switches` e `involuntary_switches` son los contadores
        del kernel (`getrusage`) de la unidad: explican `cpu_system_s` (memoria contra
        candados y red). `mmap_threshold` es el umbral de `mmap` de glibc con que corrió
        (`MALLOC_MMAP_THRESHOLD_` o 33 554 432; `None` si quedó dinámico) y `trim_threshold`, el
        de recorte del heap (`MALLOC_TRIM_THRESHOLD_`; `None` si no está fijado), para que
        cada corrida diga con qué corrió. `rss_before_trim_mib` y `rss_after_trim_mib` son el
        RSS actual del proceso (`VmRSS`, no el pico) justo antes y después de que la unidad
        devolviera al kernel lo libre con `malloc_trim(0)` (ITSC-371); el pico (`rss_peak_mib`)
        no baja, y el que sube de unidad en unidad es este. `inflight_bytes` es el tope de
        bytes de eventos abiertos (`L3_INFLIGHT_BYTES`); `inflight_events_max` es el máximo
        de eventos entregados al motor que aún no cerraron e `inflight_bytes_max`, lo mismo
        por los bytes de un evento (ITSC-372). El tope se mira antes de empujar, así que
        `inflight_bytes_max` puede pasar de `inflight_bytes` hasta en un row group sin que
        ninguno lo pase solo: es lo normal, no una señal. El límite conocido es
        `inflight_alone_bytes_max`, los bytes abiertos tras empujar un row group a un motor
        vacío: si pasa de `inflight_bytes`, un solo row group trajo más eventos que el tope
        (y salió el aviso `L3_INFLIGHT_BYTES`). `spawn_s` es lo que tardó en arrancar el proceso hijo
        de la unidad (intérprete, importaciones y argumentos), `null` si corrió en el proceso
        de la CLI (`L3_UNIT_PROCESS=0` o `monthly`).
        """
        p = self.phases
        seconds = {
            "wall_s": self.wall_s,
            "gate_s": p.gate_s,
            "setup_s": p.setup_s,
            "read_s": p.read_s,
            "read_wall_s": p.read.clock.seconds,
            "aggregate_s": p.aggregate_s,
            "write_s": p.write.seconds,
            "publish_s": p.publish_s,
            "reconcile_s": p.reconcile_s,
            "other_s": self.other_s,
        }
        counts = self.row_groups
        utilization = self.cpu_utilization
        return {
            **{name: round(value, 3) for name, value in seconds.items()},
            "threads": p.threads,
            "engine_threads": p.engine_threads,
            "readers": p.readers,
            "inflight": p.inflight,
            "inflight_max": p.inflight_max,
            "inflight_bytes": p.inflight_bytes,
            "inflight_events_max": p.inflight_events_max,
            "inflight_bytes_max": p.inflight_bytes_max,
            "inflight_alone_bytes_max": p.inflight_alone_bytes_max,
            "writers": p.writers,
            "engine_busy": round(p.engine_busy, 3),
            "items": p.items,
            "read_bytes": p.read.bytes,
            "boundaries_peak_mib": round(p.retained.peak / MIB, 1),
            "rss_peak_mib": round(self.rss_peak_bytes / MIB, 1),
            "rss_peak_scope": self.rss_scope,
            "row_groups_month": counts.month,
            "row_groups_previous": counts.previous,
            "row_groups_repeated": counts.repeated,
            "row_groups_month_total": counts.month_total,
            "bytes_in": counts.bytes_in,
            "thetas": len(self.thetas),
            "events": self.events,
            "ticks": self.ticks,
            "theta_stats": [t.details() for t in self.thetas],
            "cores": self.cpu.cores,
            "cores_visible": self.cpu.visible,
            "cores_source": self.cpu.source,
            "cpu_throttled_s": (
                None if self.cpu_throttled_s is None else round(self.cpu_throttled_s, 3)
            ),
            "cpu_user_s": round(self.cpu_user_s, 3),
            "cpu_system_s": round(self.cpu_system_s, 3),
            "cpu_s": round(self.cpu_s, 3),
            "cpu_utilization": None if utilization is None else round(utilization, 3),
            "minor_faults": self.kernel.minor_faults,
            "major_faults": self.kernel.major_faults,
            "voluntary_switches": self.kernel.voluntary_switches,
            "involuntary_switches": self.kernel.involuntary_switches,
            "mmap_threshold": self.mmap_threshold,
            "trim_threshold": self.trim_threshold,
            "rss_before_trim_mib": _mib(self.rss_before_trim_bytes),
            "rss_after_trim_mib": _mib(self.rss_after_trim_bytes),
            "spawn_s": None if self.spawn_s is None else round(self.spawn_s, 3),
        }
