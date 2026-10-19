"""Lo que mide una unidad de L3 (un mes) para el hallazgo `unit_timing` (TRD-L3 §9.3).

Es lo que ITSC-335 necesita para el umbral 3 de §10.3: pared total y por fase, RSS pico,
row groups de L1 decodificados (del mes y de meses anteriores), eventos y ticks por θ y
el límite efectivo de CPU. Todo se acumula en objetos que crea la unidad (`Phases`,
`RowGroupCount`) y que mueren con ella: ningún módulo guarda cuentas, así que dos
unidades en el mismo proceso, en hilos o tareas distintas, no se pisan.

Las fases son pared del hilo que las corre y, en serie, suman a lo más `wall_s`; lo que
falta es `other_s`. Si un día los θ de un row group se agregan en varios hilos,
`write_s` (la suma de lo que cada θ gastó escribiendo) puede pasar de la pared:
`aggregate_s` se calcula restándola y queda en 0 como mínimo.

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
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import pyarrow.parquet as pq
from dc_frames import lake

from l3_dc_descriptors.cpu import CpuLimit

MIB = 1 << 20


class Phases:
    """Segundos de cada fase de la unidad. Los llena la unidad (hilo principal); solo
    `write_s` lo suma cada θ (`_Lane`) sobre su propio contador.
    """

    __slots__ = (
        "aggregate_s",
        "gate_s",
        "lineage_s",
        "read_s",
        "reconcile_s",
        "write_s",
    )

    def __init__(self) -> None:
        self.gate_s = 0.0
        self.read_s = 0.0
        self.aggregate_s = 0.0
        self.write_s = 0.0
        self.reconcile_s = 0.0
        self.lineage_s = 0.0


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
    _seen: set[tuple[str, int]] = field(default_factory=set, repr=False)

    def opened(self, path: str, row_groups: int) -> None:
        if path == self.month_path:
            self.month_total = row_groups

    def decoded(self, path: str, index: int, size: int) -> None:
        if path == self.month_path:
            self.month += 1
        else:
            self.previous += 1
        if (path, index) in self._seen:
            self.repeated += 1
        self._seen.add((path, index))
        self.bytes_in += size


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


def _open_counting(path: str, what: str) -> pq.ParquetFile:
    """Sustituto de `dc_frames.lake.open_parquet`: igual al original, y cuenta los
    row groups de `consolidated.parquet` si la unidad en curso trae un contador.
    """
    parquet = _OPEN(path, what)
    count = _COUNT.get()
    if count is None or not path.endswith(lake.CONSOLIDATED):
        return parquet
    count.opened(path, parquet.num_row_groups)
    read = parquet.read_row_group

    def read_row_group(i, columns=None, **kwargs):
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
        explained = (
            p.gate_s
            + p.read_s
            + p.aggregate_s
            + p.write_s
            + p.reconcile_s
            + p.lineage_s
        )
        return max(0.0, self.wall_s - explained)

    def details(self) -> dict:
        """El payload del hallazgo `unit_timing` (segundos con ms de resolución).

        `read_s` es lectura y decodificación (lo que el hilo principal esperó al lector
        por cada row group); `aggregate_s`, la agregación de los θ; `write_s`, armar y
        escribir los row groups de salida y publicar; `reconcile_s`, §9.1; `lineage_s`,
        el hash de las entradas de L2; `gate_s`, la compuerta de entrada. `cpu_user_s` y
        `cpu_system_s` son la CPU del proceso (todos sus hilos) durante la unidad y
        `cpu_utilization`, `cpu_s / (wall_s × cores)`.
        """
        p = self.phases
        seconds = {
            "wall_s": self.wall_s,
            "gate_s": p.gate_s,
            "read_s": p.read_s,
            "aggregate_s": p.aggregate_s,
            "write_s": p.write_s,
            "reconcile_s": p.reconcile_s,
            "lineage_s": p.lineage_s,
            "other_s": self.other_s,
        }
        counts = self.row_groups
        utilization = self.cpu_utilization
        return {
            **{name: round(value, 3) for name, value in seconds.items()},
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
        }
