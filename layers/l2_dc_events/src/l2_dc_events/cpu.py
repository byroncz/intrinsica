"""Límite efectivo de CPU del proceso, su estrangulamiento y el paralelismo
que de ahí se deriva (ITSC-289, ITSC-290).

`os.sched_getaffinity` dice qué cores ve el proceso, no cuántos puede usar: en
Cloud Run la cuota de vCPU del job es un límite del cgroup (`cpu.max`) y la
máquina reporta más cores (6, 6 y 9 con 2, 4 y 8 vCPU, ITSC-281). El límite
efectivo es el menor de los dos.

Este módulo es el único que decide cuántos hilos usa la unidad. Ni
`thread::available_parallelism` de Rust (que divide cuota entre período con
enteros: una cuota de 1,97 vCPU da 1 hilo) ni `len(sched_getaffinity)` (5 o 6
hilos sobre una cuota de 2, que desalojan al hilo principal) sirven de guía.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")


@dataclass(frozen=True)
class CpuLimit:
    """`cores` es el límite efectivo (puede ser fraccionario); `visible`, los
    cores que ve el proceso. `source` dice quién fijó `cores`: `cgroup-v2` o
    `cgroup-v1` si su cuota es la que limita, `affinity` si no hay cuota o no
    baja de los cores visibles.
    """

    cores: float
    visible: int
    source: str

    @property
    def fanout_threads(self) -> int:
        """Hilos del fan-out: `ceil(cuota)` sin pasar de los cores visibles.

        Se redondea hacia arriba porque Cloud Run entrega cuotas como 1,97 para
        2 vCPU y `floor` dejaría al detector con un hilo. El hilo de más
        compite por una fracción de CPU, pero es menos costoso que dejar el
        fan-out sin paralelismo.
        """
        return max(1, min(self.visible, math.ceil(self.cores)))

    @property
    def writers(self) -> int:
        """Hilos de escritura de Parquet: `round(cuota)`, al menos 1.

        Más escritores que cuota sobresuscriben el cgroup y desalojan al hilo
        principal, que es quien alimenta el detector.
        """
        return max(1, round(self.cores))


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _v2_dirs(root: Path, proc_cgroup: Path) -> list[Path]:
    """Del cgroup del proceso hasta la raíz: la cuota de un ancestro también limita."""
    text = _read(proc_cgroup) or ""
    relative = next(
        (line[3:] for line in text.splitlines() if line.startswith("0::")), "/"
    )
    leaf = root / relative.lstrip("/")
    dirs = [leaf, *leaf.parents] if leaf != root else [root]
    return [d for d in dirs if d == root or root in d.parents]


def _quota_v2(root: Path, proc_cgroup: Path) -> float | None:
    quotas = []
    for directory in _v2_dirs(root, proc_cgroup):
        fields = (_read(directory / "cpu.max") or "").split()
        if len(fields) == 2 and fields[0] != "max":
            quotas.append(int(fields[0]) / int(fields[1]))
    return min(quotas) if quotas else None


def _v1_dirs(root: Path, proc_cgroup: Path) -> list[Path]:
    """Directorios del controlador `cpu` (v1) del proceso, de su cgroup a la raíz.

    La raíz (`root/cpu`) rara vez trae cuota: la cuota está en el cgroup del
    proceso o en uno de sus ancestros. Si la ruta de `/proc/self/cgroup` no
    existe bajo `root` (el contenedor monta solo su subárbol), se usa la raíz
    del controlador.
    """
    text = _read(proc_cgroup) or ""
    relative = "/"
    for line in text.splitlines():
        _, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if "cpu" in controllers.split(","):
            relative = path
            break
    for base in (root / "cpu", root / "cpu,cpuacct"):
        if not base.is_dir():
            continue
        leaf = base / relative.lstrip("/")
        chain = [leaf, *leaf.parents] if leaf != base else [base]
        found = [d for d in chain if (d == base or base in d.parents) and d.is_dir()]
        return found or [base]
    return []


def _quota_v1(root: Path, proc_cgroup: Path) -> float | None:
    quotas = []
    for directory in _v1_dirs(root, proc_cgroup):
        quota = _read(directory / "cpu.cfs_quota_us")
        period = _read(directory / "cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            quotas.append(int(quota) / int(period))
    return min(quotas) if quotas else None


def cpu_limit(
    root: Path = CGROUP_ROOT,
    proc_cgroup: Path = PROC_CGROUP,
    visible: int | None = None,
) -> CpuLimit:
    """El límite efectivo de CPU: cuota del cgroup o, sin ella, los cores visibles."""
    visible = visible if visible is not None else len(os.sched_getaffinity(0))
    for source, quota in (
        ("cgroup-v2", _quota_v2(root, proc_cgroup)),
        ("cgroup-v1", _quota_v1(root, proc_cgroup)),
    ):
        if quota is not None and quota < visible:
            return CpuLimit(quota, visible, source)
    return CpuLimit(float(visible), visible, "affinity")


def throttled_s(
    root: Path = CGROUP_ROOT, proc_cgroup: Path = PROC_CGROUP
) -> float | None:
    """Segundos que el cgroup del proceso lleva sin poder correr por agotar su
    cuota, o `None` si no se puede leer.

    Sale de `cpu.stat`: `throttled_usec` en cgroup v2 y `throttled_time`
    (nanosegundos) en v1. Es acumulado del cgroup: la sonda usa la diferencia
    entre el inicio y el final de la unidad.
    """
    stat = _read(_v2_dirs(root, proc_cgroup)[0] / "cpu.stat")
    value = _stat_value(stat, "throttled_usec")
    if value is not None:
        return value / 1e6
    v1 = _v1_dirs(root, proc_cgroup)
    value = _stat_value(_read(v1[0] / "cpu.stat"), "throttled_time") if v1 else None
    return None if value is None else value / 1e9


def _stat_value(stat: str | None, name: str) -> int | None:
    for line in (stat or "").splitlines():
        key, _, value = line.partition(" ")
        if key == name:
            return int(value)
    return None
