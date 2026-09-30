"""Límite efectivo de CPU del proceso y su estrangulamiento (ITSC-289).

`os.sched_getaffinity` dice qué cores ve el proceso, no cuántos puede usar: en
Cloud Run la cuota de vCPU del job es un límite del cgroup (`cpu.max`) y la
máquina reporta más cores (6, 6 y 9 con 2, 4 y 8 vCPU, ITSC-281). El límite
efectivo es el menor de los dos.
"""

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


def _quota_v1(root: Path) -> float | None:
    for directory in (root / "cpu", root / "cpu,cpuacct"):
        quota = _read(directory / "cpu.cfs_quota_us")
        period = _read(directory / "cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            return int(quota) / int(period)
    return None


def cpu_limit(
    root: Path = CGROUP_ROOT,
    proc_cgroup: Path = PROC_CGROUP,
    visible: int | None = None,
) -> CpuLimit:
    """El límite efectivo de CPU: cuota del cgroup o, sin ella, los cores visibles."""
    visible = visible if visible is not None else len(os.sched_getaffinity(0))
    for source, quota in (
        ("cgroup-v2", _quota_v2(root, proc_cgroup)),
        ("cgroup-v1", _quota_v1(root)),
    ):
        if quota is not None and quota < visible:
            return CpuLimit(quota, visible, source)
    return CpuLimit(float(visible), visible, "affinity")


def throttled_s(
    root: Path = CGROUP_ROOT, proc_cgroup: Path = PROC_CGROUP
) -> float | None:
    """Segundos que el cgroup lleva sin poder correr por agotar su cuota
    (`throttled_usec` de `cpu.stat`, cgroup v2), o `None` si no se puede leer.

    Es acumulado del cgroup: la sonda usa la diferencia entre el inicio y el
    final de la unidad.
    """
    stat = _read(_v2_dirs(root, proc_cgroup)[0] / "cpu.stat")
    for line in (stat or "").splitlines():
        name, _, value = line.partition(" ")
        if name == "throttled_usec":
            return int(value) / 1e6
    return None
