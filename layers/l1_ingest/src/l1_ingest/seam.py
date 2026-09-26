"""Costura entre particiones mensuales consecutivas, leyendo solo el footer Parquet.

Nunca se cargan row groups de datos: `agg_trade_id` se toma de las estadísticas
de cada row group (ADR-L1-07), así que el costo es una lectura del footer por
partición y la memoria no depende del tamaño del mes.
"""

import logging
import re
from itertools import pairwise

import pyarrow.fs as pafs
import pyarrow.parquet as pq
from dq import Finding, Severity, Status, emit_findings

from l1_ingest.checks import CheckResult
from l1_ingest.manifest import resolve_fs
from l1_ingest.pipeline import RunContext, Unit, _finding
from l1_ingest.write import CONSOLIDATED, partition_path

logger = logging.getLogger(__name__)

CHECK_TYPE = "seam_discontinuity"
_PROVISIONAL = re.compile(r"provisional-day=(\d{2})\.parquet")


def find_partition(root: str, unit: Unit, *, last: bool) -> str | None:
    """Ruta del archivo que representa el mes, o None si no existe.

    Prefiere `consolidated.parquet`; si no, el último (`last`) o el primer
    `provisional-day=DD.parquet`, ordenados por el día del nombre.
    """

    def path(name: str) -> str:
        return partition_path(
            root, unit.provider, unit.market, unit.asset, unit.year, unit.month, name
        )

    consolidated = path(CONSOLIDATED)
    fs, resolved = resolve_fs(consolidated)
    if fs.get_file_info(resolved).type == pafs.FileType.File:
        return consolidated
    directory = resolved.rsplit("/", 1)[0]
    if fs.get_file_info(directory).type == pafs.FileType.NotFound:
        return None
    days = sorted(
        int(m.group(1))
        for info in fs.get_file_info(pafs.FileSelector(directory))
        if (m := _PROVISIONAL.fullmatch(info.base_name))
    )
    if not days:
        return None
    return path(f"provisional-day={days[-1 if last else 0]:02d}.parquet")


def id_bounds(path: str) -> tuple[int, int]:
    """(min, max) de `agg_trade_id` según las estadísticas de los row groups."""
    fs, resolved = resolve_fs(path)
    meta = pq.ParquetFile(resolved, filesystem=fs).metadata
    column = meta.schema.names.index("agg_trade_id")
    stats = [
        meta.row_group(i).column(column).statistics for i in range(meta.num_row_groups)
    ]
    if not stats or any(s is None or not s.has_min_max for s in stats):
        raise ValueError(f"{path} no trae estadísticas de agg_trade_id")
    return min(s.min for s in stats), max(s.max for s in stats)


def check_seam(
    prev_path: str | None, next_path: str | None, expected: dict
) -> CheckResult:
    """Compara max(id) de la partición previa con min(id) de la siguiente.

    `expected` mapea "prev" y "next" a la ruta esperada, para reportar la que falta.
    """
    missing = [
        expected[k] for k, p in (("prev", prev_path), ("next", next_path)) if p is None
    ]
    if missing:
        return CheckResult(
            CHECK_TYPE, Severity.ERROR, Status.FAIL, None, {"missing": missing}
        )
    prev_max = id_bounds(prev_path)[1]
    next_min = id_bounds(next_path)[0]
    gap = next_min - prev_max - 1
    details = {
        "prev_max": prev_max,
        "next_min": next_min,
        "prev_path": prev_path,
        "next_path": next_path,
    }
    if gap == 0:
        return CheckResult(CHECK_TYPE, Severity.INFO, Status.PASS, 0, details)
    return CheckResult(CHECK_TYPE, Severity.WARNING, Status.FAIL, gap, details)


def _months(year: int, month: int, last: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    while (year, month) <= last:
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def run_seam_check(
    start: tuple[int, int], end: tuple[int, int], asset: str, ctx: RunContext
) -> list[Finding]:
    """Recorre los bordes (M, M+1) del rango y emite un hallazgo por borde."""
    months = _months(*start, end)
    findings = []
    for prev, nxt in pairwise(months):
        prev_unit = Unit(*prev, asset=asset)
        next_unit = Unit(*nxt, asset=asset)
        expected = {
            "prev": partition_path(
                ctx.landing_root,
                prev_unit.provider,
                prev_unit.market,
                asset,
                prev[0],
                prev[1],
                CONSOLIDATED,
            ),
            "next": partition_path(
                ctx.landing_root,
                next_unit.provider,
                next_unit.market,
                asset,
                nxt[0],
                nxt[1],
                CONSOLIDATED,
            ),
        }
        result = check_seam(
            find_partition(str(ctx.landing_root), prev_unit, last=True),
            find_partition(str(ctx.landing_root), next_unit, last=False),
            expected,
        )
        logger.info("costura %s -> %s: %s", prev, nxt, result.status.value)
        findings.append(_finding(result, next_unit, ctx))
    emit_findings(findings, ctx.dq_root)
    return findings
