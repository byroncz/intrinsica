"""Costura entre particiones mensuales consecutivas, leyendo solo el footer Parquet.

Nunca se cargan row groups de datos: `agg_trade_id` se toma de las estadísticas
de cada row group (ADR-L1-07), así que el costo es una lectura del footer por
partición y la memoria no depende del tamaño del mes.
"""

import logging
import re
from datetime import date, timedelta
from itertools import pairwise

import pyarrow.fs as pafs
import pyarrow.parquet as pq
from dq import Finding, Severity, Status, emit_findings

from l1_ingest.checks import CheckResult
from l1_ingest.manifest import resolve_fs
from l1_ingest.pipeline import RunContext, Unit, _finding
from l1_ingest.write import CONSOLIDATED, day_filename, partition_path

logger = logging.getLogger(__name__)

CHECK_TYPE = "seam_discontinuity"
SEAM_SKIPPED = "seam_skipped"
_PROVISIONAL = re.compile(r"provisional-day=(\d{2})\.parquet")


def list_provisionals(root: str, unit: Unit) -> dict[int, str]:
    """Día → ruta de cada `provisional-day=DD.parquet` de la partición del mes."""

    def path(name: str) -> str:
        return partition_path(
            root, unit.provider, unit.market, unit.asset, unit.year, unit.month, name
        )

    fs, resolved = resolve_fs(path(CONSOLIDATED))
    directory = resolved.rsplit("/", 1)[0]
    if fs.get_file_info(directory).type == pafs.FileType.NotFound:
        return {}
    return {
        int(m.group(1)): path(info.base_name)
        for info in fs.get_file_info(pafs.FileSelector(directory))
        if (m := _PROVISIONAL.fullmatch(info.base_name))
    }


def find_partition(root: str, unit: Unit, *, last: bool) -> str | None:
    """Ruta del archivo que representa el mes, o None si no existe.

    Prefiere `consolidated.parquet`; si no, el último (`last`) o el primer
    `provisional-day=DD.parquet`, ordenados por el día del nombre.
    """
    consolidated = partition_path(
        root,
        unit.provider,
        unit.market,
        unit.asset,
        unit.year,
        unit.month,
        CONSOLIDATED,
    )
    fs, resolved = resolve_fs(consolidated)
    if fs.get_file_info(resolved).type == pafs.FileType.File:
        return consolidated
    provisionals = list_provisionals(root, unit)
    if not provisionals:
        return None
    return provisionals[max(provisionals) if last else min(provisionals)]


def footer_stats(path: str) -> tuple[int, int, int]:
    """(num_rows, min, max) de `agg_trade_id` según el footer, sin leer datos."""
    fs, resolved = resolve_fs(path)
    meta = pq.ParquetFile(resolved, filesystem=fs).metadata
    column = meta.schema.names.index("agg_trade_id")
    stats = [
        meta.row_group(i).column(column).statistics for i in range(meta.num_row_groups)
    ]
    if not stats or any(s is None or not s.has_min_max for s in stats):
        raise ValueError(f"{path} no trae estadísticas de agg_trade_id")
    return meta.num_rows, min(s.min for s in stats), max(s.max for s in stats)


def id_bounds(path: str) -> tuple[int, int]:
    """(min, max) de `agg_trade_id` según las estadísticas de los row groups."""
    _, low, high = footer_stats(path)
    return low, high


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


def run_daily_seam(unit: Unit, ctx: RunContext) -> Finding | None:
    """Costura del día recién escrito con su día previo, solo por los footers.

    El previo es `provisional-day=DD-1` del mismo mes o, el día 1, la última
    partición de M-1 (su consolidated si existe; si no, su último provisional).
    Con varias tareas corriendo en paralelo, el día previo puede no estar
    escrito todavía (ITSC-233): no es un error de datos, la costura solo no
    se puede evaluar. Se deja un WARNING en el log y se emite un hallazgo
    `seam_skipped` (INFO, pass) para que quede en el lago de hallazgos.
    """

    def path(u: Unit, name: str) -> str:
        return partition_path(
            ctx.landing_root, u.provider, u.market, u.asset, u.year, u.month, name
        )

    next_path = path(unit, day_filename(unit.day))
    if unit.day > 1:
        expected = path(unit, day_filename(unit.day - 1))
        fs, resolved = resolve_fs(expected)
        found = fs.get_file_info(resolved).type == pafs.FileType.File
        prev_path = expected if found else None
    else:
        before = date(unit.year, unit.month, 1) - timedelta(days=1)
        prev_unit = Unit(
            before.year,
            before.month,
            provider=unit.provider,
            market=unit.market,
            asset=unit.asset,
        )
        expected = path(prev_unit, CONSOLIDATED)
        prev_path = find_partition(str(ctx.landing_root), prev_unit, last=True)
    if prev_path is None:
        logger.warning("costura omitida unidad=%s: no existe %s", unit, expected)
        skipped = CheckResult(
            SEAM_SKIPPED,
            Severity.INFO,
            Status.PASS,
            None,
            {"reason": "previous_missing", "expected_path": expected},
        )
        finding = _finding(skipped, unit, ctx)
        emit_findings([finding], ctx.dq_root)
        return finding
    result = check_seam(prev_path, next_path, {})
    logger.info("costura %s -> %s: %s", prev_path, next_path, result.status.value)
    finding = _finding(result, unit, ctx)
    emit_findings([finding], ctx.dq_root)
    return finding
