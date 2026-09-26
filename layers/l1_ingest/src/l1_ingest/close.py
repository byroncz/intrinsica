"""Cierre mensual (§8.4 del TRD-L1): consolidado, drift contra los provisionales y borrado.

De los provisionales solo se leen footers Parquet, así que la memoria de la
comparación no depende del tamaño del mes; el núcleo del mes ya es O(lote).
"""

import calendar
import logging
import re

import pyarrow.fs as pafs
from dq import Finding, Severity, Status, emit_findings

from l1_ingest.checks import CheckResult
from l1_ingest.manifest import resolve_fs
from l1_ingest.pipeline import RunContext, Unit, _exists, _finding, process_unit
from l1_ingest.seam import check_seam, find_partition, footer_stats
from l1_ingest.write import CONSOLIDATED, partition_path

logger = logging.getLogger(__name__)

CHECK_TYPE = "daily_monthly_drift"
_PROVISIONAL = re.compile(r"provisional-day=(\d{2})\.parquet")


def _path(ctx: RunContext, unit: Unit, name: str) -> str:
    return partition_path(
        ctx.landing_root,
        unit.provider,
        unit.market,
        unit.asset,
        unit.year,
        unit.month,
        name,
    )


def list_provisionals(ctx: RunContext, unit: Unit) -> dict[int, str]:
    """Día → ruta de cada `provisional-day=DD.parquet` de la partición del mes."""
    fs, resolved = resolve_fs(_path(ctx, unit, CONSOLIDATED))
    directory = resolved.rsplit("/", 1)[0]
    if fs.get_file_info(directory).type == pafs.FileType.NotFound:
        return {}
    root = _path(ctx, unit, "").rstrip("/")
    return {
        int(m.group(1)): f"{root}/{info.base_name}"
        for info in fs.get_file_info(pafs.FileSelector(directory))
        if (m := _PROVISIONAL.fullmatch(info.base_name))
    }


def check_drift(
    consolidated: str, provisionals: dict[int, str], days: int
) -> CheckResult:
    """Compara filas y rango de ids del consolidado con los de los provisionales.

    Lee solo el footer de cada archivo. Sin provisionales (mes que nunca tuvo
    daily) no hay nada que comparar: pass.
    """
    if not provisionals:
        return CheckResult(
            CHECK_TYPE, Severity.INFO, Status.PASS, 0, {"provisionals": 0}
        )
    rows, low, high = footer_stats(consolidated)
    stats = [footer_stats(p) for p in provisionals.values()]
    prov_rows = sum(s[0] for s in stats)
    prov_range = [min(s[1] for s in stats), max(s[2] for s in stats)]
    missing = [d for d in range(1, days + 1) if d not in provisionals]
    details = {
        "provisionals": len(provisionals),
        "provisional_rows": prov_rows,
        "consolidated_rows": rows,
        "provisional_id_range": prov_range,
        "consolidated_id_range": [low, high],
        "missing_days": missing,
    }
    if prov_rows == rows and prov_range == [low, high] and not missing:
        return CheckResult(CHECK_TYPE, Severity.INFO, Status.PASS, 0, details)
    return CheckResult(
        CHECK_TYPE, Severity.WARNING, Status.FAIL, rows - prov_rows, details
    )


def _seam_with_previous(ctx: RunContext, unit: Unit) -> CheckResult:
    """Borde del consolidado con la última partición de M-1 (solo footers)."""
    prev_year, prev_month = (
        (unit.year - 1, 12) if unit.month == 1 else (unit.year, unit.month - 1)
    )
    previous = Unit(
        prev_year,
        prev_month,
        provider=unit.provider,
        market=unit.market,
        asset=unit.asset,
    )
    expected = {"prev": _path(ctx, previous, CONSOLIDATED)}
    return check_seam(
        find_partition(str(ctx.landing_root), previous, last=True),
        _path(ctx, unit, CONSOLIDATED),
        {**expected, "next": _path(ctx, unit, CONSOLIDATED)},
    )


def run_monthly_close(unit: Unit, ctx: RunContext) -> list[Finding]:
    """Cierra el mes: consolidado, drift, costura con M-1 y borrado de provisionales.

    Si consolidated.parquet ya existe la partición está cerrada: no se descarga
    ni se escribe, salvo `force`. Los provisionales se borran solo después del
    commit del consolidado y del hallazgo de drift (RL1-08).
    """
    consolidated = _path(ctx, unit, CONSOLIDATED)
    if _exists(consolidated) and not ctx.force:
        logger.info("mes ya cerrado unidad=%s: existe %s", unit, consolidated)
        return []
    result = process_unit(unit, ctx)
    provisionals = list_provisionals(ctx, unit)
    days = calendar.monthrange(unit.year, unit.month)[1]
    extra = [
        _finding(check_drift(result.path, provisionals, days), unit, ctx),
        _finding(_seam_with_previous(ctx, unit), unit, ctx),
    ]
    emit_findings(extra, ctx.dq_root)
    fs, _ = resolve_fs(consolidated)
    for path in provisionals.values():
        fs.delete_file(resolve_fs(path)[1])
    logger.info("cierre unidad=%s: %d provisionales borrados", unit, len(provisionals))
    return [*result.findings, *extra]
