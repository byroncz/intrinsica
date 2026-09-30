"""Cierre mensual (§8.4 del TRD-L1): consolidado, drift contra los provisionales y borrado.

De los provisionales solo se leen footers Parquet, así que la memoria de la
comparación no depende del tamaño del mes; el núcleo del mes ya es O(lote).
"""

import calendar
import logging

from dq import Finding, Severity, Status, emit_findings
from pyutils import resolve_fs

from l1_ingest.checks import CheckResult
from l1_ingest.pipeline import RunContext, Unit, _exists, _finding, process_unit
from l1_ingest.seam import check_seam, find_partition, footer_stats, list_provisionals
from l1_ingest.write import CONSOLIDATED, partition_path

logger = logging.getLogger(__name__)

CHECK_TYPE = "daily_monthly_drift"


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
    return check_seam(
        find_partition(str(ctx.landing_root), previous, last=True),
        _path(ctx, unit, CONSOLIDATED),
        {
            "prev": _path(ctx, previous, CONSOLIDATED),
            "next": _path(ctx, unit, CONSOLIDATED),
        },
    )


def run_monthly_close(unit: Unit, ctx: RunContext) -> list[Finding]:
    """Cierra el mes: consolidado, drift, costura con M-1 y borrado de provisionales.

    Si consolidated.parquet ya existe no se descarga ni se escribe, salvo
    `force`; si además quedan provisionales (cierre interrumpido o backfill),
    se comparan, se borran y se emiten los hallazgos. Los provisionales se
    borran solo después del commit del consolidado y del hallazgo de drift
    (RL1-08).
    """
    consolidated = _path(ctx, unit, CONSOLIDATED)
    findings: list[Finding] = []
    if _exists(consolidated) and not ctx.force:
        if not list_provisionals(str(ctx.landing_root), unit):
            logger.info("mes ya cerrado unidad=%s: existe %s", unit, consolidated)
            return []
        # Cierre a medias (reintento o backfill con provisionales): se repara.
    else:
        findings = process_unit(unit, ctx).findings
    provisionals = list_provisionals(str(ctx.landing_root), unit)
    days = calendar.monthrange(unit.year, unit.month)[1]
    extra = [
        _finding(check_drift(consolidated, provisionals, days), unit, ctx),
        _finding(_seam_with_previous(ctx, unit), unit, ctx),
    ]
    emit_findings(extra, ctx.dq_root)
    fs, _ = resolve_fs(consolidated)
    for path in provisionals.values():
        fs.delete_file(resolve_fs(path)[1])
    logger.info(
        "cierre completado unidad=%s: %d provisionales borrados",
        unit,
        len(provisionals),
    )
    return [*findings, *extra]
