"""Hallazgos de DQ de L2 (TRD-L2 §9.3), emitidos con `shared/dq`.

Todos llevan `layer = "l2"` y `stage = "canonical"`: L2 no tiene noción de
provisional (ADR-L2-09). El θ viaja en `details`, no en una columna (§9.4).
"""

import logging

from dq import Finding, Severity, Stage, Status, emit_findings

from l2_dc_events.context import RunContext, Unit
from l2_dc_events.timing import Timing

logger = logging.getLogger(__name__)

LAYER = "l2"


def finding(
    ctx: RunContext,
    unit: Unit,
    check_type: str,
    *,
    severity: Severity,
    status: Status,
    details: dict,
    metric_value: float | None = None,
) -> Finding:
    return Finding(
        layer=LAYER,
        mode=ctx.mode,
        check_type=check_type,
        severity=severity,
        stage=Stage.CANONICAL,
        status=status,
        provider=unit.provider,
        market=unit.market,
        asset=unit.asset,
        year=unit.year,
        month=unit.month,
        metric_value=metric_value,
        details=details,
        run_id=ctx.run_id,
        image_version=ctx.image_version,
    )


def failure(ctx: RunContext, unit: Unit, check_type: str, details: dict) -> Finding:
    """Un chequeo que falló y que detiene (o debería detener) la unidad."""
    severity = (
        Severity.WARNING if check_type == "theta_config_drift" else Severity.ERROR
    )
    return finding(
        ctx, unit, check_type, severity=severity, status=Status.FAIL, details=details
    )


def events_summary(
    ctx: RunContext,
    unit: Unit,
    *,
    theta: int,
    events: int,
    has_pending_event: bool,
    events_content_hash: str,
    carry_over_content_hash: str,
) -> Finding:
    """Resumen de lo que escribió un θ: cuántos eventos y con qué hash."""
    return finding(
        ctx,
        unit,
        "events_summary",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=float(events),
        details={
            "theta": theta,
            "events": events,
            "has_pending_event": has_pending_event,
            "events_content_hash": events_content_hash,
            "carry_over_content_hash": carry_over_content_hash,
        },
    )


def unit_timing(ctx: RunContext, unit: Unit, timing: Timing) -> Finding:
    """Tiempo por fase de la unidad (uno por mes; `metric_value` es la pared).

    Va aparte de `events_summary` porque este es por θ y debe dar el mismo
    `details` en dos corridas del mes; los tiempos nunca coinciden.
    """
    return finding(
        ctx,
        unit,
        "unit_timing",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=round(timing.wall_s, 3),
        details=timing.details(),
    )


def zero_tick_discarded(
    ctx: RunContext, unit: Unit, *, theta: int, discarded: int
) -> Finding:
    """La guarda de §9.1 descartó eventos: un defecto del detector, no del mercado."""
    return finding(
        ctx,
        unit,
        "dc_zero_tick_discarded",
        severity=Severity.ERROR,
        status=Status.FAIL,
        metric_value=float(discarded),
        details={"theta": theta, "discarded": discarded},
    )


def emit(ctx: RunContext, findings: list[Finding]) -> None:
    if findings:
        emit_findings(findings, ctx.dq_root)
