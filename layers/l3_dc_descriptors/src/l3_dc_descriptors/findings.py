"""Hallazgos de DQ de L3 (TRD-L3 §9.3), emitidos con `shared/dq`.

Todos llevan `layer = "l3"` y `stage = "canonical"`. El θ viaja en `details`, como en L2,
pero como texto de la partición (`"0.00010000"`), igual que en las excepciones del
resumidor, de las que sale la mayoría de los `details`.
"""

import logging
import signal

from dq import Finding, Severity, Stage, Status, emit_findings

from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.errors import SummarizeError
from l3_dc_descriptors.timing import Timing

logger = logging.getLogger(__name__)

LAYER = "l3"


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
    return finding(
        ctx,
        unit,
        check_type,
        severity=Severity.ERROR,
        status=Status.FAIL,
        details=details,
    )


def from_error(ctx: RunContext, unit: Unit, error: SummarizeError) -> Finding | None:
    """El hallazgo de una excepción del resumidor, con los `details` que ya trae.

    `None` si el error no tiene `check_type` (`PhaseValueError`): aborta la unidad sin
    hallazgo propio.
    """
    if error.check_type is None:
        return None
    return failure(ctx, unit, error.check_type, error.details)


def l2_input_missing(ctx: RunContext, unit: Unit, *, theta: str, file: str) -> Finding:
    """Falta `events.parquet` o `carry_over.parquet` de L2 de `(θ, mes)`: detiene el rango
    del θ (ADR-L3-05). Un hallazgo por archivo, con los `details` de `L2InputMissing`.
    """
    return failure(
        ctx,
        unit,
        "l2_input_missing",
        {"theta": theta, "month": f"{unit.year:04d}-{unit.month:02d}", "file": file},
    )


def family_skeleton_missing(
    ctx: RunContext, unit: Unit, *, theta: str, file: str
) -> Finding:
    """Falta `summaries.parquet` de `(θ, mes)`, el esqueleto de la familia de la corrida: el
    θ sale del rango (§7.3, RL3-11) y no se escribe nada de la familia.
    """
    return failure(
        ctx,
        unit,
        "family_skeleton_missing",
        {
            "theta": theta,
            "month": f"{unit.year:04d}-{unit.month:02d}",
            "family": ctx.family,
            "file": file,
        },
    )


def input_missing(
    ctx: RunContext, unit: Unit, check_type: str, details: dict
) -> Finding:
    """`input_missing` o `input_provisional_only`: L1 no tiene el consolidado del mes."""
    return failure(ctx, unit, check_type, details)


def theta_catalog_invalid(
    ctx: RunContext, unit: Unit, *, source: str | None, problems: list[str]
) -> Finding:
    """El catálogo de θ no se pudo leer o no cumple TRD-L2 §7.3: la corrida no escribe."""
    return failure(
        ctx,
        unit,
        "theta_catalog_invalid",
        {"source": source, "problems": problems},
    )


def theta_behind_frontier(
    ctx: RunContext,
    unit: Unit,
    *,
    theta: str,
    frontier: str | None,
    reason: str,
) -> Finding:
    """`monthly` no procesó el θ (§8.3). `reason` es `l3_frontier` (su serie de L3 no
    llega al mes previo), `l2_missing` (L2 no tiene el mes de ese θ) o `summaries_missing`
    (una familia distinta de S, y `summaries.parquet` del mes no existe). `family` dice
    cuál de las dos pasadas mensuales (S o T) dejó el θ rezagado.
    """
    return finding(
        ctx,
        unit,
        "theta_behind_frontier",
        severity=Severity.WARNING,
        status=Status.FAIL,
        details={
            "theta": theta,
            "frontier": frontier,
            "reason": reason,
            "family": ctx.family,
        },
    )


def theta_config_drift(
    ctx: RunContext, unit: Unit, *, theta: str, months: int, last_month: str
) -> Finding:
    """El lago de L3 tiene particiones de un θ que el catálogo ya no incluye.

    Es informativo: quitar un θ del catálogo no borra nada.
    """
    return finding(
        ctx,
        unit,
        "theta_config_drift",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=float(months),
        details={"theta": theta, "months": months, "last_month": last_month},
    )


def summaries_summary(
    ctx: RunContext,
    unit: Unit,
    *,
    theta: str,
    rows: int,
    content_hash: str,
    l2_events_content_hash: str,
    l2_carry_over_content_hash: str,
    unassigned_ticks: int | None = None,
) -> Finding:
    """Resumen de lo que escribió un θ: filas, hash del archivo y linaje contra L2.

    `unassigned_ticks` es de la conciliación de S (§9.1): la pasada de las demás familias no
    concilia y no lo publica.
    """
    details = {
        "theta": theta,
        "family": ctx.family,
        "rows": rows,
        "content_hash": content_hash,
        "l2_events_content_hash": l2_events_content_hash,
        "l2_carry_over_content_hash": l2_carry_over_content_hash,
    }
    if unassigned_ticks is not None:
        details["unassigned_ticks"] = unassigned_ticks
    return finding(
        ctx,
        unit,
        "summaries_summary",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=float(rows),
        details=details,
    )


def unit_timing(ctx: RunContext, unit: Unit, timing: Timing) -> Finding:
    """Tiempo y medidas de la unidad (uno por mes; `metric_value` es la pared).

    Va aparte de `summaries_summary` porque este es por θ y debe dar el mismo `details`
    en dos corridas del mes; los tiempos nunca coinciden.
    """
    return finding(
        ctx,
        unit,
        "unit_timing",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=round(timing.wall_s, 3),
        details={"family": ctx.family, **timing.details()},
    )


def unit_process_died(
    ctx: RunContext,
    unit: Unit,
    *,
    exit_signal: int | None,
    exit_code: int | None,
    error: str | None = None,
) -> Finding:
    """El proceso hijo de la unidad terminó sin devolver su resultado (ITSC-372).

    `exit_signal` es la señal que lo mató (9 es el OOM del cgroup); `exit_code`, su código de
    salida si terminó solo (una excepción que no es de la unidad), con `error` como motivo.
    La unidad queda fallida y no deja archivos publicados.
    """
    try:
        name = signal.Signals(exit_signal).name if exit_signal else None
    except ValueError:  # señal de tiempo real: sin nombre en `signal.Signals`
        name = None
    return failure(
        ctx,
        unit,
        "unit_process_died",
        {
            "month": f"{unit.year:04d}-{unit.month:02d}",
            "exit_signal": exit_signal,
            "signal_name": name,
            "exit_code": exit_code,
            "error": error,
        },
    )


def unit_profile(
    ctx: RunContext, unit: Unit, *, wall_s: float, top: dict[str, list[dict]]
) -> Finding:
    """Las funciones más caras de la unidad, solo con `L3_PROFILE=1` (uno por unidad).

    `wall_s` es la pared de la unidad perfilada, inflada por el perfilador: sirve para
    proporciones, no para absolutos.
    """
    return finding(
        ctx,
        unit,
        "unit_profile",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=round(wall_s, 3),
        details={"wall_s": round(wall_s, 3), **top},
    )


def emit(ctx: RunContext, findings: list[Finding]) -> None:
    if findings:
        emit_findings(findings, ctx.dq_root)
