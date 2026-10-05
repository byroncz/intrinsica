"""Hallazgos de DQ de viz (TRD-viz §9.3), emitidos con `shared/dq`.

Todos llevan `layer = "viz"`, `mode = "tiles"` y `stage = "canonical"`: viz solo
lee el consolidado de L1 (ADR-L2-09). El día va en `details.day`, no en una columna.
"""

from datetime import date

from dq import Finding, Severity, Stage, Status, emit_findings

from viz_tiles.context import RunContext

LAYER = "viz"
MODE = "tiles"


def finding(
    ctx: RunContext,
    day: date,
    check_type: str,
    *,
    severity: Severity,
    status: Status,
    details: dict,
    metric_value: float | None = None,
) -> Finding:
    return Finding(
        layer=LAYER,
        mode=MODE,
        check_type=check_type,
        severity=severity,
        stage=Stage.CANONICAL,
        status=status,
        provider=ctx.provider,
        market=ctx.market,
        asset=ctx.asset,
        year=day.year,
        month=day.month,
        metric_value=metric_value,
        details={"day": day.isoformat(), **details},
        run_id=ctx.run_id,
        image_version=ctx.image_version,
    )


def tiles_summary(
    ctx: RunContext,
    day: date,
    *,
    ticks: int,
    skipped: bool,
    tiles_version: str,
    input_hash: str,
    content_hash: str,
    objects: int,
    size: int,
    levels: list[int],
    thetas: list[str],
    provisional_thetas: list[str],
    missing_thetas: list[str],
) -> Finding:
    """Uno por día, construido o saltado. `metric_value` son los ticks del día."""
    return finding(
        ctx,
        day,
        "tiles_summary",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=float(ticks),
        details={
            "input_hash": input_hash,
            "content_hash": content_hash,
            "tiles_version": tiles_version,
            "skipped": skipped,
            "objects": objects,
            "bytes": size,
            "levels": levels,
            "thetas": thetas,
            # La cola del día usó el extremo provisional del pendiente.
            "provisional_tail": bool(provisional_thetas),
            "provisional_thetas": provisional_thetas,
            "missing_thetas": missing_thetas,
        },
    )


def input_missing(ctx: RunContext, day: date, what: str, **details: object) -> Finding:
    """Falta una entrada (`what`: `l1`, `events`, `carry_over` o `ticks`).

    Es un hecho del mes y no del día, salvo `ticks`: el hallazgo lleva el primer
    día pedido en `details.day` y cuántos días afecta en `details.days`.
    """
    return finding(
        ctx,
        day,
        "input_missing",
        severity=Severity.ERROR,
        status=Status.FAIL,
        details={"what": what, **details},
    )


def price_rounded(
    ctx: RunContext, day: date, *, count: int, max_abs_delta_int: int
) -> Finding:
    """Ticks fuera del tick del activo: el tile los redondeó y el día se escribió."""
    return finding(
        ctx,
        day,
        "price_rounded",
        severity=Severity.WARNING,
        status=Status.PASS,
        metric_value=float(count),
        details={"count": count, "max_abs_delta_int": max_abs_delta_int},
    )


def price_unrepresentable(
    ctx: RunContext, day: date, *, price_scale: int, max_price_int: int
) -> Finding:
    """El precio máximo no cabe en `int32`: el día no se escribe."""
    return finding(
        ctx,
        day,
        "price_unrepresentable",
        severity=Severity.ERROR,
        status=Status.FAIL,
        details={"price_scale": price_scale, "max_price_int": max_price_int},
    )


def emit(ctx: RunContext, findings: list[Finding]) -> None:
    if findings:
        emit_findings(findings, ctx.dq_root)
