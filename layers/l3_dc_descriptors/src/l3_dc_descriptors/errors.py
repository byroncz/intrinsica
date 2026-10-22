"""Errores tipados del resumidor (TRD-L3 §9.3).

El núcleo no emite hallazgos: lanza estas excepciones y quien lo invoca las convierte
en hallazgos de DQ. `check_type` es el del hallazgo y `details`, su payload (JSON:
cadenas, enteros y booleanos). Un error sin `check_type` no tiene hallazgo propio: la
unidad aborta igual y no escribe nada.
"""

from __future__ import annotations

from dc_frames.lake import Month


def month_text(month: Month) -> str:
    return f"{month[0]:04d}-{month[1]:02d}"


class SummarizeError(Exception):
    check_type: str | None = None

    def __init__(self, message: str, details: dict[str, object]) -> None:
        super().__init__(message)
        self.details = details


class L2InputMissing(SummarizeError):
    """Falta `events.parquet` o `carry_over.parquet` de L2 de `(θ, mes)` (§9.2)."""

    check_type = "l2_input_missing"

    def __init__(self, theta: str, month: Month, file: str, path: str) -> None:
        super().__init__(
            f"falta {file} de L2 del θ {theta} en {month_text(month)}: {path}",
            {"theta": theta, "month": month_text(month), "file": file},
        )


class L2LineageMissing(SummarizeError):
    """La partición de L2 de `(θ, mes)` no trae su `content_hash.json`, o no es válido (§9.2).

    L3 no recalcula el linaje: lo lee de lo que L2 publicó (ADR-L3-16). Sin ese objeto la
    unidad falla, y se resuelve corriendo el backfill de ITSC-368 (o reprocesando el mes en L2).
    """

    check_type = "l2_lineage_missing"

    def __init__(self, theta: str, month: Month, path: str, reason: str) -> None:
        super().__init__(
            f"falta el linaje de L2 del θ {theta} en {month_text(month)} ({reason}): {path}",
            {
                "theta": theta,
                "month": month_text(month),
                "file": "content_hash.json",
                "reason": reason,
            },
        )


class L2StateInconsistent(SummarizeError):
    """L2 contradice a L1 o a sí mismo: una frontera que no es tick, una fila que viola
    un invariante de §7.2 o un carry-over con `direction ≠ 0` sin evento pendiente.
    """

    check_type = "l2_state_inconsistent"

    @classmethod
    def mismatch(
        cls, theta: str, month: Month, field: str, got: object, wanted: object
    ) -> L2StateInconsistent:
        return cls(
            f"L2 incoherente en el θ {theta}, {month_text(month)}: {field} vale {got} "
            f"y debería valer {wanted}",
            {
                "theta": theta,
                "month": month_text(month),
                "field": field,
                "got": str(got),
                "wanted": str(wanted),
            },
        )

    @classmethod
    def no_pending(
        cls, theta: str, month: Month, direction: int, has_pending_event: bool
    ) -> L2StateInconsistent:
        return cls(
            f"L2 incoherente en el θ {theta}, {month_text(month)}: el carry-over tiene "
            f"direction {direction} y has_pending_event {has_pending_event}",
            {
                "theta": theta,
                "month": month_text(month),
                "direction": direction,
                "has_pending_event": has_pending_event,
            },
        )


class PhaseQuantityMismatch(SummarizeError):
    """`buy + sell` de una fase no es la suma de `quantity` de sus ticks (§7.2)."""

    check_type = "phase_quantity_mismatch"

    def __init__(
        self,
        theta: str,
        confirm_agg_trade_id: int,
        phase: str,
        expected: str,
        got: str,
    ) -> None:
        super().__init__(
            f"θ {theta}, evento con confirmación {confirm_agg_trade_id}: en la fase "
            f"{phase}, buy + sell vale {got} y la suma de quantity {expected}",
            {
                "theta": theta,
                "confirm_agg_trade_id": confirm_agg_trade_id,
                "phase": phase,
                "expected": expected,
                "got": got,
            },
        )


class PhaseTicksMismatch(SummarizeError):
    """No cuadra la conciliación de ticks del mes (§9.1)."""

    check_type = "phase_ticks_mismatch"

    def __init__(
        self,
        theta: str,
        month: Month,
        n: int,
        closed: int,
        before_month: int,
        tail: int,
        unassigned: int,
    ) -> None:
        super().__init__(
            f"θ {theta}, {month_text(month)}: N_M = {n} y (Σ_cerrados − P_M) + Cola_M "
            f"+ U_M = ({closed} − {before_month}) + {tail} + {unassigned}",
            {
                "theta": theta,
                "month": month_text(month),
                "N_M": n,
                "sigma_closed": closed,
                "P_M": before_month,
                "tail_M": tail,
                "U_M": unassigned,
            },
        )


class PhaseValueError(SummarizeError):
    """Un valor de una fase no cabe en su columna o no se puede calcular (RL3-05).

    La conversión falla en vez de truncar. No tiene `check_type` en §9.3.
    """

    def __init__(self, theta: str, confirm_agg_trade_id: int, message: str) -> None:
        super().__init__(
            f"θ {theta}, evento con confirmación {confirm_agg_trade_id}: {message}",
            {"theta": theta, "confirm_agg_trade_id": confirm_agg_trade_id},
        )


class SkeletonMismatch(SummarizeError):
    """El archivo de la familia no repite el esqueleto de L2: filas, orden o row groups (§9.3)."""

    check_type = "family_skeleton_mismatch"

    def __init__(self, theta: str, month: Month, family: str, detail: str) -> None:
        super().__init__(
            f"θ {theta}, {month_text(month)}: {family} no repite el esqueleto de L2: "
            f"{detail}",
            {
                "theta": theta,
                "month": month_text(month),
                "family": family,
                "detail": detail,
            },
        )
