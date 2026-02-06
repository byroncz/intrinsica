"""Validador de umbral (Nivel 5).

Valida cumplimiento del umbral θ en eventos Silver.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult


class ThresholdValidator(BaseValidator):
    """Validador de cumplimiento de umbral θ."""

    name = "threshold"
    level = 5

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Valida cumplimiento de umbral de un día."""
        results = []

        results.append(self._test_dc_magnitude_meets_theta(df, day, theta))
        results.append(self._test_slippage_bounded(df, day, theta))
        results.append(self._test_direction_matches_event_type(df, day))

        return results

    def _test_dc_magnitude_meets_theta(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> ValidationResult:
        """Verifica que la magnitud DC cumple el umbral θ."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_magnitude_meets_theta[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="No events to validate",
            )

        tolerance = self.config.theta_tolerance

        # Calcular magnitud DC
        magnitude = (
            (df["confirm_price"] - df["reference_price"]).abs() / df["reference_price"]
        )

        # Verificar que cumple θ - ε
        invalid_mask = magnitude < (theta - tolerance)
        invalid_count = invalid_mask.sum()

        affected = []
        if invalid_count > 0:
            affected = [i for i, m in enumerate(invalid_mask.to_list()) if m]

        # Estadísticas
        min_mag = magnitude.min()
        avg_mag = magnitude.mean()

        return self._make_result(
            test_name=f"dc_magnitude_meets_theta[{day}]",
            passed=invalid_count == 0,
            severity=Severity.HIGH,
            message=(
                f"{invalid_count} events with DC magnitude < θ (min={min_mag:.6f})"
                if invalid_count > 0
                else f"All DC magnitudes meet θ (avg={avg_mag:.6f})"
            ),
            details={
                "day": str(day),
                "theta": theta,
                "min_magnitude": float(min_mag) if min_mag is not None else None,
                "avg_magnitude": float(avg_mag) if avg_mag is not None else None,
            },
            affected_events=affected,
        )

    def _test_slippage_bounded(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> ValidationResult:
        """Verifica que el slippage está dentro de límites razonables."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"slippage_bounded[{day}]",
                passed=True,
                severity=Severity.MEDIUM,
                message="No events to validate",
            )

        # slippage = |confirm_price - expected_threshold|
        # expected_threshold = reference_price * (1 + event_type * θ)
        expected_threshold = df["reference_price"] * (
            1 + df["event_type"].cast(pl.Float64) * theta
        )
        slippage = (df["confirm_price"] - expected_threshold).abs()

        # Límite: slippage < max_slippage_factor * θ * reference_price
        max_allowed = self.config.max_slippage_factor * theta * df["reference_price"]

        invalid_mask = slippage > max_allowed
        invalid_count = invalid_mask.sum()

        affected = []
        if invalid_count > 0:
            affected = [i for i, m in enumerate(invalid_mask.to_list()) if m]

        max_slippage = slippage.max()
        avg_slippage = slippage.mean()

        return self._make_result(
            test_name=f"slippage_bounded[{day}]",
            passed=invalid_count == 0,
            severity=Severity.MEDIUM,
            message=(
                f"{invalid_count} events with excessive slippage (max={max_slippage:.6f})"
                if invalid_count > 0
                else f"All slippage within bounds (avg={avg_slippage:.6f})"
            ),
            details={
                "day": str(day),
                "max_slippage": float(max_slippage) if max_slippage is not None else None,
                "avg_slippage": float(avg_slippage) if avg_slippage is not None else None,
                "max_slippage_factor": self.config.max_slippage_factor,
            },
            affected_events=affected,
        )

    def _test_direction_matches_event_type(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica coherencia entre event_type y movimiento de precio."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"direction_matches_event_type[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        price_change = df["confirm_price"] - df["reference_price"]

        # Para upturn (+1): confirm_price > reference_price → price_change > 0
        # Para downturn (-1): confirm_price < reference_price → price_change < 0
        # Verificar: sign(price_change) == event_type

        # Usar tolerancia para evitar falsos positivos por precisión numérica
        tolerance = self.config.price_tolerance

        # Upturn debería tener price_change > 0
        upturn_invalid = (df["event_type"] == 1) & (price_change < -tolerance)
        # Downturn debería tener price_change < 0
        downturn_invalid = (df["event_type"] == -1) & (price_change > tolerance)

        invalid_mask = upturn_invalid | downturn_invalid
        invalid_count = invalid_mask.sum()

        affected = []
        if invalid_count > 0:
            affected = [i for i, m in enumerate(invalid_mask.to_list()) if m]

        return self._make_result(
            test_name=f"direction_matches_event_type[{day}]",
            passed=invalid_count == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{invalid_count} events with direction mismatch"
                if invalid_count > 0
                else "All directions match event_type"
            ),
            details={"day": str(day)},
            affected_events=affected,
        )
