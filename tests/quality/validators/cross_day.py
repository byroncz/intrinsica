"""Validador cross-día (Nivel 7).

Valida continuidad y consistencia entre días.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult, ValidatorReport


class CrossDayValidator(BaseValidator):
    """Validador de consistencia cross-día."""

    name = "cross_day"
    level = 7

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Este validador necesita contexto de múltiples días.

        La validación real se hace en validate_month().
        """
        # Retornar resultado vacío - la validación se hace en validate_month
        return []

    def validate_month(
        self, data: dict[date, pl.DataFrame], theta: float
    ) -> ValidatorReport:
        """Valida consistencia entre días de un mes.

        Override del método base para acceder a múltiples días.
        """
        all_results = []
        sorted_days = sorted(data.keys())

        # Validar cadena de referencia entre días
        all_results.extend(self._test_reference_chain_across_days(data, sorted_days))

        # Validar continuidad de tendencia
        all_results.extend(self._test_trend_continuity(data, sorted_days))

        # Validar eventos provisionales
        all_results.extend(self._test_last_event_extreme_provisional(data, sorted_days))

        return ValidatorReport(
            validator_name=self.name,
            level=self.level,
            results=all_results,
        )

    def _test_reference_chain_across_days(
        self, data: dict[date, pl.DataFrame], sorted_days: list[date]
    ) -> list[ValidationResult]:
        """Verifica continuidad de precios de referencia entre días."""
        results = []
        tolerance = self.config.price_tolerance

        for i in range(1, len(sorted_days)):
            prev_day = sorted_days[i - 1]
            curr_day = sorted_days[i]

            prev_df = data[prev_day]
            curr_df = data[curr_day]

            if len(prev_df) == 0 or len(curr_df) == 0:
                continue

            # Último extreme del día anterior
            last_extreme = prev_df.row(-1)[prev_df.columns.index("extreme_price")]

            # Primer reference del día actual
            first_reference = curr_df.row(0)[curr_df.columns.index("reference_price")]

            # Solo validar si extreme no es provisional
            if last_extreme != -1.0:
                passed = abs(first_reference - last_extreme) <= tolerance

                results.append(
                    self._make_result(
                        test_name=f"reference_chain_across_days[{prev_day}->{curr_day}]",
                        passed=passed,
                        severity=Severity.MEDIUM,
                        message=(
                            f"Reference chain broken: {last_extreme:.6f} → {first_reference:.6f}"
                            if not passed
                            else "Reference chain intact"
                        ),
                        details={
                            "prev_day": str(prev_day),
                            "curr_day": str(curr_day),
                            "prev_extreme": last_extreme,
                            "curr_reference": first_reference,
                        },
                    )
                )

        return results

    def _test_trend_continuity(
        self, data: dict[date, pl.DataFrame], sorted_days: list[date]
    ) -> list[ValidationResult]:
        """Verifica coherencia de tendencia entre días (informativo)."""
        results = []

        for i in range(1, len(sorted_days)):
            prev_day = sorted_days[i - 1]
            curr_day = sorted_days[i]

            prev_df = data[prev_day]
            curr_df = data[curr_day]

            if len(prev_df) == 0 or len(curr_df) == 0:
                continue

            last_event_type = prev_df.row(-1)[prev_df.columns.index("event_type")]
            first_event_type = curr_df.row(0)[curr_df.columns.index("event_type")]

            # Esperamos alternancia, pero reversiones intra-día son posibles
            expected_reversal = last_event_type != first_event_type

            results.append(
                self._make_result(
                    test_name=f"trend_continuity[{prev_day}->{curr_day}]",
                    passed=True,  # Siempre pasa, es informativo
                    severity=Severity.INFO,
                    message=(
                        f"Trend {'reversed' if expected_reversal else 'continued'}: "
                        f"{last_event_type} → {first_event_type}"
                    ),
                    details={
                        "prev_day": str(prev_day),
                        "curr_day": str(curr_day),
                        "prev_event_type": last_event_type,
                        "curr_event_type": first_event_type,
                        "reversed": expected_reversal,
                    },
                )
            )

        return results

    def _test_last_event_extreme_provisional(
        self, data: dict[date, pl.DataFrame], sorted_days: list[date]
    ) -> list[ValidationResult]:
        """Documenta eventos con extreme_price provisional."""
        results = []

        for day in sorted_days:
            df = data[day]

            if len(df) == 0:
                continue

            last_extreme = df.row(-1)[df.columns.index("extreme_price")]
            is_provisional = last_extreme == -1.0

            results.append(
                self._make_result(
                    test_name=f"last_event_extreme_provisional[{day}]",
                    passed=True,  # Siempre pasa, es informativo
                    severity=Severity.INFO,
                    message=(
                        f"Last event extreme is {'PROVISIONAL' if is_provisional else 'confirmed'}"
                    ),
                    details={
                        "day": str(day),
                        "extreme_price": last_extreme,
                        "is_provisional": is_provisional,
                    },
                )
            )

        return results
