"""Validador de precios (Nivel 4).

Valida consistencia de precios en eventos Silver.
"""

import math
from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult


class PriceValidator(BaseValidator):
    """Validador de consistencia de precios."""

    name = "price"
    level = 4

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Valida consistencia de precios de un día."""
        results = []

        results.append(self._test_prices_positive(df, day))
        results.append(self._test_prices_finite(df, day))
        results.append(self._test_confirm_price_in_dc_range(df, day))
        results.append(self._test_confirm_price_matches_dc(df, day))
        results.append(self._test_reference_price_chain(df, day))
        results.append(self._test_extreme_price_in_os_range(df, day))

        return results

    def _test_prices_positive(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que todos los precios son positivos."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"prices_positive[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        # Verificar columnas escalares (excepto extreme_price que puede ser -1)
        scalar_cols = ["reference_price", "confirm_price"]
        invalid_scalar = {}

        for col in scalar_cols:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                invalid_scalar[col] = invalid_count

        # Verificar listas
        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            price_dc_idx = df.columns.index("price_dc")
            price_os_idx = df.columns.index("price_os")

            price_dc = row[price_dc_idx]
            price_os = row[price_os_idx]

            if price_dc:
                for p in price_dc:
                    if p is not None and p <= 0:
                        invalid_events.append(idx)
                        break

            if price_os and idx not in invalid_events:
                for p in price_os:
                    if p is not None and p <= 0:
                        invalid_events.append(idx)
                        break

        passed = len(invalid_scalar) == 0 and len(invalid_events) == 0

        return self._make_result(
            test_name=f"prices_positive[{day}]",
            passed=passed,
            severity=Severity.CRITICAL,
            message=(
                f"Non-positive prices found (scalars: {invalid_scalar}, "
                f"list events: {len(invalid_events)})"
                if not passed
                else "All prices positive"
            ),
            details={"invalid_scalars": invalid_scalar, "day": str(day)},
            affected_events=invalid_events,
        )

    def _test_prices_finite(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que no hay NaN ni Inf en precios."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"prices_finite[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        # Verificar columnas escalares
        scalar_cols = ["reference_price", "extreme_price", "confirm_price"]
        invalid_scalar = {}

        for col in scalar_cols:
            # extreme_price puede ser -1.0 (provisional), pero no NaN/Inf
            nan_count = df[col].is_nan().sum()
            inf_count = df[col].is_infinite().sum()
            if nan_count > 0 or inf_count > 0:
                invalid_scalar[col] = {"nan": nan_count, "inf": inf_count}

        # Verificar listas
        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            price_dc_idx = df.columns.index("price_dc")
            price_os_idx = df.columns.index("price_os")

            price_dc = row[price_dc_idx]
            price_os = row[price_os_idx]

            if price_dc:
                for p in price_dc:
                    if p is not None and (math.isnan(p) or math.isinf(p)):
                        invalid_events.append(idx)
                        break

            if price_os and idx not in invalid_events:
                for p in price_os:
                    if p is not None and (math.isnan(p) or math.isinf(p)):
                        invalid_events.append(idx)
                        break

        passed = len(invalid_scalar) == 0 and len(invalid_events) == 0

        return self._make_result(
            test_name=f"prices_finite[{day}]",
            passed=passed,
            severity=Severity.CRITICAL,
            message=(
                f"NaN/Inf prices found (scalars: {invalid_scalar}, "
                f"list events: {len(invalid_events)})"
                if not passed
                else "All prices finite"
            ),
            details={"invalid_scalars": invalid_scalar, "day": str(day)},
            affected_events=invalid_events,
        )

    def _test_confirm_price_in_dc_range(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que confirm_price está dentro del rango de price_dc."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"confirm_price_in_dc_range[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            confirm_price_idx = df.columns.index("confirm_price")
            price_dc_idx = df.columns.index("price_dc")

            confirm_price = row[confirm_price_idx]
            price_dc = row[price_dc_idx]

            if price_dc and len(price_dc) > 0:
                min_dc = min(price_dc)
                max_dc = max(price_dc)

                if confirm_price < min_dc or confirm_price > max_dc:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"confirm_price_in_dc_range[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.HIGH,
            message=(
                f"{len(invalid_events)} events with confirm_price outside DC range"
                if invalid_events
                else "All confirm_price within DC range"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_confirm_price_matches_dc(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que confirm_price aparece en price_dc."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"confirm_price_matches_dc[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []
        tolerance = self.config.price_tolerance

        for idx in range(len(df)):
            row = df.row(idx)
            confirm_price_idx = df.columns.index("confirm_price")
            price_dc_idx = df.columns.index("price_dc")

            confirm_price = row[confirm_price_idx]
            price_dc = row[price_dc_idx]

            if price_dc and len(price_dc) > 0:
                # Verificar si confirm_price está en la lista (con tolerancia)
                found = False
                for p in price_dc:
                    if abs(p - confirm_price) < tolerance:
                        found = True
                        break

                if not found:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"confirm_price_matches_dc[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events where confirm_price not in price_dc"
                if invalid_events
                else "All confirm_price found in price_dc"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_reference_price_chain(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica la cadena de precios de referencia intra-día."""
        if len(df) <= 1:
            return self._make_result(
                test_name=f"reference_price_chain[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="Not enough events to validate chain",
            )

        invalid_events = []
        tolerance = self.config.price_tolerance

        for idx in range(1, len(df)):
            row = df.row(idx)
            prev_row = df.row(idx - 1)

            ref_price_idx = df.columns.index("reference_price")
            ext_price_idx = df.columns.index("extreme_price")

            ref_price = row[ref_price_idx]
            prev_extreme = prev_row[ext_price_idx]

            # Solo validar si el extreme anterior no es provisional
            if prev_extreme != -1.0:
                if abs(ref_price - prev_extreme) > tolerance:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"reference_price_chain[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.HIGH,
            message=(
                f"{len(invalid_events)} events with broken reference chain"
                if invalid_events
                else "Reference price chain intact"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_extreme_price_in_os_range(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que extreme_price está dentro del rango de price_os."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"extreme_price_in_os_range[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="No events to validate",
            )

        invalid_events = []
        tolerance = self.config.price_tolerance

        for idx in range(len(df)):
            row = df.row(idx)
            extreme_price_idx = df.columns.index("extreme_price")
            price_os_idx = df.columns.index("price_os")
            event_type_idx = df.columns.index("event_type")

            extreme_price = row[extreme_price_idx]
            price_os = row[price_os_idx]
            event_type = row[event_type_idx]

            # Solo validar si extreme_price no es provisional y OS no está vacío
            if extreme_price != -1.0 and price_os and len(price_os) > 0:
                if event_type == 1:  # Upturn: extreme debe ser max(OS)
                    max_os = max(price_os)
                    if abs(extreme_price - max_os) > tolerance:
                        invalid_events.append(idx)
                elif event_type == -1:  # Downturn: extreme debe ser min(OS)
                    min_os = min(price_os)
                    if abs(extreme_price - min_os) > tolerance:
                        invalid_events.append(idx)

        return self._make_result(
            test_name=f"extreme_price_in_os_range[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.HIGH,
            message=(
                f"{len(invalid_events)} events with extreme_price not matching OS extreme"
                if invalid_events
                else "All extreme_price matches OS extreme"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )
