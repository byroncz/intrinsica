"""Validador de colisiones (Nivel 6).

Valida el manejo correcto de timestamps simultáneos.
Este validador verifica la corrección del bug crítico de colisión de ticks.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult


class CollisionValidator(BaseValidator):
    """Validador de colisiones de timestamps."""

    name = "collision"
    level = 6

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Valida manejo de colisiones de un día."""
        results = []

        results.append(self._test_no_os_ticks_at_confirm_time(df, day))
        results.append(self._test_all_confirm_instant_in_dc(df, day))
        results.append(self._test_dc_os_no_timestamp_overlap(df, day))
        results.append(self._test_conservative_price_selection(df, day, theta))

        return results

    def _test_no_os_ticks_at_confirm_time(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """CRÍTICO: Verifica que ningún tick del OS tiene el mismo timestamp que confirm_time."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"no_os_ticks_at_confirm_time[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            confirm_time_idx = df.columns.index("confirm_time")
            time_os_idx = df.columns.index("time_os")

            confirm_time = row[confirm_time_idx]
            time_os = row[time_os_idx]

            if time_os and len(time_os) > 0:
                for t in time_os:
                    if t == confirm_time:
                        invalid_events.append(idx)
                        break

        return self._make_result(
            test_name=f"no_os_ticks_at_confirm_time[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"COLLISION BUG: {len(invalid_events)} events with OS ticks at confirm_time"
                if invalid_events
                else "No OS ticks at confirm_time"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_all_confirm_instant_in_dc(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """CRÍTICO: Verifica que TODOS los ticks con timestamp == confirm_time están en DC."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"all_confirm_instant_in_dc[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []
        collision_details = []

        for idx in range(len(df)):
            row = df.row(idx)
            confirm_time_idx = df.columns.index("confirm_time")
            time_dc_idx = df.columns.index("time_dc")
            time_os_idx = df.columns.index("time_os")

            confirm_time = row[confirm_time_idx]
            time_dc = row[time_dc_idx]
            time_os = row[time_os_idx]

            # Contar ticks en DC con confirm_time
            dc_count = 0
            if time_dc:
                dc_count = sum(1 for t in time_dc if t == confirm_time)

            # Contar ticks en OS con confirm_time (debería ser 0)
            os_count = 0
            if time_os:
                os_count = sum(1 for t in time_os if t == confirm_time)

            if os_count > 0:
                invalid_events.append(idx)
                collision_details.append({
                    "event": idx,
                    "confirm_time": confirm_time,
                    "dc_at_confirm": dc_count,
                    "os_at_confirm": os_count,
                })

        return self._make_result(
            test_name=f"all_confirm_instant_in_dc[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"COLLISION BUG: {len(invalid_events)} events with confirm-instant ticks in OS"
                if invalid_events
                else "All confirm-instant ticks in DC"
            ),
            details={"day": str(day), "collisions": collision_details[:10]},
            affected_events=invalid_events,
        )

    def _test_dc_os_no_timestamp_overlap(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que no hay solapamiento de timestamps entre DC y OS."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_os_no_timestamp_overlap[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            time_dc_idx = df.columns.index("time_dc")
            time_os_idx = df.columns.index("time_os")

            time_dc = row[time_dc_idx]
            time_os = row[time_os_idx]

            # Solo validar si ambas listas tienen elementos
            if time_dc and time_os and len(time_dc) > 0 and len(time_os) > 0:
                max_dc = max(time_dc)
                min_os = min(time_os)

                # Estrictamente menor (no menor-igual como en temporal)
                if max_dc >= min_os:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"dc_os_no_timestamp_overlap[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.HIGH,
            message=(
                f"{len(invalid_events)} events with DC/OS timestamp overlap"
                if invalid_events
                else "No DC/OS timestamp overlap"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_conservative_price_selection(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> ValidationResult:
        """Verifica que se aplicó la regla conservadora para ticks simultáneos."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"conservative_price_selection[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="No events to validate",
            )

        invalid_events = []
        tolerance = self.config.price_tolerance

        for idx in range(len(df)):
            row = df.row(idx)
            confirm_time_idx = df.columns.index("confirm_time")
            confirm_price_idx = df.columns.index("confirm_price")
            reference_price_idx = df.columns.index("reference_price")
            time_dc_idx = df.columns.index("time_dc")
            price_dc_idx = df.columns.index("price_dc")
            event_type_idx = df.columns.index("event_type")

            confirm_time = row[confirm_time_idx]
            confirm_price = row[confirm_price_idx]
            reference_price = row[reference_price_idx]
            time_dc = row[time_dc_idx]
            price_dc = row[price_dc_idx]
            event_type = row[event_type_idx]

            if not time_dc or not price_dc:
                continue

            # Encontrar todos los precios al momento de confirmación
            prices_at_confirm = []
            for i, t in enumerate(time_dc):
                if t == confirm_time:
                    prices_at_confirm.append(price_dc[i])

            # Si hay múltiples precios, verificar regla conservadora
            if len(prices_at_confirm) > 1:
                # Calcular umbral
                if event_type == 1:  # Upturn
                    threshold = reference_price * (1 + theta)
                    # Solo precios que cruzan umbral
                    crossing_prices = [p for p in prices_at_confirm if p >= threshold - tolerance]
                    if crossing_prices:
                        expected_confirm = min(crossing_prices)  # Mínimo para upturn
                        if abs(confirm_price - expected_confirm) > tolerance:
                            invalid_events.append(idx)
                else:  # Downturn
                    threshold = reference_price * (1 - theta)
                    crossing_prices = [p for p in prices_at_confirm if p <= threshold + tolerance]
                    if crossing_prices:
                        expected_confirm = max(crossing_prices)  # Máximo para downturn
                        if abs(confirm_price - expected_confirm) > tolerance:
                            invalid_events.append(idx)

        return self._make_result(
            test_name=f"conservative_price_selection[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.HIGH,
            message=(
                f"{len(invalid_events)} events with incorrect conservative price selection"
                if invalid_events
                else "Conservative price selection verified"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )
