"""Validador temporal (Nivel 3).

Valida consistencia temporal de eventos Silver.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult


class TemporalValidator(BaseValidator):
    """Validador de consistencia temporal."""

    name = "temporal"
    level = 3

    def validate_day(self, df: pl.DataFrame, day: date, theta: float) -> list[ValidationResult]:
        """Valida consistencia temporal de eventos de un día."""
        results = []

        results.append(self._test_dc_timestamps_monotonic(df, day))
        results.append(self._test_os_timestamps_monotonic(df, day))
        results.append(self._test_dc_before_os(df, day))
        results.append(self._test_reference_before_confirm(df, day))
        results.append(self._test_confirm_before_extreme(df, day))
        results.append(self._test_zero_os_rate(df, day))
        results.append(self._test_flash_event_rate(df, day))
        results.append(self._test_confirm_time_equals_last_dc_time(df, day))
        results.append(self._test_timestamps_positive(df, day))

        return results

    def _test_dc_timestamps_monotonic(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que timestamps dentro de time_dc son monotónicamente crecientes."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_timestamps_monotonic[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            time_dc_idx = df.columns.index("time_dc")
            time_dc = row[time_dc_idx]

            if time_dc is not None and len(time_dc) > 1:
                for i in range(len(time_dc) - 1):
                    if time_dc[i] > time_dc[i + 1]:
                        invalid_events.append(idx)
                        break

        return self._make_result(
            test_name=f"dc_timestamps_monotonic[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events with non-monotonic DC timestamps"
                if invalid_events
                else "All DC timestamps monotonic"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_os_timestamps_monotonic(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que timestamps dentro de time_os son monotónicamente crecientes."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"os_timestamps_monotonic[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            time_os_idx = df.columns.index("time_os")
            time_os = row[time_os_idx]

            if time_os is not None and len(time_os) > 1:
                for i in range(len(time_os) - 1):
                    if time_os[i] > time_os[i + 1]:
                        invalid_events.append(idx)
                        break

        return self._make_result(
            test_name=f"os_timestamps_monotonic[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events with non-monotonic OS timestamps"
                if invalid_events
                else "All OS timestamps monotonic"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_dc_before_os(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que la fase DC termina antes de que comience la fase OS."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_before_os[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
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

                if max_dc > min_os:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"dc_before_os[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events where DC overlaps OS"
                if invalid_events
                else "All DC phases before OS phases"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_reference_before_confirm(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica orden temporal: reference_time vs confirm_time.

        Casos válidos:
        - reference_time < confirm_time: evento normal
        - reference_time == confirm_time Y es flash event: válido

        Un flash event es cuando toda la fase DC ocurre en un solo instante
        (todos los ticks de time_dc tienen el mismo timestamp).
        Ver DC_FRAMEWORK.md §8.2.
        """
        if len(df) == 0:
            return self._make_result(
                test_name=f"reference_before_confirm[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []
        for idx, row in enumerate(df.iter_rows(named=True)):
            ref_t = row["reference_time"]
            confirm_t = row["confirm_time"]
            time_dc = row.get("time_dc") or []

            if ref_t > confirm_t:
                # Violación directa
                invalid_events.append(idx)
            elif ref_t == confirm_t:
                # Verificar si es flash event legítimo:
                # Todos los ticks de time_dc tienen el mismo timestamp
                if len(time_dc) > 0:
                    is_flash_event = all(t == time_dc[0] for t in time_dc)
                    if not is_flash_event:
                        invalid_events.append(idx)
                # Si time_dc está vacío con ref_t == confirm_t, también es sospechoso
                else:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"reference_before_confirm[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events with invalid reference/confirm ordering"
                if invalid_events
                else "All reference_time ordering valid (including flash events)"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_confirm_before_extreme(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica orden temporal: confirm_time vs extreme_time.

        Casos válidos:
        - confirm_time < extreme_time: evento normal con OS
        - confirm_time == extreme_time Y es Zero OS: válido (reversión inmediata)

        Casos inválidos:
        - confirm_time > extreme_time: violación temporal
        - confirm_time == extreme_time pero NO es Zero OS: data corruption

        Ver DC_FRAMEWORK.md §3.5 para definición de Zero OS.
        """
        if len(df) == 0:
            return self._make_result(
                test_name=f"confirm_before_extreme[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        # Filtrar eventos con extreme_time válido (no provisional)
        valid_extreme = df.filter(pl.col("extreme_time") != -1)

        if len(valid_extreme) == 0:
            return self._make_result(
                test_name=f"confirm_before_extreme[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events with valid extreme_time to validate",
            )

        # Obtener índices originales para mapeo
        valid_indices = df.with_row_index().filter(pl.col("extreme_time") != -1)["index"].to_list()

        invalid_events = []
        for local_idx, row in enumerate(valid_extreme.iter_rows(named=True)):
            confirm_t = row["confirm_time"]
            extreme_t = row["extreme_time"]
            price_os = row.get("price_os") or []

            if confirm_t > extreme_t:
                # Violación directa: confirm después de extreme
                invalid_events.append(valid_indices[local_idx])
            elif confirm_t == extreme_t:
                # Verificar que es Zero OS legítimo:
                # Solo verificar price_os vacío.
                # Nota: extreme_price puede diferir de confirm_price debido a
                # la regla conservadora (múltiples ticks en mismo timestamp).
                # Ver DC_FRAMEWORK.md §8.2.1-8.2.2.
                is_zero_os = len(price_os) == 0
                if not is_zero_os:
                    invalid_events.append(valid_indices[local_idx])

        return self._make_result(
            test_name=f"confirm_before_extreme[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events with invalid temporal ordering"
                if invalid_events
                else "All temporal ordering valid (including Zero OS)"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_zero_os_rate(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Reporta tasa de eventos con Zero OS (extreme_time == confirm_time).

        Este test es informativo (INFO) y no falla.
        Tasas muy altas pueden indicar problemas con θ o datos.
        """
        if len(df) == 0:
            return self._make_result(
                test_name=f"zero_os_rate[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No events to analyze",
            )

        # Solo contar eventos con extreme_time válido
        valid = df.filter(pl.col("extreme_time") != -1)
        if len(valid) == 0:
            return self._make_result(
                test_name=f"zero_os_rate[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No events with valid extreme_time",
            )

        # Contar Zero OS: extreme_time == confirm_time
        zero_os_count = valid.filter(pl.col("extreme_time") == pl.col("confirm_time")).height
        zero_os_rate = zero_os_count / len(valid)

        return self._make_result(
            test_name=f"zero_os_rate[{day}]",
            passed=True,  # Siempre pasa (informativo)
            severity=Severity.INFO,
            message=f"Zero OS rate: {zero_os_rate:.1%} ({zero_os_count}/{len(valid)})",
            details={
                "day": str(day),
                "zero_os_count": zero_os_count,
                "total_valid": len(valid),
                "zero_os_rate": round(zero_os_rate, 4),
            },
        )

    def _test_flash_event_rate(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Reporta tasa de flash events (reference_time == confirm_time).

        Un flash event ocurre cuando toda la fase DC sucede en un solo instante,
        tipicamente durante gaps de apertura o flash crashes.
        Este test es informativo (INFO) y no falla.
        """
        if len(df) == 0:
            return self._make_result(
                test_name=f"flash_event_rate[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No events to analyze",
            )

        # Contar flash events: reference_time == confirm_time
        flash_count = df.filter(pl.col("reference_time") == pl.col("confirm_time")).height
        flash_rate = flash_count / len(df)

        return self._make_result(
            test_name=f"flash_event_rate[{day}]",
            passed=True,  # Siempre pasa (informativo)
            severity=Severity.INFO,
            message=f"Flash event rate: {flash_rate:.1%} ({flash_count}/{len(df)})",
            details={
                "day": str(day),
                "flash_count": flash_count,
                "total_events": len(df),
                "flash_rate": round(flash_rate, 4),
            },
        )

    def _test_confirm_time_equals_last_dc_time(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que confirm_time == time_dc[-1]."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"confirm_time_equals_last_dc_time[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            confirm_time_idx = df.columns.index("confirm_time")
            time_dc_idx = df.columns.index("time_dc")

            confirm_time = row[confirm_time_idx]
            time_dc = row[time_dc_idx]

            if time_dc and len(time_dc) > 0:
                last_dc_time = time_dc[-1]
                if confirm_time != last_dc_time:
                    invalid_events.append(idx)

        return self._make_result(
            test_name=f"confirm_time_equals_last_dc_time[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events where confirm_time != last DC time"
                if invalid_events
                else "All confirm_time equals last DC time"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_timestamps_positive(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que todos los timestamps son positivos."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"timestamps_positive[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        # Verificar columnas escalares
        scalar_cols = ["reference_time", "confirm_time"]
        invalid_scalar = False

        for col in scalar_cols:
            if (df[col] <= 0).any():
                invalid_scalar = True
                break

        # extreme_time puede ser -1 (provisional), así que excluirlo de la verificación
        # de timestamps negativos

        # Verificar listas
        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            time_dc_idx = df.columns.index("time_dc")
            time_os_idx = df.columns.index("time_os")

            time_dc = row[time_dc_idx]
            time_os = row[time_os_idx]

            if time_dc:
                for t in time_dc:
                    if t <= 0:
                        invalid_events.append(idx)
                        break

            if time_os and idx not in invalid_events:
                for t in time_os:
                    if t <= 0:
                        invalid_events.append(idx)
                        break

        return self._make_result(
            test_name=f"timestamps_positive[{day}]",
            passed=not invalid_scalar and len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"Found non-positive timestamps (scalar: {invalid_scalar}, "
                f"list events: {len(invalid_events)})"
                if invalid_scalar or invalid_events
                else "All timestamps positive"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )
