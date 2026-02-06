"""Validador de leyes de escala (Nivel 8).

Valida propiedades estadísticas esperadas del paradigma DC.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult, ValidatorReport


class ScaleLawsValidator(BaseValidator):
    """Validador de leyes de escala DC."""

    name = "scale_laws"
    level = 8

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Valida leyes de escala de un día."""
        results = []

        results.append(self._test_avg_os_magnitude_near_theta(df, day, theta))
        results.append(self._test_avg_tmv_near_2theta(df, day, theta))
        results.append(self._test_zero_os_rate_bounded(df, day))
        results.append(self._test_dc_duration_distribution(df, day))

        return results

    def _test_avg_os_magnitude_near_theta(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> ValidationResult:
        """Verifica la ley del factor 2: avg(|OS|) ≈ θ."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"avg_os_magnitude_near_theta[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No events to validate",
            )

        # Calcular magnitud OS para cada evento
        os_magnitudes = []

        for idx in range(len(df)):
            row = df.row(idx)
            extreme_price_idx = df.columns.index("extreme_price")
            confirm_price_idx = df.columns.index("confirm_price")
            price_os_idx = df.columns.index("price_os")

            extreme_price = row[extreme_price_idx]
            confirm_price = row[confirm_price_idx]
            price_os = row[price_os_idx]

            # Solo calcular si extreme no es provisional y OS no está vacío
            if extreme_price != -1.0 and price_os and len(price_os) > 0:
                os_mag = abs(extreme_price - confirm_price) / confirm_price
                os_magnitudes.append(os_mag)

        if len(os_magnitudes) == 0:
            return self._make_result(
                test_name=f"avg_os_magnitude_near_theta[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No valid OS phases to analyze",
            )

        avg_os = sum(os_magnitudes) / len(os_magnitudes)

        # Verificar límites
        min_expected = self.config.os_magnitude_factor_min * theta
        max_expected = self.config.os_magnitude_factor_max * theta

        passed = min_expected < avg_os < max_expected

        return self._make_result(
            test_name=f"avg_os_magnitude_near_theta[{day}]",
            passed=passed,
            severity=Severity.INFO,
            message=(
                f"avg(|OS|) = {avg_os:.6f} ({'within' if passed else 'outside'} "
                f"[{min_expected:.6f}, {max_expected:.6f}])"
            ),
            details={
                "day": str(day),
                "avg_os_magnitude": avg_os,
                "theta": theta,
                "min_expected": min_expected,
                "max_expected": max_expected,
                "n_events_analyzed": len(os_magnitudes),
            },
        )

    def _test_avg_tmv_near_2theta(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> ValidationResult:
        """Verifica que TMV promedio ≈ 2θ."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"avg_tmv_near_2theta[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No events to validate",
            )

        tmv_values = []

        for idx in range(len(df)):
            row = df.row(idx)
            reference_price_idx = df.columns.index("reference_price")
            confirm_price_idx = df.columns.index("confirm_price")
            extreme_price_idx = df.columns.index("extreme_price")

            reference_price = row[reference_price_idx]
            confirm_price = row[confirm_price_idx]
            extreme_price = row[extreme_price_idx]

            # TMV = |DC magnitude| + |OS magnitude|
            dc_mag = abs(confirm_price - reference_price) / reference_price

            if extreme_price != -1.0:
                os_mag = abs(extreme_price - confirm_price) / confirm_price
                tmv = dc_mag + os_mag
                tmv_values.append(tmv)

        if len(tmv_values) == 0:
            return self._make_result(
                test_name=f"avg_tmv_near_2theta[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No complete events to calculate TMV",
            )

        avg_tmv = sum(tmv_values) / len(tmv_values)

        # Verificar límites
        min_expected = self.config.tmv_factor_min * theta
        max_expected = self.config.tmv_factor_max * theta

        passed = min_expected < avg_tmv < max_expected

        return self._make_result(
            test_name=f"avg_tmv_near_2theta[{day}]",
            passed=passed,
            severity=Severity.INFO,
            message=(
                f"avg(TMV) = {avg_tmv:.6f} ({'within' if passed else 'outside'} "
                f"[{min_expected:.6f}, {max_expected:.6f}])"
            ),
            details={
                "day": str(day),
                "avg_tmv": avg_tmv,
                "theta": theta,
                "expected_tmv": 2 * theta,
                "min_expected": min_expected,
                "max_expected": max_expected,
                "n_events_analyzed": len(tmv_values),
            },
        )

    def _test_zero_os_rate_bounded(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que la tasa de OS vacíos no es excesiva."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"zero_os_rate_bounded[{day}]",
                passed=True,
                severity=Severity.MEDIUM,
                message="No events to validate",
            )

        # Contar OS vacíos
        os_lengths = df["price_os"].list.len()
        zero_os_count = (os_lengths == 0).sum()
        total_events = len(df)

        zero_os_rate = zero_os_count / total_events

        passed = zero_os_rate < self.config.max_zero_os_rate

        return self._make_result(
            test_name=f"zero_os_rate_bounded[{day}]",
            passed=passed,
            severity=Severity.MEDIUM,
            message=(
                f"Zero OS rate = {zero_os_rate:.1%} "
                f"({'OK' if passed else f'exceeds {self.config.max_zero_os_rate:.0%}'})"
            ),
            details={
                "day": str(day),
                "zero_os_count": zero_os_count,
                "total_events": total_events,
                "zero_os_rate": zero_os_rate,
                "max_allowed": self.config.max_zero_os_rate,
            },
        )

    def _test_dc_duration_distribution(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Reporta estadísticas de duración DC (informativo)."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_duration_distribution[{day}]",
                passed=True,
                severity=Severity.INFO,
                message="No events to validate",
            )

        # Calcular duración DC en nanosegundos
        dc_durations = (df["confirm_time"] - df["reference_time"]).to_list()

        # Convertir a segundos
        dc_durations_sec = [d / 1e9 for d in dc_durations]

        min_dur = min(dc_durations_sec)
        max_dur = max(dc_durations_sec)
        avg_dur = sum(dc_durations_sec) / len(dc_durations_sec)

        # Calcular mediana
        sorted_durs = sorted(dc_durations_sec)
        n = len(sorted_durs)
        median_dur = (
            sorted_durs[n // 2]
            if n % 2 == 1
            else (sorted_durs[n // 2 - 1] + sorted_durs[n // 2]) / 2
        )

        return self._make_result(
            test_name=f"dc_duration_distribution[{day}]",
            passed=True,  # Siempre pasa, es informativo
            severity=Severity.INFO,
            message=(
                f"DC duration: min={min_dur:.1f}s, avg={avg_dur:.1f}s, "
                f"median={median_dur:.1f}s, max={max_dur:.1f}s"
            ),
            details={
                "day": str(day),
                "n_events": len(dc_durations_sec),
                "min_duration_sec": min_dur,
                "avg_duration_sec": avg_dur,
                "median_duration_sec": median_dur,
                "max_duration_sec": max_dur,
            },
        )
