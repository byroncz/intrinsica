"""Validador intra-evento (Nivel 2).

Valida consistencia interna de cada evento Silver.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult


class IntraEventValidator(BaseValidator):
    """Validador de consistencia intra-evento."""

    name = "intra_event"
    level = 2

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Valida consistencia interna de eventos de un día."""
        results = []

        results.append(self._test_dc_list_lengths_consistent(df, day))
        results.append(self._test_os_list_lengths_consistent(df, day))
        results.append(self._test_event_type_valid_values(df, day))
        results.append(self._test_dc_not_empty(df, day))
        results.append(self._test_directions_valid_values(df, day))
        results.append(self._test_quantities_positive(df, day))

        return results

    def _test_dc_list_lengths_consistent(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que las 4 listas DC tienen la misma longitud para cada evento."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_list_lengths_consistent[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        # Calcular longitudes
        lengths_df = df.select(
            pl.col("price_dc").list.len().alias("len_price"),
            pl.col("time_dc").list.len().alias("len_time"),
            pl.col("qty_dc").list.len().alias("len_qty"),
            pl.col("dir_dc").list.len().alias("len_dir"),
        )

        # Verificar consistencia por fila
        inconsistent = lengths_df.filter(
            (pl.col("len_price") != pl.col("len_time"))
            | (pl.col("len_price") != pl.col("len_qty"))
            | (pl.col("len_price") != pl.col("len_dir"))
        )

        affected = []
        if len(inconsistent) > 0:
            affected = list(range(len(df)))
            # Encontrar índices reales
            mask = (
                (lengths_df["len_price"] != lengths_df["len_time"])
                | (lengths_df["len_price"] != lengths_df["len_qty"])
                | (lengths_df["len_price"] != lengths_df["len_dir"])
            )
            affected = [i for i, m in enumerate(mask.to_list()) if m]

        return self._make_result(
            test_name=f"dc_list_lengths_consistent[{day}]",
            passed=len(inconsistent) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(inconsistent)} events with inconsistent DC list lengths"
                if len(inconsistent) > 0
                else "All DC list lengths consistent"
            ),
            details={"day": str(day)},
            affected_events=affected,
        )

    def _test_os_list_lengths_consistent(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que las 4 listas OS tienen la misma longitud para cada evento."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"os_list_lengths_consistent[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        lengths_df = df.select(
            pl.col("price_os").list.len().alias("len_price"),
            pl.col("time_os").list.len().alias("len_time"),
            pl.col("qty_os").list.len().alias("len_qty"),
            pl.col("dir_os").list.len().alias("len_dir"),
        )

        inconsistent = lengths_df.filter(
            (pl.col("len_price") != pl.col("len_time"))
            | (pl.col("len_price") != pl.col("len_qty"))
            | (pl.col("len_price") != pl.col("len_dir"))
        )

        affected = []
        if len(inconsistent) > 0:
            mask = (
                (lengths_df["len_price"] != lengths_df["len_time"])
                | (lengths_df["len_price"] != lengths_df["len_qty"])
                | (lengths_df["len_price"] != lengths_df["len_dir"])
            )
            affected = [i for i, m in enumerate(mask.to_list()) if m]

        return self._make_result(
            test_name=f"os_list_lengths_consistent[{day}]",
            passed=len(inconsistent) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(inconsistent)} events with inconsistent OS list lengths"
                if len(inconsistent) > 0
                else "All OS list lengths consistent"
            ),
            details={"day": str(day)},
            affected_events=affected,
        )

    def _test_event_type_valid_values(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que event_type solo contiene 1 (upturn) o -1 (downturn)."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"event_type_valid_values[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_mask = ~df["event_type"].is_in([1, -1])
        invalid_count = invalid_mask.sum()

        affected = []
        if invalid_count > 0:
            affected = [i for i, m in enumerate(invalid_mask.to_list()) if m]

        unique_values = df["event_type"].unique().to_list()

        return self._make_result(
            test_name=f"event_type_valid_values[{day}]",
            passed=invalid_count == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{invalid_count} events with invalid event_type (found: {unique_values})"
                if invalid_count > 0
                else "All event_type values valid"
            ),
            details={"unique_values": unique_values, "day": str(day)},
            affected_events=affected,
        )

    def _test_dc_not_empty(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que la fase DC nunca está vacía."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"dc_not_empty[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        dc_lengths = df["price_dc"].list.len()
        empty_mask = dc_lengths == 0
        empty_count = empty_mask.sum()

        affected = []
        if empty_count > 0:
            affected = [i for i, m in enumerate(empty_mask.to_list()) if m]

        return self._make_result(
            test_name=f"dc_not_empty[{day}]",
            passed=empty_count == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{empty_count} events with empty DC phase"
                if empty_count > 0
                else "All DC phases non-empty"
            ),
            details={"day": str(day)},
            affected_events=affected,
        )

    def _test_directions_valid_values(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que direction solo contiene 1 (buy) o -1 (sell)."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"directions_valid_values[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            # Obtener índices de columnas
            dir_dc_idx = df.columns.index("dir_dc")
            dir_os_idx = df.columns.index("dir_os")

            dir_dc = row[dir_dc_idx]
            dir_os = row[dir_os_idx]

            # Verificar valores DC
            if dir_dc is not None:
                for val in dir_dc:
                    if val not in (1, -1):
                        invalid_events.append(idx)
                        break

            # Verificar valores OS
            if dir_os is not None and idx not in invalid_events:
                for val in dir_os:
                    if val not in (1, -1):
                        invalid_events.append(idx)
                        break

        return self._make_result(
            test_name=f"directions_valid_values[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"{len(invalid_events)} events with invalid direction values"
                if invalid_events
                else "All direction values valid"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )

    def _test_quantities_positive(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que todas las cantidades son positivas."""
        if len(df) == 0:
            return self._make_result(
                test_name=f"quantities_positive[{day}]",
                passed=True,
                severity=Severity.HIGH,
                message="No events to validate",
            )

        invalid_events = []

        for idx in range(len(df)):
            row = df.row(idx)
            qty_dc_idx = df.columns.index("qty_dc")
            qty_os_idx = df.columns.index("qty_os")

            qty_dc = row[qty_dc_idx]
            qty_os = row[qty_os_idx]

            # Verificar cantidades DC
            if qty_dc is not None:
                for val in qty_dc:
                    if val is not None and val <= 0:
                        invalid_events.append(idx)
                        break

            # Verificar cantidades OS
            if qty_os is not None and idx not in invalid_events:
                for val in qty_os:
                    if val is not None and val <= 0:
                        invalid_events.append(idx)
                        break

        return self._make_result(
            test_name=f"quantities_positive[{day}]",
            passed=len(invalid_events) == 0,
            severity=Severity.HIGH,
            message=(
                f"{len(invalid_events)} events with non-positive quantities"
                if invalid_events
                else "All quantities positive"
            ),
            details={"day": str(day)},
            affected_events=invalid_events,
        )
