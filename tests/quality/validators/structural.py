"""Validador estructural (Nivel 1).

Valida la estructura básica de los datos Silver.
"""

from datetime import date

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult


class StructuralValidator(BaseValidator):
    """Validador de estructura básica de datos Silver."""

    name = "structural"
    level = 1

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Valida estructura de un día de datos Silver."""
        results = []

        results.append(self._test_required_columns(df, day))
        results.append(self._test_column_types(df, day))
        results.append(self._test_no_null_scalars(df, day))
        results.append(self._test_non_empty_dataframe(df, day))

        return results

    def _test_required_columns(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que todas las columnas requeridas estén presentes."""
        missing = set(self.config.required_columns) - set(df.columns)

        return self._make_result(
            test_name=f"required_columns[{day}]",
            passed=len(missing) == 0,
            severity=Severity.CRITICAL,
            message=f"Missing columns: {missing}" if missing else "All columns present",
            details={"missing": list(missing), "found": df.columns, "day": str(day)},
        )

    def _test_column_types(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica los tipos de datos de cada columna."""
        type_errors = {}

        for col, expected_type_str in self.config.expected_types.items():
            if col not in df.columns:
                continue

            actual_type = str(df.schema[col])

            # Normalizar para comparación
            expected_normalized = expected_type_str.lower().replace(" ", "")
            actual_normalized = actual_type.lower().replace(" ", "")

            # Comparación flexible para tipos List
            if "list" in expected_normalized:
                # Extraer tipo interno
                if "list" in actual_normalized:
                    # Ambos son List, verificar compatibilidad
                    exp_inner = expected_normalized.replace("list(", "").replace(")", "")
                    # Polars puede mostrar como "List(Int8)" o "list[i8]"
                    if exp_inner not in actual_normalized and exp_inner.replace(
                        "int", "i"
                    ).replace("float", "f") not in actual_normalized:
                        type_errors[col] = {
                            "expected": expected_type_str,
                            "actual": actual_type,
                        }
                else:
                    type_errors[col] = {
                        "expected": expected_type_str,
                        "actual": actual_type,
                    }
            else:
                # Tipos escalares
                if expected_normalized not in actual_normalized:
                    type_errors[col] = {
                        "expected": expected_type_str,
                        "actual": actual_type,
                    }

        return self._make_result(
            test_name=f"column_types[{day}]",
            passed=len(type_errors) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"Type mismatches: {list(type_errors.keys())}"
                if type_errors
                else "All types correct"
            ),
            details={"type_errors": type_errors, "day": str(day)},
        )

    def _test_no_null_scalars(self, df: pl.DataFrame, day: date) -> ValidationResult:
        """Verifica que no hay nulls en columnas escalares."""
        scalar_cols = [
            "event_type",
            "reference_price",
            "reference_time",
            "extreme_price",
            "extreme_time",
            "confirm_price",
            "confirm_time",
        ]

        null_counts = {}
        for col in scalar_cols:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    null_counts[col] = null_count

        return self._make_result(
            test_name=f"no_null_scalars[{day}]",
            passed=len(null_counts) == 0,
            severity=Severity.CRITICAL,
            message=(
                f"Null values found: {null_counts}"
                if null_counts
                else "No null values"
            ),
            details={"null_counts": null_counts, "day": str(day)},
        )

    def _test_non_empty_dataframe(
        self, df: pl.DataFrame, day: date
    ) -> ValidationResult:
        """Verifica que el DataFrame no está vacío."""
        n_events = len(df)

        return self._make_result(
            test_name=f"non_empty_dataframe[{day}]",
            passed=n_events > 0,
            severity=Severity.HIGH,
            message=(
                f"DataFrame has {n_events} events"
                if n_events > 0
                else "Empty DataFrame"
            ),
            details={"n_events": n_events, "day": str(day)},
        )
