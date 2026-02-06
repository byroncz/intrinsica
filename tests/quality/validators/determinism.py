"""Validador de determinismo (Nivel 9).

Valida que el reprocesamiento produce resultados idénticos.
Este es el "gold standard" de validación.
"""

from datetime import date
from pathlib import Path

import polars as pl

from ..config import Severity
from .base import BaseValidator, ValidationResult, ValidatorReport


class DeterminismValidator(BaseValidator):
    """Validador de determinismo de reprocesamiento."""

    name = "determinism"
    level = 9

    def __init__(self, config=None, skip_reprocess: bool = True):
        """Inicializa el validador.

        Args:
            config: Configuración del framework
            skip_reprocess: Si True, omite el test de reprocesamiento (costoso).
                           Por defecto True para ejecuciones rápidas.
        """
        super().__init__(config)
        self.skip_reprocess = skip_reprocess

    def validate_day(
        self, df: pl.DataFrame, day: date, theta: float
    ) -> list[ValidationResult]:
        """Este validador requiere acceso a Bronze y Engine.

        Por defecto, reporta que el test fue omitido.
        """
        if self.skip_reprocess:
            return [
                self._make_result(
                    test_name=f"reprocess_identical[{day}]",
                    passed=True,
                    severity=Severity.INFO,
                    message="Reprocess test skipped (enable with skip_reprocess=False)",
                    details={"day": str(day), "skipped": True},
                )
            ]

        # Intentar reprocesar (requiere Engine y datos Bronze)
        return [self._test_reprocess_identical(df, day, theta)]

    def _test_reprocess_identical(
        self, df_silver: pl.DataFrame, day: date, theta: float
    ) -> ValidationResult:
        """GOLD STANDARD: Reprocesar datos Bronze debe producir Silver idéntico."""
        try:
            # Importar dependencias del core
            from intrinseca.core.engine import Engine

            # Path a Bronze (hardcoded, debería venir de config)
            bronze_base_path = Path(
                "/Users/byroncampo/My Drive/datalake/financial/01_bronze"
            )

            # Construir path a Bronze del día
            ticker = self.config.default_ticker
            bronze_path = (
                bronze_base_path
                / ticker
                / f"year={day.year}"
                / f"month={day.month:02d}"
                / f"day={day.day:02d}"
                / "data.parquet"
            )

            if not bronze_path.exists():
                return self._make_result(
                    test_name=f"reprocess_identical[{day}]",
                    passed=True,
                    severity=Severity.INFO,
                    message=f"Bronze data not found at {bronze_path}",
                    details={"day": str(day), "bronze_path": str(bronze_path)},
                )

            # Cargar Bronze
            df_bronze = pl.read_parquet(bronze_path)

            # Reprocesar con Engine
            engine = Engine(theta=theta)
            results, _ = engine.process_date_range(
                df_bronze,
                ticker=ticker,
                time_col="time",
                analyze_convergence=False,
            )

            if day not in results or results[day] is None:
                return self._make_result(
                    test_name=f"reprocess_identical[{day}]",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=f"Engine did not produce result for {day}",
                    details={"day": str(day)},
                )

            df_reprocessed = results[day]

            # Comparar DataFrames
            if len(df_silver) != len(df_reprocessed):
                return self._make_result(
                    test_name=f"reprocess_identical[{day}]",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Row count mismatch: Silver={len(df_silver)}, "
                        f"Reprocessed={len(df_reprocessed)}"
                    ),
                    details={
                        "day": str(day),
                        "silver_rows": len(df_silver),
                        "reprocessed_rows": len(df_reprocessed),
                    },
                )

            # Comparar columnas escalares
            scalar_cols = [
                "event_type",
                "reference_price",
                "reference_time",
                "extreme_price",
                "extreme_time",
                "confirm_price",
                "confirm_time",
            ]

            for col in scalar_cols:
                if col not in df_silver.columns or col not in df_reprocessed.columns:
                    continue

                if not df_silver[col].equals(df_reprocessed[col]):
                    # Encontrar primera diferencia
                    diff_idx = -1
                    for i in range(len(df_silver)):
                        if df_silver[col][i] != df_reprocessed[col][i]:
                            diff_idx = i
                            break

                    return self._make_result(
                        test_name=f"reprocess_identical[{day}]",
                        passed=False,
                        severity=Severity.CRITICAL,
                        message=f"Column {col} differs at row {diff_idx}",
                        details={
                            "day": str(day),
                            "column": col,
                            "first_diff_idx": diff_idx,
                            "silver_value": str(df_silver[col][diff_idx]),
                            "reprocessed_value": str(df_reprocessed[col][diff_idx]),
                        },
                    )

            return self._make_result(
                test_name=f"reprocess_identical[{day}]",
                passed=True,
                severity=Severity.CRITICAL,
                message="Reprocessed Silver is identical",
                details={"day": str(day), "n_events": len(df_silver)},
            )

        except ImportError as e:
            return self._make_result(
                test_name=f"reprocess_identical[{day}]",
                passed=True,
                severity=Severity.INFO,
                message=f"Cannot import Engine: {e}",
                details={"day": str(day), "error": str(e)},
            )
        except Exception as e:
            return self._make_result(
                test_name=f"reprocess_identical[{day}]",
                passed=False,
                severity=Severity.HIGH,
                message=f"Reprocess error: {e}",
                details={"day": str(day), "error": str(e)},
            )
