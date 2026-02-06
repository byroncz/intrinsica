"""Clase base para todos los validadores.

Todos los validadores DEBEN heredar de BaseValidator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from ..config import CONFIG, Severity


@dataclass
class ValidationResult:
    """Resultado de una validación individual."""

    test_name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    affected_events: list[int] = field(default_factory=list)

    @property
    def is_assertion(self) -> bool:
        """True si el test es una aserción (puede fallar)."""
        return self.severity != Severity.INFO

    @property
    def is_statistic(self) -> bool:
        """True si el test es una observación estadística (INFO)."""
        return self.severity == Severity.INFO

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "affected_events": self.affected_events[:100],  # Limitar para reporte
            "n_affected": len(self.affected_events),
            "is_assertion": self.is_assertion,
        }


@dataclass
class ValidatorReport:
    """Reporte agregado de un validador."""

    validator_name: str
    level: int  # 1-9
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True si todas las aserciones pasaron (excluye INFO)."""
        return all(r.passed for r in self.results if r.is_assertion)

    @property
    def n_passed(self) -> int:
        """Cantidad de aserciones que pasaron (excluye INFO)."""
        return sum(1 for r in self.results if r.is_assertion and r.passed)

    @property
    def n_failed(self) -> int:
        """Cantidad de aserciones que fallaron (excluye INFO)."""
        return sum(1 for r in self.results if r.is_assertion and not r.passed)

    @property
    def n_statistics(self) -> int:
        """Cantidad de observaciones estadísticas (INFO)."""
        return sum(1 for r in self.results if r.is_statistic)

    @property
    def statistics(self) -> list[ValidationResult]:
        """Lista de observaciones estadísticas."""
        return [r for r in self.results if r.is_statistic]

    @property
    def has_critical_failures(self) -> bool:
        return any(not r.passed and r.severity == Severity.CRITICAL for r in self.results)

    def to_dict(self) -> dict:
        return {
            "validator_name": self.validator_name,
            "level": self.level,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "has_critical_failures": self.has_critical_failures,
            "results": [r.to_dict() for r in self.results],
        }


class BaseValidator(ABC):
    """Clase base abstracta para validadores.

    Todos los validadores deben:
    1. Heredar de esta clase
    2. Implementar `validate_day()`
    3. Definir `name` y `level`
    """

    name: str = "base"
    level: int = 0

    def __init__(self, config: type = CONFIG):
        """Inicializa el validador con configuración."""
        self.config = config

    @abstractmethod
    def validate_day(self, df: pl.DataFrame, day: date, theta: float) -> list[ValidationResult]:
        """Valida un día de datos Silver.

        Args:
            df: DataFrame Silver del día
            day: Fecha del día
            theta: Umbral DC usado

        Returns:
            Lista de resultados de validación
        """
        pass

    def validate_month(self, data: dict[date, pl.DataFrame], theta: float) -> ValidatorReport:
        """Valida un mes completo de datos.

        Args:
            data: Dict {date: DataFrame} con datos del mes
            theta: Umbral DC

        Returns:
            Reporte agregado del validador
        """
        all_results = []

        for day, df in sorted(data.items()):
            day_results = self.validate_day(df, day, theta)
            all_results.extend(day_results)

        return ValidatorReport(
            validator_name=self.name,
            level=self.level,
            results=all_results,
        )

    def _make_result(
        self,
        test_name: str,
        passed: bool,
        severity: Severity,
        message: str,
        details: dict | None = None,
        affected_events: list[int] | None = None,
    ) -> ValidationResult:
        """Helper para crear resultados de validación."""
        return ValidationResult(
            test_name=f"{self.name}.{test_name}",
            passed=passed,
            severity=severity,
            message=message,
            details=details or {},
            affected_events=affected_events or [],
        )
