"""Silver Data Quality Framework.

Framework de validación exhaustiva para datos Silver del motor Intrinseca.
Ejecutar con: python -m tests.quality --year YYYY --month MM
"""

from .config import CONFIG, QualityConfig, Severity, ValidationLevel
from .runner import run_quality_check

__all__ = [
    "CONFIG",
    "QualityConfig",
    "Severity",
    "ValidationLevel",
    "run_quality_check",
]
