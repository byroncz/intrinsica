"""Validadores del framework de calidad.

Cada validador hereda de BaseValidator e implementa validate_day().
"""

from .base import BaseValidator, ValidationResult, ValidatorReport

__all__ = [
    "BaseValidator",
    "ValidationResult",
    "ValidatorReport",
]
