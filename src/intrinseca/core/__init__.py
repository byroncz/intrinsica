"""
Núcleo de cálculo de alto rendimiento para Directional Change.

Este módulo contiene los algoritmos optimizados con Numba para detección
de eventos DC e indicadores derivados. No requiere dependencias gráficas.
"""

from intrinseca.core.event_detector import DCDetector, DCResult
from intrinseca.core.indicators import DCIndicators

__all__ = [
    "DCDetector",
    "DCResult",
    "DCIndicators",
]