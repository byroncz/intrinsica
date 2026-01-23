"""
Núcleo de cálculo de alto rendimiento para Directional Change.

Este módulo contiene los algoritmos optimizados con Numba para detección
de eventos DC e indicadores derivados. No requiere dependencias gráficas.

Incluye también el motor Silver Layer para materialización de eventos anidados.
"""

from intrinseca.core.event_detector import DCDetector, DCResult
from intrinseca.core.indicators import DCIndicators
from intrinseca.core.engine import Engine
from intrinseca.core.state import DCState

__all__ = [
    # Bronze Layer (análisis tick-a-tick)
    "DCDetector",
    "DCResult",
    "DCIndicators",
    # Silver Layer (eventos anidados)
    "Engine",
    "DCState",
]