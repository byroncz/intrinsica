"""
Núcleo de cálculo de alto rendimiento para Directional Change.

Este módulo contiene los algoritmos optimizados con Numba para detección
de eventos DC e indicadores derivados. No requiere dependencias gráficas.
"""

from intrinseca.core.event_detector import DCDetector, DCEvent, TrendState
from intrinseca.core.indicators import DCIndicators
from intrinseca.core.bridging import to_numpy, to_polars, extract_price_column

__all__ = [
    "DCDetector",
    "DCEvent",
    "TrendState",
    "DCIndicators",
    "to_numpy",
    "to_polars",
    "extract_price_column",
]