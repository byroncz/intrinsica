"""
Núcleo de cálculo de alto rendimiento para Directional Change.

Este módulo contiene los algoritmos optimizados con Numba para detección
de eventos DC. Arquitectura Silver Layer para materialización de eventos anidados.

Componentes:
    - Engine: Motor de transformación Bronze → Silver
    - kernel: Kernel Numba JIT para segmentación de eventos
    - DCState: Estado persistente para stitching entre días
    - Convergence: Análisis de convergencia entre ejecuciones
"""

from intrinseca.core.engine import Engine
from intrinseca.core.kernel import segment_events_kernel, warmup_kernel
from intrinseca.core.state import DCState
from intrinseca.core.convergence import (
    ConvergenceResult,
    ConvergenceReport,
    compare_dc_events,
)

__all__ = [
    # Silver Layer Engine
    "Engine",
    "segment_events_kernel",
    "warmup_kernel",
    "DCState",
    # Convergencia
    "ConvergenceResult",
    "ConvergenceReport",
    "compare_dc_events",
]