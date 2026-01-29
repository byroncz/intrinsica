"""
Núcleo de cálculo de alto rendimiento para Directional Change.

Este módulo contiene los algoritmos optimizados con Numba para detección
de eventos DC. Arquitectura Silver Layer para materialización de eventos anidados.

Componentes:
    - Engine: Motor de transformación Bronze → Silver
    - DayResult: Resultado estructurado del procesamiento de un día
    - EngineStats: Estadísticas de rendimiento del Engine
    - kernel: Kernel Numba JIT para segmentación de eventos
    - DCState: Estado persistente para stitching entre días
    - Convergence: Análisis de convergencia entre ejecuciones
"""

from intrinseca.core.engine import Engine, DayResult, EngineStats, EngineConfig
from intrinseca.core.kernel import (
    segment_events_kernel,
    warmup_kernel,
    verify_nopython_mode,
    estimate_memory_usage,
    benchmark_kernel,
)
from intrinseca.core.state import DCState
from intrinseca.core.convergence import (
    ConvergenceResult,
    ConvergenceReport,
    compare_dc_events,
)

__all__ = [
    # Silver Layer Engine
    "Engine",
    "DayResult",
    "EngineStats",
    "EngineConfig",
    # Kernel
    "segment_events_kernel",
    "warmup_kernel",
    "verify_nopython_mode",
    "estimate_memory_usage",
    "benchmark_kernel",
    # Estado
    "DCState",
    # Convergencia
    "ConvergenceResult",
    "ConvergenceReport",
    "compare_dc_events",
]