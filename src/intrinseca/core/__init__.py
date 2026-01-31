"""Núcleo de cálculo de alto rendimiento para Directional Change.

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

from intrinseca.core.convergence import (
    DEFAULT_TOLERANCE_NS,
    MAX_DISCREPANCY_DETAILS,
    ConvergenceReport,
    ConvergenceResult,
    DiscrepancyDetail,
    compare_dc_events,
    compare_reports,
    load_report,
)
from intrinseca.core.engine import DayResult, Engine, EngineConfig, EngineStats
from intrinseca.core.kernel import (
    benchmark_kernel,
    estimate_memory_usage,
    segment_events_kernel,
    verify_nopython_mode,
    warmup_kernel,
)
from intrinseca.core.state import (
    MAX_LOOKBACK_DAYS,
    DCState,
    build_state_path,
    cleanup_old_states,
    create_empty_state,
    find_previous_state,
    format_theta,
    get_state_stats,
    list_available_states,
    load_state,
    save_state,
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
    "format_theta",
    "build_state_path",
    "save_state",
    "load_state",
    "find_previous_state",
    "create_empty_state",
    "list_available_states",
    "get_state_stats",
    "cleanup_old_states",
    "MAX_LOOKBACK_DAYS",
    # Convergencia
    "ConvergenceResult",
    "ConvergenceReport",
    "DiscrepancyDetail",
    "compare_dc_events",
    "load_report",
    "compare_reports",
    "MAX_DISCREPANCY_DETAILS",
    "DEFAULT_TOLERANCE_NS",
]
