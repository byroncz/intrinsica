"""
Módulo de visualización para análisis Directional Change.

Este módulo es OPCIONAL. Requiere instalación con:
    pip install intrinseca[plot]

Para visualización interactiva:
    pip install intrinseca[interactive]

Funciones disponibles:
    - plot_dc_events: Gráfico estático de precios con eventos DC
    - plot_dc_summary: Panel resumen con múltiples métricas
    - create_interactive_chart: Gráfico interactivo Plotly
"""

from intrinseca.visualization.static_plots import (
    plot_dc_events,
    plot_dc_summary,
    plot_coastline,
    plot_event_distribution,
)

__all__ = [
    "plot_dc_events",
    "plot_dc_summary",
    "plot_coastline",
    "plot_event_distribution",
]


def _check_matplotlib_available() -> bool:
    """Verifica si matplotlib está instalado."""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def _check_plotly_available() -> bool:
    """Verifica si plotly está instalado."""
    try:
        import plotly
        return True
    except ImportError:
        return False