"""
Módulo de visualización para análisis Directional Change (DC).

Este módulo es modular y admite visualizaciones estáticas e interactivas.
Requiere instalaciones opcionales:
    pip install intrinseca[plot]         # Para gráficos estáticos
    pip install intrinseca[interactive]  # Para dashboards interactivos
"""

# Importaciones de Gráficos Estáticos (Matplotlib/Plotly)
from intrinseca.visualization.static_plots import (
    plot_dc_events,
    plot_dc_summary,
    plot_coastline,
    plot_event_distribution,
)

# Importaciones de Gráficos Interactivos (Panel/HoloViews/Datashader)
from intrinseca.visualization.interactive import (
    create_dashboard_app,
    create_dual_axis_dashboard,
    serve_dashboard
)

# Configuración del módulo
from intrinseca.visualization.config import (
    MAX_WINDOW_HOURS,
    INITIAL_WINDOW_HOURS,
)

# Definición de la API pública del módulo
__all__ = [
    "plot_dc_events",
    "plot_dc_summary",
    "plot_coastline",
    "plot_event_distribution",
    "create_dashboard_app",
    "create_dual_axis_dashboard",
    "serve_dashboard",
    "MAX_WINDOW_HOURS",
    "INITIAL_WINDOW_HOURS",
]


def _check_matplotlib_available() -> bool:
    """Verifica si matplotlib está instalado para gráficos estáticos."""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def _check_plotly_available() -> bool:
    """Verifica si plotly está instalado para gráficos enriquecidos."""
    try:
        import plotly
        return True
    except ImportError:
        return False


def _check_interactive_available() -> bool:
    """Verifica si las dependencias interactivas están instaladas."""
    try:
        import panel
        import holoviews
        import datashader
        return True
    except ImportError:
        return False