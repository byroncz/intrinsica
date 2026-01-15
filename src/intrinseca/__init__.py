"""
Módulo de visualización para análisis Directional Change (DC).

Este módulo implementa una arquitectura modular para visualización de alta frecuencia,
separando los gráficos estáticos de los dashboards interactivos dinámicos.

Componentes principales:
    - Dashboards Interactivos (Panel/HoloViz): Dual-Axis y Estándar.
    - Gráficos Estáticos (Matplotlib/Plotly): Resúmenes y Distribuciones.
"""

# 1. Exportación de Gráficos Estáticos (Capa de Reporte)
from intrinseca.visualization.static_plots import (
    plot_dc_events,
    plot_dc_summary,
    plot_coastline,
    plot_event_distribution,
)

# 2. Exportación de Gráficos Interactivos (Capa de Exploración)
from intrinseca.visualization.interactive import (
    create_dashboard_app,
    create_dual_axis_dashboard,  # Nuevo: Sistema de sincronización n <-> t
    serve_dashboard
)

# 3. API Pública del Módulo
__all__ = [
    # Dashboards
    "create_dashboard_app",
    "create_dual_axis_dashboard",
    "serve_dashboard",
    
    # Estáticos
    "plot_dc_events",
    "plot_dc_summary",
    "plot_coastline",
    "plot_event_distribution",
]

# --- Helpers de Verificación de Entorno ---

def _check_interactive_dependencies():
    """Verifica si el stack de HoloViz está instalado correctamente."""
    try:
        import panel as pn
        import holoviews as hv
        import datashader as ds
        return True
    except ImportError as e:
        print(f"⚠️ Faltan dependencias interactivas: {e}")
        print("Instale con: pip install intrinseca[interactive]")
        return False

def _check_static_dependencies():
    """Verifica dependencias para gráficos estáticos."""
    try:
        import matplotlib
        import plotly
        return True
    except ImportError:
        print("⚠️ Faltan dependencias estáticas. Instale con: pip install intrinseca[plot]")
        return False