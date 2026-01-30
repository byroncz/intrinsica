"""
Intrinseca: Biblioteca para análisis Directional Change (DC).

Submódulos disponibles:
    - intrinseca.core: Motor de detección de eventos DC
    - intrinseca.indicators: Sistema de indicadores sobre datos Silver
    - intrinseca.visualization: Dashboards interactivos y gráficos estáticos

Uso:
    # Para indicadores (sin cargar visualización)
    from intrinseca.indicators import compute
    
    # Para visualización (carga Panel/HoloViz)
    from intrinseca.visualization import create_dual_axis_dashboard
"""

# No importar submódulos automáticamente para evitar cargar dependencias pesadas
# El usuario importa explícitamente lo que necesita:
#   from intrinseca.indicators import compute
#   from intrinseca.visualization import create_dual_axis_dashboard

__version__ = "0.1.0"

__all__ = [
    "core",
    "indicators", 
    "visualization",
]