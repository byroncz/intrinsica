"""
Intrinseca: Motor de Transformación Directional Change (DC)
============================================================

Librería de alto rendimiento para análisis de series temporales financieras
mediante el paradigma de Directional Change.

Uso básico (live-trading):
    >>> import intrinseca as dc
    >>> from intrinseca import DCDetector
    >>> detector = DCDetector(theta=0.01)
    >>> events = detector.detect(prices)

Con visualización (requiere: pip install intrinseca[plot]):
    >>> from intrinseca.visualization import plot_dc_events
    >>> plot_dc_events(prices, events)
"""

from intrinseca.core.event_detector import DCDetector, DCEvent, TrendState
from intrinseca.core.indicators import DCIndicators
from intrinseca.core.bridging import to_numpy, to_polars, extract_price_column

__version__ = "0.1.0"
__author__ = "Tu Nombre"

__all__ = [
    # Clases principales
    "DCDetector",
    "DCEvent",
    "TrendState",
    "DCIndicators",
    # Utilidades de bridging
    "to_numpy",
    "to_polars",
    "extract_price_column",
    # Metadata
    "__version__",
]


def get_version() -> str:
    """Retorna la versión actual de la librería."""
    return __version__


def has_visualization() -> bool:
    """Verifica si el módulo de visualización está disponible."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def has_interactive() -> bool:
    """Verifica si el módulo interactivo (Plotly) está disponible."""
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False