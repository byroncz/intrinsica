"""
Gráficos interactivos con Plotly para análisis DC.

Requiere: pip install intrinseca[interactive]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from intrinseca.core.event_detector import DCResult
    import plotly.graph_objects as go


def _ensure_plotly():
    """Verifica que plotly esté disponible."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "plotly no está instalado. "
            "Instala con: pip install intrinseca[interactive]"
        )


def create_interactive_chart(
    prices: NDArray[np.float64],
    result: "DCResult",
    title: str = "Análisis DC Interactivo",
    height: int = 600,
    show_volume: bool = False,
    volume: Optional[NDArray] = None
) -> "go.Figure":
    """
    Crea gráfico interactivo de análisis DC con Plotly.
    
    Permite zoom, pan, y hover con información detallada
    de cada evento.
    
    Args:
        prices: Array de precios.
        result: Resultado DC.
        title: Título del gráfico.
        height: Altura en píxeles.
        show_volume: Mostrar volumen en panel secundario.
        volume: Array de volumen (requerido si show_volume=True).
    
    Returns:
        Figura Plotly.
    
    Example:
        >>> from intrinseca.visualization.interactive import create_interactive_chart
        >>> fig = create_interactive_chart(prices, result)
        >>> fig.show()  # Abre en navegador
        >>> fig.write_html("dc_interactive.html")  # Guarda como HTML
    """
    go = _ensure_plotly()
    from plotly.subplots import make_subplots
    
    # Configurar subplots si hay volumen
    if show_volume and volume is not None:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    x = np.arange(len(prices))
    
    # Línea de precios
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prices,
            mode="lines",
            name="Precio",
            line=dict(color="#3498db", width=1.5),
            hovertemplate="Índice: %{x}<br>Precio: %{y:.4f}<extra></extra>"
        ),
        row=1 if show_volume else None,
        col=1 if show_volume else None
    )
    
    # Eventos upturn
    upturn_events = [e for e in result.events if e.event_type == "upturn"]
    if upturn_events:
        fig.add_trace(
            go.Scatter(
                x=[e.index for e in upturn_events],
                y=[e.price for e in upturn_events],
                mode="markers",
                name="Upturn",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#2ecc71",
                    line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    "Upturn<br>"
                    "Índice: %{x}<br>"
                    "Precio: %{y:.4f}<br>"
                    "<extra></extra>"
                )
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    
    # Eventos downturn
    downturn_events = [e for e in result.events if e.event_type == "downturn"]
    if downturn_events:
        fig.add_trace(
            go.Scatter(
                x=[e.index for e in downturn_events],
                y=[e.price for e in downturn_events],
                mode="markers",
                name="Downturn",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#e74c3c",
                    line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    "Downturn<br>"
                    "Índice: %{x}<br>"
                    "Precio: %{y:.4f}<br>"
                    "<extra></extra>"
                )
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    
    # Líneas de tendencia DC (conectando extremo a evento)
    for event in result.events:
        color = "#2ecc71" if event.event_type == "upturn" else "#e74c3c"
        fig.add_trace(
            go.Scatter(
                x=[event.extreme_index, event.index],
                y=[event.extreme_price, event.price],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    
    # Volumen
    if show_volume and volume is not None:
        colors = ["#2ecc71" if result.trends[i] == 1 else "#e74c3c" 
                  for i in range(len(volume))]
        fig.add_trace(
            go.Bar(
                x=x,
                y=volume,
                name="Volumen",
                marker_color=colors,
                opacity=0.5
            ),
            row=2,
            col=1
        )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height,
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis_title="Observación",
        yaxis_title="Precio"
    )
    
    # Botones de zoom
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=100, label="100", step="all"),
                dict(count=500, label="500", step="all"),
                dict(count=1000, label="1K", step="all"),
                dict(step="all", label="Todo")
            ])
        )
    )
    
    return fig


def create_metrics_dashboard(
    result: "DCResult",
    rolling_metrics: dict,
    title: str = "Dashboard de Métricas DC"
) -> "go.Figure":
    """
    Dashboard interactivo con métricas rolling.
    
    Args:
        result: Resultado DC.
        rolling_metrics: Dict de DCIndicators.compute_rolling_metrics().
        title: Título.
    
    Returns:
        Figura Plotly con múltiples paneles.
    """
    go = _ensure_plotly()
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("TMV Rolling", "Duración Promedio Rolling", "Overshoot Rolling")
    )
    
    x = np.arange(len(rolling_metrics.get("rolling_tmv", [])))
    
    # TMV
    if "rolling_tmv" in rolling_metrics:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_metrics["rolling_tmv"],
                mode="lines",
                name="TMV",
                line=dict(color="#3498db")
            ),
            row=1, col=1
        )
    
    # Duración
    if "rolling_duration" in rolling_metrics:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_metrics["rolling_duration"],
                mode="lines",
                name="Duración",
                line=dict(color="#9b59b6")
            ),
            row=2, col=1
        )
    
    # Overshoot
    if "rolling_overshoot" in rolling_metrics:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_metrics["rolling_overshoot"],
                mode="lines",
                name="Overshoot",
                line=dict(color="#f39c12")
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        title=title,
        height=700,
        showlegend=True
    )
    
    return fig