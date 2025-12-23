"""
Gráficos interactivos con Plotly para análisis DC.

Requiere: pip install intrinseca[interactive]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl
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


def create_interactive_chart_from_polars(
    df_ticks: pl.DataFrame,
    df_events: pl.DataFrame,
    title: str = "Análisis DC (Diagnóstico Full Stack)",
    height: int = 900,  # Aumentamos altura para acomodar dos paneles
    show_volume: bool = False
) -> "go.Figure":
    """
    Crea Dashboard de Diagnóstico DC con dos paneles sincronizados.
    
    Panel Superior (Diagnóstico de Estado):
    - Colorea cada tick según su Tendencia (Verde Claro/Rosa)
    - Resalta los Overshoots/Runs (Verde Oscuro/Rojo Oscuro)
    
    Panel Inferior (Eventos DC):
    - Visualización clásica de Eventos y Líneas de Tendencia.
    """
    go = _ensure_plotly()
    from plotly.subplots import make_subplots
    import numpy as np

    # Validaciones básicas
    if "time" not in df_ticks.columns or "time" not in df_events.columns:
        raise ValueError("Se requiere columna 'time' en ambos DataFrames")
    
    required_cols = ["dc_trend", "dc_run_flag"]
    for col in required_cols:
        if col not in df_ticks.columns:
            raise ValueError(f"El DataFrame de ticks debe tener la columna '{col}' para este diagnóstico.")

    # 1. Asegurar Orden Cronológico (Siempre)
    df_ticks = df_ticks.sort("time")
    df_events = df_events.sort("time")

    # 2. Configuración de Subplots (2 Filas)
    # Fila 1: Diagnóstico de Estado (Trend/Runs)
    # Fila 2: Eventos DC (La gráfica anterior)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.5, 0.5],
        subplot_titles=("Diagnóstico de Estado (Ticks)", "Eventos DC Confirmados")
    )

    # --- PREPARACIÓN DE DATOS ---
    times = df_ticks["time"].to_numpy()
    prices = df_ticks["price"].to_numpy()
    trends = df_ticks["dc_trend"].to_numpy()
    runs = df_ticks["dc_run_flag"].to_numpy()

    # =========================================================================
    # PANEL 1 (ARRIBA): DIAGNÓSTICO DE ESTADO (Colores Tick-a-Tick)
    # =========================================================================
    
    # Estrategia de Color:
    # 1. Base: Gris neutro
    # 2. Tendencia: Verde Claro (1) o Rosa (-1)
    # 3. Runs (Prioridad): Verde Oscuro (>0) o Rojo Oscuro (<0)
    
    # Usamos un array de strings para colores vectorizados (mucho más rápido)
    # Inicializamos con gris
    colors = np.full(len(times), "#bdc3c7", dtype=object) # Grey default
    
    # Aplicar Colores de Tendencia
    # Trend 1 -> Verde Claro (#a3e4d7 - Mint)
    colors[trends == 1] = "#90ee90" 
    # Trend -1 -> Rosa (#f5b7b1 - Light Red)
    colors[trends == -1] = "#ffb6c1"
    
    # Aplicar Colores de Runs (Sobrescriben Tendencia)
    # Run > 0 -> Verde Oscuro (#006400)
    colors[runs > 0] = "#006400"
    # Run < 0 -> Rojo Oscuro (#8b0000)
    colors[runs < 0] = "#8b0000"

    # Graficamos usando Scattergl (WebGL) para rendimiento con +500k puntos
    # Trace de Línea de base (Gris fina) para continuidad visual
    fig.add_trace(
        go.Scattergl(
            x=times, y=prices, mode="lines", 
            line=dict(color="#bdc3c7", width=0.5),
            hoverinfo="skip", name="Conectividad"
        ),
        row=1, col=1
    )
    
    # Trace de Puntos Coloreados
    fig.add_trace(
        go.Scattergl(
            x=times, y=prices, mode="markers",
            marker=dict(
                size=3, # Puntos pequeños pero visibles
                color=colors,
                line=dict(width=0)
            ),
            name="Estado del Motor",
            hovertemplate="<b>Tick Estado</b><br>Precio: %{y:.2f}<br>Trend: %{text}<extra></extra>",
            text=[f"T:{t} | R:{r}" for t, r in zip(trends, runs)] # Info extra en hover
        ),
        row=1, col=1
    )

    # =========================================================================
    # PANEL 2 (ABAJO): EVENTOS DC (Lógica Anterior)
    # =========================================================================
    
    min_visible_time = times[0]
    
    # Línea de precios limpia
    fig.add_trace(
        go.Scattergl( # Usamos GL también aquí para consistencia
            x=times, y=prices, mode="lines", name="Precio Base",
            line=dict(color="#3498db", width=1.5),
            hovertemplate="Fecha: %{x}<br>Precio: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    # Marcadores de Eventos
    for e_type, color, name, symbol in [
        (1, "#2ecc71", "Upturn (Evt)", "triangle-up"),
        (-1, "#e74c3c", "Downturn (Evt)", "triangle-down")
    ]:
        subset = df_events.filter(pl.col("type") == e_type)
        if not subset.is_empty():
            fig.add_trace(
                go.Scatter(
                    x=subset["time"].to_numpy(),
                    y=subset["price"].to_numpy(),
                    mode="markers", name=name,
                    marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color="white")),
                    hovertemplate=f"{name}<br>Fecha: %{{x}}<br>Precio: %{{y:.2f}}<extra></extra>"
                ),
                row=2, col=1
            )

    # Líneas de Tendencia (Con Clipping)
    if not df_events.is_empty() and "ext_time" in df_events.columns:
        for e_type, color in [(1, "#2ecc71"), (-1, "#e74c3c")]:
            subset = df_events.filter(
                (pl.col("type") == e_type) & 
                (pl.col("ext_time") >= min_visible_time)
            )
            
            if not subset.is_empty():
                n = len(subset)
                x_vals = np.empty(n * 3, dtype='datetime64[ns]')
                y_vals = np.empty(n * 3, dtype=np.float64)
                
                x_vals[0::3] = subset["ext_time"].to_numpy()
                x_vals[1::3] = subset["time"].to_numpy()
                x_vals[2::3] = np.datetime64('NaT') 
                
                y_vals[0::3] = subset["ext_price"].to_numpy()
                y_vals[1::3] = subset["price"].to_numpy()
                y_vals[2::3] = np.nan

                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_vals, mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip"
                    ),
                    row=2, col=1
                )

    # Layout Global
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height, 
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Ejes X Sincronizados
    fig.update_xaxes(title_text="Tiempo (UTC)", row=2, col=1)
    
    return fig


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
    
    if result.n_events > 0:
        # Eventos upturn
        upturn_mask = result.event_types == 1
        upturn_indices = result.event_indices[upturn_mask]
        upturn_prices = result.event_prices[upturn_mask]
        
        if len(upturn_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=upturn_indices,
                    y=upturn_prices,
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
        downturn_mask = result.event_types == -1
        downturn_indices = result.event_indices[downturn_mask]
        downturn_prices = result.event_prices[downturn_mask]
        
        if len(downturn_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=downturn_indices,
                    y=downturn_prices,
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
        # Aquí sí iteramos porque son líneas individuales (segmentos)
        # Plotly no soporta "segmentos" de forma eficiente como una sola traza fácilmente
        # sin trucos de NaNs. Iterar está bien si no son miles de eventos.
        # Si son muchos, mejor usar arrays intercalando NaNs para hacer una sola traza.
        
        if result.n_events < 2000: # Limite razonable para iterar
            for i in range(result.n_events):
                idx = result.event_indices[i]
                price = result.event_prices[i]
                etype = result.event_types[i]
                ext_idx = result.extreme_indices[i]
                ext_price = result.extreme_prices[i]
                
                color = "#2ecc71" if etype == 1 else "#e74c3c"
                fig.add_trace(
                    go.Scatter(
                        x=[ext_idx, idx],
                        y=[ext_price, price],
                        mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    row=1 if show_volume else None,
                    col=1 if show_volume else None
                )
        else:
             # Versión optimizada con NaNs para muchos eventos
             # x: [x1, x2, None, x3, x4, None, ...]
             # y: [y1, y2, None, y3, y4, None, ...]
             # Separamos upturns y downturns para color
             
             # Upturn lines
             up_mask = result.event_types == 1
             if np.any(up_mask):
                 up_ev_idx = result.event_indices[up_mask]
                 up_ev_pr = result.event_prices[up_mask]
                 up_ex_idx = result.extreme_indices[up_mask]
                 up_ex_pr = result.extreme_prices[up_mask]
                 
                 # Intercalar
                 n_up = len(up_ev_idx)
                 x_vals = np.empty(n_up * 3, dtype=float)
                 x_vals[0::3] = up_ex_idx
                 x_vals[1::3] = up_ev_idx
                 x_vals[2::3] = np.nan
                 
                 y_vals = np.empty(n_up * 3, dtype=float)
                 y_vals[0::3] = up_ex_pr
                 y_vals[1::3] = up_ev_pr
                 y_vals[2::3] = np.nan
                 
                 fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        line=dict(color="#2ecc71", width=1, dash="dot"),
                        name="Trend Lines (Up)",
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    row=1 if show_volume else None,
                    col=1 if show_volume else None
                )
             
             # Downturn lines
             dn_mask = result.event_types == -1
             if np.any(dn_mask):
                 dn_ev_idx = result.event_indices[dn_mask]
                 dn_ev_pr = result.event_prices[dn_mask]
                 dn_ex_idx = result.extreme_indices[dn_mask]
                 dn_ex_pr = result.extreme_prices[dn_mask]
                 
                 n_dn = len(dn_ev_idx)
                 x_vals = np.empty(n_dn * 3, dtype=float)
                 x_vals[0::3] = dn_ex_idx
                 x_vals[1::3] = dn_ev_idx
                 x_vals[2::3] = np.nan
                 
                 y_vals = np.empty(n_dn * 3, dtype=float)
                 y_vals[0::3] = dn_ex_pr
                 y_vals[1::3] = dn_ev_pr
                 y_vals[2::3] = np.nan
                 
                 fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        line=dict(color="#e74c3c", width=1, dash="dot"),
                        name="Trend Lines (Down)",
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    row=1 if show_volume else None,
                    col=1 if show_volume else None
                )
    
    # Volumen
    if show_volume and volume is not None:
        # Colorear volumen según tendencia actual
        # trends array (tick-by-tick)
        if len(result.trend_states) == len(volume):
            colors = ["#2ecc71" if t == 1 else "#e74c3c" if t == -1 else "#95a5a6" 
                      for t in result.trend_states]
        else:
            colors = "#95a5a6"

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