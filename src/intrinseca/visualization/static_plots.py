"""
Gráficos estáticos con Matplotlib/Seaborn para análisis DC.

Estos gráficos están diseñados para documentación, reportes
y análisis exploratorio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from intrinseca.core.event_detector import DCResult
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _ensure_matplotlib():
    """Verifica que matplotlib esté disponible."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib no está instalado. "
            "Instala con: pip install intrinseca[plot]"
        )


def plot_dc_events(
    prices: NDArray[np.float64],
    result: "DCResult",
    title: str = "Análisis Directional Change",
    figsize: tuple[int, int] = (14, 7),
    show_extremes: bool = True,
    show_trend_background: bool = True,
    upturn_color: str = "#2ecc71",
    downturn_color: str = "#e74c3c",
    price_color: str = "#3498db",
    ax: Optional["Axes"] = None
) -> "Figure":
    """
    Visualiza serie de precios con eventos DC superpuestos.
    
    Args:
        prices: Array de precios.
        result: Resultado de DCDetector.detect().
        title: Título del gráfico.
        figsize: Tamaño de la figura (ancho, alto).
        show_extremes: Mostrar puntos de máximos/mínimos locales.
        show_trend_background: Colorear fondo según tendencia.
        upturn_color: Color para eventos upturn.
        downturn_color: Color para eventos downturn.
        price_color: Color de la línea de precios.
        ax: Axes existente (opcional).
    
    Returns:
        Figura de matplotlib.
    
    Example:
        >>> from intrinseca import DCDetector
        >>> from intrinseca.visualization import plot_dc_events
        >>> detector = DCDetector(theta=0.02)
        >>> result = detector.detect(prices)
        >>> fig = plot_dc_events(prices, result)
        >>> fig.savefig("dc_analysis.png", dpi=150)
    """
    plt = _ensure_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    x = np.arange(len(prices))
    
    # Fondo coloreado por tendencia
    if show_trend_background and len(result.trends) > 0:
        trends = result.trends
        for i in range(1, len(trends)):
            if trends[i] == 1:  # Uptrend
                ax.axvspan(i-1, i, alpha=0.1, color=upturn_color, linewidth=0)
            elif trends[i] == -1:  # Downtrend
                ax.axvspan(i-1, i, alpha=0.1, color=downturn_color, linewidth=0)
    
    # Línea de precios
    ax.plot(x, prices, color=price_color, linewidth=1.2, label="Precio", zorder=2)
    
    # Extremos locales
    if show_extremes:
        if len(result.extremes_high) > 0:
            ax.scatter(
                result.extremes_high,
                prices[result.extremes_high],
                color=downturn_color,
                marker="v",
                s=60,
                label="Máximo local",
                zorder=3,
                alpha=0.7
            )
        if len(result.extremes_low) > 0:
            ax.scatter(
                result.extremes_low,
                prices[result.extremes_low],
                color=upturn_color,
                marker="^",
                s=60,
                label="Mínimo local",
                zorder=3,
                alpha=0.7
            )
    
    # Eventos DC
    for event in result.events:
        color = upturn_color if event.event_type == "upturn" else downturn_color
        ax.axvline(
            event.index,
            color=color,
            linestyle="--",
            alpha=0.6,
            linewidth=1
        )
        # Línea conectando extremo con evento
        ax.plot(
            [event.extreme_index, event.index],
            [event.extreme_price, event.price],
            color=color,
            linewidth=1.5,
            alpha=0.8,
            zorder=4
        )
    
    ax.set_xlabel("Observación", fontsize=11)
    ax.set_ylabel("Precio", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Anotación de estadísticas
    stats_text = (
        f"Eventos: {result.n_events}\n"
        f"Upturns: {result.n_upturns}\n"
        f"Downturns: {result.n_downturns}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    return fig


def plot_dc_summary(
    prices: NDArray[np.float64],
    result: "DCResult",
    title: str = "Resumen DC",
    figsize: tuple[int, int] = (16, 10)
) -> "Figure":
    """
    Panel resumen con múltiples visualizaciones DC.
    
    Incluye:
    - Serie de precios con eventos
    - Distribución de retornos DC
    - Distribución de duraciones
    - Evolución del overshoot
    
    Args:
        prices: Array de precios.
        result: Resultado DC.
        title: Título general.
        figsize: Tamaño de figura.
    
    Returns:
        Figura con múltiples subplots.
    """
    plt = _ensure_matplotlib()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Panel 1: Precios con eventos
    plot_dc_events(prices, result, title="Precios y Eventos DC", ax=axes[0, 0])
    
    # Panel 2: Distribución de retornos
    if result.n_events > 0:
        returns = [e.dc_return for e in result.events]
        upturn_returns = [e.dc_return for e in result.events if e.event_type == "upturn"]
        downturn_returns = [e.dc_return for e in result.events if e.event_type == "downturn"]
        
        ax = axes[0, 1]
        bins = np.linspace(min(returns), max(returns), 25)
        ax.hist(upturn_returns, bins=bins, alpha=0.7, label="Upturn", color="#2ecc71")
        ax.hist(downturn_returns, bins=bins, alpha=0.7, label="Downturn", color="#e74c3c")
        ax.set_xlabel("Retorno DC")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Retornos DC")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Panel 3: Distribución de duraciones
    if result.n_events > 0:
        durations = [e.dc_duration for e in result.events]
        
        ax = axes[1, 0]
        ax.hist(durations, bins=30, color="#9b59b6", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(durations), color="red", linestyle="--", label=f"Media: {np.mean(durations):.1f}")
        ax.set_xlabel("Duración (observaciones)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Duraciones DC")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Panel 4: Overshoot a lo largo del tiempo
    if len(result.overshoot_values) > 0:
        ax = axes[1, 1]
        event_indices = [e.index for e in result.events]
        ax.bar(range(len(result.overshoot_values)), result.overshoot_values, 
               color="#f39c12", alpha=0.7)
        ax.set_xlabel("Número de evento")
        ax.set_ylabel("Overshoot")
        ax.set_title("Overshoot por Evento")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_coastline(
    coastline: dict[float, float],
    title: str = "Coastline del Mercado",
    figsize: tuple[int, int] = (10, 6),
    log_scale: bool = True
) -> "Figure":
    """
    Visualiza la coastline (TMV vs theta) del mercado.
    
    La coastline revela la estructura fractal: mercados con
    comportamiento fractal muestran una relación lineal en
    escala log-log.
    
    Args:
        coastline: Diccionario {theta: tmv} de DCIndicators.compute_coastline().
        title: Título del gráfico.
        figsize: Tamaño de figura.
        log_scale: Usar escala logarítmica en ambos ejes.
    
    Returns:
        Figura matplotlib.
    """
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    thetas = sorted(coastline.keys())
    tmvs = [coastline[t] for t in thetas]
    
    ax.plot(thetas, tmvs, "o-", color="#3498db", markersize=8, linewidth=2)
    
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    ax.set_xlabel("Theta (umbral DC)", fontsize=11)
    ax.set_ylabel("TMV (Total Movement Variable)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    
    # Ajuste lineal en log-log para estimar dimensión fractal
    if log_scale and len(thetas) > 2:
        log_thetas = np.log(thetas)
        log_tmvs = np.log(tmvs)
        slope, intercept = np.polyfit(log_thetas, log_tmvs, 1)
        
        fit_line = np.exp(intercept) * np.array(thetas) ** slope
        ax.plot(thetas, fit_line, "--", color="#e74c3c", 
                label=f"Ajuste: pendiente = {slope:.2f}")
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_event_distribution(
    result: "DCResult",
    title: str = "Distribución Temporal de Eventos",
    figsize: tuple[int, int] = (12, 5)
) -> "Figure":
    """
    Visualiza la distribución temporal de eventos DC.
    
    Muestra dónde ocurren los eventos a lo largo de la serie,
    útil para identificar clusters de actividad.
    
    Args:
        result: Resultado DC.
        title: Título.
        figsize: Tamaño de figura.
    
    Returns:
        Figura matplotlib.
    """
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    upturn_indices = [e.index for e in result.events if e.event_type == "upturn"]
    downturn_indices = [e.index for e in result.events if e.event_type == "downturn"]
    
    # Crear "rug plot" de eventos
    ax.eventplot([upturn_indices], colors=["#2ecc71"], lineoffsets=0.5, 
                 linelengths=0.4, label="Upturn")
    ax.eventplot([downturn_indices], colors=["#e74c3c"], lineoffsets=-0.5, 
                 linelengths=0.4, label="Downturn")
    
    ax.set_yticks([0.5, -0.5])
    ax.set_yticklabels(["Upturn", "Downturn"])
    ax.set_xlabel("Índice de observación", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    return fig