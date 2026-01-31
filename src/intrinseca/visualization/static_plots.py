"""Gráficos estáticos con Matplotlib/Seaborn para análisis DC.

Estos gráficos están diseñados para documentación, reportes
y análisis exploratorio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import (
    STATIC_COLOR_DOWNTURN,
    STATIC_COLOR_DURATION,
    STATIC_COLOR_OVERSHOOT,
    STATIC_COLOR_PRICE,
    STATIC_COLOR_UPTURN,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from intrinseca.core.event_detector import DCResult


def _ensure_matplotlib():
    """Verifica que matplotlib esté disponible."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as err:
        raise ImportError(
            "matplotlib no está instalado. " "Instala con: pip install intrinseca[plot]"
        ) from err


def plot_dc_events(
    prices: NDArray[np.float64],
    result: DCResult,
    title: str = "Análisis Directional Change",
    figsize: tuple[int, int] = (14, 7),
    show_extremes: bool = True,
    show_trend_background: bool = True,
    upturn_color: str = STATIC_COLOR_UPTURN,
    downturn_color: str = STATIC_COLOR_DOWNTURN,
    price_color: str = STATIC_COLOR_PRICE,
    ax: Axes | None = None,
) -> Figure:
    """Visualiza serie de precios con eventos DC superpuestos.

    Args:
    ----
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
    -------
        Figura de matplotlib.

    Example:
    -------
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
    # trend_states: 1 (Alza), -1 (Baja), 0 (Indefinido)
    if show_trend_background and len(result.trend_states) > 0:
        trends = result.trend_states
        # Identificar cambios de tendencia para pintar bloques
        # Optimizacion: no pintar iterando 1 a 1 si es muy grande.
        # Pero para plots estáticos, iterar cambios es aceptable.

        # Encontrar donde cambia el estado
        diffs = np.diff(trends)
        changes = np.where(diffs != 0)[0] + 1

        # Agregamos inicio y fin
        boundaries = [0] + list(changes) + [len(trends)]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            trend_val = trends[start]

            if trend_val == 1:
                ax.axvspan(start, end, alpha=0.1, color=upturn_color, linewidth=0)
            elif trend_val == -1:
                ax.axvspan(start, end, alpha=0.1, color=downturn_color, linewidth=0)

    # Línea de precios
    ax.plot(x, prices, color=price_color, linewidth=1.2, label="Precio", zorder=2)

    # Extremos locales
    # Extremos están en result.extreme_indices asociadas a cada evento
    # Upturn (1) -> Evento confirma mínimo (Low). El extremo asociado es un Low.
    # Downturn (-1) -> Evento confirma máximo (High). El extremo asociado es un High.
    # filter extremes

    if show_extremes and result.n_events > 0:
        # Indices de eventos upturn (generados por un Low previo)
        upturn_mask = result.event_types == 1
        downturn_mask = result.event_types == -1

        # Lows (asociados a Upturns)
        low_indices = result.extreme_indices[upturn_mask]
        low_prices = result.extreme_prices[upturn_mask]

        # Highs (asociados a Downturns)
        high_indices = result.extreme_indices[downturn_mask]
        high_prices = result.extreme_prices[downturn_mask]

        if len(high_indices) > 0:
            ax.scatter(
                high_indices,
                high_prices,
                color=downturn_color,
                marker="v",
                s=60,
                label="Máximo local",
                zorder=3,
                alpha=0.7,
            )
        if len(low_indices) > 0:
            ax.scatter(
                low_indices,
                low_prices,
                color=upturn_color,
                marker="^",
                s=60,
                label="Mínimo local",
                zorder=3,
                alpha=0.7,
            )

    # Eventos DC
    if result.n_events > 0:
        for i in range(result.n_events):
            idx = result.event_indices[i]
            price = result.event_prices[i]
            etype = result.event_types[i]
            ext_idx = result.extreme_indices[i]
            ext_price = result.extreme_prices[i]

            color = upturn_color if etype == 1 else downturn_color

            # Linea vertical de evento
            ax.axvline(idx, color=color, linestyle="--", alpha=0.6, linewidth=1)
            # Línea conectando extremo con evento
            ax.plot(
                [ext_idx, idx], [ext_price, price], color=color, linewidth=1.5, alpha=0.8, zorder=4
            )

    ax.set_xlabel("Observación", fontsize=11)
    ax.set_ylabel("Precio", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Anotación de estadísticas
    n_events = result.n_events
    n_upturns = np.sum(result.event_types == 1)
    n_downturns = np.sum(result.event_types == -1)

    stats_text = f"Eventos: {n_events}\n" f"Upturns: {n_upturns}\n" f"Downturns: {n_downturns}"
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    return fig


def plot_dc_summary(
    prices: NDArray[np.float64],
    result: DCResult,
    title: str = "Resumen DC",
    figsize: tuple[int, int] = (16, 10),
) -> Figure:
    """Panel resumen con múltiples visualizaciones DC.

    Incluye:
    - Serie de precios con eventos
    - Distribución de retornos DC
    - Distribución de duraciones
    - Evolución del overshoot

    Args:
    ----
        prices: Array de precios.
        result: Resultado DC.
        title: Título general.
        figsize: Tamaño de figura.

    Returns:
    -------
        Figura con múltiples subplots.

    """
    plt = _ensure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Precios con eventos
    plot_dc_events(prices, result, title="Precios y Eventos DC", ax=axes[0, 0])

    # Panel 2: Distribución de retornos
    if result.n_events > 0:
        returns = result.returns_price
        upturn_returns = returns[result.event_types == 1]
        downturn_returns = returns[result.event_types == -1]

        ax = axes[0, 1]
        # Evitar bins vacíos si pocos datos
        bins = np.linspace(min(returns), max(returns), 25) if len(returns) > 1 else 10

        ax.hist(upturn_returns, bins=bins, alpha=0.7, label="Upturn", color=STATIC_COLOR_UPTURN)
        ax.hist(
            downturn_returns, bins=bins, alpha=0.7, label="Downturn", color=STATIC_COLOR_DOWNTURN
        )
        ax.set_xlabel("Retorno DC")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Retornos DC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: Distribución de duraciones
    if result.n_events > 0:
        durations = result.event_durations

        ax = axes[1, 0]
        ax.hist(durations, bins=30, color=STATIC_COLOR_DURATION, alpha=0.7, edgecolor="white")
        ax.axvline(
            np.mean(durations),
            color="red",
            linestyle="--",
            label=f"Media: {np.mean(durations):.1f}",
        )
        ax.set_xlabel("Duración (observaciones)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Duraciones DC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 4: Overshoot a lo largo del tiempo
    if len(result.overshoots) > 0:
        ax = axes[1, 1]
        ax.bar(
            range(len(result.overshoots)),
            result.overshoots,
            color=STATIC_COLOR_OVERSHOOT,
            alpha=0.7,
        )
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
    log_scale: bool = True,
) -> Figure:
    """Visualiza la coastline (TMV vs theta) del mercado.

    La coastline revela la estructura fractal: mercados con
    comportamiento fractal muestran una relación lineal en
    escala log-log.

    Args:
    ----
        coastline: Diccionario {theta: tmv} de DCIndicators.compute_coastline().
        title: Título del gráfico.
        figsize: Tamaño de figura.
        log_scale: Usar escala logarítmica en ambos ejes.

    Returns:
    -------
        Figura matplotlib.

    """
    plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    thetas = sorted(coastline.keys())
    tmvs = [coastline[t] for t in thetas]

    ax.plot(thetas, tmvs, "o-", color=STATIC_COLOR_PRICE, markersize=8, linewidth=2)

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
        ax.plot(
            thetas,
            fit_line,
            "--",
            color=STATIC_COLOR_DOWNTURN,
            label=f"Ajuste: pendiente = {slope:.2f}",
        )
        ax.legend()

    plt.tight_layout()
    return fig


def plot_event_distribution(
    result: DCResult,
    title: str = "Distribución Temporal de Eventos",
    figsize: tuple[int, int] = (12, 5),
) -> Figure:
    """Visualiza la distribución temporal de eventos DC.

    Muestra dónde ocurren los eventos a lo largo de la serie,
    útil para identificar clusters de actividad.

    Args:
    ----
        result: Resultado DC.
        title: Título.
        figsize: Tamaño de figura.

    Returns:
    -------
        Figura matplotlib.

    """
    plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    upturn_indices = result.event_indices[result.event_types == 1]
    downturn_indices = result.event_indices[result.event_types == -1]

    # Crear "rug plot" de eventos
    ax.eventplot(
        [upturn_indices],
        colors=[STATIC_COLOR_UPTURN],
        lineoffsets=0.5,
        linelengths=0.4,
        label="Upturn",
    )
    ax.eventplot(
        [downturn_indices],
        colors=[STATIC_COLOR_DOWNTURN],
        lineoffsets=-0.5,
        linelengths=0.4,
        label="Downturn",
    )

    ax.set_yticks([0.5, -0.5])
    ax.set_yticklabels(["Upturn", "Downturn"])
    ax.set_xlabel("Índice de observación", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig
