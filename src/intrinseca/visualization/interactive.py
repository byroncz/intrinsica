import panel as pn
import holoviews as hv
import polars as pl
from datetime import datetime, timezone, timedelta

from .config import (
    MAX_WINDOW_HOURS, INITIAL_WINDOW_HOURS, 
    MAIN_PANEL_HEIGHT, SECONDARY_PANEL_HEIGHT
)
from .core_plots import _build_hv_plot, _build_intrinsic_panel, _build_physical_panel
from .hooks import RangeUpdateStream, create_mouseup_sync_hook, _apply_x_zoom_hook
from .utils import prepare_dual_axis_data

# Inicialización de extensiones
pn.extension('mathjax') 
hv.extension('bokeh')


def create_dashboard_app(df_ticks, df_events, title="Análisis DC", theta=0.001, height=None):
    """
    Dashboard interactivo con tiempo físico y actualización por botón.
    
    Args:
        df_ticks: DataFrame Polars con ticks augmentados
        df_events: DataFrame Polars con eventos DC
        title: Título del dashboard
        theta: Umbral DC para mostrar en título
        height: Altura del panel (usa MAIN_PANEL_HEIGHT si no se especifica)
    """
    height = height or MAIN_PANEL_HEIGHT
    
    # Rangos para los widgets
    abs_min = df_ticks["time"].min().replace(tzinfo=None)
    abs_max = df_ticks["time"].max().replace(tzinfo=None)
    init_start = abs_max - timedelta(hours=INITIAL_WINDOW_HOURS)

    # Widgets
    start_p = pn.widgets.DatetimePicker(name="Inicio", value=init_start, start=abs_min, end=abs_max)
    end_p = pn.widgets.DatetimePicker(name="Fin", value=abs_max, start=abs_min, end=abs_max)
    update_btn = pn.widgets.Button(name="🚀 Actualizar", button_type="primary", width=150)

    def _update_view(clicks):
        s_utc = start_p.value.replace(tzinfo=timezone.utc)
        e_utc = end_p.value.replace(tzinfo=timezone.utc)
        
        # Validar ventana máxima
        window_hours = (e_utc - s_utc).total_seconds() / 3600
        if window_hours > MAX_WINDOW_HOURS:
            return pn.pane.Markdown(
                f"### ⚠️ Ventana muy amplia ({window_hours:.0f}h). Máximo: {MAX_WINDOW_HOURS}h"
            )
        
        df_tk_win = df_ticks.filter(pl.col("time").is_between(s_utc, e_utc))
        df_ev_win = df_events.filter(pl.col("time").is_between(s_utc, e_utc))

        if df_tk_win.is_empty():
            return pn.pane.Markdown("### ⚠️ Sin datos en el rango.")

        return _build_hv_plot(df_tk_win, df_ev_win, title, theta, height)

    interactive_plot = pn.bind(_update_view, update_btn)

    return pn.Column(
        pn.pane.Markdown(f"### {title}"), 
        pn.Row(
            start_p, end_p, 
            pn.Column(pn.Spacer(height=15), update_btn),
            styles={"background": "#f8f9fa", "padding": "10px"}
        ),
        pn.panel(interactive_plot, loading_indicator=True),
        sizing_mode="stretch_both"
    )


def serve_dashboard(app, port=5006):
    pn.serve(app, port=port, threaded=True)


def create_dual_axis_dashboard(df_ticks, df_events, title="Dual-Axis DC"):
    """
    Dashboard sincronizado de doble eje (intrínseco + físico).
    
    Panel A (superior): Tiempo intrínseco basado en índice de eventos
    Panel B (inferior): Tiempo físico, actualizado en mouseup del Panel A
    
    Ambos paneles se re-renderizan al cambiar el rango visible para auto-ajustar Y.
    
    Args:
        df_ticks: DataFrame Polars con ticks augmentados
        df_events: DataFrame Polars con eventos DC
        title: Título del dashboard
    """
    # 1. Límites temporales del dataset
    abs_max_t = df_ticks["time"].max()
    abs_min_t = df_ticks["time"].min()
    init_start = abs_max_t - timedelta(hours=INITIAL_WINDOW_HOURS)
    
    # 2. Preparar datos de eventos (ligero: ~12K registros con índice secuencial)
    pdf_segments = prepare_dual_axis_data(df_events)
    total_events = len(pdf_segments)
    
    # 3. Calcular rango inicial para ambos paneles
    mask = (pdf_segments['time'] >= init_start) & (pdf_segments['time'] <= abs_max_t)
    init_segment_data = pdf_segments[mask]
    
    if not init_segment_data.empty:
        init_n_min = int(init_segment_data['x0'].min())
        init_n_max = int(init_segment_data['x0'].max())
    else:
        init_n_max = total_events - 1
        init_n_min = max(0, init_n_max - 100)
    
    # 4. Contenedores Panel para ambos gráficos (permiten actualización dinámica)
    from .core_plots import _build_intrinsic_panel, _build_physical_panel
    
    # Renderizar vistas iniciales
    init_intrinsic = _render_intrinsic_filtered(pdf_segments, init_n_min, init_n_max)
    intrinsic_pane = pn.pane.HoloViews(init_intrinsic, sizing_mode="stretch_both")
    
    init_physical = _render_physical_panel(
        df_ticks, df_events, init_n_min, init_n_max,
        fallback_start=init_start, fallback_end=abs_max_t
    )
    physical_pane = pn.pane.HoloViews(init_physical, sizing_mode="stretch_both")
    physical_pane.loading_indicator = True
    
    # 5. Stream de actualización discreta (mouseup)
    range_stream = RangeUpdateStream()
    mouseup_hook = create_mouseup_sync_hook(range_stream)
    
    # Aplicar hook al panel intrínseco inicial
    intrinsic_pane.object = intrinsic_pane.object.opts(hooks=[mouseup_hook])
    
    # 6. Callback que actualiza AMBOS paneles cuando cambia el rango
    def on_range_change(event):
        if event.new is None:
            return
        
        n_min, n_max = int(max(0, event.new[0])), int(min(event.new[1], total_events))
        
        # Actualizar Panel A (intrínseco) - filtrado para Y auto-range
        new_intrinsic = _render_intrinsic_filtered(pdf_segments, n_min, n_max)
        new_intrinsic = new_intrinsic.opts(hooks=[mouseup_hook])
        intrinsic_pane.object = new_intrinsic
        
        # Actualizar Panel B (físico)
        new_physical = _render_physical_panel(
            df_ticks, df_events,
            n_min=n_min, n_max=n_max,
            fallback_start=init_start, fallback_end=abs_max_t
        )
        physical_pane.object = new_physical
    
    range_stream.param.watch(on_range_change, 'x_range')
    
    # Divisor visual
    divider = pn.pane.HTML('<hr style="margin: 10px 0; border: 0; border-top: 1px solid #ddd;">')

    return pn.Column(
        pn.pane.Markdown(f"## {title}"),
        intrinsic_pane,
        divider,
        physical_pane,
        sizing_mode="stretch_both"
    )


def _render_intrinsic_filtered(pdf_segments, n_min, n_max):
    """
    Renderiza Panel A (intrínseco) filtrado a un rango de eventos.
    El Y-range se calcula solo sobre datos visibles.
    """
    from .core_plots import _build_intrinsic_panel
    
    # Filtrar segmentos al rango visible
    mask = (pdf_segments['x0'] >= n_min) & (pdf_segments['x1'] <= n_max + 1)
    filtered = pdf_segments[mask]
    
    if filtered.empty:
        # Fallback: mostrar algo si el filtro está vacío
        filtered = pdf_segments.iloc[-10:]
    
    return _build_intrinsic_panel(filtered, height=MAIN_PANEL_HEIGHT)


def _render_physical_panel(df_ticks, df_events, n_min, n_max, fallback_start, fallback_end):
    """
    Renderiza Panel B (físico) para un rango de eventos dado.
    
    Args:
        df_ticks: DataFrame completo de ticks
        df_events: DataFrame completo de eventos
        n_min, n_max: Rango de índices SECUENCIALES de eventos (0,1,2,...N)
        fallback_start, fallback_end: Tiempos por defecto si el rango está vacío
    
    Returns:
        Objeto HoloViews renderizable
    """
    # Mapear índice secuencial a tiempo físico usando slicing por filas
    n_min = max(0, n_min)
    n_max = min(n_max, len(df_events))
    
    if n_min >= n_max or n_max <= 0:
        t_start, t_end = fallback_start, fallback_end
    else:
        window_ev = df_events.slice(n_min, n_max - n_min)
        if window_ev.is_empty():
            t_start, t_end = fallback_start, fallback_end
        else:
            t_start = window_ev["time"].min()
            t_end = window_ev["time"].max()
    
    # Asegurar ventana mínima de 1 hora (evita t_start == t_end)
    min_window = timedelta(hours=1)
    if (t_end - t_start) < min_window:
        mid_time = t_start + (t_end - t_start) / 2
        t_start = mid_time - min_window / 2
        t_end = mid_time + min_window / 2
    
    # Validar ventana máxima
    window_hours = (t_end - t_start).total_seconds() / 3600
    if window_hours > MAX_WINDOW_HOURS:
        # Limitar al máximo permitido, centrado en el rango solicitado
        mid_time = t_start + (t_end - t_start) / 2
        half_window = timedelta(hours=MAX_WINDOW_HOURS / 2)
        t_start = mid_time - half_window
        t_end = mid_time + half_window
    
    # Filtrar ticks y preparar con categorías DC para colorización
    df_win = df_ticks.filter(pl.col("time").is_between(t_start, t_end))
    
    from .utils import _prepare_price_data
    pdf_win = _prepare_price_data(df_win)
    
    return _build_physical_panel(pdf_win, height=SECONDARY_PANEL_HEIGHT)