import panel as pn
import holoviews as hv
import polars as pl
from datetime import datetime, timezone, timedelta
import math

from .config import (
    MAX_WINDOW_HOURS, INITIAL_WINDOW_HOURS, 
    MAIN_PANEL_HEIGHT, SECONDARY_PANEL_HEIGHT
)
from .core_plots import _build_hv_plot, _build_intrinsic_panel, _build_physical_panel
from .hooks import RangeUpdateStream, create_mouseup_sync_hook, _apply_x_zoom_hook, _apply_integer_xticks_hook
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


def create_dual_axis_dashboard(df_ticks, df_events, title="Dual-Axis DC", event_range=None):
    """
    Dashboard sincronizado de doble eje (intrínseco + físico).
    
    Panel A (superior): Tiempo intrínseco basado en índice de eventos
    Panel B (inferior): Tiempo físico, actualizado en mouseup del Panel A
    
    Ambos paneles se re-renderizan al cambiar el rango visible para auto-ajustar Y.
    
    Args:
        df_ticks: DataFrame Polars con ticks augmentados
        df_events: DataFrame Polars con eventos DC
        title: Título del dashboard
        event_range: Tupla opcional (n_min, n_max) para especificar la franja 
                     de eventos a visualizar inicialmente. Si es None, se calcula
                     automáticamente basándose en INITIAL_WINDOW_HOURS.
    """
    # 1. Límites temporales del dataset
    abs_max_t = df_ticks["time"].max()
    abs_min_t = df_ticks["time"].min()
    init_start = abs_max_t - timedelta(hours=INITIAL_WINDOW_HOURS)
    
    # 2. Preparar datos de eventos (ligero: ~12K registros con índice secuencial)
    pdf_segments = prepare_dual_axis_data(df_events)
    total_events = len(pdf_segments)
    
    # 3. Calcular rango inicial para ambos paneles
    if event_range is not None:
        # Usar rango especificado por el usuario, validando límites
        init_n_min = max(0, int(event_range[0]))
        init_n_max = min(int(event_range[1]), total_events - 1)
        # Validar que el rango sea coherente
        if init_n_min > init_n_max:
            init_n_min, init_n_max = init_n_max, init_n_min
    else:
        # Comportamiento por defecto: calcular basándose en ventana temporal
        mask = (pdf_segments['time'] >= init_start) & (pdf_segments['time'] <= abs_max_t)
        init_segment_data = pdf_segments[mask]
        
        if not init_segment_data.empty:
            init_n_min = int(init_segment_data['seq_idx'].min())
            init_n_max = int(init_segment_data['seq_idx'].max())
        else:
            init_n_max = total_events - 1
            init_n_min = max(0, init_n_max - 100)
    
    # 4. Contenedores Panel para ambos gráficos (permiten actualización dinámica)
    from .core_plots import _build_intrinsic_panel, _build_physical_panel
    
    # Función helper para calcular Y-limits unificados
    def _calculate_shared_ylim(n_min, n_max, t_start=None, t_end=None):
        """Calcula ylim unificado para Chart A y B basado en datos visibles."""
        # 1. Rango de precios en Eventos (Chart A)
        # Filtrar eventos visibles
        mask_ev = (pdf_segments['seq_idx'] >= n_min) & (pdf_segments['seq_idx'] <= n_max)
        vis_events = pdf_segments[mask_ev]
        
        if vis_events.empty:
            y_min_ev, y_max_ev = float('inf'), float('-inf')
        else:
            # Considerar precios de extremos y confirmaciones para Change y Overshoot
            p_cols = ['ext_price', 'price', 'next_ext_price']
            # Usar min/max global de las columnas relevantes
            # Nota: next_ext_price puede tener NaNs si es el último evento, ignorarlos
            y_min_ev = min(vis_events[c].min() for c in p_cols if c in vis_events.columns)
            y_max_ev = max(vis_events[c].max() for c in p_cols if c in vis_events.columns)

        # 2. Rango de precios en Ticks (Chart B)
        # Si no se pasan tiempos explícitos, derivarlos de los eventos visibles
        if t_start is None or t_end is None:
            if not vis_events.empty:
                t_start = vis_events['ext_time'].min()
                # Para el final, intentar usar next_ext_time del último, o fallback a time
                if 'next_ext_time' in vis_events.columns and not vis_events['next_ext_time'].isnull().all():
                     t_end = vis_events['next_ext_time'].max()
                else:
                     t_end = vis_events['time'].max()
            else:
                 # Fallback total
                 t_start, t_end = init_start, abs_max_t
        
        # Filtrar ticks en el rango de tiempo
        # Usar lazy evaluation si fuera muy grande, pero aquí df_ticks es el full dataframe
        # Hacer un filter rápido de Polars
        ticks_win = df_ticks.filter(pl.col("time").is_between(t_start, t_end))
        if ticks_win.is_empty():
            y_min_tk, y_max_tk = float('inf'), float('-inf')
        else:
            y_min_tk = ticks_win['price'].min()
            y_max_tk = ticks_win['price'].max()
            
        # 3. Combinar rangos
        y_min = min(y_min_ev, y_min_tk)
        y_max = max(y_max_ev, y_max_tk)
        
        if y_min == float('inf') or y_max == float('-inf'):
            return None # No data
            
        # 4. Agregar padding para etiquetas (estimado fijo o proporcional)
        # En core_plots se usa y_padding_fixed = 15 + texto (~35) = 50 unids aprox? 
        # Mejor usar un % del rango total para ser agnóstico a la escala del precio, 
        # pero asegurando un mínimo para las etiquetas de texto fijas.
        # Definimos un padding generoso para acomodar etiquetas arriba y abajo.
        
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 10 # Default range si es plano
            
        # Padding proporcional (e.g. 10%) + offset fijo para etiquetas
        # Las etiquetas 'overshoot' consumen espacio vertical fijo en pixels/unidades
        # Asumiremos un padding del 5-8% arriba y abajo
        pad = y_range * 0.08
        
        # Asegurar espacio mínimo fijo si el rango es muy pequeño (para que quepa el texto)
        # Esto depende de la escala de precios del activo. 
        # Si el activo es 4000 (SPX), 15 pts es poco. Si es 1.05 (EURUSD), 15 es enorme.
        # El código original usaba padding fijo de 15. Asumiremos que es adecuado para este activo.
        FIXED_LABEL_PAD = 15 + 20 # 15 offset + 20 texto
        
        # Usar el mayor entre el proporcional y el fijo (si el rango es muy chico, domina el fijo)
        final_pad = max(pad, FIXED_LABEL_PAD)
        
        return (y_min - final_pad, y_max + final_pad)

    # Calcular ylim inicial
    # Para el cálculo inicial necesitamos el t_start/t_end físico
    # Simular la lógica de _render_physical_panel para obtener tiempos
    # (Simplificado: min ext_time a max next_ext_time de la ventana de eventos)
    mask_init = (pdf_segments['seq_idx'] >= init_n_min) & (pdf_segments['seq_idx'] <= init_n_max)
    vis_init = pdf_segments[mask_init]
    if not vis_init.empty:
        t_s = vis_init['ext_time'].min()
        # Intentar obtener el tiempo final real (incluyendo el overshoot del último evento)
        # Esto requiere mirar el evento N+1 si existe, lógica similar a _render_physical_panel
        if init_n_max + 1 < total_events:
            # Buscar el sig evento en el df completo
            next_row = pdf_segments.iloc[init_n_max + 1]
            t_e = next_row['ext_time']
        else:
            t_e = vis_init['time'].max()
    else:
        t_s, t_e = init_start, abs_max_t

    shared_ylim = _calculate_shared_ylim(init_n_min, init_n_max, t_s, t_e)

    # Renderizar vistas iniciales con altura responsiva (50% viewport cada uno)
    init_intrinsic = _render_intrinsic_filtered(pdf_segments, init_n_min, init_n_max, ylim=shared_ylim)
    intrinsic_pane = pn.pane.HoloViews(
        init_intrinsic, sizing_mode="stretch_both", min_height=200,
        linked_axes=False  # Disable auto-linking: Panel A uses int indices, Panel B uses datetime
    )
    
    init_physical = _render_physical_panel(
        df_ticks, df_events, init_n_min, init_n_max,
        fallback_start=init_start, fallback_end=abs_max_t,
        ylim=shared_ylim
    )
    physical_pane = pn.pane.HoloViews(
        init_physical, sizing_mode="stretch_both", min_height=200,
        linked_axes=False  # Disable auto-linking to prevent dtype conflicts
    )
    physical_pane.loading_indicator = True
    
    # 5. Stream de actualización discreta (mouseup) con rango inicial para Reset
    range_stream = RangeUpdateStream()
    mouseup_hook = create_mouseup_sync_hook(range_stream, init_n_min, init_n_max)
    
    # Aplicar hooks al panel intrínseco inicial (integer ticks + mouseup sync)
    intrinsic_pane.object = intrinsic_pane.object.opts(hooks=[_apply_integer_xticks_hook, mouseup_hook])
    
    # 6. Callback con debouncing para evitar race conditions en actualizaciones rápidas
    import threading
    pending_update = {'timer': None, 'lock': threading.Lock()}
    
    def _do_update(n_min, n_max):
        """Ejecuta la actualización real de ambos paneles (thread-safe)."""
        def _apply_updates():
            # 1. Calcular tiempos físicos correspondientes al nuevo rango de eventos N
            # Necesario para filtrar ticks y obtener el rango Y completo real
            # Replicar lógica de tiempos de physical_panel
            n_min_c = max(0, n_min)
            n_max_c = min(n_max, total_events - 1)
            
            # Obtener t_start y t_end estimados
            # Usamos un slice ligero
            # slice(offset, length) -> iloc[start : start + length]
            # n_max_c - n_min_c + 1 es length.
            # start + length = n_min_c + (n_max_c - n_min_c + 1) = n_max_c + 1
            win = pdf_segments.iloc[n_min_c : n_max_c + 1]
            if win.is_empty():
                 t_start_upd, t_end_upd = init_start, abs_max_t
            else:
                t_start_upd = win['ext_time'].min()
                # Fin: overshoot del último (inicio evento N+1)
                if n_max_c + 1 < total_events:
                     next_r = pdf_segments.iloc[n_max_c + 1]
                     t_end_upd = next_r['ext_time']
                else:
                     t_end_upd = win['time'].max()
            
            # 2. Calcular límites Y unificados usando estos tiempos
            new_ylim = _calculate_shared_ylim(n_min, n_max, t_start_upd, t_end_upd)
            
            # Actualizar Panel A (intrínseco) - filtrado para Y auto-range
            new_intrinsic = _render_intrinsic_filtered(pdf_segments, n_min, n_max, ylim=new_ylim)
            new_intrinsic = new_intrinsic.opts(hooks=[_apply_integer_xticks_hook, mouseup_hook])
            intrinsic_pane.object = new_intrinsic
            
            # Actualizar Panel B (físico)
            new_physical = _render_physical_panel(
                df_ticks, df_events,
                n_min=n_min, n_max=n_max,
                fallback_start=init_start, fallback_end=abs_max_t,
                ylim=new_ylim
            )
            physical_pane.object = new_physical
        
        # Ejecutar en el thread principal de Panel para evitar race conditions
        pn.state.execute(_apply_updates)
        
        with pending_update['lock']:
            pending_update['timer'] = None
    
    def on_range_change(event):
        if event.new is None:
            return
        
        # Redondeo semántico: ceil para n_min, floor para n_max
        # Esto asegura que solo se incluyan eventos cuyo CENTRO esté dentro de la selección
        n_min = int(max(0, math.ceil(event.new[0])))
        n_max = int(min(math.floor(event.new[1]), total_events - 1))
        
        # Validar que el rango sea válido (puede ser inválido si la selección es muy pequeña)
        if n_min > n_max:
            return
        
        with pending_update['lock']:
            # Cancelar timer pendiente si existe
            if pending_update['timer'] is not None:
                pending_update['timer'].cancel()
            
            # Programar nueva actualización con 100ms de delay
            pending_update['timer'] = threading.Timer(0.1, _do_update, args=(n_min, n_max))
            pending_update['timer'].start()
    
    range_stream.param.watch(on_range_change, 'x_range')
    
    # Divisor visual minimalista
    divider = pn.pane.HTML('<hr style="margin: 2px 0; border: 0; border-top: 1px solid #ddd;">')

    # Título compacto con altura fija
    title_pane = pn.pane.Markdown(f"## {title}", styles={'margin': '0', 'padding': '5px 10px'})

    # Container FlexBox: paneles con flex: 1 para ocupar 50-50 del espacio disponible
    return pn.FlexBox(
        title_pane,
        pn.Row(intrinsic_pane, sizing_mode="stretch_both", styles={'flex': '1', 'min-height': '0'}),
        divider,
        pn.Row(physical_pane, sizing_mode="stretch_both", styles={'flex': '1', 'min-height': '0'}),
        flex_direction='column',
        sizing_mode="stretch_both",
        styles={'height': '100vh', 'width': '100%', 'overflow': 'hidden'}
    )


def _render_intrinsic_filtered(pdf_segments, n_min, n_max, ylim=None):
    """
    Renderiza Panel A (intrínseco) filtrado a un rango de eventos.
    El Y-range se calcula solo sobre datos visibles.
    """
    from .core_plots import _build_intrinsic_panel
    
    # Filtrar segmentos al rango visible usando seq_idx
    mask = (pdf_segments['seq_idx'] >= n_min) & (pdf_segments['seq_idx'] <= n_max)
    filtered = pdf_segments[mask]
    
    if filtered.empty:
        # Fallback: mostrar algo si el filtro está vacío
        filtered = pdf_segments.iloc[-10:]
    
    # Sin altura fija: permite que el contenedor flex controle el tamaño
    # Pasar ylim si existe
    return _build_intrinsic_panel(filtered, height=None, ylim=ylim)


def _render_physical_panel(df_ticks, df_events, n_min, n_max, fallback_start, fallback_end, ylim=None):
    """
    Renderiza Panel B (físico) para un rango de eventos dado.
    
    El rango de tiempo se calcula según definiciones Tsang:
    - t_start: ext_time del primer evento (inicio DC Event)
    - t_end: ext_time del evento siguiente al último (fin Overshoot)
    
    Args:
        df_ticks: DataFrame completo de ticks
        df_events: DataFrame completo de eventos
        n_min, n_max: Rango de índices SECUENCIALES de eventos (0,1,2,...N)
        fallback_start, fallback_end: Tiempos por defecto si el rango está vacío
    
    Returns:
        Objeto HoloViews renderizable
    """
    # Mapear índice secuencial a tiempo físico usando slicing por filas
    # n_min y n_max son INCLUSIVOS (consistente con Panel A)
    n_min = max(0, n_min)
    n_max = min(n_max, len(df_events) - 1)  # -1 porque es inclusivo
    
    if n_min > n_max or n_max < 0:
        t_start, t_end = fallback_start, fallback_end
        event_markers = []
        event_segments = []
    else:
        # slice(offset, length): length = n_max - n_min + 1 para incluir ambos extremos
        window_ev = df_events.slice(n_min, n_max - n_min + 1)
        if window_ev.is_empty():
            t_start, t_end = fallback_start, fallback_end
            event_markers = []
            event_segments = []
        else:
            # t_start: ext_time del primer evento (inicio del DC Event del primero)
            t_start = window_ev["ext_time"].min()
            
            # t_end: ext_time del evento SIGUIENTE al último visible (fin del Overshoot del último)
            # n_max es INCLUSIVO, así que el evento siguiente es n_max + 1
            if n_max + 1 < len(df_events):
                next_event = df_events.slice(n_max + 1, 1)
                if not next_event.is_empty():
                    t_end = next_event["ext_time"][0]
                else:
                    # Fallback: usar time del último evento visible
                    t_end = window_ev["time"].max()
            else:
                # No hay evento siguiente, usar time del último visible como aproximación
                t_end = window_ev["time"].max()
            
            # Extraer timestamps, índices y tipo de eventos para marcadores
            event_markers = [
                (row["time"], n_min + i, row["type_desc"]) 
                for i, row in enumerate(window_ev.iter_rows(named=True))
            ]
            
            # ================================================================
            # Construir event_segments para VSpan bands
            # Cada segmento contiene: ext_time, time, next_ext_time, type_desc
            # ================================================================
            # Agregar columna next_ext_time usando shift(-1)
            window_with_next = window_ev.with_columns([
                pl.col("ext_time").shift(-1).alias("next_ext_time")
            ])
            
            # Si el último evento visible tiene un siguiente (n_max + 1), usar su ext_time
            if n_max + 1 < len(df_events):
                next_event = df_events.slice(n_max + 1, 1)
                if not next_event.is_empty():
                    last_next_ext_time = next_event["ext_time"][0]
                    # Actualizar el último registro con el next_ext_time correcto
                    # Convertir a lista, modificar, y usar
                    segments_list = window_with_next.to_dicts()
                    if segments_list:
                        segments_list[-1]["next_ext_time"] = last_next_ext_time
                    event_segments = segments_list
                else:
                    event_segments = window_with_next.to_dicts()
            else:
                event_segments = window_with_next.to_dicts()
    
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
    
    # Sin altura fija: permite que el contenedor flex controle el tamaño
    return _build_physical_panel(
        pdf_win, event_markers, event_segments=event_segments, 
        xlim=(t_start, t_end), height=None, ylim=ylim
    )