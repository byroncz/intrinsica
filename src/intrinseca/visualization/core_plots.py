import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, spread, dynspread
from holoviews import opts
from bokeh.models import HoverTool

hv.extension('bokeh')

from .utils import _prepare_price_data, _prepare_event_data, _get_vlines
from .hooks import _apply_x_zoom_hook, _apply_integer_xticks_hook, _apply_30min_xticks_hook

def _build_hv_plot(df_ticks, df_events, title, theta, height):
    custom_hover = HoverTool(
        tooltips=[('Fecha', '$x{%F %T}'), ('Precio', '$y{0,0.00}')],
        formatters={'$x': 'datetime', '$y': 'numeral'}
    )

    pdf = _prepare_price_data(df_ticks)
    edf = _prepare_event_data(df_events)

    # Capas de Datashader
    price_dots = hv.Points(pdf, ['time', 'price'], vdims=['status_cat'])
    price_shaded = spread(
        datashade(price_dots, aggregator=ds.count_cat('status_cat'),
                  color_key={'Upward': '#006400', 'Upturn': '#90ee90', 'Downward': '#8b0000', 
                             'Downturn': '#ffb6c1', 'Neutral': '#bdc3c7'}),
        px=1
    ).opts(xlabel="Time (UTC)", ylabel="Price", show_legend=False)

    # Capas de Eventos
    up_ticks = hv.Scatter(pdf[pdf['status_cat'] == 'Upward'], ['time'], ['price']).opts(color='#006400', size=2)
    down_ticks = hv.Scatter(pdf[pdf['status_cat'] == 'Downward'], ['time'], ['price']).opts(color='#8b0000', size=2)
    
    # Líneas de tiempo
    mid, noon = _get_vlines(df_ticks)
    v_lines = hv.Overlay([hv.VLine(t).opts(color='gray', line_width=0.5) for t in mid] + 
                        [hv.VLine(t).opts(color='gray', line_dash='dotted', line_width=0.5) for t in noon])

    layout = (price_shaded * up_ticks * down_ticks * v_lines)

    return layout.opts(
        opts.RGB(height=height, responsive=True, bgcolor='white',
                 tools=[custom_hover, 'crosshair', 'xbox_zoom', 'xwheel_zoom', 'reset'],
                 hooks=[_apply_x_zoom_hook])
    )

def _build_intrinsic_panel(pdf_segments, height=400):
    """
    Panel Superior: Cajas bi-particionadas por evento DC (definiciones Tsang).
    
    Cada evento N se visualiza como dos rectángulos apilados verticalmente:
    - DC Event N (change): ext_price(N) → price(N) [extremo → confirmación]
    - Overshoot N: price(N) → next_ext_price [confirmación → extremo siguiente]
    
    Colores:
    - Upturn: verde claro (change) / verde oscuro (overshoot)
    - Downturn: rosa (change) / rojo oscuro (overshoot)
    
    Args:
        pdf_segments: DataFrame con columnas seq_idx, ext_price, price, next_ext_price, 
                      next_ext_time, type_desc, time, ext_time
        height: Altura en píxeles, o None para modo responsivo completo
    """
    import numpy as np
    
    # Parámetros de visualización
    box_half_width = 0.35  # Mitad del ancho de la caja
    
    # Colores por tipo de evento
    colors = {
        'upturn': {'change': '#90ee90', 'overshoot': '#006400'},   # Verdes
        'downturn': {'change': '#ffb6c1', 'overshoot': '#8b0000'}  # Rosas/Rojos
    }
    
    # =========================================================================
    # DC Event N (change): ext_price(N) → price(N)
    # Representa el movimiento desde el extremo hasta la confirmación
    # =========================================================================
    change_data = pdf_segments.copy()
    change_data['x0'] = change_data['seq_idx'] - box_half_width
    change_data['x1'] = change_data['seq_idx'] + box_half_width
    # y0/y1 son el rango de precios (min/max de ext_price y price)
    change_data['y0'] = np.minimum(change_data['ext_price'], change_data['price'])
    change_data['y1'] = np.maximum(change_data['ext_price'], change_data['price'])
    change_data['color'] = change_data['type_desc'].map(
        lambda t: colors.get(t, {}).get('change', '#bdc3c7')
    )
    change_data['part'] = 'DC Event'
    
    # Tiempo: el DC Event va desde ext_time hasta time (confirmación)
    if 'ext_time' in change_data.columns:
        change_data['start_time'] = change_data['ext_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        change_data['start_time'] = 'N/A'
    if 'time' in change_data.columns:
        change_data['end_time'] = change_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        change_data['end_time'] = 'N/A'
    
    # Delta P y Delta T para DC Event
    change_data['delta_p'] = np.abs(change_data['price'] - change_data['ext_price'])
    if 'time' in change_data.columns and 'ext_time' in change_data.columns:
        change_data['delta_t'] = (change_data['time'] - change_data['ext_time']).dt.total_seconds() / 3600  # en horas
        change_data['delta_t_str'] = change_data['delta_t'].apply(lambda h: f"{h:.2f} h")
    else:
        change_data['delta_t_str'] = 'N/A'
    
    # =========================================================================
    # Overshoot N: price(N) → next_ext_price
    # Representa el movimiento desde confirmación hasta el siguiente extremo
    # =========================================================================
    os_data = pdf_segments.copy()
    os_data['x0'] = os_data['seq_idx'] - box_half_width
    os_data['x1'] = os_data['seq_idx'] + box_half_width
    # y0/y1 son el rango de precios (min/max de price y next_ext_price)
    os_data['y0'] = np.minimum(os_data['price'], os_data['next_ext_price'])
    os_data['y1'] = np.maximum(os_data['price'], os_data['next_ext_price'])
    os_data['color'] = os_data['type_desc'].map(
        lambda t: colors.get(t, {}).get('overshoot', '#bdc3c7')
    )
    os_data['part'] = 'Overshoot'
    
    # Tiempo: el Overshoot va desde time (confirmación) hasta next_ext_time (extremo del evento N+1)
    if 'time' in os_data.columns:
        os_data['start_time'] = os_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        os_data['start_time'] = 'N/A'
    if 'next_ext_time' in os_data.columns:
        os_data['end_time'] = os_data['next_ext_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        os_data['end_time'] = 'N/A'
    
    # Delta P y Delta T para Overshoot
    os_data['delta_p'] = np.abs(os_data['next_ext_price'] - os_data['price'])
    if 'next_ext_time' in os_data.columns and 'time' in os_data.columns:
        os_data['delta_t'] = (os_data['next_ext_time'] - os_data['time']).dt.total_seconds() / 3600  # en horas
        os_data['delta_t_str'] = os_data['delta_t'].apply(lambda h: f"{h:.2f} h")
    else:
        os_data['delta_t_str'] = 'N/A'
    
    # Custom HoverTool con formato legible
    custom_hover = HoverTool(tooltips=[
        ('Tipo', '@type_desc'),
        ('Parte', '@part'),
        ('Inicio', '@start_time'),
        ('Fin', '@end_time'),
        ('Precio Alto', '@y1{0.0,00}'),
        ('Precio Bajo', '@y0{0.0,00}'),
        ('ΔP', '@delta_p{0.0,00}'),
        ('ΔT', '@delta_t_str'),
    ])
    
    # Crear rectángulos HoloViews
    change_rects = hv.Rectangles(
        change_data, 
        ['x0', 'y0', 'x1', 'y1'], 
        ['type_desc', 'part', 'start_time', 'end_time', 'delta_p', 'delta_t_str', 'color']
    ).opts(
        color='color',
        line_width=0.5,
        line_color='#666666',
        alpha=0.85
    )
    
    os_rects = hv.Rectangles(
        os_data, 
        ['x0', 'y0', 'x1', 'y1'], 
        ['type_desc', 'part', 'start_time', 'end_time', 'delta_p', 'delta_t_str', 'color']
    ).opts(
        color='color',
        line_width=0.5,
        line_color='#444444',
        alpha=0.95
    )
    
    # Combinar overlays
    overlay = change_rects * os_rects
    
    # Opciones base (usar custom_hover en lugar de 'hover' genérico)
    base_opts = dict(
        responsive=True,
        padding=0.02,
        tools=['xbox_select', 'xwheel_zoom', 'reset', custom_hover],
        hooks=[_apply_integer_xticks_hook],
        xlabel='Event Index (n)',
        ylabel='Price',
        xformatter='%d',
        yformatter='%.0f'
    )
    
    # Altura explícita solo si se especifica
    if height is not None:
        base_opts['height'] = height
    
    return overlay.opts(opts.Rectangles(**base_opts))

def _build_physical_panel(pdf_ticks_win, event_markers=None, event_segments=None, xlim=None, height=250):
    """
    Panel Inferior: Recibe ticks con columna status_cat para colorización.
    
    Args:
        pdf_ticks_win: DataFrame Pandas con time, price, status_cat
        event_markers: Lista de tuplas (timestamp, event_idx) para VLines
        event_segments: Lista de dicts con datos de eventos para VSpan bands:
            - ext_time: inicio del DC Event
            - time: fin del DC Event / inicio del Overshoot
            - next_ext_time: fin del Overshoot
            - type_desc: 'upturn' o 'downturn'
        xlim: Tupla (t_start, t_end) para limitar el eje X
        height: Altura en píxeles, o None para modo responsivo completo
    """
    # Colores para VSpan bands (consistentes con Panel A)
    vspan_colors = {
        'upturn': {'dc_event': '#90ee90', 'overshoot': '#006400'},    # Verde claro / oscuro
        'downturn': {'dc_event': '#ffb6c1', 'overshoot': '#8b0000'}   # Rosa / Rojo oscuro
    }
    
    # =========================================================================
    # 1. Crear VSpan bands para DC Events y Overshoots (capa inferior)
    # =========================================================================
    vspan_elements = []
    if event_segments:
        for seg in event_segments:
            type_desc = seg.get('type_desc', '').lower()
            colors = vspan_colors.get(type_desc, {'dc_event': '#bdc3c7', 'overshoot': '#888888'})
            
            # Remover timezone de los timestamps
            def naive(ts):
                return ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            
            ext_time = naive(seg['ext_time'])
            time = naive(seg['time'])
            next_ext_time = seg.get('next_ext_time')
            
            # DC Event band: ext_time → time
            if ext_time and time and ext_time < time:
                dc_vspan = hv.VSpan(ext_time, time).opts(
                    color=colors['dc_event'], alpha=0.3
                )
                vspan_elements.append(dc_vspan)
            
            # Overshoot band: time → next_ext_time
            if next_ext_time:
                next_ext_time = naive(next_ext_time)
                if time and next_ext_time and time < next_ext_time:
                    os_vspan = hv.VSpan(time, next_ext_time).opts(
                        color=colors['overshoot'], alpha=0.3
                    )
                    vspan_elements.append(os_vspan)
    
    # =========================================================================
    # 2. Datashader layer (ticks colorizados)
    # =========================================================================
    points = hv.Points(pdf_ticks_win, ['time', 'price'], vdims=['status_cat'])
    
    shaded = spread(
        datashade(points, aggregator=ds.count_cat('status_cat'),
                  color_key={'Upward': '#006400', 'Upturn': '#90ee90', 
                             'Downward': '#8b0000', 'Downturn': '#ffb6c1', 
                             'Neutral': '#bdc3c7'}),
        px=1
    )
    
    # =========================================================================
    # 3. Preparar datos para etiquetas de eventos (fondo blanco via Bokeh hook)
    # =========================================================================
    label_data = []
    if event_markers:
        max_markers = 30
        num_events = len(event_markers)
        
        # Densidad adaptativa: mostrar cada N eventos si hay muchos
        step = max(1, (num_events + max_markers - 1) // max_markers)
        sampled_markers = event_markers[::step]
        
        # Obtener rango de precios para posicionar etiquetas DENTRO del área visible
        if not pdf_ticks_win.empty:
            y_min = pdf_ticks_win['price'].min()
            y_max = pdf_ticks_win['price'].max()
            y_range = y_max - y_min
            y_pos = y_max - 0.05 * y_range  # 5% debajo del máximo (dentro del rango)
        else:
            y_pos = 0
        
        # Colores por tipo de evento
        event_colors = {
            'upturn': '#006400',   # Verde oscuro
            'downturn': '#8b0000'  # Rojo oscuro
        }
        
        for ts, idx, type_desc in sampled_markers:
            # Remover timezone si existe
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            color = event_colors.get(type_desc.lower() if type_desc else '', '#555555')
            label_data.append({'x': ts_naive, 'y': y_pos, 'text': f' {idx} ', 'color': color})
    
    # =========================================================================
    # 4. Combinar layers: VSpans (fondo) → Datashader (medio)
    # =========================================================================
    if vspan_elements:
        vspan_overlay = hv.Overlay(vspan_elements)
        layout = vspan_overlay * shaded
    else:
        layout = shaded
    
    # Custom HoverTool para Panel B con formato de tiempo y precio
    panel_b_hover = HoverTool(tooltips=[
        ('Tiempo', '$x{%F %H:%M:%S}'),
        ('Precio', '$y{0.0,00}'),
    ], formatters={
        '$x': 'datetime'
    })
    
    # Crear lista de hooks: siempre incluir 30min ticks, opcionalmente labels
    from .hooks import create_event_labels_hook
    hooks_list = [_apply_30min_xticks_hook]
    if label_data:
        hooks_list.append(create_event_labels_hook(label_data))
    
    # Opciones base para RGB
    rgb_opts = dict(
        responsive=True, 
        xlabel="Time (UTC)",
        tools=['xwheel_zoom', 'reset', 'crosshair', panel_b_hover],
        hooks=hooks_list,
        show_legend=False,  # Ocultar leyenda para evitar obstruir ticks
        padding=0  # Sin padding para ajustar exactamente al rango de datos
    )
    
    # Altura explícita solo si se especifica
    if height is not None:
        rgb_opts['height'] = height
    
    # Limitar eje X si se proporciona xlim
    if xlim is not None:
        t_start, t_end = xlim
        # Remover timezone si existe
        if hasattr(t_start, 'tzinfo') and t_start.tzinfo:
            t_start = t_start.replace(tzinfo=None)
        if hasattr(t_end, 'tzinfo') and t_end.tzinfo:
            t_end = t_end.replace(tzinfo=None)
        rgb_opts['xlim'] = (t_start, t_end)
    
    return layout.opts(opts.RGB(**rgb_opts))