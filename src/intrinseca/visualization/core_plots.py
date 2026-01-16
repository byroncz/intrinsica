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
        pdf_segments: DataFrame con columnas seq_idx, ext_price, price, next_ext_price, type_desc, time, ext_time
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
    
    # Tiempo: el Overshoot va desde time (confirmación) hasta el ext_time del siguiente evento
    # Como no tenemos ext_time(N+1) directamente, mostramos la confirmación como inicio
    if 'time' in os_data.columns:
        os_data['start_time'] = os_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        os_data['start_time'] = 'N/A'
    os_data['end_time'] = '→ siguiente extremo'
    
    # Custom HoverTool con formato legible
    custom_hover = HoverTool(tooltips=[
        ('Tipo', '@type_desc'),
        ('Parte', '@part'),
        ('Inicio', '@start_time'),
        ('Fin', '@end_time'),
        ('Precio Alto', '@y1{0.0,00}'),
        ('Precio Bajo', '@y0{0.0,00}'),
    ])
    
    # Crear rectángulos HoloViews
    change_rects = hv.Rectangles(
        change_data, 
        ['x0', 'y0', 'x1', 'y1'], 
        ['type_desc', 'part', 'start_time', 'end_time', 'color']
    ).opts(
        color='color',
        line_width=0.5,
        line_color='#666666',
        alpha=0.85
    )
    
    os_rects = hv.Rectangles(
        os_data, 
        ['x0', 'y0', 'x1', 'y1'], 
        ['type_desc', 'part', 'start_time', 'end_time', 'color']
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

def _build_physical_panel(pdf_ticks_win, event_markers=None, height=250):
    """
    Panel Inferior: Recibe ticks con columna status_cat para colorización.
    
    Args:
        pdf_ticks_win: DataFrame Pandas con time, price, status_cat
        event_markers: Lista de tuplas (timestamp, event_idx) para VLines
        height: Altura en píxeles, o None para modo responsivo completo
    """
    points = hv.Points(pdf_ticks_win, ['time', 'price'], vdims=['status_cat'])
    
    # Datashader con agregación por categoría DC
    shaded = spread(
        datashade(points, aggregator=ds.count_cat('status_cat'),
                  color_key={'Upward': '#006400', 'Upturn': '#90ee90', 
                             'Downward': '#8b0000', 'Downturn': '#ffb6c1', 
                             'Neutral': '#bdc3c7'}),
        px=1
    )
    
    # Crear overlay de VLines para marcadores de eventos
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
            y_pos = y_max - 0.05 * y_range  # 2% debajo del máximo (dentro del rango)
        else:
            y_pos = 0
        
        # Crear VLines y etiquetas
        elements = []
        for ts, idx in sampled_markers:
            # Remover timezone si existe
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            
            # Línea vertical
            vline = hv.VLine(ts_naive).opts(
                color='gray', line_width=0.8, line_dash='dashed', alpha=0.4
            )
            elements.append(vline)
            
            # Etiqueta con índice del evento (fondo blanco, borde gris)
            label = hv.Text(ts_naive, y_pos, f' n={idx} ', fontsize=7).opts(
                text_color='gray', text_alpha=0.9,
                bgcolor='white',
                border_line_color='gray', border_line_alpha=0.7
            )
            elements.append(label)
        
        if elements:
            event_overlay = hv.Overlay(elements)
            layout = shaded * event_overlay
        else:
            layout = shaded
    else:
        layout = shaded
    
    # Custom HoverTool para Panel B con formato de tiempo y precio
    panel_b_hover = HoverTool(tooltips=[
        ('Tiempo', '$x{%F %H:%M:%S}'),
        ('Precio', '$y{0.0,00}'),
    ], formatters={
        '$x': 'datetime'
    })
    
    # Opciones base para RGB
    rgb_opts = dict(
        responsive=True, 
        xlabel="Time (UTC)",
        tools=['xwheel_zoom', 'reset', 'crosshair', panel_b_hover],
        hooks=[_apply_30min_xticks_hook],
        show_legend=False  # Ocultar leyenda para evitar obstruir ticks
    )
    
    # Altura explícita solo si se especifica
    if height is not None:
        rgb_opts['height'] = height
    
    return layout.opts(opts.RGB(**rgb_opts))