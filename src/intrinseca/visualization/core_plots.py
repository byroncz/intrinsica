import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, spread, dynspread
from holoviews import opts
from bokeh.models import HoverTool

hv.extension('bokeh')

from .utils import _prepare_price_data, _prepare_event_data, _get_vlines
from .hooks import _apply_x_zoom_hook, _apply_integer_xticks_hook

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
    Panel Superior: Basado en Índices de Eventos (n).
    
    Args:
        pdf_segments: DataFrame con segmentos de eventos
        height: Altura en píxeles, o None para modo responsivo completo
    """
    # Opciones base
    base_opts = dict(
        color='type_desc', 
        cmap={'upturn': '#006400', 'downturn': '#8b0000'},
        line_width=2, 
        responsive=True,
        padding=0,
        tools=['xbox_select', 'xwheel_zoom', 'reset'], 
        hooks=[_apply_integer_xticks_hook],
        xformatter='%d',
        yformatter='%.0f'
    )
    
    # Altura explícita solo si se especifica
    if height is not None:
        base_opts['height'] = height
    
    return hv.Segments(
        pdf_segments, 
        [hv.Dimension('x0', label='Event Index (n)'), 
         hv.Dimension('y0', label='Price'), 
         'x1', 'y1'], 
        ['type_desc', 'is_overshoot']
    ).opts(**base_opts)

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
    
    # Opciones base para RGB
    rgb_opts = dict(
        responsive=True, 
        xlabel="Time (UTC)",
        tools=['xwheel_zoom', 'reset'],
        hooks=[_apply_x_zoom_hook]
    )
    
    # Altura explícita solo si se especifica
    if height is not None:
        rgb_opts['height'] = height
    
    return layout.opts(opts.RGB(**rgb_opts))