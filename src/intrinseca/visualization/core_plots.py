import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, spread, dynspread
from holoviews import opts
from bokeh.models import HoverTool

hv.extension('bokeh')

from .utils import _prepare_price_data, _prepare_event_data, _get_vlines
from .hooks import _apply_x_zoom_hook

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
    """Panel Superior: Basado en Índices de Eventos (n)."""
    return hv.Segments(
        pdf_segments, 
        [hv.Dimension('x0', label='Event (n)'), 
         hv.Dimension('y0', label='Price'), 
         'x1', 'y1'], 
        ['type_desc', 'is_overshoot']
    ).opts(
        color='type_desc', 
        cmap={'upturn': '#006400', 'downturn': '#8b0000'},
        line_width=2, 
        height=height, 
        responsive=True,
        padding=0,  # Sin espacio extra en los bordes
        tools=['xbox_select', 'xwheel_zoom', 'reset'], 
        hooks=[_apply_x_zoom_hook],
        # Formateo de ejes: evitar notación científica
        xformatter='%d',  # Enteros para índice de eventos
        yformatter='%.0f'  # Precios sin decimales (para BTC ~90000)
    )

def _build_physical_panel(pdf_ticks_win, height=250):
    """Panel Inferior: Recibe ticks con columna status_cat para colorización."""
    points = hv.Points(pdf_ticks_win, ['time', 'price'], vdims=['status_cat'])
    
    # Datashader con agregación por categoría DC
    shaded = spread(
        datashade(points, aggregator=ds.count_cat('status_cat'),
                  color_key={'Upward': '#006400', 'Upturn': '#90ee90', 
                             'Downward': '#8b0000', 'Downturn': '#ffb6c1', 
                             'Neutral': '#bdc3c7'}),
        px=1
    )
    
    return shaded.opts(
        height=height, responsive=True, xlabel="Time (UTC)",
        tools=['xwheel_zoom', 'reset'],
        hooks=[_apply_x_zoom_hook]
    )