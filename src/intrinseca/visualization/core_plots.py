import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, spread
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
    ).opts(xlabel="Tiempo (UTC)", ylabel="Precio", show_legend=False)

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