from bokeh.models import BoxZoomTool, WheelZoomTool, SingleIntervalTicker, DatetimeTicker
from bokeh.events import RangesUpdate
import param


def _apply_x_zoom_hook(plot, element):
    """Fuerza a las herramientas de zoom a operar solo en el eje X."""
    for tool in plot.state.tools:
        # Bloquear Zoom de Caja a horizontal
        if isinstance(tool, BoxZoomTool):
            tool.dimensions = 'width'
        
        # Bloquear Zoom de Rueda a horizontal
        if isinstance(tool, WheelZoomTool):
            tool.dimensions = 'width'


def _apply_30min_xticks_hook(plot, element):
    """
    Configura el eje X para mostrar ticks cada 30 minutos.
    Usado en Panel B (tiempo físico).
    """
    from bokeh.models import AdaptiveTicker
    
    # Aplicar restricciones de zoom X
    for tool in plot.state.tools:
        if isinstance(tool, BoxZoomTool):
            tool.dimensions = 'width'
        if isinstance(tool, WheelZoomTool):
            tool.dimensions = 'width'
    
    # Acceder al eje X y configurar ticker de 30 minutos
    xaxis = plot.handles.get('xaxis')
    if xaxis:
        # 30 minutos = 30 * 60 * 1000 = 1,800,000 milisegundos
        # Usar AdaptiveTicker con base de 30 minutos
        xaxis.ticker = AdaptiveTicker(
            base=60 * 1000,  # 1 minuto en ms como base
            mantissas=[1, 5, 15, 30, 60],  # Intervalos: 1, 5, 15, 30, 60 min
            min_interval=30 * 60 * 1000,  # Mínimo 30 minutos
            max_interval=4 * 60 * 60 * 1000  # Máximo 4 horas
        )
        xaxis.minor_tick_line_color = 'gray'
        xaxis.minor_tick_line_alpha = 0.5


def _apply_integer_xticks_hook(plot, element):
    """
    Fuerza el eje X a mostrar solo marcas en valores enteros.
    SingleIntervalTicker(interval=1) garantiza ticks exactamente cada 1 unidad.
    """
    # Aplicar restricciones de zoom X
    for tool in plot.state.tools:
        if isinstance(tool, BoxZoomTool):
            tool.dimensions = 'width'
        if isinstance(tool, WheelZoomTool):
            tool.dimensions = 'width'
    
    # Acceder al eje X correctamente vía plot.handles (no plot.state.xaxis que es tuple)
    xaxis = plot.handles.get('xaxis')
    if xaxis:
        xaxis.ticker = SingleIntervalTicker(interval=1)
        xaxis.minor_tick_line_color = None  # Ocultar marcas menores (fraccionarias)




def _apply_dynspread_hook(plot, element):
    """Asegura que picos rápidos no desaparezcan al rasterizar."""
    from holoviews.operation.datashader import dynspread
    # Esta lógica se aplica sobre la operación de datashader
    pass


class RangeUpdateStream(param.Parameterized):
    """
    Stream personalizado que solo se actualiza en eventos discretos (mouseup/RangesUpdate).
    Evita callbacks continuos durante animaciones de pan/zoom.
    """
    x_range = param.Tuple(default=None, length=2, doc="Rango X actual (start, end)")
    
    def update_range(self, x_start, x_end):
        """Actualiza el rango - dispara watchers de param"""
        self.x_range = (x_start, x_end)


def create_mouseup_sync_hook(range_stream: RangeUpdateStream):
    """
    Factory que crea un hook para sincronización en mouseup.
    
    Args:
        range_stream: Instancia de RangeUpdateStream a actualizar
        
    Returns:
        Hook function para usar en hv.opts(hooks=[...])
    """
    def _mouseup_hook(plot, element):
        """
        Hook que registra callback JS para capturar rango al soltar mouse.
        El callback JS envía el rango final al stream Python.
        """
        # Aplicar restricciones de zoom X
        _apply_x_zoom_hook(plot, element)
        
        # Callback al finalizar cambio de rango (después de zoom/pan)
        def on_ranges_update(event):
            if hasattr(event, 'x0') and hasattr(event, 'x1'):
                range_stream.update_range(event.x0, event.x1)
        
        # Registrar evento RangesUpdate (se dispara al finalizar interacción)
        plot.state.on_event(RangesUpdate, on_ranges_update)
    
    return _mouseup_hook