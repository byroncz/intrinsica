from bokeh.models import BoxZoomTool, WheelZoomTool

def _apply_x_zoom_hook(plot, element):
    """Fuerza a las herramientas de zoom a operar solo en el eje X."""
    for tool in plot.state.tools:
        # Bloquear Zoom de Caja a horizontal
        if isinstance(tool, BoxZoomTool):
            tool.dimensions = 'width'
        
        # Bloquear Zoom de Rueda a horizontal
        if isinstance(tool, WheelZoomTool):
            tool.dimensions = 'width'