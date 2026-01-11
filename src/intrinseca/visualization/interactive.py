"""
Visualización de Alta Frecuencia con Datashader + Panel (Arquitectura Dinámica).
"""

from __future__ import annotations
from typing import Dict

import polars as pl
import pandas as pd
import holoviews as hv
import datashader as ds
import panel as pn 
from holoviews.operation.datashader import datashade, dynspread, spread
from holoviews import opts
from bokeh.models import HoverTool
import datetime


pn.extension('mathjax') 
hv.extension('bokeh')


def _prepare_price_data(df: pl.DataFrame) -> pd.DataFrame:
    """
    Preprocesa y categoriza los datos usando la columna 'dc_interval_type'
    generada por la nueva implementación optimizada.
    """
    df_proc = df.with_columns([
        pl.col("time").dt.replace_time_zone(None).alias("time"),
        
        # Mapeo de los nuevos Intervalos DC a las categorías visuales antiguas
        pl.when(pl.col("dc_interval_type") == "Upward Overshoot").then(pl.lit("Upward"))
          .when(pl.col("dc_interval_type") == "Downward Overshoot").then(pl.lit("Downward"))
          .when(pl.col("dc_interval_type") == "Upturn Event").then(pl.lit("Upturn"))
          .when(pl.col("dc_interval_type") == "Downturn Event").then(pl.lit("Downturn"))
          .otherwise(pl.lit("Neutral"))
          .cast(pl.Categorical)
          .alias("status_cat")
    ])
    
    return df_proc.select(["time", "price", "status_cat"]).to_pandas()


def _prepare_event_data(df: pl.DataFrame) -> pd.DataFrame:
    """
    Preprocesa y categoriza los datos para el agregador categórico de Datashader.
    """
    df_proc = df.with_columns([
        pl.col("time").dt.replace_time_zone(None).alias("time"),
    ])
    
    return df_proc.select(["time", "price", "ext_time", "ext_price", "type_desc"]).to_pandas()


def _get_vlines(df):
    # Obtener fechas mínimas y máximas
    min_date = df["time"].min()
    max_date = df["time"].max()
    
    # 1. Líneas Continuas (Medianoche - 00:00)
    # Redondeamos al inicio del día
    start_midnight = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
    midnights = []
    curr = start_midnight
    while curr <= max_date:
        midnights.append(curr)
        curr += datetime.timedelta(days=1)
        
    # 2. Líneas Punteadas (Mediodía - 12:00)
    start_noon = min_date.replace(hour=12, minute=0, second=0, microsecond=0)
    noons = []
    curr = start_noon
    while curr <= max_date:
        noons.append(curr)
        curr += datetime.timedelta(days=1)
        
    return midnights, noons


def create_dashboard_app(
    df_ticks: pl.DataFrame,
    df_events: pl.DataFrame,
    title: str = "Análisis DC (Live Server)",
    theta: float = 0.001,
    height: int = 800
) -> pn.Column:
    """
    Crea una aplicación Panel reactiva usando DynamicMap + RangeXY.
    """
    
    # 0. Configuración Hover Tool
    custom_hover = HoverTool(
        tooltips=[
            ('Fecha', '$x{%F %T}'),    # Formato: YYYY-MM-DD HH:MM:SS
            ('Precio [USDT]', '$y{0,0.00}')   # Formato: 2 decimales y separador de miles
        ],
        formatters={
            '$x': 'datetime',          # Le dice a Bokeh que x es tiempo
            '$y': 'numeral',            # Le dice a Bokeh que use formato numérico
        }
    )

    # 1. Preparación de Datos
    pdf = _prepare_price_data(df_ticks)
    edf = _prepare_event_data(df_events)

    # 2. Definición de Elementos HoloViews
    # 2.1 Pipeline Dinámico basado en Puntos (Datashader) 
    price_dots = hv.Points(pdf, ['time', 'price'], vdims=['status_cat'])
    price_shaded = spread(
        datashade(
            price_dots,
            aggregator=ds.count_cat('status_cat'),
            color_key={
                'Upward':     '#006400',
                'Upturn':   '#90ee90',
                'Downward':   '#8b0000',
                'Downturn': '#ffb6c1',
                'Neutral':    '#bdc3c7' 
            },
            min_alpha=255,
            dynamic=True
        ),
        # threshold=0.5,
        # max_px=5 
        px=1,    # <--- Radio en píxeles. px=2 crea un punto de 5x5 píxeles. px=3 es 7x7.
        shape='circle' # Opcional: Intenta redondear los bordes
    ).opts(
        xlabel="Tiempo (UTC)", 
        ylabel="Precio [USDT]",
        yformatter='%d',
        show_legend=False,
        # tools=[custom_hover, 'crosshair', 'xbox_zoom', 'reset']
    )

    # 2.2 Overshots
    # 2.2.1 Upward Ticks
    upwards_ticks = hv.Scatter(
        pdf[pdf['status_cat'] == 'Upward'], 
        kdims=['time'], 
        vdims=['price', 'status_cat']
    ).opts(
        color='#006400',
        marker='circle',
        size=2,
        show_legend=True,
    )
    # 2.2.2 Downward Ticks
    downward_ticks = hv.Scatter(
        pdf[pdf['status_cat'] == 'Downward'], 
        kdims=['time'], 
        vdims=['price', 'status_cat']
    ).opts(
        color='#8b0000',
        marker='circle',
        size=2,
        show_legend=True,
    )

    # 2.3 Critical Points
    # 2.3.1 Upturn Events
    upturn_confirmations = hv.Scatter(
        edf[edf['type_desc'] == 'upturn'], 
        kdims=['time'], 
        vdims=['price', 'type_desc']
    ).opts(
        color='#006400',
        marker='triangle',
        size=12,
        show_legend=True,
    )
    extreme_lows = hv.Scatter(
        edf[edf['type_desc'] == 'upturn'], 
        kdims=['ext_time'], 
        vdims=['ext_price', 'type_desc']
    ).opts(
        color='#90ee90',
        marker='triangle',
        size=12,
        show_legend=True,
    )

    # 2.3.2 Downturn Events
    downturn_confirmations = hv.Scatter(
        edf[edf['type_desc'] == 'downturn'], 
        kdims=['time'], 
        vdims=['price', 'type_desc']
    ).opts(
        color='#8b0000', 
        marker='inverted_triangle',
        size=12,
        show_legend=True,
    )
    extreme_highs = hv.Scatter(
        edf[edf['type_desc'] == 'downturn'], 
        kdims=['ext_time'], 
        vdims=['ext_price', 'type_desc']
    ).opts(
        color='#ffb6c1',
        marker='inverted_triangle',
        size=12,
        show_legend=True,
    )

    # Dentro de la lógica de creación del gráfico:
    midnights, noons = _get_vlines(df_ticks)

    # Líneas continuas para medianoche
    midnight_lines = hv.Overlay([hv.VLine(t).opts(color='gray', line_width=0.5) for t in midnights])

    # Líneas punteadas para mediodía
    noon_lines = hv.Overlay([hv.VLine(t).opts(color='gray', line_dash='dotted', line_width=0.5) for t in noons])

    # 4. Composición Interactiva
    plot = (
        # Base Plot
        price_shaded 

        # Overshots Ticks
        * upwards_ticks
        * downward_ticks

        # Critical Points
        * upturn_confirmations
        * extreme_lows  
        * downturn_confirmations
        * extreme_highs

        # Plot Configs
        * midnight_lines 
        * noon_lines 
    ).opts(
        opts.RGB(
            height=height, 
            responsive=True,
            bgcolor='white',
            tools=[custom_hover, 'crosshair', 'xbox_zoom', 'reset'] 
            # tools=['hover', 'crosshair', 'xbox_zoom', 'reset'] 
        )
    )
    
    # 5. Envoltura en Panel con String Matemático Robusto
    # Agregamos '\n' alrededor de la ecuación de bloque para asegurar que el parser
    # de Markdown la identifique correctamente.
    return pn.Column(
        pn.pane.Markdown(
            rf"""
            ## {title} 
            Parámetros: \\(\theta = {theta}\\)
            
            Fórmula:
            $$DC = \Delta P / P$$
            """
        ),
        plot,
        sizing_mode="stretch_width"
    )

def serve_dashboard(app, port: int = 5006):
    print(f"🚀 Iniciando servidor de visualización en http://localhost:{port}")
    pn.serve(
        app, 
        port=port, 
        # show=True, 
        threaded=True
    )