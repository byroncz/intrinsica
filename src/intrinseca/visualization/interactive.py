"""
Visualización de Alta Frecuencia con Datashader + Panel (Arquitectura Dinámica).
Basado en: Propuesta Arquitectónica Integral para Visualización Financiera.
"""

from __future__ import annotations
from typing import Dict

import polars as pl
import pandas as pd
import holoviews as hv
import datashader as ds
import panel as pn # Crítico: Motor de servidor para actualizaciones en tiempo real
from holoviews.operation.datashader import datashade, dynspread
from holoviews import opts

# Inicializar extensión con Panel habilitado
hv.extension('bokeh')
pn.extension() # Inicializa el backend de Panel

def _prepare_data_for_datashader(df: pl.DataFrame) -> pd.DataFrame:
    """
    Preprocesa y categoriza los datos para el agregador categórico de Datashader.
    Prioridad de Color: Run > Trend > Base.
    """
    # Lógica vectorizada en Polars
    # Eliminamos la zona horaria para compatibilidad con Datashader/Numba
    df_proc = df.with_columns([
        pl.col("time").dt.replace_time_zone(None).alias("time"),
        
        pl.when(pl.col("dc_run_flag") > 0).then(pl.lit("Run_Up"))
          .when(pl.col("dc_run_flag") < 0).then(pl.lit("Run_Down"))
          .when(pl.col("dc_trend") == 1).then(pl.lit("Trend_Up"))
          .when(pl.col("dc_trend") == -1).then(pl.lit("Trend_Down"))
          .otherwise(pl.lit("Neutral"))
          .cast(pl.Categorical)
          .alias("status_cat")
    ])
    
    return df_proc.select(["time", "price", "status_cat"]).to_pandas()

def create_dashboard_app(
    df_ticks: pl.DataFrame,
    title: str = "Análisis DC (Live Server)",
    height: int = 800
) -> pn.Column:
    """
    Crea una aplicación Panel reactiva.
    
    Arquitectura DynamicMap + RangeXY[cite: 32, 39]:
    A diferencia de un HTML estático, esta función devuelve un objeto que se suscribe
    al stream del rango de ejes (RangeXY). Cada evento de zoom dispara
    un re-renderizado en el servidor.
    """
    
    # 1. Preparación de Datos (Persistencia en RAM)
    pdf = _prepare_data_for_datashader(df_ticks)
    
    # Definir Claves de Color (Estilo DC) [cite: 69]
    color_key: Dict[str, str] = {
        'Run_Up':     '#006400',
        'Run_Down':   '#8b0000',
        'Trend_Up':   '#90ee90',
        'Trend_Down': '#ffb6c1',
        'Neutral':    '#bdc3c7' 
    }

    # 2. Definición de Elementos HoloViews
    # Estos son los objetos semánticos puros, no la imagen [cite: 32]
    curve = hv.Curve(pdf, 'time', 'price')
    points = hv.Points(pdf, ['time', 'price'], vdims=['status_cat'])

    # 3. Pipeline Dinámico (Datashader)
    # datashade() crea implícitamente un DynamicMap que escucha cambios de zoom.
    # Al estar dentro de Panel, este DynamicMap se mantiene vivo.
    
    # A. Precio (Línea Sólida, No-Heatmap) [cite: 49, 66]
    shader_price = datashade(
        curve,
        cmap=['#bdc3c7'], 
        min_alpha=255, # Opacidad total forzada
        dynamic=True   # Habilita el re-cálculo por streams (Default en datashade)
    ).opts(
        # title=title,
        xlabel="Tiempo (UTC)", 
        ylabel="Precio"
    )

    # B. Estados (Puntos Expandidos) [cite: 100, 102]
    shader_status = dynspread(
        datashade(
            points,
            aggregator=ds.count_cat('status_cat'),
            color_key=color_key,
            min_alpha=255,
            dynamic=True
        ),
        threshold=0.5,
        max_px=5
    )

    # 4. Composición Interactiva [cite: 161]
    # Tools: 'xbox_zoom' es ideal para series temporales (zoom solo en eje X)
    plot = (shader_price * shader_status).opts(
        opts.RGB(
            height=height, 
            responsive=True, # Se adapta al ancho del navegador
            bgcolor='white',
            tools=['hover', 'crosshair', 'xbox_zoom', 'reset'] 
        )
    )
    
    # 5. Envoltura en Panel
    # Retornamos un layout de Panel, no un objeto HoloViews plano.
    # Esto prepara el "documento" para ser servido.
    return pn.Column(
        pn.pane.Markdown(f"## {title}"),
        plot,
        sizing_mode="stretch_width"
    )

def serve_dashboard(app, port: int = 5006):
    """
    Inicia el servidor WebSocket local.
    Sustituye a 'save_chart'. Bloquea la terminal mientras corre.
    """
    print(f"🚀 Iniciando servidor de visualización en http://localhost:{port}")
    print("Para detener: Ctrl+C")
    pn.serve(
        app, 
        port=port, 
        show=True, # Abre el navegador automáticamente
        threaded=True
    )