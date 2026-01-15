import polars as pl
import pandas as pd
from datetime import timedelta

def _prepare_price_data(df: pl.DataFrame) -> pd.DataFrame:
    df_proc = df.with_columns([
        pl.col("time").dt.replace_time_zone(None).alias("time"),
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
    df_proc = df.with_columns([
        pl.col("time").dt.replace_time_zone(None).alias("time"),
    ])
    return df_proc.select(["time", "price", "ext_time", "ext_price", "type_desc"]).to_pandas()

def _get_vlines(df: pl.DataFrame):
    min_date = df["time"].min()
    max_date = df["time"].max()
    
    # Medianoches
    start_midnight = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
    midnights = []
    curr = start_midnight
    while curr <= max_date:
        midnights.append(curr)
        curr += timedelta(days=1)
        
    # Mediodías
    start_noon = min_date.replace(hour=12, minute=0, second=0, microsecond=0)
    noons = []
    curr = start_noon
    while curr <= max_date:
        noons.append(curr)
        curr += timedelta(days=1)
        
    return midnights, noons


def prepare_dual_axis_data(df_events: pl.DataFrame):
    """
    Prepara solo los eventos (Panel A). 
    Los ticks se procesarán de forma dinámica después.
    
    Nota: Usamos row_nr como índice secuencial de eventos (0,1,2,...N)
    en lugar de event_idx que es el índice del tick donde ocurrió.
    """
    # Añadir índice secuencial de eventos
    df_with_seq = df_events.with_row_index("seq_idx")
    
    # Panel Intrínseco: Basado en índices secuenciales
    df_segments = df_with_seq.with_columns([
        pl.col("price").shift(1).alias("prev_price")
    ]).fill_null(strategy="backward")
    
    pdf_segments = df_segments.select([
        pl.col("seq_idx").alias("x0"),
        pl.col("prev_price").alias("y0"),
        (pl.col("seq_idx") + 1).alias("x1"),
        pl.col("price").alias("y1"),
        pl.col("type_desc"),
        pl.col("overshoot").cast(pl.Boolean).alias("is_overshoot"),
        pl.col("time"),  # Preservar para mapeo
        pl.col("event_idx")  # Preservar para referencia
    ]).to_pandas()
    
    return pdf_segments