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
    Prepara datos de eventos para Panel A (visualización intrínseca).
    
    Genera datos para cajas bi-particionadas según definiciones Tsang:
    - DC Event N (change): ext_price(N) → price(N) [extremo → confirmación]
    - Overshoot N: price(N) → next_ext_price [confirmación → extremo siguiente]
    
    Columnas generadas:
    - ext_price: Extremo del evento N (inicio del DC Event N)
    - price: Confirmación del evento N (fin DC Event N, inicio Overshoot N)
    - next_ext_price: Extremo del evento N+1 (fin del Overshoot N)
    
    Nota: Usamos row_nr como índice secuencial (0,1,2,...N).
    """
    # Añadir índice secuencial de eventos
    df_with_seq = df_events.with_row_index("seq_idx")
    
    # Calcular next_ext_price = ext_price del evento N+1 (para Overshoot N)
    # Usamos shift(-1) para obtener el valor de la fila siguiente
    df_segments = df_with_seq.with_columns([
        pl.col("ext_price").shift(-1).alias("next_ext_price")
    ])
    
    # La última fila no tiene next_ext_price (no hay evento N+1)
    # Usamos fill_null con forward para el último registro
    df_segments = df_segments.with_columns([
        pl.col("next_ext_price").fill_null(strategy="forward")
    ])
    
    pdf_segments = df_segments.select([
        pl.col("seq_idx"),
        pl.col("ext_price"),       # Extremo del evento N (inicio DC Event N)
        pl.col("price"),           # Confirmación del evento N (fin DC Event N)
        pl.col("next_ext_price"),  # Extremo del evento N+1 (fin Overshoot N)
        pl.col("type_desc"),
        pl.col("overshoot").cast(pl.Boolean).alias("is_overshoot"),
        pl.col("time"),            # Tiempo de confirmación
        pl.col("ext_time"),        # Tiempo del extremo
        pl.col("event_idx")        # Índice del tick de confirmación
    ]).to_pandas()
    
    return pdf_segments