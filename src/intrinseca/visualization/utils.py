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