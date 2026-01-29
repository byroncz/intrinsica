"""
Dynamics Metrics.

Indicators related to time, velocity, and speed of events.
"""

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata
import polars as pl

class Duration(BaseIndicator):
    name = "duration_ns"
    metadata = IndicatorMetadata(
        description="Duration of the DC phase (Extreme -> Confirmation) in nanoseconds.",
        category="dynamics"
    )

    def get_expression(self) -> pl.Expr:
        # Duration = Confirmation timestamp - Extreme timestamp
        # time_os.first() = confirmation timestamp (start of OS phase)
        # time_dc.first() = extreme timestamp (start of DC phase)
        return pl.col("time_os").list.first() - pl.col("time_dc").list.first()

class Velocity(BaseIndicator):
    name = "velocity"
    metadata = IndicatorMetadata(
        description="Speed of the DC price change (Price Change / Duration).",
        category="dynamics"
    )
    dependencies = ["duration_ns", "price", "ext_price"]

    def get_expression(self) -> pl.Expr:
        # Velocity = (Price - ExtPrice) / (Duration in seconds)
        # Duration is in ns. 
        duration_sec = pl.col("duration_ns") / 1_000_000_000.0
        price_change = pl.col("price") - pl.col("ext_price")
        
        return pl.when(duration_sec > 0)\
                 .then(price_change / duration_sec)\
                 .otherwise(0.0)
