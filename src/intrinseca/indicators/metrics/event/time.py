"""Time Metrics.

Indicators related to time, velocity, and speed of events.

All indicators use Silver Layer columns directly:
- extreme_time: Timestamp at event origin (nanoseconds)
- confirm_time: Timestamp at event confirmation (nanoseconds)
- extreme_price: Price at event origin
- confirm_price: Price at event confirmation
"""

import polars as pl

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata


class Duration(BaseIndicator):
    """Duration: Time elapsed during the DC phase.

    Formula: Confirm Time - Extreme Time (in nanoseconds)
    """

    name = "duration_ns"
    metadata = IndicatorMetadata(
        description="Duration of the DC phase (Extreme -> Confirmation) in nanoseconds.",
        category="event/time",
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for duration calculation."""
        return pl.col("confirm_time") - pl.col("extreme_time")


class Velocity(BaseIndicator):
    """Velocity: Speed of price change during the DC phase.

    Formula: (Confirm Price - Extreme Price) / Duration (in seconds)
    """

    name = "velocity"
    metadata = IndicatorMetadata(
        description="Speed of the DC price change (Price Change / Duration).", category="event/time"
    )
    dependencies = ["duration_ns"]  # Needs duration calculated first

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for velocity calculation."""
        duration_sec = pl.col("duration_ns") / 1_000_000_000.0
        price_change = pl.col("confirm_price") - pl.col("extreme_price")

        return pl.when(duration_sec > 0).then(price_change / duration_sec).otherwise(0.0)
