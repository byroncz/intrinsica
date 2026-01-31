"""Summary Statistics.

Global indicators that collapse the event DataFrame into summary statistics.
These metrics are computed using select() rather than with_columns().
"""

import polars as pl

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata


class TMV(BaseIndicator):
    """Total Movement Value - sum of absolute DC returns."""

    name = "tmv"
    metadata = IndicatorMetadata(
        description="Total Movement Value - sum of absolute returns.",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = ["dc_return"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for TMV calculation."""
        return pl.col("dc_return").abs().sum()


class AvgDuration(BaseIndicator):
    """Average duration of DC phases."""

    name = "avg_duration"
    metadata = IndicatorMetadata(
        description="Average duration of DC phases in nanoseconds.",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = ["duration_ns"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for average duration."""
        return pl.col("duration_ns").mean()


class AvgReturn(BaseIndicator):
    """Average return of DC phases."""

    name = "avg_return"
    metadata = IndicatorMetadata(
        description="Average return of DC phases.", category="summary/stats", is_event_level=False
    )
    dependencies = ["dc_return"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for average return."""
        return pl.col("dc_return").mean()


class AvgOvershoot(BaseIndicator):
    """Average overshoot magnitude."""

    name = "avg_overshoot"
    metadata = IndicatorMetadata(
        description="Average overshoot magnitude.", category="summary/stats", is_event_level=False
    )
    dependencies = ["overshoot"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for average overshoot."""
        return pl.col("overshoot").mean()


class VolatilityDC(BaseIndicator):
    """Volatility measured as standard deviation of DC returns."""

    name = "volatility_dc"
    metadata = IndicatorMetadata(
        description="Volatility measured as standard deviation of DC returns.",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = ["dc_return"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for volatility."""
        return pl.col("dc_return").std()


class UpturnRatio(BaseIndicator):
    """Ratio of upturn events to total events."""

    name = "upturn_ratio"
    metadata = IndicatorMetadata(
        description="Ratio of upturn events to total events.",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = []  # Uses Silver column directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for upturn ratio."""
        return (pl.col("event_type") == 1).mean()
