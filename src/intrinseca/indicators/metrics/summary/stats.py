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


class AvgDcTime(BaseIndicator):
    """Average duration of DC phases."""

    name = "avg_dc_time"
    metadata = IndicatorMetadata(
        description="Average duration of DC phases in nanoseconds.",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = ["dc_time"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for average DC time."""
        return pl.col("dc_time").mean()


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


class AvgOsMagnitude(BaseIndicator):
    """Average OS magnitude (overshoot)."""

    name = "avg_os_magnitude"
    metadata = IndicatorMetadata(
        description="Average OS magnitude.", category="summary/stats", is_event_level=False
    )
    dependencies = ["os_magnitude"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for average OS magnitude."""
        return pl.col("os_magnitude").mean()


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
