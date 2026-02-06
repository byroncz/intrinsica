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


class Ndc(BaseIndicator):
    """NDC: Number of Directional Change events.

    Formula: NDC = count()

    Fundamental measure of volatility in intrinsic time.
    Higher NDC with same theta indicates a more "nervous" market.

    References: Guillaume et al. (1997), Aloud et al. (2012)
    """

    name = "ndc"
    metadata = IndicatorMetadata(
        description="Number of Directional Change events.",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = []

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for NDC calculation."""
        return pl.len()


class Cdc(BaseIndicator):
    """CDC (Coastline): Sum of absolute TMV values.

    Formula: CDC = sum(|tmv_event|)

    Inspired by Mandelbrot's fractal coastline paradox.
    Represents the total "energy" dissipated by the market
    and the maximum theoretical return if every trend were captured.

    References: Glattfelder et al. (2011)
    """

    name = "cdc"
    metadata = IndicatorMetadata(
        description="Coastline: sum of absolute TMV (fractal energy).",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = ["tmv_event"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for CDC calculation."""
        return pl.col("tmv_event").abs().sum()


class AccumulatedTime(BaseIndicator):
    """AT: Accumulated Time asymmetry between upturns and downturns.

    Formula: AT = sum(dc_time for upturns) - sum(dc_time for downturns)

    Interpretation:
    - Positive AT: Uptrends are slower than downtrends
    - Negative AT: Downtrends are slower than uptrends
    - Useful for detecting directional bias in market dynamics

    References: Kampouridis (2025)
    """

    name = "accumulated_time"
    metadata = IndicatorMetadata(
        description="Time asymmetry: sum(upturn time) - sum(downturn time).",
        category="summary/stats",
        is_event_level=False,
    )
    dependencies = ["dc_time"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for accumulated time calculation."""
        uptime = pl.when(pl.col("event_type") == 1).then(pl.col("dc_time")).otherwise(0).sum()
        downtime = pl.when(pl.col("event_type") == -1).then(pl.col("dc_time")).otherwise(0).sum()
        return uptime - downtime
