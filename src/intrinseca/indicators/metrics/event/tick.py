"""Tick Metrics.

Indicators related to tick-level dynamics, runs, and volume profiles.
"""

import polars as pl

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata


def _count_runs(prices: list, event_type: int, theta: float = 0.005) -> int:
    """Count directional grid crossings during the event.

    Args:
    ----
        prices: List of prices during the OS phase
        event_type: 1 for upturn, -1 for downturn
        theta: Threshold for grid crossing (default 0.5%)

    Returns:
    -------
        Number of runs (grid crossings)

    """
    if prices is None or len(prices) < 2:
        return 0

    ref = prices[0]
    mult = (1.0 + theta) if event_type == 1 else (1.0 - theta)
    count = 0

    for p in prices[1:]:
        threshold = ref * mult
        if (event_type == 1 and p >= threshold) or (event_type == -1 and p <= threshold):
            count += 1
            ref = p

    return count


class RunsCount(BaseIndicator):
    """Count directional grid crossings (runs) during the OS phase.

    A "run" occurs when the price moves at least theta in the direction
    of the trend from the last reference point.

    Note: This indicator requires theta as a parameter to match the
    threshold used in DC processing.
    """

    name = "runs_count"
    metadata = IndicatorMetadata(
        description="Number of directional grid crossings during the event.",
        category="event/tick",
    )
    dependencies = []

    def __init__(self, theta: float = 0.005):
        """Initialize Runs Count indicator.

        Args:
            theta: DC threshold for grid crossing (default: 0.5%)
        """
        self.theta = theta

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for runs count calculation."""
        theta = self.theta
        return pl.struct(["price_os", "event_type"]).map_elements(
            lambda row: _count_runs(row["price_os"], row["event_type"], theta),
            return_dtype=pl.Int64,
        )
