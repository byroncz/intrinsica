"""Price Metrics.

Indicators related to price movement magnitude, returns, and spatial extent.

All indicators use Silver Layer columns directly:
- reference_price: Price at DC phase origin (= extreme_price of previous event)
- extreme_price: Price at OS phase end (last tick of price_os)
- confirm_price: Price at DC phase end / OS phase start (DCC, last tick of price_dc)

Temporal structure of event N:
    reference_price[N] → DC phase → confirm_price[N] → OS phase → extreme_price[N]
                                                                         ↓
                                                              = reference_price[N+1]
"""

import polars as pl

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata


class DcMagnitude(BaseIndicator):
    """DC Magnitude: Absolute price change during the DC phase.

    Definition (Adegboye et al., 2017 - Attribute A1):
        DC Magnitude[N] = confirm_price[N] - reference_price[N]

    This is the raw (non-normalized) price change during the DC phase.
    Equivalent to A1 (DC magnitude) in the canonical taxonomy.

    Sign indicates direction:
    - Positive for upturns (confirm_price > reference_price)
    - Negative for downturns (confirm_price < reference_price)

    Note: |DC Magnitude| / reference_price = |DC Return|
    """

    name = "dc_magnitude"
    metadata = IndicatorMetadata(
        description="Absolute price change during DC phase (A1). Reference -> DCC.",
        category="event/price",
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for DC magnitude calculation."""
        return pl.col("confirm_price") - pl.col("reference_price")


class OsMagnitude(BaseIndicator):
    """OS Magnitude: Absolute price change during the Overshoot phase.

    Definition (Directional Change Literature):
        OS Magnitude[N] = extreme_price[N] - confirm_price[N]

    The Overshoot phase starts at the confirmation point (DCC) and ends at
    the extreme point of the same event. Both values are in the same row:
    - confirm_price[N]: DCC, last tick of price_dc[N]
    - extreme_price[N]: Last tick of price_os[N], filled retrospectively
                        when event N+1 is confirmed

    Note: extreme_price[N] = reference_price[N+1] by construction.
    The last event may have extreme_price = -1.0 (provisional) if the
    OS phase has not been closed by a subsequent confirmation.
    """

    name = "os_magnitude"
    metadata = IndicatorMetadata(
        description="Absolute price change during OS phase. DCC -> Extreme.",
        category="event/price",
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for OS magnitude calculation."""
        # Both extreme_price and confirm_price belong to the same event row.
        return pl.col("extreme_price") - pl.col("confirm_price")


class DcReturn(BaseIndicator):
    """DC Return: Relative price change during the DC phase.

    Formula: dc_magnitude[N] / reference_price[N]
           = (confirm_price[N] - reference_price[N]) / reference_price[N]

    This measures the return from the start of the DC phase (reference)
    to the end of the DC phase (confirmation/DCC). The sign indicates direction:
    - Positive for upturns (confirm_price > reference_price)
    - Negative for downturns (confirm_price < reference_price)

    By construction: |DC Return| >= theta (the DC threshold).
    """

    name = "dc_return"
    metadata = IndicatorMetadata(
        description="Relative return of the DC phase. Reference -> DCC.",
        category="event/price",
    )
    dependencies = ["dc_magnitude"]  # Reutiliza dc_magnitude calculado

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for DC return calculation."""
        return pl.col("dc_magnitude") / pl.col("reference_price")


class OsReturn(BaseIndicator):
    """OS Return: Relative price change during the Overshoot phase.

    Formula: os_magnitude[N] / confirm_price[N]
           = (extreme_price[N] - confirm_price[N]) / confirm_price[N]

    This measures the return from the DCC to the end of the OS phase,
    both within the same event row.
    """

    name = "os_return"
    metadata = IndicatorMetadata(
        description="Relative return of the OS phase. DCC -> Extreme.",
        category="event/price",
    )
    dependencies = ["os_magnitude"]  # Depends on calculated OS Magnitude

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for OS return calculation."""
        return pl.col("os_magnitude") / pl.col("confirm_price")


class EventMagnitude(BaseIndicator):
    """Event Magnitude: Total price change across the complete event.

    Formula: dc_magnitude[N] + os_magnitude[N]
           = (confirm_price - reference_price) + (extreme_price - confirm_price)
           = extreme_price[N] - reference_price[N]

    This is the raw (non-normalized) total price movement from the start
    of the DC phase to the end of the OS phase.

    Sign indicates direction:
    - Positive for upturns
    - Negative for downturns

    Note: EventMagnitude = DC Magnitude + OS Magnitude by construction.
    """

    name = "event_magnitude"
    metadata = IndicatorMetadata(
        description="Total price change across the complete event. Reference -> Extreme.",
        category="event/price",
    )
    dependencies = ["dc_magnitude", "os_magnitude"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for event magnitude calculation."""
        return pl.col("dc_magnitude") + pl.col("os_magnitude")


class DcSlippage(BaseIndicator):
    """DC Slippage (Facial): Difference between actual and theoretical confirmation price.

    Formula:
        For upturn:   slippage = confirm_price - reference_price × (1 + θ)
        For downturn: slippage = confirm_price - reference_price × (1 - θ)

    Combined formula using event_type:
        slippage = confirm_price - reference_price × (1 + event_type × θ)

    Where:
    - confirm_price: Actual price at DCC (conservative selection)
    - reference_price: Extreme price of previous event (= EXT)
    - θ: DC threshold used in processing
    - event_type: +1 for upturn, -1 for downturn

    Units: Price units of the underlying asset.

    Interpretation:
    - Positive slippage: Price overshot the theoretical threshold
    - Slippage magnitude indicates market gap/jump at confirmation
    - In continuous markets, slippage approaches zero
    - High slippage indicates discrete jumps (gaps, flash events)

    Note: This is "facial" slippage because it uses the conservatively selected
    confirmation price. "Real" slippage (using the worst price at confirmation
    instant) requires kernel modifications to capture alternative prices.
    """

    name = "dc_slippage"
    metadata = IndicatorMetadata(
        description="Slippage: actual vs theoretical confirmation price.",
        category="event/price",
    )
    dependencies = []  # Uses Silver columns directly

    def __init__(self, theta: float = 0.005):
        """Initialize DC Slippage indicator.

        Args:
            theta: DC threshold used in processing (default: 0.5%)
        """
        self.theta = theta

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for DC slippage calculation."""
        # Theoretical confirmation price: reference_price × (1 + event_type × θ)
        # For upturn (+1):  reference_price × (1 + θ)
        # For downturn (-1): reference_price × (1 - θ)
        theoretical_confirm = pl.col("reference_price") * (
            1.0 + pl.col("event_type").cast(pl.Float64) * self.theta
        )

        return pl.col("confirm_price") - theoretical_confirm


def _compute_worst_confirm_price(
    price_dc: list[float] | None,
    time_dc: list[int] | None,
    confirm_time: int,
    event_type: int,
) -> float | None:
    """Find the worst (most distant from threshold) price at confirmation instant.

    For upturn (+1): worst = maximum price (furthest above threshold)
    For downturn (-1): worst = minimum price (furthest below threshold)

    Args:
        price_dc: List of prices during DC phase
        time_dc: List of timestamps during DC phase
        confirm_time: Timestamp of confirmation (last timestamp in DC)
        event_type: +1 for upturn, -1 for downturn

    Returns:
        Worst price at confirmation instant, or None if no prices found
    """
    if price_dc is None or time_dc is None or len(price_dc) == 0:
        return None

    # Filter prices at confirmation timestamp
    prices_at_confirm = [p for p, t in zip(price_dc, time_dc) if t == confirm_time]

    if not prices_at_confirm:
        return None

    # For upturn: max price is worst (furthest from lower threshold)
    # For downturn: min price is worst (furthest from upper threshold)
    if event_type == 1:
        return max(prices_at_confirm)
    else:
        return min(prices_at_confirm)


class DcSlippageReal(BaseIndicator):
    """DC Slippage (Real): Worst-case slippage using all prices at confirmation instant.

    Unlike DcSlippage (Facial) which uses the conservatively selected confirm_price,
    this indicator finds the WORST price among all ticks at the confirmation timestamp
    and calculates the slippage from that worst-case scenario.

    Formula:
        worst_price = max(prices at confirm_time) for upturn
        worst_price = min(prices at confirm_time) for downturn
        slippage_real = worst_price - reference_price × (1 + event_type × θ)

    Interpretation:
    - Measures the maximum possible slippage a trader could have experienced
    - Difference between Real and Facial slippage indicates price dispersion
      at the confirmation instant (market microstructure noise)
    - If Real == Facial, there was only one price at confirmation instant

    Requirements:
    - Kernel must include ALL ticks of the confirmation instant in price_dc
      (fixed in kernel.py via last_same_ts_idx correction)

    Units: Price units of the underlying asset.
    """

    name = "dc_slippage_real"
    metadata = IndicatorMetadata(
        description="Worst-case slippage using all prices at confirmation instant.",
        category="event/price",
    )
    dependencies = []  # Uses Silver columns directly

    def __init__(self, theta: float = 0.005):
        """Initialize DC Slippage Real indicator.

        Args:
            theta: DC threshold used in processing (default: 0.5%)
        """
        self.theta = theta

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for real DC slippage calculation."""
        theta = self.theta

        # Use struct to pass multiple columns to map_elements
        worst_price = pl.struct(
            ["price_dc", "time_dc", "confirm_time", "event_type"]
        ).map_elements(
            lambda row: _compute_worst_confirm_price(
                row["price_dc"],
                row["time_dc"],
                row["confirm_time"],
                row["event_type"],
            ),
            return_dtype=pl.Float64,
        )

        # Theoretical confirmation price
        theoretical_confirm = pl.col("reference_price") * (
            1.0 + pl.col("event_type").cast(pl.Float64) * theta
        )

        return worst_price - theoretical_confirm
