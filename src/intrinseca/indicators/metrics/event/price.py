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
