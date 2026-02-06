"""Memory Metrics.

Indicators that capture context from previous events using shift().
Used for ML feature vectors (Adegboye et al., 2017).

These indicators provide "memory" of previous events, enabling models
to detect patterns like higher highs, alternating overshoots, and flash events.
"""

import polars as pl

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata


class A4PrevDccPrice(BaseIndicator):
    """A4: DCC price of the immediately preceding event.

    Formula: A4[N] = confirm_price[N-1]

    Interpretation:
    - Enables detection of "higher highs / lower lows" patterns
    - Null for the first event (no previous event exists)

    References: Adegboye et al. (2017) - Attribute A4
    """

    name = "a4_prev_dcc_price"
    metadata = IndicatorMetadata(
        description="DCC price of previous event (A4). Null for first event.",
        category="event/memory",
    )
    dependencies = []  # Uses Silver column directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for previous DCC price."""
        return pl.col("confirm_price").shift(1)


class A5PrevOsFlag(BaseIndicator):
    """A5: Binary flag indicating if previous event had overshoot.

    Formula: A5[N] = 1 if |os_magnitude[N-1]| > 0, else 0

    Interpretation:
    - Captures alternating patterns of events with/without overshoot
    - Null for the first event (no previous event exists)

    References: Adegboye et al. (2017) - Attribute A5
    """

    name = "a5_prev_os_flag"
    metadata = IndicatorMetadata(
        description="Binary flag if previous event had overshoot (A5).",
        category="event/memory",
    )
    dependencies = ["os_magnitude"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for previous OS flag."""
        prev_os = pl.col("os_magnitude").shift(1)
        return pl.when(prev_os.abs() > 0).then(1).otherwise(0)


class A6FlashEvent(BaseIndicator):
    """A6: Flash Event flag when DC phase duration is zero.

    Formula: A6[N] = 1 if dc_time[N] = 0, else 0

    Interpretation:
    - Detects opening gaps and flash crashes
    - These events represent statistical discontinuities requiring special handling

    References: Adegboye et al. (2017) - Attribute A6
    """

    name = "a6_flash_event"
    metadata = IndicatorMetadata(
        description="Flash event flag when DC time is zero (A6).",
        category="event/memory",
    )
    dependencies = ["dc_time"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for flash event detection."""
        return pl.when(pl.col("dc_time") == 0).then(1).otherwise(0)
