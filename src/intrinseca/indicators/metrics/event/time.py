"""Time Metrics.

Indicators related to time, velocity, and duration of DC/OS phases.

All indicators use Silver Layer columns directly:
- reference_time: Timestamp at DC phase start (nanoseconds)
- confirm_time: Timestamp at DC phase end / OS phase start (nanoseconds)
- extreme_time: Timestamp at OS phase end (nanoseconds)
- reference_price, confirm_price: Prices at corresponding timestamps
"""

import polars as pl

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata


class DcTime(BaseIndicator):
    """DC Time: Duration of the DC phase in nanoseconds.

    Formula: confirm_time[N] - reference_time[N]

    The DC phase starts at reference_time (= extreme_time of previous event)
    and ends at confirm_time (DCC). Both are in the same row.

    Equivalent to A2 (DCtime) in Adegboye et al. (2017).
    """

    name = "dc_time"
    metadata = IndicatorMetadata(
        description="Duration of the DC phase (Reference -> DCC) in nanoseconds.",
        category="event/time",
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for DC time calculation."""
        return pl.col("confirm_time") - pl.col("reference_time")


class OsTime(BaseIndicator):
    """OS Time: Duration of the Overshoot phase in nanoseconds.

    Formula: extreme_time[N] - confirm_time[N]

    The OS phase starts at confirm_time (DCC) and ends at extreme_time
    (the point where the next DC is triggered). Both are in the same row.

    Note: For the last event, extreme_time may be provisional (-1),
    resulting in an invalid (negative) os_time.
    """

    name = "os_time"
    metadata = IndicatorMetadata(
        description="Duration of the OS phase (DCC -> Extreme) in nanoseconds.",
        category="event/time",
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for OS time calculation."""
        return pl.col("extreme_time") - pl.col("confirm_time")


class EventTime(BaseIndicator):
    """Event Time: Total duration of a DC event (DC + OS phases).

    Formula: dc_time[N] + os_time[N] = extreme_time[N] - reference_time[N]

    Represents the complete lifecycle of an event from the reference point
    to the extreme point.
    """

    name = "event_time"
    metadata = IndicatorMetadata(
        description="Total event duration (DC + OS phases) in nanoseconds.",
        category="event/time",
    )
    dependencies = ["dc_time", "os_time"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for total event time."""
        return pl.col("dc_time") + pl.col("os_time")


class DcVelocity(BaseIndicator):
    """DC Velocity: Speed of price change during the DC phase.

    Formula: dc_magnitude[N] / dc_time[N]
           = (confirm_price[N] - reference_price[N]) / dc_time[N]

    Measures the rate of price change per unit time during the DC phase.
    Positive for upturns, negative for downturns.

    Equivalent to A3 (σ₀) in Adegboye et al. (2017).
    """

    name = "dc_velocity"
    metadata = IndicatorMetadata(
        description="Speed of DC phase price change (DC magnitude / DC time).",
        category="event/time",
    )
    dependencies = ["dc_time", "dc_magnitude"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for DC velocity calculation."""
        dc_time_sec = pl.col("dc_time") / 1_000_000_000.0

        return pl.when(dc_time_sec > 0).then(pl.col("dc_magnitude") / dc_time_sec).otherwise(0.0)


class OsVelocity(BaseIndicator):
    """OS Velocity: Speed of price change during the Overshoot phase.

    Formula: os_magnitude[N] / os_time[N]
           = (extreme_price[N] - confirm_price[N]) / os_time[N]

    Measures the rate of price change per unit time during the OS phase.
    Positive for upturns, negative for downturns.

    Note: For OS time = 0 (instant overshoot), returns 0.0.
    """

    name = "os_velocity"
    metadata = IndicatorMetadata(
        description="Speed of OS phase price change (OS magnitude / OS time).",
        category="event/time",
    )
    dependencies = ["os_time", "os_magnitude"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for OS velocity calculation."""
        os_time_sec = pl.col("os_time") / 1_000_000_000.0

        return pl.when(os_time_sec > 0).then(pl.col("os_magnitude") / os_time_sec).otherwise(0.0)


class EventVelocity(BaseIndicator):
    """Event Velocity: Speed of price change for the complete event.

    Formula: event_magnitude[N] / event_time[N]
           = (extreme_price[N] - reference_price[N]) / event_time[N]

    Measures the rate of total price change per unit time across
    both DC and OS phases.
    """

    name = "event_velocity"
    metadata = IndicatorMetadata(
        description="Speed of total event price change (Event magnitude / Event time).",
        category="event/time",
    )
    dependencies = ["event_time", "event_magnitude"]

    def get_expression(self) -> pl.Expr:
        """Return Polars expression for event velocity calculation."""
        event_time_sec = pl.col("event_time") / 1_000_000_000.0

        return (
            pl.when(event_time_sec > 0)
            .then(pl.col("event_magnitude") / event_time_sec)
            .otherwise(0.0)
        )
