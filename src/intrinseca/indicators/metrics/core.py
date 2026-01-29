"""
Core Metrics.

Basic extractions from the raw Silver events (Price, Time, Type).
"""

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata
import polars as pl

class EventTime(BaseIndicator):
    name = "time"
    metadata = IndicatorMetadata(
        description="Timestamp of event confirmation (End of DC / Start of OS).",
        category="core"
    )

    def get_expression(self) -> pl.Expr:
        # The event time is the last timestamp of the DC phase
        # Or the first timestamp of the OS phase?
        # Definition: Confirmation point.
        # In Silver: dc_times lists timestamps from Extreme to Confirmation (exclusive of T-confirmation?)
        # Let's look at kernel.py:
        # dc_times[dc_ptr] = timestamps[i] for i in range(prev_ext_idx, t)
        # where t is confirmation index.
        # So dc_times contains points UP TO confirmation.
        # The confirmation happens AT 't'.
        # However, Silver event row structure might implicitly store the confirmation info.
        # Let's assume for now the 'time' column in df_events represents the confirmation time.
        # If we look at existing logic: `event_timestamps[tao] = ts_val` where ts_val is at `t`.
        # In Silver, `time_dc` ends right before `t`.
        # BUT, wait. `price_dc` is `prev_ext_idx` to `t`.
        # So `price_dc.last()` is `prices[t-1]`.
        # The confirmation price `p` at `t` is NOT in `price_dc`?
        # Let's check kernel.py carefully.
        # `dc_prices` stores `prices[i]` for `i` in `prev_ext_idx` to `t`.
        # So it stores up to `t-1`.
        # The price AT `t` (confirmation) is `last_os_ref` for the NEXT checks, but for THIS event...
        # In `event_detector.py`: `event_prices[tao] = p` (price at t).
        # In Silver `kernel.py` logic: validation code might be useful.
        # The Silver data seems to store the WAVE (DC phase).
        # The confirmation point `t` is actually the START of the Overshoot phase.
        # So `time` should be `time_os.first()`.
        # Let's verify `kernel.py`: 
        # `prev_os_start = t`.
        # `os_times` stores `range(prev_os_start, ...)` -> `t` is the first element of OS.
        return pl.col("time_os").list.first()

class EventPrice(BaseIndicator):
    name = "price"
    metadata = IndicatorMetadata(
        description="Price at event confirmation.",
        category="core"
    )

    def get_expression(self) -> pl.Expr:
        # Based on logic above: Confirmation Price is the first price of the OS phase.
        return pl.col("price_os").list.first()

class EventTypeDesc(BaseIndicator):
    name = "type_desc"
    metadata = IndicatorMetadata(
        description="Text description of event type (upturn/downturn).",
        category="core"
    )

    def get_expression(self) -> pl.Expr:
        # event_type is 1 or -1
        return pl.when(pl.col("event_type") == 1)\
                 .then(pl.lit("upturn"))\
                 .otherwise(pl.lit("downturn"))

class ExtremePrice(BaseIndicator):
    name = "ext_price"
    metadata = IndicatorMetadata(
        description="Price at the event origin (The Extreme point).",
        category="core"
    )
    
    def get_expression(self) -> pl.Expr:
        # The Extreme is the START of the DC phase.
        return pl.col("price_dc").list.first()

class ExtremeTime(BaseIndicator):
    name = "ext_time"
    metadata = IndicatorMetadata(
        description="Timestamp at the event origin (The Extreme point).",
        category="core"
    )
    
    def get_expression(self) -> pl.Expr:
        return pl.col("time_dc").list.first()
