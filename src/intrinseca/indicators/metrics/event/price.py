"""
Price Metrics.

Indicators related to price movement magnitude, returns, and spatial extent.

All indicators use Silver Layer columns directly:
- extreme_price: Price at event origin (extreme point)
- extreme_time: Timestamp at event origin
- confirm_price: Price at event confirmation
- confirm_time: Timestamp at event confirmation
"""

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata
import polars as pl


class Overshoot(BaseIndicator):
    """
    Overshoot: Magnitude of price movement during the OS phase.
    
    Definition (Directional Change Literature):
        Overshoot = Next Extreme Price - Confirmation Price
        
    The Overshoot phase starts at the confirmation point (end of DC event)
    and ends at the next extreme (start of the next DC event).
    
    Note: The last event in a series will have null overshoot since
    there is no subsequent extreme to close the OS phase.
    """
    name = "overshoot"
    metadata = IndicatorMetadata(
        description="Magnitude of the OS phase (Confirmation -> Next Extreme).",
        category="event/price"
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        # extreme_price already exists in Silver (precalculated in kernel)
        # shift(-1) gets the next event's extreme = end of this OS phase
        next_ext_price = pl.col("extreme_price").shift(-1)
        return next_ext_price - pl.col("confirm_price")


class DcReturn(BaseIndicator):
    """
    DC Return: Relative price change during the DC phase.
    
    Formula: (Extreme Price - Confirm Price) / Confirm Price
    """
    name = "dc_return"
    metadata = IndicatorMetadata(
        description="Return of the DC phase (Extreme -> Confirmation).",
        category="event/price"
    )
    dependencies = []  # Uses Silver columns directly

    def get_expression(self) -> pl.Expr:
        return (pl.col("extreme_price") - pl.col("confirm_price")) / pl.col("confirm_price")


class OsReturn(BaseIndicator):
    """
    OS Return: Relative price change during the Overshoot phase.
    
    Formula: Overshoot / Confirm Price
    """
    name = "os_return"
    metadata = IndicatorMetadata(
        description="Return of the OS phase (Confirmation -> Next Extreme).",
        category="event/price"
    )
    dependencies = ["overshoot"]  # Depends on calculated Overshoot

    def get_expression(self) -> pl.Expr:
        return pl.col("overshoot") / pl.col("confirm_price")
