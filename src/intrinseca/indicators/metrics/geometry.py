"""
Geometry Metrics.

Indicators related to price movement magnitude, returns, and spatial extent.
"""

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata
import polars as pl

class Overshoot(BaseIndicator):
    name = "overshoot"
    metadata = IndicatorMetadata(
        description="Magnitude of the price movement during the DC phase (Extreme -> Confirmation).",
        category="geometry"
    )

    def get_expression(self) -> pl.Expr:
        # Overshoot = Extreme Price - Confirmation Price
        # price_dc.first() = extreme price (start of DC phase)
        # price_os.first() = confirmation price (end of DC phase, start of OS phase)
        return pl.col("price_dc").list.first() - pl.col("price_os").list.first()

class DcReturn(BaseIndicator):
    name = "return"
    metadata = IndicatorMetadata(
        description="Return of the DC phase (Extreme -> Confirmation).",
        category="geometry"
    )
    dependencies = ["ext_price", "price"]

    def get_expression(self) -> pl.Expr:
        # Return = (Extreme Price - Confirmation Price) / Confirmation Price
        # Legacy calculates (Ext - Conf) / Conf
        # ext_price = extreme price, price = confirmation price
        return (pl.col("ext_price") - pl.col("price")) / pl.col("price")

class OsReturn(BaseIndicator):
    name = "os_return"
    metadata = IndicatorMetadata(
        description="Return of the OS phase (Confirmation -> Next Extreme).",
        category="geometry"
    )
    dependencies = ["overshoot", "price"]

    def get_expression(self) -> pl.Expr:
        return pl.col("overshoot") / pl.col("price")
