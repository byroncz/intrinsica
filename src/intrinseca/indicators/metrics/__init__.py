"""Metrics Package.

Contains all indicator implementations organized by category:
- event/price: Price movement magnitude and returns
- event/time: Time and velocity metrics
- event/tick: Tick-level analysis
- event/series: Signal processing (Fourier, wavelets, etc.)
- summary/stats: Summary statistics
"""

from .event.price import DcMagnitude, DcReturn, OsMagnitude, OsReturn
from .event.tick import RunsCount
from .event.time import DcTime, DcVelocity, EventTime, EventVelocity, OsTime, OsVelocity
from .summary.stats import TMV, AvgDcTime, AvgOsMagnitude, AvgReturn, UpturnRatio, VolatilityDC


def register_all(registry):
    """Registers all standard metrics to the provided registry."""
    # Event/Price (event-level)
    registry.register(DcMagnitude())
    registry.register(OsMagnitude())
    registry.register(DcReturn())
    registry.register(OsReturn())

    # Event/Time (event-level)
    registry.register(DcTime())
    registry.register(OsTime())
    registry.register(EventTime())
    registry.register(DcVelocity())
    registry.register(OsVelocity())
    registry.register(EventVelocity())

    # Event/Tick (event-level)
    registry.register(RunsCount())

    # Summary/Stats (aggregation)
    registry.register(TMV())
    registry.register(AvgDcTime())
    registry.register(AvgReturn())
    registry.register(AvgOsMagnitude())
    registry.register(VolatilityDC())
    registry.register(UpturnRatio())


__all__ = [
    # Event/Price
    "DcMagnitude",
    "OsMagnitude",
    "DcReturn",
    "OsReturn",
    # Event/Time
    "DcTime",
    "OsTime",
    "EventTime",
    "DcVelocity",
    "OsVelocity",
    "EventVelocity",
    # Event/Tick
    "RunsCount",
    # Summary/Stats
    "TMV",
    "AvgDcTime",
    "AvgReturn",
    "AvgOsMagnitude",
    "VolatilityDC",
    "UpturnRatio",
    # Registration
    "register_all",
]
