"""Metrics Package.

Contains all indicator implementations organized by category:
- event/price: Price movement magnitude and returns
- event/time: Time and velocity metrics
- event/tick: Tick-level analysis
- event/series: Signal processing (Fourier, wavelets, etc.)
- summary/stats: Summary statistics
"""

from .event.price import DcReturn, OsReturn, Overshoot
from .event.tick import RunsCount
from .event.time import Duration, Velocity
from .summary.stats import TMV, AvgDuration, AvgOvershoot, AvgReturn, UpturnRatio, VolatilityDC


def register_all(registry):
    """Registers all standard metrics to the provided registry."""
    # Event/Price (event-level)
    registry.register(Overshoot())
    registry.register(DcReturn())
    registry.register(OsReturn())

    # Event/Time (event-level)
    registry.register(Duration())
    registry.register(Velocity())

    # Event/Tick (event-level)
    registry.register(RunsCount())

    # Summary/Stats (aggregation)
    registry.register(TMV())
    registry.register(AvgDuration())
    registry.register(AvgReturn())
    registry.register(AvgOvershoot())
    registry.register(VolatilityDC())
    registry.register(UpturnRatio())


__all__ = [
    # Event/Price
    "Overshoot",
    "DcReturn",
    "OsReturn",
    # Event/Time
    "Duration",
    "Velocity",
    # Event/Tick
    "RunsCount",
    # Summary/Stats
    "TMV",
    "AvgDuration",
    "AvgReturn",
    "AvgOvershoot",
    "VolatilityDC",
    "UpturnRatio",
    # Registration
    "register_all",
]
