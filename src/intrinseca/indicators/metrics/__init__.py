"""
Metrics Package.

Contains all indicator implementations organized by category:
- geometry: Price movement magnitude and returns
- dynamics: Time and velocity metrics
- microstructure: Tick-level analysis
- aggregation: Summary statistics
"""

from .geometry import Overshoot, DcReturn, OsReturn
from .dynamics import Duration, Velocity
from .microstructure import RunsCount
from .aggregation import TMV, AvgDuration, AvgReturn, AvgOvershoot, VolatilityDC, UpturnRatio


def register_all(registry):
    """Registers all standard metrics to the provided registry."""
    # Geometry (event-level)
    registry.register(Overshoot())
    registry.register(DcReturn())
    registry.register(OsReturn())

    # Dynamics (event-level)
    registry.register(Duration())
    registry.register(Velocity())

    # Microstructure (event-level)
    registry.register(RunsCount())

    # Aggregation (summary statistics)
    registry.register(TMV())
    registry.register(AvgDuration())
    registry.register(AvgReturn())
    registry.register(AvgOvershoot())
    registry.register(VolatilityDC())
    registry.register(UpturnRatio())


__all__ = [
    # Geometry
    "Overshoot", "DcReturn", "OsReturn",
    # Dynamics
    "Duration", "Velocity",
    # Microstructure
    "RunsCount",
    # Aggregation
    "TMV", "AvgDuration", "AvgReturn", "AvgOvershoot", "VolatilityDC", "UpturnRatio",
    # Registration
    "register_all",
]
