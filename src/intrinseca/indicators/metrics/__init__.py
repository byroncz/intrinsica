"""
Metrics Package.
"""

from .core import EventTime, EventPrice, EventTypeDesc, ExtremePrice, ExtremeTime
from .geometry import Overshoot, DcReturn, OsReturn
from .dynamics import Duration, Velocity
from .microstructure import RunsCount
from .aggregation import TMV, AvgDuration, AvgReturn, AvgOvershoot, VolatilityDC, UpturnRatio


def register_all(registry):
    """Registers all standard metrics to the provided registry."""
    # Core
    registry.register(EventTime())
    registry.register(EventPrice())
    registry.register(EventTypeDesc())
    registry.register(ExtremePrice())
    registry.register(ExtremeTime())

    # Geometry
    registry.register(Overshoot())
    registry.register(DcReturn())
    registry.register(OsReturn())

    # Dynamics
    registry.register(Duration())
    registry.register(Velocity())

    # Microstructure
    registry.register(RunsCount())

    # Aggregation
    registry.register(TMV())
    registry.register(AvgDuration())
    registry.register(AvgReturn())
    registry.register(AvgOvershoot())
    registry.register(VolatilityDC())
    registry.register(UpturnRatio())
