"""Metrics Package.

Contains all indicator implementations organized by category:
- event/price: Price movement magnitude and returns
- event/time: Time and velocity metrics
- event/tick: Tick-level analysis
- event/memory: Context from previous events (ML features)
- event/series: Signal processing (Fourier, wavelets, etc.)
- summary/stats: Summary statistics
"""

import warnings

from intrinseca.indicators.registry import IndicatorRegistry

from .event.memory import A4PrevDccPrice, A5PrevOsFlag, A6FlashEvent
from .event.price import (
    A1DcPriceAbs,
    DcMagnitude,
    DcReturn,
    DcSlippage,
    DcSlippageReal,
    EventMagnitude,
    OsMagnitude,
    OsReturn,
    OsvEvent,
    Tmv,
    TotalMove,
)
from .event.tick import RunsCount
from .event.time import DcTime, DcVelocity, EventTime, EventVelocity, OsTime, OsVelocity
from .summary.stats import (
    AccumulatedTime,
    AvgDcTime,
    AvgOsMagnitude,
    AvgReturn,
    Cdc,
    Ndc,
    TmvAggregated,
    VolatilityDC,
)


def create_registry(theta: float) -> IndicatorRegistry:
    """Create an IndicatorRegistry with the specified theta.

    This is the recommended way to create a registry when working with
    DC event data. The theta should be extracted from the Parquet
    metadata using read_dc_events() or read_dc_month().

    Args:
        theta: DC threshold used to generate the event data.

    Returns:
        Configured IndicatorRegistry with all standard indicators.

    Example:
        >>> from intrinseca.core import read_dc_month
        >>> dataset = read_dc_month(data_path, "BTCUSDT", 2025, 11)
        >>> registry = create_registry(dataset.theta)
        >>> result = registry.compute(dataset.df.lazy(), ["tmv"]).collect()
    """
    registry = IndicatorRegistry()

    # Event/Price (event-level) - theta-independent
    registry.register(DcMagnitude())
    registry.register(OsMagnitude())
    registry.register(EventMagnitude())
    registry.register(DcReturn())
    registry.register(OsReturn())
    registry.register(TotalMove())
    registry.register(A1DcPriceAbs())

    # Event/Price (event-level) - theta-dependent
    registry.register(DcSlippage(theta=theta))
    registry.register(DcSlippageReal(theta=theta))
    registry.register(Tmv(theta=theta))
    registry.register(OsvEvent(theta=theta))

    # Event/Time (event-level)
    registry.register(DcTime())
    registry.register(OsTime())
    registry.register(EventTime())
    registry.register(DcVelocity())
    registry.register(OsVelocity())
    registry.register(EventVelocity())

    # Event/Tick (event-level) - theta-dependent
    registry.register(RunsCount(theta=theta))

    # Event/Memory (ML context features)
    registry.register(A4PrevDccPrice())
    registry.register(A5PrevOsFlag())
    registry.register(A6FlashEvent())

    # Summary/Stats (aggregation)
    registry.register(TmvAggregated())
    registry.register(AvgDcTime())
    registry.register(AvgReturn())
    registry.register(AvgOsMagnitude())
    registry.register(VolatilityDC())
    registry.register(Ndc())
    registry.register(Cdc())
    registry.register(AccumulatedTime())

    return registry


def register_all(registry: IndicatorRegistry) -> None:
    """Register all standard metrics to the provided registry.

    .. deprecated::
        Use create_registry(theta) instead. This function uses a default
        theta=0.005 which may not match your DC event data.
    """
    warnings.warn(
        "register_all() is deprecated. Use create_registry(theta) instead "
        "to ensure theta matches your DC event data.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Use default theta for backwards compatibility
    default_theta = 0.005

    # Event/Price (event-level)
    registry.register(DcMagnitude())
    registry.register(OsMagnitude())
    registry.register(EventMagnitude())
    registry.register(DcReturn())
    registry.register(OsReturn())
    registry.register(DcSlippage(theta=default_theta))
    registry.register(DcSlippageReal(theta=default_theta))
    registry.register(TotalMove())
    registry.register(Tmv(theta=default_theta))
    registry.register(OsvEvent(theta=default_theta))
    registry.register(A1DcPriceAbs())

    # Event/Time (event-level)
    registry.register(DcTime())
    registry.register(OsTime())
    registry.register(EventTime())
    registry.register(DcVelocity())
    registry.register(OsVelocity())
    registry.register(EventVelocity())

    # Event/Tick (event-level)
    registry.register(RunsCount(theta=default_theta))

    # Event/Memory (ML context features)
    registry.register(A4PrevDccPrice())
    registry.register(A5PrevOsFlag())
    registry.register(A6FlashEvent())

    # Summary/Stats (aggregation)
    registry.register(TmvAggregated())
    registry.register(AvgDcTime())
    registry.register(AvgReturn())
    registry.register(AvgOsMagnitude())
    registry.register(VolatilityDC())
    registry.register(Ndc())
    registry.register(Cdc())
    registry.register(AccumulatedTime())


__all__ = [
    # Event/Price
    "DcMagnitude",
    "OsMagnitude",
    "EventMagnitude",
    "DcReturn",
    "OsReturn",
    "DcSlippage",
    "DcSlippageReal",
    "TotalMove",
    "TmvEvent",
    "OsvEvent",
    "A1DcPriceAbs",
    # Event/Time
    "DcTime",
    "OsTime",
    "EventTime",
    "DcVelocity",
    "OsVelocity",
    "EventVelocity",
    # Event/Tick
    "RunsCount",
    # Event/Memory
    "A4PrevDccPrice",
    "A5PrevOsFlag",
    "A6FlashEvent",
    # Summary/Stats
    "TMV",
    "AvgDcTime",
    "AvgReturn",
    "AvgOsMagnitude",
    "VolatilityDC",
    "Ndc",
    "Cdc",
    "AccumulatedTime",
    # Registration
    "create_registry",
    "register_all",  # Deprecated
]
