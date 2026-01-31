"""Intrinsica Indicators Module.

Provides a modular, Zero-Copy system for calculating Event-Level
and Aggregated metrics from Silver Layer data.

Usage:
    from intrinseca.indicators import registry, compute

    # Option 1: Use registry directly (all indicators pre-loaded)
    df_result = registry.compute(df_silver.lazy(), ["overshoot", "duration"]).collect()

    # Option 2: Use convenience function
    df_result = compute(df_silver, ["overshoot", "duration"])
"""

import polars as pl

from .base import BaseIndicator, IndicatorMetadata
from .metrics import register_all
from .registry import IndicatorRegistry, registry

# Auto-register all standard indicators at import time
register_all(registry)


def compute(
    df: pl.DataFrame | pl.LazyFrame,
    indicators: list[str] | str = "all",
) -> pl.DataFrame:
    """Compute indicators on a Silver Layer DataFrame.

    Args:
    ----
        df: Silver Layer DataFrame (eager or lazy) with nested list columns.
        indicators: List of indicator names to compute, or "all" for all available.

    Returns:
    -------
        pl.DataFrame: Original data with indicator columns added.

    Example:
    -------
        >>> from intrinseca.indicators import compute
        >>> df_with_overshoot = compute(df_silver, ["overshoot"])

    """
    lazy_df = df.lazy() if isinstance(df, pl.DataFrame) else df
    return registry.compute(lazy_df, indicators).collect()


__all__ = [
    "BaseIndicator",
    "IndicatorMetadata",
    "registry",
    "IndicatorRegistry",
    "compute",
]
