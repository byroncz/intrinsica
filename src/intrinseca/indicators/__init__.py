"""
Intrinsica Indicators Module.

Provides a modular, Zero-Copy system for calculating Event-Level 
and Aggregated metrics from Silver Layer data.
"""

from .base import BaseIndicator, IndicatorMetadata
from .registry import registry, IndicatorRegistry

__all__ = ["BaseIndicator", "IndicatorMetadata", "registry", "IndicatorRegistry"]
