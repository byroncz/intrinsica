from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl


@dataclass
class IndicatorMetadata:
    """Metadata for an indicator including description and categorization."""

    description: str
    category: str  # 'core', 'geometry', 'dynamics', 'microstructure'
    is_event_level: bool = True  # True = 1 row per event, False = Aggregated


class BaseIndicator(ABC):
    """Abstract Base Class for all Intrinsica Indicators.

    Designed for Zero-Copy operations on Silver Layer Data (Nested Parquet).
    Each indicator defines a Polars Expression that transforms the input schema
    into a target result column.
    """

    name: str
    metadata: IndicatorMetadata
    dependencies: list[str] = []

    @abstractmethod
    def get_expression(self) -> pl.Expr:
        """Returns the Polars Expression to compute this indicator.

        The expression must be compatible with the Silver Layer Schema:
        - event_type: Int8
        - price_dc: List[Float64]
        - price_os: List[Float64]
        - time_dc: List[Int64]
        - time_os: List[Int64]

        Returns:
        -------
            pl.Expr: Expression resulting in the indicator column.

        """
        pass

    def validate_dependencies(self, available_indicators: set[str]) -> bool:
        """Checks if all dependencies are present in the registry."""
        return all(dep in available_indicators for dep in self.dependencies)
