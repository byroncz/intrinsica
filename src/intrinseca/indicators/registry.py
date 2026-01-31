from __future__ import annotations

import polars as pl

from intrinseca.indicators.base import BaseIndicator


class IndicatorRegistry:
    """Central registry for Intrinsica Indicators.
    Manages registration, dependency resolution, and DataFrame construction.
    """

    def __init__(self):
        self._indicators: dict[str, BaseIndicator] = {}

    def register(self, indicator: BaseIndicator):
        """Registers a new indicator instance."""
        if indicator.name in self._indicators:
            raise ValueError(f"Indicator '{indicator.name}' is already registered.")

        # Verify dependencies (simple check, topological sort could be added if needed)
        # For now, we assume registration order or runtime check
        self._indicators[indicator.name] = indicator

    def get_indicator(self, name: str) -> BaseIndicator:
        """Retrieve an indicator by name."""
        if name not in self._indicators:
            raise KeyError(f"Indicator '{name}' not found.")
        return self._indicators[name]

    def list_indicators(self) -> list[str]:
        """Return list of all registered indicator names."""
        return list(self._indicators.keys())

    def compute(self, df: pl.LazyFrame, indicators: list[str] | str = "all") -> pl.LazyFrame:
        """Applies requested indicators to the LazyFrame using Zero-Copy expressions.
        Resolves dependencies and executes in topological order.

        Args:
        ----
            df: Silver Layer LazyFrame (with nested list columns).
            indicators: List of indicator names to compute, or "all".

        Returns:
        -------
            pl.LazyFrame: The original frame with new indicator columns added.

        """
        selected_names = list(self._indicators.keys()) if indicators == "all" else indicators

        # 1. Expand dependencies (Transitive Closure)
        final_set = set(selected_names)
        queue = list(selected_names)
        while queue:
            curr = queue.pop(0)
            ind = self.get_indicator(curr)
            for dep in ind.dependencies:
                if dep not in final_set:
                    final_set.add(dep)
                    queue.append(dep)

        # 2. Topological Sort
        # Build graph: name -> set of dependencies that are also in final_set
        # (We only care about dependencies that we are computing)
        # However, if a dep is NOT in final_set (e.g. valid column in input df),
        # we treat it as satisfied.
        # But here, dependencies usually refer to other INDICATORS.
        # If a dependency is not an indicator (e.g. 'price_os'), it won't be in registry.
        # So we filter deps to only those in registry.

        registry_keys = set(self._indicators.keys())
        graph = {}
        for name in final_set:
            ind = self.get_indicator(name)
            # Only wait for dependencies that are calculated indicators
            deps = {d for d in ind.dependencies if d in registry_keys}
            graph[name] = deps

        ordered = []
        while graph:
            # Find nodes with 0 unsatisfied dependencies
            # (Satisfied means not in the current graph keys)
            ready = {name for name, deps in graph.items() if not deps}

            if not ready:
                # Cycle detected
                remain = list(graph.keys())
                raise ValueError(f"Circular dependency detected involving: {remain}")

            # Sort for deterministic behavior
            for name in sorted(ready):
                ordered.append(name)
                del graph[name]

            # Remove ready nodes from other dependencies (conceptually they are done)
            for name in graph:
                graph[name] -= ready

        # 3. Apply sequentially
        # Polars lazy frames optimize chained with_columns well.
        for name in ordered:
            ind = self.get_indicator(name)
            df = df.with_columns(ind.get_expression().alias(name))

        return df


# Global singleton
registry = IndicatorRegistry()
