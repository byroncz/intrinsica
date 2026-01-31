"""
Tests para el módulo de indicadores.

Cubre:
- Registry de indicadores
- BaseIndicator
- Indicadores de precio (Overshoot, DcReturn, OsReturn)
- Indicadores de tiempo
- Indicadores de ticks
- Función compute()
"""

import numpy as np
import polars as pl
import pytest

from intrinseca.indicators import (
    registry,
    compute,
    BaseIndicator,
    IndicatorMetadata,
    IndicatorRegistry,
)
from intrinseca.indicators.metrics.event.price import Overshoot, DcReturn, OsReturn


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_silver_df():
    """DataFrame Silver de ejemplo para tests."""
    return pl.DataFrame({
        "event_type": [1, -1, 1, -1, 1],
        "reference_price": [100.0, 102.0, 99.0, 103.0, 98.0],
        "extreme_price": [102.0, 99.0, 103.0, 98.0, 105.0],
        "confirm_price": [101.0, 100.0, 101.5, 99.5, 103.0],
        "reference_time": [1000, 2000, 3000, 4000, 5000],
        "extreme_time": [1100, 2100, 3100, 4100, 5100],
        "confirm_time": [1200, 2200, 3200, 4200, 5200],
        # Columnas de listas anidadas (microestructura)
        "price_dc": [[100.0, 100.5, 101.0], [102.0, 101.0, 100.0], [99.0, 100.0, 101.5], [103.0, 101.0, 99.5], [98.0, 100.0, 103.0]],
        "price_os": [[101.0, 101.5, 102.0], [100.0, 99.5, 99.0], [101.5, 102.0, 103.0], [99.5, 99.0, 98.0], [103.0, 104.0, 105.0]],
        "time_dc": [[1000, 1050, 1100], [2000, 2050, 2100], [3000, 3050, 3100], [4000, 4050, 4100], [5000, 5050, 5100]],
        "time_os": [[1100, 1150, 1200], [2100, 2150, 2200], [3100, 3150, 3200], [4100, 4150, 4200], [5100, 5150, 5200]],
    })


@pytest.fixture
def empty_registry():
    """Registry vacío para tests aislados."""
    return IndicatorRegistry()


# =============================================================================
# TESTS: IndicatorRegistry
# =============================================================================

class TestIndicatorRegistry:
    """Tests para el IndicatorRegistry."""
    
    def test_registry_singleton_exists(self):
        """Test que el singleton existe."""
        assert registry is not None
        assert isinstance(registry, IndicatorRegistry)
    
    def test_registry_has_indicators(self):
        """Test que el registry tiene indicadores cargados."""
        indicators = registry.list_indicators()
        assert len(indicators) > 0
    
    def test_registry_contains_standard_indicators(self):
        """Test que contiene indicadores estándar."""
        indicators = registry.list_indicators()
        
        assert "overshoot" in indicators
        assert "dc_return" in indicators
        assert "os_return" in indicators
    
    def test_register_custom_indicator(self, empty_registry):
        """Test registro de indicador personalizado."""
        class CustomIndicator(BaseIndicator):
            name = "custom_test"
            metadata = IndicatorMetadata(
                description="Test indicator",
                category="test"
            )
            dependencies = []
            
            def get_expression(self):
                return pl.lit(42)
        
        indicator = CustomIndicator()
        empty_registry.register(indicator)
        
        assert "custom_test" in empty_registry.list_indicators()
    
    def test_register_duplicate_raises(self, empty_registry):
        """Test que registrar duplicado lanza error."""
        class TestIndicator(BaseIndicator):
            name = "duplicate"
            metadata = IndicatorMetadata(description="test", category="test")
            dependencies = []
            def get_expression(self): return pl.lit(1)
        
        empty_registry.register(TestIndicator())
        
        with pytest.raises(ValueError, match="already registered"):
            empty_registry.register(TestIndicator())
    
    def test_get_indicator(self):
        """Test obtener indicador por nombre."""
        indicator = registry.get_indicator("overshoot")
        
        assert indicator is not None
        assert indicator.name == "overshoot"
    
    def test_get_nonexistent_indicator_raises(self):
        """Test que obtener indicador inexistente lanza error."""
        with pytest.raises(KeyError, match="not found"):
            registry.get_indicator("nonexistent_indicator")


# =============================================================================
# TESTS: BaseIndicator
# =============================================================================

class TestBaseIndicator:
    """Tests para BaseIndicator."""
    
    def test_indicator_has_name(self):
        """Test que indicador tiene nombre."""
        indicator = Overshoot()
        assert hasattr(indicator, "name")
        assert indicator.name == "overshoot"
    
    def test_indicator_has_metadata(self):
        """Test que indicador tiene metadata."""
        indicator = Overshoot()
        assert hasattr(indicator, "metadata")
        assert indicator.metadata.description is not None
        assert indicator.metadata.category is not None
    
    def test_indicator_has_dependencies(self):
        """Test que indicador tiene dependencias."""
        indicator = OsReturn()
        assert hasattr(indicator, "dependencies")
        assert "overshoot" in indicator.dependencies
    
    def test_indicator_returns_expression(self):
        """Test que get_expression retorna pl.Expr."""
        indicator = Overshoot()
        expr = indicator.get_expression()
        
        assert isinstance(expr, pl.Expr)


# =============================================================================
# TESTS: Price Indicators
# =============================================================================

class TestPriceIndicators:
    """Tests para indicadores de precio."""
    
    def test_overshoot_calculation(self, sample_silver_df):
        """Test cálculo de overshoot."""
        df = sample_silver_df
        
        result = compute(df, ["overshoot"])
        
        assert "overshoot" in result.columns
        
        # Verificar cálculo manual para primer evento
        # overshoot[0] = extreme_price[1] - confirm_price[0]
        # = 99.0 - 101.0 = -2.0
        expected_first = 99.0 - 101.0
        assert result["overshoot"][0] == expected_first
    
    def test_overshoot_last_is_null(self, sample_silver_df):
        """Test que el último overshoot es null (no hay siguiente extremo)."""
        df = sample_silver_df
        
        result = compute(df, ["overshoot"])
        
        # El último evento no tiene siguiente, debe ser null
        assert result["overshoot"][-1] is None
    
    def test_dc_return_calculation(self, sample_silver_df):
        """Test cálculo de DC return."""
        df = sample_silver_df
        
        result = compute(df, ["dc_return"])
        
        assert "dc_return" in result.columns
        
        # Verificar primer valor: (102.0 - 101.0) / 101.0
        expected_first = (102.0 - 101.0) / 101.0
        assert abs(result["dc_return"][0] - expected_first) < 1e-10
    
    def test_os_return_depends_on_overshoot(self, sample_silver_df):
        """Test que os_return depende de overshoot."""
        indicator = OsReturn()
        assert "overshoot" in indicator.dependencies
    
    def test_os_return_calculation(self, sample_silver_df):
        """Test cálculo de OS return."""
        df = sample_silver_df
        
        result = compute(df, ["os_return"])
        
        assert "os_return" in result.columns
        assert "overshoot" in result.columns  # Dependencia incluida


# =============================================================================
# TESTS: compute() Function
# =============================================================================

class TestComputeFunction:
    """Tests para la función compute()."""
    
    def test_compute_single_indicator(self, sample_silver_df):
        """Test computar un solo indicador."""
        result = compute(sample_silver_df, ["overshoot"])
        
        assert "overshoot" in result.columns
    
    def test_compute_multiple_indicators(self, sample_silver_df):
        """Test computar múltiples indicadores."""
        result = compute(sample_silver_df, ["overshoot", "dc_return"])
        
        assert "overshoot" in result.columns
        assert "dc_return" in result.columns
    
    def test_compute_all_indicators(self, sample_silver_df):
        """Test computar todos los indicadores."""
        result = compute(sample_silver_df, "all")
        
        # Debe tener al menos los indicadores estándar
        assert "overshoot" in result.columns
        assert "dc_return" in result.columns
        assert "os_return" in result.columns
    
    def test_compute_preserves_original_columns(self, sample_silver_df):
        """Test que compute preserva columnas originales."""
        original_cols = set(sample_silver_df.columns)
        
        result = compute(sample_silver_df, ["overshoot"])
        
        for col in original_cols:
            assert col in result.columns
    
    def test_compute_returns_dataframe(self, sample_silver_df):
        """Test que compute retorna DataFrame."""
        result = compute(sample_silver_df, ["overshoot"])
        
        assert isinstance(result, pl.DataFrame)
    
    def test_compute_accepts_lazy_frame(self, sample_silver_df):
        """Test que compute acepta LazyFrame."""
        lazy_df = sample_silver_df.lazy()
        
        result = compute(lazy_df, ["overshoot"])
        
        assert isinstance(result, pl.DataFrame)
        assert "overshoot" in result.columns


# =============================================================================
# TESTS: Dependency Resolution
# =============================================================================

class TestDependencyResolution:
    """Tests para resolución de dependencias."""
    
    def test_computes_dependencies(self, sample_silver_df):
        """Test que computa dependencias automáticamente."""
        # os_return depende de overshoot
        result = compute(sample_silver_df, ["os_return"])
        
        # overshoot debe estar presente
        assert "overshoot" in result.columns
        assert "os_return" in result.columns
    
    def test_no_duplicate_dependencies(self, sample_silver_df):
        """Test que no duplica dependencias."""
        # Pedir ambos explícitamente
        result = compute(sample_silver_df, ["overshoot", "os_return"])
        
        # overshoot debe aparecer solo una vez
        overshoot_cols = [c for c in result.columns if c == "overshoot"]
        assert len(overshoot_cols) == 1


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestIndicatorEdgeCases:
    """Tests para casos edge."""
    
    def test_empty_dataframe(self):
        """Test con DataFrame vacío."""
        empty_df = pl.DataFrame({
            "event_type": [],
            "reference_price": [],
            "extreme_price": [],
            "confirm_price": [],
            "reference_time": [],
            "extreme_time": [],
            "confirm_time": [],
        }).cast({
            "event_type": pl.Int8,
            "reference_price": pl.Float64,
            "extreme_price": pl.Float64,
            "confirm_price": pl.Float64,
            "reference_time": pl.Int64,
            "extreme_time": pl.Int64,
            "confirm_time": pl.Int64,
        })
        
        result = compute(empty_df, ["overshoot"])
        
        assert len(result) == 0
        assert "overshoot" in result.columns
    
    def test_single_event(self):
        """Test con un solo evento."""
        single_df = pl.DataFrame({
            "event_type": [1],
            "reference_price": [100.0],
            "extreme_price": [105.0],
            "confirm_price": [103.0],
            "reference_time": [1000],
            "extreme_time": [1500],
            "confirm_time": [1200],
        }).cast({
            "event_type": pl.Int8,
        })
        
        result = compute(single_df, ["overshoot", "dc_return"])
        
        assert len(result) == 1
        # overshoot es null porque no hay siguiente evento
        assert result["overshoot"][0] is None
        # dc_return sí tiene valor
        assert result["dc_return"][0] is not None
