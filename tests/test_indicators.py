"""Tests para el módulo de indicadores.

Cubre:
- Registry de indicadores
- BaseIndicator
- Indicadores de precio (DcMagnitude, OsMagnitude, DcReturn, OsReturn)
- Indicadores de tiempo (DcTime, OsTime, EventTime, DcVelocity, OsVelocity, EventVelocity)
- Indicadores de ticks
- Función compute()
"""

import polars as pl
import pytest
from intrinseca.indicators import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorRegistry,
    compute,
    registry,
)
from intrinseca.indicators.metrics.event.price import OsMagnitude, OsReturn

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_silver_df():
    """DataFrame Silver de ejemplo para tests.

    Orden cronológico correcto: reference_time -> confirm_time -> extreme_time
    - reference_time: Inicio de fase DC
    - confirm_time: Fin de fase DC / Inicio de fase OS
    - extreme_time: Fin de fase OS
    """
    return pl.DataFrame(
        {
            "event_type": [1, -1, 1, -1, 1],
            "reference_price": [100.0, 102.0, 99.0, 103.0, 98.0],
            "confirm_price": [101.0, 100.0, 101.5, 99.5, 103.0],
            "extreme_price": [102.0, 99.0, 103.0, 98.0, 105.0],
            # Timestamps in correct chronological order: ref -> confirm -> extreme
            "reference_time": [1000, 2000, 3000, 4000, 5000],
            "confirm_time": [1100, 2100, 3100, 4100, 5100],
            "extreme_time": [1200, 2200, 3200, 4200, 5200],
            # Columnas de listas anidadas (microestructura)
            "price_dc": [
                [100.0, 100.5, 101.0],
                [102.0, 101.0, 100.0],
                [99.0, 100.0, 101.5],
                [103.0, 101.0, 99.5],
                [98.0, 100.0, 103.0],
            ],
            "price_os": [
                [101.0, 101.5, 102.0],
                [100.0, 99.5, 99.0],
                [101.5, 102.0, 103.0],
                [99.5, 99.0, 98.0],
                [103.0, 104.0, 105.0],
            ],
            "time_dc": [
                [1000, 1050, 1100],
                [2000, 2050, 2100],
                [3000, 3050, 3100],
                [4000, 4050, 4100],
                [5000, 5050, 5100],
            ],
            "time_os": [
                [1100, 1150, 1200],
                [2100, 2150, 2200],
                [3100, 3150, 3200],
                [4100, 4150, 4200],
                [5100, 5150, 5200],
            ],
        }
    )


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

        assert "dc_magnitude" in indicators
        assert "os_magnitude" in indicators
        assert "dc_return" in indicators
        assert "os_return" in indicators
        assert "dc_time" in indicators
        assert "os_time" in indicators
        assert "event_time" in indicators
        assert "dc_velocity" in indicators
        assert "os_velocity" in indicators
        assert "event_velocity" in indicators

    def test_register_custom_indicator(self, empty_registry):
        """Test registro de indicador personalizado."""

        class CustomIndicator(BaseIndicator):
            name = "custom_test"
            metadata = IndicatorMetadata(description="Test indicator", category="test")
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

            def get_expression(self):
                return pl.lit(1)

        empty_registry.register(TestIndicator())

        with pytest.raises(ValueError, match="already registered"):
            empty_registry.register(TestIndicator())

    def test_get_indicator(self):
        """Test obtener indicador por nombre."""
        indicator = registry.get_indicator("os_magnitude")

        assert indicator is not None
        assert indicator.name == "os_magnitude"

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
        indicator = OsMagnitude()
        assert hasattr(indicator, "name")
        assert indicator.name == "os_magnitude"

    def test_indicator_has_metadata(self):
        """Test que indicador tiene metadata."""
        indicator = OsMagnitude()
        assert hasattr(indicator, "metadata")
        assert indicator.metadata.description is not None
        assert indicator.metadata.category is not None

    def test_indicator_has_dependencies(self):
        """Test que indicador tiene dependencias."""
        indicator = OsReturn()
        assert hasattr(indicator, "dependencies")
        assert "os_magnitude" in indicator.dependencies

    def test_indicator_returns_expression(self):
        """Test que get_expression retorna pl.Expr."""
        indicator = OsMagnitude()
        expr = indicator.get_expression()

        assert isinstance(expr, pl.Expr)


# =============================================================================
# TESTS: Price Indicators
# =============================================================================


class TestPriceIndicators:
    """Tests para indicadores de precio."""

    def test_dc_magnitude_calculation(self, sample_silver_df):
        """Test cálculo de dc_magnitude (A1).

        dc_magnitude = confirm_price - reference_price
        """
        df = sample_silver_df

        result = compute(df, ["dc_magnitude"])

        assert "dc_magnitude" in result.columns

        # dc_magnitude[0] = 101.0 - 100.0 = 1.0
        expected_first = 101.0 - 100.0
        assert result["dc_magnitude"][0] == expected_first

        # dc_magnitude[1] = 100.0 - 102.0 = -2.0 (downturn)
        expected_second = 100.0 - 102.0
        assert result["dc_magnitude"][1] == expected_second

    def test_os_magnitude_calculation(self, sample_silver_df):
        """Test cálculo de os_magnitude.

        os_magnitude = extreme_price - confirm_price (same row, no shift)
        """
        df = sample_silver_df

        result = compute(df, ["os_magnitude"])

        assert "os_magnitude" in result.columns

        # os_magnitude[0] = extreme_price[0] - confirm_price[0]
        # = 102.0 - 101.0 = 1.0
        expected_first = 102.0 - 101.0
        assert result["os_magnitude"][0] == expected_first

        # Segundo evento (downturn)
        # os_magnitude[1] = 99.0 - 100.0 = -1.0
        expected_second = 99.0 - 100.0
        assert result["os_magnitude"][1] == expected_second

    def test_dc_return_calculation(self, sample_silver_df):
        """Test cálculo de DC return.

        dc_return = (confirm_price - reference_price) / reference_price
        """
        df = sample_silver_df

        result = compute(df, ["dc_return"])

        assert "dc_return" in result.columns

        # Verificar primer valor: (101.0 - 100.0) / 100.0 = 0.01
        expected_first = (101.0 - 100.0) / 100.0
        assert abs(result["dc_return"][0] - expected_first) < 1e-10

        # Segundo valor (downturn): (100.0 - 102.0) / 102.0
        expected_second = (100.0 - 102.0) / 102.0
        assert abs(result["dc_return"][1] - expected_second) < 1e-10

    def test_os_return_depends_on_os_magnitude(self, sample_silver_df):
        """Test que os_return depende de os_magnitude."""
        indicator = OsReturn()
        assert "os_magnitude" in indicator.dependencies

    def test_os_return_calculation(self, sample_silver_df):
        """Test cálculo de OS return."""
        df = sample_silver_df

        result = compute(df, ["os_return"])

        assert "os_return" in result.columns
        assert "os_magnitude" in result.columns  # Dependencia incluida


# =============================================================================
# TESTS: Time Indicators
# =============================================================================


class TestTimeIndicators:
    """Tests para indicadores de tiempo."""

    def test_dc_time_calculation(self, sample_silver_df):
        """Test cálculo de dc_time.

        dc_time = confirm_time - reference_time
        """
        df = sample_silver_df

        result = compute(df, ["dc_time"])

        assert "dc_time" in result.columns

        # dc_time[0] = confirm_time[0] - reference_time[0] = 1100 - 1000 = 100
        assert result["dc_time"][0] == 100
        # dc_time[1] = 2100 - 2000 = 100
        assert result["dc_time"][1] == 100

    def test_os_time_calculation(self, sample_silver_df):
        """Test cálculo de os_time.

        os_time = extreme_time - confirm_time
        """
        df = sample_silver_df

        result = compute(df, ["os_time"])

        assert "os_time" in result.columns

        # os_time[0] = extreme_time[0] - confirm_time[0] = 1200 - 1100 = 100
        assert result["os_time"][0] == 100
        # os_time[1] = 2200 - 2100 = 100
        assert result["os_time"][1] == 100

    def test_event_time_calculation(self, sample_silver_df):
        """Test cálculo de event_time.

        event_time = dc_time + os_time
        """
        df = sample_silver_df

        result = compute(df, ["event_time"])

        assert "event_time" in result.columns
        assert "dc_time" in result.columns  # Dependencia
        assert "os_time" in result.columns  # Dependencia

        # event_time[0] = dc_time[0] + os_time[0] = 100 + 100 = 200
        assert result["event_time"][0] == 200

    def test_dc_velocity_calculation(self, sample_silver_df):
        """Test cálculo de dc_velocity.

        dc_velocity = dc_magnitude / dc_time_in_seconds
        """
        df = sample_silver_df

        result = compute(df, ["dc_velocity"])

        assert "dc_velocity" in result.columns
        assert "dc_time" in result.columns  # Dependencia
        assert "dc_magnitude" in result.columns  # Dependencia

        # dc_velocity[0] = dc_magnitude[0] / (dc_time[0] / 1e9)
        # = 1.0 / (100 / 1e9) = 1.0 / 1e-7 = 1e7
        expected_first = 1.0 / (100 / 1_000_000_000.0)
        assert abs(result["dc_velocity"][0] - expected_first) < 1e-5

    def test_os_velocity_calculation(self, sample_silver_df):
        """Test cálculo de os_velocity.

        os_velocity = os_magnitude / os_time_in_seconds
        """
        df = sample_silver_df

        result = compute(df, ["os_velocity"])

        assert "os_velocity" in result.columns
        assert "os_time" in result.columns  # Dependencia
        assert "os_magnitude" in result.columns  # Dependencia

        # os_velocity[0] = os_magnitude[0] / (os_time[0] / 1e9)
        # = 1.0 / (100 / 1e9) = 1.0 / 1e-7 = 1e7
        expected_first = 1.0 / (100 / 1_000_000_000.0)
        assert abs(result["os_velocity"][0] - expected_first) < 1e-5

    def test_event_velocity_calculation(self, sample_silver_df):
        """Test cálculo de event_velocity.

        event_velocity = (extreme_price - reference_price) / event_time_in_seconds
        """
        df = sample_silver_df

        result = compute(df, ["event_velocity"])

        assert "event_velocity" in result.columns
        assert "event_time" in result.columns  # Dependencia

        # event_velocity[0] = (102.0 - 100.0) / (200 / 1e9)
        # = 2.0 / 2e-7 = 1e7
        expected_first = (102.0 - 100.0) / (200 / 1_000_000_000.0)
        assert abs(result["event_velocity"][0] - expected_first) < 1e-5

    def test_event_time_depends_on_dc_and_os_time(self, sample_silver_df):
        """Test que event_time depende de dc_time y os_time."""
        from intrinseca.indicators.metrics.event.time import EventTime

        indicator = EventTime()
        assert "dc_time" in indicator.dependencies
        assert "os_time" in indicator.dependencies


# =============================================================================
# TESTS: compute() Function
# =============================================================================


class TestComputeFunction:
    """Tests para la función compute()."""

    def test_compute_single_indicator(self, sample_silver_df):
        """Test computar un solo indicador."""
        result = compute(sample_silver_df, ["os_magnitude"])

        assert "os_magnitude" in result.columns

    def test_compute_multiple_indicators(self, sample_silver_df):
        """Test computar múltiples indicadores."""
        result = compute(sample_silver_df, ["os_magnitude", "dc_return"])

        assert "os_magnitude" in result.columns
        assert "dc_return" in result.columns

    def test_compute_all_indicators(self, sample_silver_df):
        """Test computar todos los indicadores."""
        result = compute(sample_silver_df, "all")

        # Debe tener al menos los indicadores estándar
        assert "os_magnitude" in result.columns
        assert "dc_magnitude" in result.columns
        assert "dc_return" in result.columns
        assert "os_return" in result.columns

    def test_compute_preserves_original_columns(self, sample_silver_df):
        """Test que compute preserva columnas originales."""
        original_cols = set(sample_silver_df.columns)

        result = compute(sample_silver_df, ["os_magnitude"])

        for col in original_cols:
            assert col in result.columns

    def test_compute_returns_dataframe(self, sample_silver_df):
        """Test que compute retorna DataFrame."""
        result = compute(sample_silver_df, ["os_magnitude"])

        assert isinstance(result, pl.DataFrame)

    def test_compute_accepts_lazy_frame(self, sample_silver_df):
        """Test que compute acepta LazyFrame."""
        lazy_df = sample_silver_df.lazy()

        result = compute(lazy_df, ["os_magnitude"])

        assert isinstance(result, pl.DataFrame)
        assert "os_magnitude" in result.columns


# =============================================================================
# TESTS: Dependency Resolution
# =============================================================================


class TestDependencyResolution:
    """Tests para resolución de dependencias."""

    def test_computes_dependencies(self, sample_silver_df):
        """Test que computa dependencias automáticamente."""
        # os_return depende de os_magnitude
        result = compute(sample_silver_df, ["os_return"])

        # os_magnitude debe estar presente
        assert "os_magnitude" in result.columns
        assert "os_return" in result.columns

    def test_no_duplicate_dependencies(self, sample_silver_df):
        """Test que no duplica dependencias."""
        # Pedir ambos explícitamente
        result = compute(sample_silver_df, ["os_magnitude", "os_return"])

        # os_magnitude debe aparecer solo una vez
        os_magnitude_cols = [c for c in result.columns if c == "os_magnitude"]
        assert len(os_magnitude_cols) == 1


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestIndicatorEdgeCases:
    """Tests para casos edge."""

    def test_empty_dataframe(self):
        """Test con DataFrame vacío."""
        empty_df = pl.DataFrame(
            {
                "event_type": [],
                "reference_price": [],
                "extreme_price": [],
                "confirm_price": [],
                "reference_time": [],
                "extreme_time": [],
                "confirm_time": [],
            }
        ).cast(
            {
                "event_type": pl.Int8,
                "reference_price": pl.Float64,
                "extreme_price": pl.Float64,
                "confirm_price": pl.Float64,
                "reference_time": pl.Int64,
                "extreme_time": pl.Int64,
                "confirm_time": pl.Int64,
            }
        )

        result = compute(empty_df, ["os_magnitude"])

        assert len(result) == 0
        assert "os_magnitude" in result.columns

    def test_single_event(self):
        """Test con un solo evento."""
        single_df = pl.DataFrame(
            {
                "event_type": [1],
                "reference_price": [100.0],
                "extreme_price": [105.0],
                "confirm_price": [103.0],
                "reference_time": [1000],
                "extreme_time": [1500],
                "confirm_time": [1200],
            }
        ).cast(
            {
                "event_type": pl.Int8,
            }
        )

        result = compute(single_df, ["os_magnitude", "dc_return"])

        assert len(result) == 1
        # os_magnitude = extreme_price - confirm_price = 105.0 - 103.0 = 2.0
        assert result["os_magnitude"][0] == 2.0
        # dc_return sí tiene valor
        assert result["dc_return"][0] is not None
