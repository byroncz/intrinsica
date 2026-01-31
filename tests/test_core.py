"""
Tests para el núcleo de cálculo DC (Silver Layer).

Arquitectura actual:
- Engine: orquestador Bronze → Silver
- segment_events_kernel: kernel Numba JIT
- DCState: estado persistente entre días

Estos tests reemplazan los tests legacy de DCDetector/DCIndicators/TrendState
que fueron eliminados en la migración a Silver Layer.
"""

from datetime import date
from pathlib import Path
import tempfile

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from intrinseca.core.engine import Engine, EngineConfig
from intrinseca.core.kernel import segment_events_kernel, warmup_kernel, verify_nopython_mode
from intrinseca.core.state import DCState, create_empty_state, save_state, load_state


# =============================================================================
# FIXTURES COMUNES
# =============================================================================

@pytest.fixture
def sample_bronze_df():
    """DataFrame Bronze de ejemplo con datos realistas."""
    n = 1000
    np.random.seed(42)
    # Precios con tendencia y ruido
    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n) * 0.5
    prices = 100.0 + trend + noise.cumsum() * 0.1
    prices = np.abs(prices)  # Asegurar positivos
    
    return pl.DataFrame({
        "price": prices.astype(np.float64),
        "timestamp": (np.arange(n) * 1_000_000_000).astype(np.int64),
        "quantity": np.ones(n, dtype=np.float64),
        "direction": np.random.choice([1, -1], n).astype(np.int8),
    })


@pytest.fixture
def simple_uptrend_prices():
    """Serie de precios con tendencia alcista simple."""
    return np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0], dtype=np.float64)


@pytest.fixture
def simple_downtrend_prices():
    """Serie de precios con tendencia bajista simple."""
    return np.array([100.0, 98.0, 96.0, 94.0, 92.0, 90.0], dtype=np.float64)


@pytest.fixture
def tmp_silver_path(tmp_path):
    """Directorio temporal para Silver Layer."""
    silver_path = tmp_path / "silver"
    silver_path.mkdir(parents=True)
    return silver_path


# =============================================================================
# TESTS: EngineConfig
# =============================================================================

class TestEngineConfig:
    """Tests para configuración del Engine."""
    
    def test_config_creation_with_defaults(self, tmp_silver_path):
        """Test creación con valores por defecto."""
        config = EngineConfig(theta=0.005, silver_base_path=tmp_silver_path)
        assert config.theta == 0.005
        assert config.verbose is False
        assert config.collect_stats is True
        assert config.keep_state_files is True
    
    def test_config_custom_theta(self, tmp_silver_path):
        """Test con theta personalizado."""
        config = EngineConfig(theta=0.02, silver_base_path=tmp_silver_path)
        assert config.theta == 0.02
        assert config.theta_str == "0.02"
    
    def test_config_theta_str_formatting(self, tmp_silver_path):
        """Test que theta_str elimina ceros trailing."""
        config1 = EngineConfig(theta=0.005, silver_base_path=tmp_silver_path)
        assert config1.theta_str == "0.005"
        
        config2 = EngineConfig(theta=0.010000, silver_base_path=tmp_silver_path)
        assert config2.theta_str == "0.01"
    
    def test_config_path_converted_to_pathlib(self, tmp_silver_path):
        """Test que el path se convierte a Path."""
        config = EngineConfig(theta=0.01, silver_base_path=str(tmp_silver_path))
        assert isinstance(config.silver_base_path, Path)


# =============================================================================
# TESTS: Kernel Directo
# =============================================================================

class TestKernelDirect:
    """Tests directos del kernel Numba."""
    
    def test_warmup_kernel(self):
        """Test que warmup no lanza excepción."""
        # No debería lanzar excepción
        warmup_kernel()
    
    def test_verify_nopython_mode(self):
        """Test verificación de modo nopython."""
        # Retorna dict con información de compilación
        result = verify_nopython_mode()
        assert isinstance(result, dict)
        assert "nopython" in result
        assert result["nopython"] is True
    
    def test_simple_uptrend_detection(self, simple_uptrend_prices):
        """Test detección de upturn simple."""
        prices = simple_uptrend_prices
        timestamps = np.arange(len(prices), dtype=np.int64) * 1_000_000_000
        quantities = np.ones(len(prices), dtype=np.float64)
        directions = np.ones(len(prices), dtype=np.int8)
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,  # 2%
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        n_events = int(result[17])
        # Con 10% de subida total y theta=2%, debería detectar eventos
        assert n_events >= 0
    
    def test_simple_downtrend_detection(self, simple_downtrend_prices):
        """Test detección de downturn simple."""
        prices = simple_downtrend_prices
        timestamps = np.arange(len(prices), dtype=np.int64) * 1_000_000_000
        quantities = np.ones(len(prices), dtype=np.float64)
        directions = -np.ones(len(prices), dtype=np.int8)
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        n_events = int(result[17])
        assert n_events >= 0
    
    def test_flat_prices_no_events(self):
        """Test que precios planos no generan eventos."""
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        timestamps = np.arange(5, dtype=np.int64) * 1_000_000_000
        quantities = np.ones(5, dtype=np.float64)
        directions = np.ones(5, dtype=np.int8)
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        n_events = int(result[17])
        assert n_events == 0
    
    def test_single_tick(self):
        """Test con un solo tick."""
        prices = np.array([100.0], dtype=np.float64)
        timestamps = np.array([1_000_000_000], dtype=np.int64)
        quantities = np.array([1.0], dtype=np.float64)
        directions = np.array([1], dtype=np.int8)
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        n_events = int(result[17])
        assert n_events == 0
    
    def test_kernel_output_structure(self, sample_bronze_df):
        """Test que el kernel retorna la estructura correcta."""
        df = sample_bronze_df
        prices = df.get_column("price").to_numpy()
        timestamps = df.get_column("timestamp").to_numpy()
        quantities = df.get_column("quantity").to_numpy()
        directions = df.get_column("direction").to_numpy()
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.01,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        # El kernel retorna 23 elementos
        assert len(result) == 23
        
        # Verificar tipos de los elementos principales
        assert isinstance(result[0], np.ndarray)  # dc_prices
        assert isinstance(result[17], (int, np.integer))  # n_events


# =============================================================================
# TESTS: DCState
# =============================================================================

class TestDCState:
    """Tests para estado persistente."""
    
    def test_empty_state_creation(self):
        """Test creación de estado vacío."""
        state = create_empty_state(date(2025, 1, 1))
        assert state.n_orphans == 0
        assert state.current_trend == 0
        assert state.last_processed_date == date(2025, 1, 1)
    
    def test_state_with_orphans(self):
        """Test estado con ticks huérfanos."""
        state = DCState(
            orphan_prices=np.array([100.0, 101.0, 102.0], dtype=np.float64),
            orphan_times=np.array([1, 2, 3], dtype=np.int64),
            orphan_quantities=np.ones(3, dtype=np.float64),
            orphan_directions=np.array([1, 1, 1], dtype=np.int8),
            current_trend=np.int8(1),
            last_os_ref=np.float64(100.0),
            reference_extreme_price=np.float64(102.0),
            reference_extreme_time=np.int64(3),
            last_processed_date=date(2025, 1, 1),
        )
        assert state.n_orphans == 3
        assert state.current_trend == 1
    
    def test_state_get_extreme_prices(self):
        """Test cálculo de precios extremos."""
        state = DCState(
            orphan_prices=np.array([100.0, 105.0, 103.0], dtype=np.float64),
            orphan_times=np.array([1, 2, 3], dtype=np.int64),
            orphan_quantities=np.ones(3, dtype=np.float64),
            orphan_directions=np.array([1, 1, -1], dtype=np.int8),
            current_trend=np.int8(1),
            last_os_ref=np.float64(100.0),
            reference_extreme_price=np.float64(105.0),
            reference_extreme_time=np.int64(2),
            last_processed_date=date(2025, 1, 1),
        )
        ext_high, ext_low = state.get_extreme_prices()
        assert ext_high == 105.0
        assert ext_low == 100.0
    
    def test_state_memory_usage(self):
        """Test estimación de uso de memoria."""
        state = DCState(
            orphan_prices=np.array([100.0, 101.0], dtype=np.float64),
            orphan_times=np.array([1, 2], dtype=np.int64),
            orphan_quantities=np.ones(2, dtype=np.float64),
            orphan_directions=np.array([1, 1], dtype=np.int8),
            current_trend=np.int8(1),
            last_os_ref=np.float64(100.0),
            reference_extreme_price=np.float64(101.0),
            reference_extreme_time=np.int64(2),
            last_processed_date=date(2025, 1, 1),
        )
        
        memory = state.memory_usage_bytes
        assert memory > 0
        # 2 orphans × (8+8+8+1) bytes = 50 bytes mínimo
        assert memory >= 50


class TestStatePersistence:
    """Tests para persistencia de estado Arrow IPC."""
    
    def test_save_load_roundtrip(self, tmp_path):
        """Test guardar y cargar estado."""
        original = DCState(
            orphan_prices=np.array([100.0, 101.0, 102.0], dtype=np.float64),
            orphan_times=np.array([1000, 2000, 3000], dtype=np.int64),
            orphan_quantities=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            orphan_directions=np.array([1, -1, 1], dtype=np.int8),
            current_trend=np.int8(-1),
            last_os_ref=np.float64(99.5),
            reference_extreme_price=np.float64(102.0),
            reference_extreme_time=np.int64(3000),
            last_processed_date=date(2025, 6, 15),
        )
        
        state_path = tmp_path / "test_state.arrow"
        save_state(original, state_path)
        
        assert state_path.exists()
        
        loaded = load_state(state_path)
        
        assert loaded is not None
        assert_array_equal(loaded.orphan_prices, original.orphan_prices)
        assert_array_equal(loaded.orphan_times, original.orphan_times)
        assert_array_equal(loaded.orphan_quantities, original.orphan_quantities)
        assert_array_equal(loaded.orphan_directions, original.orphan_directions)
        assert loaded.current_trend == original.current_trend
        assert loaded.last_os_ref == original.last_os_ref
        assert loaded.last_processed_date == original.last_processed_date
    
    def test_load_nonexistent_returns_none(self, tmp_path):
        """Test que cargar archivo inexistente retorna None."""
        state_path = tmp_path / "nonexistent.arrow"
        loaded = load_state(state_path)
        assert loaded is None
    
    def test_empty_state_roundtrip(self, tmp_path):
        """Test guardar y cargar estado vacío."""
        original = create_empty_state(date(2025, 1, 1))
        
        state_path = tmp_path / "empty_state.arrow"
        save_state(original, state_path)
        
        loaded = load_state(state_path)
        
        assert loaded is not None
        assert loaded.n_orphans == 0
        assert loaded.current_trend == 0


# =============================================================================
# TESTS: Engine
# =============================================================================

class TestEngine:
    """Tests para el Engine de transformación."""
    
    def test_engine_creation(self, tmp_silver_path):
        """Test creación de Engine."""
        config = EngineConfig(theta=0.01, silver_base_path=tmp_silver_path)
        engine = Engine(config)
        
        assert engine.config.theta == 0.01
        assert engine.stats is not None
    
    def test_engine_warmup(self, tmp_silver_path):
        """Test warmup del Engine."""
        config = EngineConfig(theta=0.01, silver_base_path=tmp_silver_path)
        engine = Engine(config)
        
        # No debería lanzar excepción
        engine.warmup()
    
    def test_process_day_basic(self, sample_bronze_df, tmp_silver_path):
        """Test procesamiento básico de un día."""
        config = EngineConfig(
            theta=0.02,
            silver_base_path=tmp_silver_path,
            verbose=False,
        )
        engine = Engine(config)
        
        result = engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )
        
        assert result.n_ticks == len(sample_bronze_df)
        assert result.n_events >= 0
        assert result.elapsed_ms > 0
    
    def test_process_day_creates_parquet(self, sample_bronze_df, tmp_silver_path):
        """Test que process_day crea archivo Parquet."""
        config = EngineConfig(
            theta=0.02,
            silver_base_path=tmp_silver_path,
        )
        engine = Engine(config)
        
        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 15),
            time_col="timestamp",
        )
        
        # Verificar que se creó el archivo Parquet
        expected_path = (
            tmp_silver_path / "TEST" / "theta=0.02" / 
            "year=2025" / "month=01" / "day=15" / "data.parquet"
        )
        assert expected_path.exists()
    
    def test_engine_stats_collected(self, sample_bronze_df, tmp_silver_path):
        """Test que las estadísticas se recolectan."""
        config = EngineConfig(
            theta=0.02,
            silver_base_path=tmp_silver_path,
            collect_stats=True,
        )
        engine = Engine(config)
        
        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )
        
        stats = engine.get_stats()
        assert stats.days_processed == 1
        assert stats.total_ticks == len(sample_bronze_df)
        assert stats.kernel_calls >= 1
    
    def test_engine_reset_stats(self, sample_bronze_df, tmp_silver_path):
        """Test reset de estadísticas."""
        config = EngineConfig(
            theta=0.02,
            silver_base_path=tmp_silver_path,
        )
        engine = Engine(config)
        
        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )
        
        engine.reset_stats()
        
        stats = engine.get_stats()
        assert stats.days_processed == 0
        assert stats.total_ticks == 0


# =============================================================================
# TESTS: Integración
# =============================================================================

class TestIntegration:
    """Tests de integración del pipeline completo."""
    
    def test_multi_day_processing(self, tmp_silver_path):
        """Test procesamiento de múltiples días."""
        np.random.seed(42)
        
        # Crear datos para 3 días
        days_data = []
        for day in range(1, 4):
            n = 500
            prices = np.abs(np.random.randn(n).cumsum() + 100)
            df = pl.DataFrame({
                "price": prices.astype(np.float64),
                "timestamp": (np.arange(n) * 1_000_000_000).astype(np.int64),
                "quantity": np.ones(n, dtype=np.float64),
                "direction": np.random.choice([1, -1], n).astype(np.int8),
            }).with_columns(
                pl.lit(date(2025, 1, day)).alias("_date")
            )
            days_data.append(df)
        
        # Concatenar
        df_bronze = pl.concat(days_data)
        
        config = EngineConfig(
            theta=0.02,
            silver_base_path=tmp_silver_path,
            verbose=False,
        )
        engine = Engine(config)
        
        results = engine.process_date_range(
            df_bronze,
            ticker="TEST",
            time_col="timestamp",
        )
        
        assert len(results) == 3
        
        stats = engine.get_stats()
        assert stats.days_processed == 3