"""Tests para el Engine de transformación Bronze → Silver.

Cubre:
- Configuración del Engine
- Procesamiento de un día
- Procesamiento de rango de fechas
- Diagnósticos y estadísticas
- Manejo de casos edge
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from intrinseca.core.engine import DayResult, Engine, EngineConfig, EngineStats

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tmp_silver_path(tmp_path):
    """Directorio temporal para Silver Layer."""
    silver_path = tmp_path / "silver"
    silver_path.mkdir(parents=True)
    return silver_path


@pytest.fixture
def engine_params(tmp_silver_path):
    """Parámetros básicos para crear un Engine."""
    return {
        "theta": 0.02,
        "silver_base_path": tmp_silver_path,
        "verbose": False,
        "collect_stats": True,
    }


@pytest.fixture
def sample_bronze_df():
    """DataFrame Bronze con datos de prueba."""
    np.random.seed(42)
    n = 500
    prices = np.abs(np.random.randn(n).cumsum() + 100)

    return pl.DataFrame(
        {
            "price": prices.astype(np.float64),
            "timestamp": (np.arange(n) * 1_000_000_000).astype(np.int64),
            "quantity": np.ones(n, dtype=np.float64),
            "direction": np.random.choice([1, -1], n).astype(np.int8),
        }
    )


@pytest.fixture
def multi_day_bronze_df():
    """DataFrame Bronze con múltiples días."""
    np.random.seed(42)
    dfs = []

    for day in range(1, 6):  # 5 días
        n = 300
        prices = np.abs(np.random.randn(n).cumsum() + 100)
        # Epoch en nanosegundos para 2025-01-XX 00:00:00 UTC
        # 2025-01-01 = 1735689600 segundos desde epoch
        base_epoch_ns = (1735689600 + (day - 1) * 86400) * 1_000_000_000
        timestamps = base_epoch_ns + np.arange(n) * 1_000_000_000
        df = pl.DataFrame(
            {
                "price": prices.astype(np.float64),
                "timestamp": timestamps.astype(np.int64),
                "quantity": np.ones(n, dtype=np.float64),
                "direction": np.random.choice([1, -1], n).astype(np.int8),
            }
        )
        dfs.append(df)

    return pl.concat(dfs)


# =============================================================================
# TESTS: EngineConfig
# =============================================================================


class TestEngineConfig:
    """Tests para EngineConfig."""

    def test_config_defaults(self, tmp_silver_path):
        """Test valores por defecto."""
        config = EngineConfig(theta=0.01, silver_base_path=tmp_silver_path)

        assert config.theta == 0.01
        assert config.verbose is False
        assert config.collect_stats is True
        assert config.keep_state_files is True

    def test_config_verbose_mode(self, tmp_silver_path):
        """Test modo verbose."""
        config = EngineConfig(
            theta=0.01,
            silver_base_path=tmp_silver_path,
            verbose=True,
        )
        assert config.verbose is True

    def test_theta_str_formatting(self, tmp_silver_path):
        """Test formateo de theta como string."""
        config1 = EngineConfig(theta=0.005, silver_base_path=tmp_silver_path)
        assert config1.theta_str == "0.005"

        config2 = EngineConfig(theta=0.02, silver_base_path=tmp_silver_path)
        assert config2.theta_str == "0.02"

        config3 = EngineConfig(theta=0.1, silver_base_path=tmp_silver_path)
        assert config3.theta_str == "0.1"

    def test_path_conversion(self, tmp_silver_path):
        """Test que string se convierte a Path."""
        config = EngineConfig(
            theta=0.01,
            silver_base_path=str(tmp_silver_path),
        )
        assert isinstance(config.silver_base_path, Path)


# =============================================================================
# TESTS: EngineStats
# =============================================================================


class TestEngineStats:
    """Tests para EngineStats."""

    def test_stats_initial_values(self):
        """Test valores iniciales."""
        stats = EngineStats()

        assert stats.days_processed == 0
        assert stats.total_events == 0
        assert stats.total_ticks == 0
        assert stats.total_time_ms == 0.0
        assert stats.kernel_calls == 0

    def test_avg_time_per_day_empty(self):
        """Test promedio con cero días."""
        stats = EngineStats()
        assert stats.avg_time_per_day_ms == 0.0

    def test_avg_time_per_day(self):
        """Test cálculo de promedio."""
        stats = EngineStats(
            days_processed=10,
            total_time_ms=1000.0,
        )
        assert stats.avg_time_per_day_ms == 100.0

    def test_throughput_calculation(self):
        """Test cálculo de throughput."""
        stats = EngineStats(
            total_ticks=1_000_000,
            total_time_ms=1000.0,  # 1 segundo
        )
        assert stats.throughput_ticks_per_sec == 1_000_000.0

    def test_events_per_day(self):
        """Test promedio de eventos por día."""
        stats = EngineStats(
            days_processed=5,
            total_events=100,
        )
        assert stats.events_per_day == 20.0

    def test_to_dict(self):
        """Test conversión a diccionario."""
        stats = EngineStats(
            days_processed=3,
            total_events=50,
            total_ticks=10000,
            total_time_ms=500.0,
            kernel_calls=3,
        )

        d = stats.to_dict()

        assert d["days_processed"] == 3
        assert d["total_events"] == 50
        assert d["throughput_ticks_per_sec"] == 20000.0


# =============================================================================
# TESTS: Engine Creation
# =============================================================================


class TestEngineCreation:
    """Tests para creación del Engine."""

    def test_engine_basic_creation(self, engine_params):
        """Test creación básica."""
        engine = Engine(**engine_params)

        assert engine.config.theta == engine_params["theta"]
        assert engine.stats is not None

    def test_engine_warmup(self, engine_params):
        """Test warmup del kernel."""
        engine = Engine(**engine_params, warmup=True)

        # Verificar que el kernel está compilado
        assert engine._compiled is True


# =============================================================================
# TESTS: Engine.process_day
# =============================================================================


class TestEngineProcessDay:
    """Tests para procesamiento de un día."""

    def test_process_day_returns_day_result(self, engine_params, sample_bronze_df):
        """Test que retorna DayResult."""
        engine = Engine(**engine_params)

        result = engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        assert isinstance(result, DayResult)

    def test_process_day_counts_ticks(self, engine_params, sample_bronze_df):
        """Test que cuenta ticks correctamente."""
        engine = Engine(**engine_params)

        result = engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        assert result.n_ticks == len(sample_bronze_df)

    def test_process_day_measures_time(self, engine_params, sample_bronze_df):
        """Test que mide tiempo de ejecución."""
        engine = Engine(**engine_params)

        result = engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        assert result.elapsed_ms > 0

    def test_process_day_creates_parquet(self, engine_params, sample_bronze_df):
        """Test que crea archivo Parquet."""
        engine = Engine(**engine_params)

        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 6, 15),
            time_col="timestamp",
        )

        expected_path = (
            engine_params["silver_base_path"]
            / "TEST"
            / f"theta={engine.config.theta_str}"
            / "year=2025"
            / "month=06"
            / "day=15"
            / "data.parquet"
        )

        assert expected_path.exists()

    def test_process_day_parquet_readable(self, engine_params, sample_bronze_df):
        """Test que el Parquet es legible."""
        engine = Engine(**engine_params)

        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        parquet_path = (
            engine_params["silver_base_path"]
            / "TEST"
            / f"theta={engine.config.theta_str}"
            / "year=2025"
            / "month=01"
            / "day=01"
            / "data.parquet"
        )

        df = pl.read_parquet(parquet_path)
        assert df is not None
        assert "event_type" in df.columns

    def test_process_day_updates_stats(self, engine_params, sample_bronze_df):
        """Test que actualiza estadísticas."""
        engine = Engine(**engine_params)

        initial_stats = engine.get_stats()
        assert initial_stats["days_processed"] == 0

        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        updated_stats = engine.get_stats()
        assert updated_stats["days_processed"] == 1
        assert updated_stats["kernel_calls"] >= 1


# =============================================================================
# TESTS: Engine.process_date_range
# =============================================================================


class TestEngineProcessDateRange:
    """Tests para procesamiento de rango de fechas."""

    def test_process_date_range_returns_tuple(self, engine_params, multi_day_bronze_df):
        """Test que retorna tupla (dict, report)."""
        engine = Engine(**engine_params)

        results, _ = engine.process_date_range(
            multi_day_bronze_df,
            ticker="TEST",
            time_col="timestamp",
        )

        assert isinstance(results, dict)

    def test_process_date_range_processes_all_days(self, engine_params, multi_day_bronze_df):
        """Test que procesa todos los días."""
        engine = Engine(**engine_params)

        results, _ = engine.process_date_range(
            multi_day_bronze_df,
            ticker="TEST",
            time_col="timestamp",
        )

        # 5 días en el fixture
        assert len(results) == 5

    def test_process_date_range_creates_files(self, engine_params, multi_day_bronze_df):
        """Test que crea archivos para cada día."""
        engine = Engine(**engine_params)

        engine.process_date_range(
            multi_day_bronze_df,
            ticker="TEST",
            time_col="timestamp",
        )

        # Verificar que existe archivo para cada día
        for day in range(1, 6):
            path = (
                engine_params["silver_base_path"]
                / "TEST"
                / f"theta={engine.config.theta_str}"
                / "year=2025"
                / "month=01"
                / f"day={day:02d}"
                / "data.parquet"
            )
            assert path.exists(), f"Falta archivo para día {day}"


# =============================================================================
# TESTS: Engine Diagnostics
# =============================================================================


class TestEngineDiagnostics:
    """Tests para métodos de diagnóstico."""

    def test_get_stats(self, engine_params):
        """Test obtener estadísticas."""
        engine = Engine(**engine_params)

        stats = engine.get_stats()

        # get_stats() retorna dict
        assert isinstance(stats, dict)

    def test_reset_stats(self, engine_params, sample_bronze_df):
        """Test reset de estadísticas."""
        engine = Engine(**engine_params)

        # Procesar algo
        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        assert engine.get_stats()["days_processed"] == 1

        # Reset
        engine.reset_stats()

        assert engine.get_stats()["days_processed"] == 0
        assert engine.get_stats()["total_ticks"] == 0

    def test_diagnose_method(self, engine_params, sample_bronze_df):
        """Test método diagnose."""
        engine = Engine(**engine_params)

        engine.process_day(
            sample_bronze_df,
            ticker="TEST",
            process_date=date(2025, 1, 1),
            time_col="timestamp",
        )

        # diagnose() debería retornar información útil
        diag = engine.diagnose()

        assert diag is not None
        assert isinstance(diag, dict)
