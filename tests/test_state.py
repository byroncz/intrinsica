"""Tests para el módulo de gestión de estado (state.py).

Cubre:
- DCState dataclass
- Persistencia Arrow IPC
- Búsqueda de estados previos
- Utilidades de diagnóstico
"""

from datetime import date

import numpy as np
import pytest
from intrinseca.core.state import (
    DCState,
    build_state_path,
    create_empty_state,
    find_previous_state,
    format_theta,
    get_state_stats,
    list_available_states,
    load_state,
    save_state,
)
from numpy.testing import assert_array_equal

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def empty_state():
    """Estado vacío para pruebas."""
    return create_empty_state(date(2025, 1, 1))


@pytest.fixture
def state_with_orphans():
    """Estado con ticks huérfanos."""
    return DCState(
        orphan_prices=np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
        orphan_times=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        orphan_quantities=np.array([1.0, 2.0, 1.5, 1.0], dtype=np.float64),
        orphan_directions=np.array([1, 1, -1, -1], dtype=np.int8),
        current_trend=np.int8(1),
        last_os_ref=np.float64(100.0),
        reference_extreme_price=np.float64(103.0),
        reference_extreme_time=np.int64(4000),
        last_processed_date=date(2025, 6, 15),
    )


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Directorio temporal para estados."""
    state_dir = tmp_path / "states"
    state_dir.mkdir(parents=True)
    return state_dir


# =============================================================================
# TESTS: DCState
# =============================================================================


class TestDCState:
    """Tests para el dataclass DCState."""

    def test_empty_state_creation(self, empty_state):
        """Test creación de estado vacío."""
        assert empty_state.n_orphans == 0
        assert empty_state.current_trend == 0
        assert empty_state.last_os_ref == 0.0

    def test_state_with_orphans_count(self, state_with_orphans):
        """Test conteo de huérfanos."""
        assert state_with_orphans.n_orphans == 4

    def test_state_current_trend(self, state_with_orphans):
        """Test tendencia actual."""
        assert state_with_orphans.current_trend == 1

    def test_state_date(self, state_with_orphans):
        """Test fecha del estado."""
        assert state_with_orphans.last_processed_date == date(2025, 6, 15)

    def test_get_extreme_prices(self, state_with_orphans):
        """Test obtención de precios extremos."""
        high, low = state_with_orphans.get_extreme_prices()

        assert high == 103.0
        assert low == 100.0

    def test_get_extreme_prices_empty(self, empty_state):
        """Test extremos con estado vacío."""
        high, low = empty_state.get_extreme_prices()

        # Valores por defecto para estado vacío
        assert high == 0.0 or np.isnan(high)

    def test_memory_usage_bytes(self, state_with_orphans):
        """Test estimación de memoria."""
        memory = state_with_orphans.memory_usage_bytes()

        # 4 orphans × (8+8+8+1) bytes = 100 bytes mínimo
        assert memory >= 100

    def test_memory_usage_empty(self, empty_state):
        """Test memoria de estado vacío."""
        memory = empty_state.memory_usage_bytes()

        # Estado vacío debería usar muy poca memoria
        assert memory >= 0


# =============================================================================
# TESTS: Utilidades
# =============================================================================


class TestFormatTheta:
    """Tests para format_theta."""

    def test_format_simple(self):
        """Test formateo simple."""
        assert format_theta(0.005) == "0.005"
        assert format_theta(0.01) == "0.01"
        assert format_theta(0.02) == "0.02"

    def test_format_removes_trailing_zeros(self):
        """Test que remueve ceros trailing."""
        assert format_theta(0.010000) == "0.01"
        assert format_theta(0.100000) == "0.1"

    def test_format_small_theta(self):
        """Test theta pequeño."""
        assert format_theta(0.001) == "0.001"
        assert format_theta(0.0005) == "0.0005"


class TestBuildStatePath:
    """Tests para build_state_path."""

    def test_path_structure(self, tmp_state_dir):
        """Test estructura del path."""
        path = build_state_path(
            tmp_state_dir,
            ticker="BTCUSDT",
            theta=0.005,
            year=2025,
            month=6,
            day=15,
        )

        assert "BTCUSDT" in str(path)
        assert "theta=0.005" in str(path)
        assert "year=2025" in str(path)
        assert "month=06" in str(path)
        assert "day=15" in str(path)

    def test_path_has_arrow_extension(self, tmp_state_dir):
        """Test que el path termina en .arrow."""
        path = build_state_path(
            tmp_state_dir,
            ticker="TEST",
            theta=0.01,
            year=2025,
            month=1,
            day=1,
        )

        assert str(path).endswith(".arrow")


# =============================================================================
# TESTS: Persistencia
# =============================================================================


class TestStatePersistence:
    """Tests para save_state y load_state."""

    def test_save_creates_file(self, state_with_orphans, tmp_path):
        """Test que save crea archivo."""
        path = tmp_path / "test_state.arrow"

        save_state(state_with_orphans, path)

        assert path.exists()

    def test_load_returns_state(self, state_with_orphans, tmp_path):
        """Test que load retorna estado."""
        path = tmp_path / "test_state.arrow"
        save_state(state_with_orphans, path)

        loaded = load_state(path)

        assert loaded is not None
        assert isinstance(loaded, DCState)

    def test_roundtrip_preserves_orphans(self, state_with_orphans, tmp_path):
        """Test que roundtrip preserva huérfanos."""
        path = tmp_path / "test_state.arrow"
        save_state(state_with_orphans, path)
        loaded = load_state(path)

        assert_array_equal(loaded.orphan_prices, state_with_orphans.orphan_prices)
        assert_array_equal(loaded.orphan_times, state_with_orphans.orphan_times)
        assert_array_equal(loaded.orphan_quantities, state_with_orphans.orphan_quantities)
        assert_array_equal(loaded.orphan_directions, state_with_orphans.orphan_directions)

    def test_roundtrip_preserves_scalars(self, state_with_orphans, tmp_path):
        """Test que roundtrip preserva escalares."""
        path = tmp_path / "test_state.arrow"
        save_state(state_with_orphans, path)
        loaded = load_state(path)

        assert loaded.current_trend == state_with_orphans.current_trend
        assert loaded.last_os_ref == state_with_orphans.last_os_ref
        assert loaded.last_processed_date == state_with_orphans.last_processed_date

    def test_roundtrip_preserves_reference_extreme(self, state_with_orphans, tmp_path):
        """Test que roundtrip preserva extremo de referencia."""
        path = tmp_path / "test_state.arrow"
        save_state(state_with_orphans, path)
        loaded = load_state(path)

        assert loaded.reference_extreme_price == state_with_orphans.reference_extreme_price
        assert loaded.reference_extreme_time == state_with_orphans.reference_extreme_time

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Test que load de archivo inexistente retorna None."""
        path = tmp_path / "nonexistent.arrow"

        loaded = load_state(path)

        assert loaded is None

    def test_empty_state_roundtrip(self, empty_state, tmp_path):
        """Test roundtrip de estado vacío."""
        path = tmp_path / "empty_state.arrow"
        save_state(empty_state, path)
        loaded = load_state(path)

        assert loaded is not None
        assert loaded.n_orphans == 0
        assert loaded.current_trend == 0


# =============================================================================
# TESTS: find_previous_state
# =============================================================================


class TestFindPreviousState:
    """Tests para find_previous_state."""

    def test_finds_previous_day(self, state_with_orphans, tmp_path):
        """Test que encuentra estado del día anterior."""
        # Guardar estado del día 15
        path = build_state_path(tmp_path, "TEST", 0.01, 2025, 6, 15)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(state_with_orphans, path)

        # Buscar desde día 16
        found = find_previous_state(
            tmp_path,
            ticker="TEST",
            theta=0.01,
            current_date=date(2025, 6, 16),
        )

        assert found is not None

    def test_finds_with_gap(self, state_with_orphans, tmp_path):
        """Test que encuentra estado con días de gap."""
        # Guardar estado del día 10
        path = build_state_path(tmp_path, "TEST", 0.01, 2025, 6, 10)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(state_with_orphans, path)

        # Buscar desde día 15 (5 días después)
        found = find_previous_state(
            tmp_path,
            ticker="TEST",
            theta=0.01,
            current_date=date(2025, 6, 15),
        )

        assert found is not None

    def test_returns_none_when_not_found(self, tmp_path):
        """Test que retorna None si no existe."""
        found = find_previous_state(
            tmp_path,
            ticker="NONEXISTENT",
            theta=0.01,
            current_date=date(2025, 6, 15),
        )

        assert found is None


# =============================================================================
# TESTS: create_empty_state
# =============================================================================


class TestCreateEmptyState:
    """Tests para create_empty_state."""

    def test_creates_with_date(self):
        """Test creación con fecha."""
        state = create_empty_state(date(2025, 12, 25))

        assert state.last_processed_date == date(2025, 12, 25)

    def test_has_zero_orphans(self):
        """Test que tiene cero huérfanos."""
        state = create_empty_state(date(2025, 1, 1))

        assert state.n_orphans == 0
        assert len(state.orphan_prices) == 0
        assert len(state.orphan_times) == 0

    def test_neutral_trend(self):
        """Test tendencia neutral."""
        state = create_empty_state(date(2025, 1, 1))

        assert state.current_trend == 0

    def test_zero_references(self):
        """Test referencias en cero."""
        state = create_empty_state(date(2025, 1, 1))

        assert state.last_os_ref == 0.0
        assert state.reference_extreme_price == 0.0


# =============================================================================
# TESTS: Diagnósticos
# =============================================================================


class TestStateDiagnostics:
    """Tests para funciones de diagnóstico."""

    def test_list_available_states(self, state_with_orphans, tmp_path):
        """Test listar estados disponibles."""
        # Crear varios estados
        for day in [10, 15, 20]:
            path = build_state_path(tmp_path, "TEST", 0.01, 2025, 6, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_state(state_with_orphans, path)

        states = list(list_available_states(tmp_path, "TEST", 0.01))

        assert len(states) == 3

    def test_get_state_stats(self, state_with_orphans, tmp_path):
        """Test obtener estadísticas de estados."""
        # Crear estado en la estructura correcta
        path = build_state_path(tmp_path, "TEST", 0.01, 2025, 6, 15)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(state_with_orphans, path)

        stats = get_state_stats(tmp_path, "TEST", 0.01)

        assert stats is not None
        assert "n_states" in stats
        assert stats["n_states"] == 1
