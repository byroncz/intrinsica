"""Tests para el sistema de reconciliación retroactiva.

Verifica:
- Detección correcta del tipo de reconciliación
- Actualización atómica de Parquet
- Backup y restauración ante errores
"""

from datetime import date

import numpy as np
import polars as pl
import pytest
from intrinseca.core.reconciliation import (
    ReconciliationType,
    check_reconciliation_needed,
    cleanup_backup,
    reconcile_previous_day,
)
from intrinseca.core.state import DCState


class TestReconciliationDetection:
    """Tests para check_reconciliation_needed."""

    @pytest.fixture
    def uptrend_state(self):
        """Estado con tendencia alcista y huérfanos."""
        return DCState(
            orphan_prices=np.array([105.0, 106.0, 107.0], dtype=np.float64),
            orphan_times=np.array([1, 2, 3], dtype=np.int64),
            orphan_quantities=np.ones(3, dtype=np.float64),
            orphan_directions=np.array([1, 1, 1], dtype=np.int8),
            current_trend=np.int8(1),
            last_os_ref=np.float64(105.0),
            reference_extreme_price=np.float64(107.0),
            reference_extreme_time=np.int64(3),
            last_processed_date=date(2025, 1, 1),
        )

    @pytest.fixture
    def downtrend_state(self):
        """Estado con tendencia bajista y huérfanos."""
        return DCState(
            orphan_prices=np.array([95.0, 94.0, 93.0], dtype=np.float64),
            orphan_times=np.array([1, 2, 3], dtype=np.int64),
            orphan_quantities=np.ones(3, dtype=np.float64),
            orphan_directions=np.array([-1, -1, -1], dtype=np.int8),
            current_trend=np.int8(-1),
            last_os_ref=np.float64(95.0),
            reference_extreme_price=np.float64(93.0),
            reference_extreme_time=np.int64(3),
            last_processed_date=date(2025, 1, 1),
        )

    def test_confirm_reversal_from_uptrend(self, uptrend_state):
        """Reversión confirmada desde tendencia alcista."""
        theta = 0.02  # 2%
        # Precio que cruza threshold hacia abajo
        first_price = 107.0 * 0.97  # Más bajo que threshold

        recon_type, _ = check_reconciliation_needed(uptrend_state, first_price, theta)

        assert recon_type == ReconciliationType.CONFIRM_REVERSAL

    def test_extend_os_from_uptrend(self, uptrend_state):
        """Falsa alarma en tendencia alcista (precio sigue subiendo)."""
        theta = 0.02
        first_price = 108.0  # Más alto que el extremo alto

        recon_type, _ = check_reconciliation_needed(uptrend_state, first_price, theta)

        assert recon_type == ReconciliationType.EXTEND_OS

    def test_confirm_reversal_from_downtrend(self, downtrend_state):
        """Reversión confirmada desde tendencia bajista."""
        theta = 0.02
        # Precio que cruza threshold hacia arriba
        first_price = 93.0 * 1.03  # Más alto que threshold

        recon_type, _ = check_reconciliation_needed(downtrend_state, first_price, theta)

        assert recon_type == ReconciliationType.CONFIRM_REVERSAL

    def test_extend_os_from_downtrend(self, downtrend_state):
        """Falsa alarma en tendencia bajista (precio sigue bajando)."""
        theta = 0.02
        first_price = 92.0  # Más bajo que el extremo bajo

        recon_type, _ = check_reconciliation_needed(downtrend_state, first_price, theta)

        assert recon_type == ReconciliationType.EXTEND_OS

    def test_no_reconciliation_when_empty(self):
        """Sin reconciliación cuando no hay huérfanos."""
        empty_state = DCState(
            orphan_prices=np.array([], dtype=np.float64),
            orphan_times=np.array([], dtype=np.int64),
            orphan_quantities=np.array([], dtype=np.float64),
            orphan_directions=np.array([], dtype=np.int8),
            current_trend=np.int8(1),
            last_os_ref=np.float64(100.0),
            reference_extreme_price=np.float64(0),
            reference_extreme_time=np.int64(0),
            last_processed_date=date(2025, 1, 1),
        )

        recon_type, _ = check_reconciliation_needed(empty_state, 100.0, 0.02)

        assert recon_type == ReconciliationType.NONE


class TestReconciliationExecution:
    """Tests para reconcile_previous_day."""

    @pytest.fixture
    def sample_silver_df(self):
        """DataFrame Silver de ejemplo."""
        return pl.DataFrame(
            {
                "event_type": [1, -1],
                "reference_price": [100.0, 105.0],
                "reference_time": [1000, 2000],
                "extreme_price": [-1.0, -1.0],  # Sentinels
                "extreme_time": [-1, -1],
                "confirm_price": [103.0, 100.0],
                "confirm_time": [1500, 2500],
            }
        )

    def test_reconciliation_updates_parquet(self, sample_silver_df, tmp_path):
        """Verifica que la reconciliación actualiza el Parquet correctamente."""
        parquet_path = tmp_path / "test.parquet"
        sample_silver_df.write_parquet(parquet_path)

        result = reconcile_previous_day(
            silver_path=parquet_path,
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            new_extreme_price=107.0,
            new_extreme_time=3000,
        )

        assert result.success
        assert result.backup_path.exists()

        # Verificar que el último evento fue actualizado
        updated_df = pl.read_parquet(parquet_path)
        assert updated_df["extreme_price"][-1] == 107.0
        assert updated_df["extreme_time"][-1] == 3000

    def test_reconciliation_creates_backup(self, sample_silver_df, tmp_path):
        """Verifica que se crea backup antes de modificar."""
        parquet_path = tmp_path / "test.parquet"
        sample_silver_df.write_parquet(parquet_path)

        result = reconcile_previous_day(
            silver_path=parquet_path,
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            new_extreme_price=107.0,
            new_extreme_time=3000,
        )

        assert result.backup_path is not None
        assert result.backup_path.exists()

        # El backup debe ser igual al original
        backup_df = pl.read_parquet(result.backup_path)
        assert backup_df["extreme_price"][-1] == -1.0  # Valor original

    def test_cleanup_backup_removes_file(self, sample_silver_df, tmp_path):
        """Verifica que cleanup_backup elimina el archivo."""
        parquet_path = tmp_path / "test.parquet"
        sample_silver_df.write_parquet(parquet_path)

        result = reconcile_previous_day(
            silver_path=parquet_path,
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            new_extreme_price=107.0,
            new_extreme_time=3000,
        )

        assert result.backup_path.exists()
        cleanup_backup(result.backup_path)
        assert not result.backup_path.exists()
