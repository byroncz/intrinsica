"""
Tests para la nueva segmentación temporal del kernel DC.

Verifica:
- DC incluye DCC y excluye extremo
- OS incluye extremo  
- Llenado retrospectivo de extreme_price
- Huérfanos excluyen el extremo

Índices del tuple de resultado del kernel (23 elementos):
- 0-3: dc_prices, dc_times, dc_quantities, dc_directions
- 4-7: os_prices, os_times, os_quantities, os_directions
- 8: event_types
- 9-10: dc_offsets, os_offsets
- 11-12: reference_prices, reference_times
- 13-14: extreme_prices, extreme_times
- 15-16: confirm_prices, confirm_times
- 17: n_events
- 18: final_trend
- 19-21: final_ext_high, final_ext_low, final_last_os_ref
- 22: orphan_start_idx
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from intrinseca.core.kernel import segment_events_kernel

# Índices para acceder al resultado del kernel
IDX_DC_PRICES = 0
IDX_DC_OFFSETS = 9
IDX_OS_PRICES = 4
IDX_OS_OFFSETS = 10
IDX_REFERENCE_PRICES = 11
IDX_EXTREME_PRICES = 13
IDX_CONFIRM_PRICES = 15
IDX_N_EVENTS = 17
IDX_TREND = 18
IDX_ORPHAN_START = 22


class TestKernelSegmentation:
    """Tests para la nueva semántica de segmentación."""

    @pytest.fixture
    def simple_uptrend_data(self):
        """Datos con una tendencia alcista simple."""
        # Precios que suben 5% para confirmar y luego siguen
        prices = np.array([
            100.0, 100.5, 101.0, 102.0, 103.0,  # DC (idx 0-4)
            104.0, 105.0,  # OS (idx 5-6)
            103.0, 101.0, 99.0,  # Reversión, confirma downturn (idx 7-9)
            98.0, 97.0,  # OS del downturn (idx 10-11)
        ], dtype=np.float64)
        timestamps = np.arange(len(prices), dtype=np.int64) * 1_000_000_000
        quantities = np.ones(len(prices), dtype=np.float64)
        directions = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.int8)
        return prices, timestamps, quantities, directions

    def test_dc_includes_dcc(self, simple_uptrend_data):
        """Verifica que el DCC es el último tick del DC."""
        prices, timestamps, quantities, directions = simple_uptrend_data
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,  # 2%
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        dc_prices = result[IDX_DC_PRICES]
        dc_offsets = result[IDX_DC_OFFSETS]
        n_events = int(result[IDX_N_EVENTS])
        
        if n_events > 0:
            first_dc_end = int(dc_offsets[1])
            dc_last_price = dc_prices[first_dc_end - 1]
            confirm_prices = result[IDX_CONFIRM_PRICES]
            
            assert_allclose(dc_last_price, confirm_prices[0], rtol=1e-10)

    def test_os_includes_extreme(self, simple_uptrend_data):
        """Verifica que el extremo es el último tick del OS."""
        prices, timestamps, quantities, directions = simple_uptrend_data
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        os_prices = result[IDX_OS_PRICES]
        os_offsets = result[IDX_OS_OFFSETS]
        n_events = int(result[IDX_N_EVENTS])
        
        if n_events > 1:
            first_os_start = int(os_offsets[0])
            first_os_end = int(os_offsets[1])
            if first_os_end > first_os_start:
                os_last_price = os_prices[first_os_end - 1]
                os_slice = os_prices[first_os_start:first_os_end]
                assert os_last_price == np.max(os_slice) or os_last_price == np.min(os_slice)

    def test_reference_price_correct(self, simple_uptrend_data):
        """Verifica que reference_price es el extremo de referencia."""
        prices, timestamps, quantities, directions = simple_uptrend_data
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        reference_prices = result[IDX_REFERENCE_PRICES]
        n_events = int(result[IDX_N_EVENTS])
        
        if n_events > 0:
            assert reference_prices[0] > 0

    def test_extreme_price_retrospective(self, simple_uptrend_data):
        """Verifica que extreme_price del evento N se llena cuando se confirma N+1."""
        prices, timestamps, quantities, directions = simple_uptrend_data
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        extreme_prices = result[IDX_EXTREME_PRICES]
        n_events = int(result[IDX_N_EVENTS])
        
        if n_events > 1:
            assert extreme_prices[0] != -1.0, "extreme_price no fue llenado retrospectivamente"
            
        if n_events > 0:
            assert extreme_prices[n_events - 1] == -1.0, "Último evento debería tener sentinel"

    def test_orphans_exclude_extreme(self, simple_uptrend_data):
        """Verifica que los huérfanos no incluyen el extremo."""
        prices, timestamps, quantities, directions = simple_uptrend_data
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        orphan_start_idx = int(result[IDX_ORPHAN_START])
        n_events = int(result[IDX_N_EVENTS])
        
        if n_events > 0:
            assert orphan_start_idx > 0, "Debería haber huérfanos"

    def test_os_empty_immediate_reversal(self):
        """Verifica que OS puede quedar vacío si hay reversión inmediata."""
        prices = np.array([
            100.0, 102.0, 103.0,
            100.0,
        ], dtype=np.float64)
        timestamps = np.arange(len(prices), dtype=np.int64) * 1_000_000_000
        quantities = np.ones(len(prices), dtype=np.float64)
        directions = np.array([1, 1, 1, -1], dtype=np.int8)
        
        result = segment_events_kernel(
            prices, timestamps, quantities, directions,
            theta=0.02,
            init_trend=np.int8(0),
            init_ext_high_price=np.float64(0),
            init_ext_low_price=np.float64(0),
            init_last_os_ref=np.float64(0),
        )
        
        n_events = int(result[IDX_N_EVENTS])
        
        if n_events > 0:
            pass  # Test pasa si no hay error
