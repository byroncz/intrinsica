"""
Tests para el núcleo de cálculo DC.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from intrinseca import DCDetector, DCIndicators, TrendState
from intrinseca.core.bridging import to_numpy, to_polars


class TestDCDetector:
    """Tests para DCDetector."""
    
    def test_initialization(self):
        """Test de inicialización básica."""
        detector = DCDetector(theta=0.01)
        assert detector.theta == 0.01
        assert detector.compute_overshoot is True
    
    def test_invalid_theta(self):
        """Test que theta inválido lanza excepción."""
        with pytest.raises(ValueError):
            DCDetector(theta=0)
        with pytest.raises(ValueError):
            DCDetector(theta=1.5)
        with pytest.raises(ValueError):
            DCDetector(theta=-0.01)
    
    def test_detect_simple_uptrend(self):
        """Test detección de upturn simple."""
        # Serie con subida del 5%
        prices = np.array([100.0, 100.5, 101.0, 102.0, 105.0, 106.0])
        detector = DCDetector(theta=0.02)  # 2%
        
        result = detector.detect(prices)
        
        # Debería detectar al menos un upturn
        assert result.n_events >= 1
        assert result.n_upturns >= 1
    
    def test_detect_simple_downtrend(self):
        """Test detección de downturn simple."""
        # Serie con caída del 5%
        prices = np.array([100.0, 99.5, 99.0, 98.0, 95.0, 94.0])
        detector = DCDetector(theta=0.02)
        
        result = detector.detect(prices)
        
        assert result.n_events >= 1
        assert result.n_downturns >= 1
    
    def test_detect_alternating(self):
        """Test con tendencias alternantes."""
        # Subida, bajada, subida
        prices = np.array([
            100.0, 101.0, 102.0, 103.0,  # Subida
            102.0, 101.0, 100.0, 99.0,   # Bajada
            100.0, 101.0, 102.0, 103.0   # Subida
        ])
        detector = DCDetector(theta=0.02)
        
        result = detector.detect(prices)
        
        # Debería detectar múltiples eventos
        assert result.n_events >= 2
    
    def test_detect_no_events(self):
        """Test con serie sin movimientos significativos."""
        # Serie casi plana
        prices = np.array([100.0, 100.1, 100.0, 100.1, 100.0])
        detector = DCDetector(theta=0.05)  # 5% - muy alto para esta serie
        
        result = detector.detect(prices)
        
        # No debería detectar eventos
        assert result.n_events == 0
    
    def test_detect_with_polars(self):
        """Test que acepta DataFrame de Polars."""
        pytest.importorskip("polars")
        import polars as pl
        
        prices = np.array([100.0, 102.0, 104.0, 103.0, 101.0, 103.0])
        df = pl.DataFrame({"close": prices, "volume": [1000] * len(prices)})
        
        detector = DCDetector(theta=0.01)
        result = detector.detect(df, price_column="close")
        
        assert result.n_events >= 0  # Puede o no tener eventos
    
    def test_trends_array_length(self):
        """Test que el array de tendencias tiene longitud correcta."""
        prices = np.random.randn(1000).cumsum() + 100
        detector = DCDetector(theta=0.01)
        
        result = detector.detect(prices)
        
        assert len(result.trends) == len(prices)
    
    def test_event_attributes(self):
        """Test que los eventos tienen todos los atributos."""
        prices = np.array([100.0, 105.0, 100.0, 105.0, 100.0])
        detector = DCDetector(theta=0.02)
        
        result = detector.detect(prices)
        
        if result.n_events > 0:
            event = result.events[0]
            assert hasattr(event, "index")
            assert hasattr(event, "price")
            assert hasattr(event, "event_type")
            assert hasattr(event, "extreme_index")
            assert hasattr(event, "extreme_price")
            assert hasattr(event, "dc_return")
            assert hasattr(event, "dc_duration")
    
    def test_streaming_detection(self):
        """Test detección en modo streaming."""
        detector = DCDetector(theta=0.02)
        
        prices = [100.0, 102.0, 104.0, 103.0, 101.0, 103.0, 105.0]
        
        state = None
        events_detected = []
        
        for price in prices:
            event, state = detector.detect_streaming(price, state)
            if event:
                events_detected.append(event)
        
        # Verificar que el estado se mantiene
        assert state is not None
        assert "trend" in state
        assert "extreme_high" in state


class TestDCIndicators:
    """Tests para DCIndicators."""
    
    def test_compute_metrics_empty(self):
        """Test métricas con resultado vacío."""
        from intrinseca.core.event_detector import DCResult
        
        result = DCResult()
        indicators = DCIndicators()
        
        metrics = indicators.compute_metrics(result)
        
        assert metrics.n_events == 0
        assert metrics.tmv == 0.0
    
    def test_compute_metrics(self):
        """Test cálculo de métricas."""
        prices = np.random.randn(500).cumsum() + 100
        prices = np.abs(prices)  # Asegurar positivos
        
        detector = DCDetector(theta=0.01)
        result = detector.detect(prices)
        
        indicators = DCIndicators()
        metrics = indicators.compute_metrics(result)
        
        # TMV debe ser positivo si hay eventos
        if result.n_events > 0:
            assert metrics.tmv > 0
            assert 0 <= metrics.upturn_ratio <= 1
    
    def test_rolling_metrics(self):
        """Test métricas rolling."""
        prices = np.random.randn(1000).cumsum() + 100
        prices = np.abs(prices)
        
        detector = DCDetector(theta=0.01)
        result = detector.detect(prices)
        
        indicators = DCIndicators()
        rolling = indicators.compute_rolling_metrics(result, window=10)
        
        assert "rolling_tmv" in rolling
        assert "rolling_duration" in rolling
    
    def test_extract_features(self):
        """Test extracción de features para ML."""
        prices = np.random.randn(500).cumsum() + 100
        prices = np.abs(prices)
        
        detector = DCDetector(theta=0.01)
        result = detector.detect(prices)
        
        indicators = DCIndicators()
        features = indicators.extract_features(result, n_last_events=5)
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1


class TestBridging:
    """Tests para funciones de bridging."""
    
    def test_to_numpy_from_list(self):
        """Test conversión desde lista."""
        data = [1.0, 2.0, 3.0]
        arr = to_numpy(data)
        
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert_array_equal(arr, [1.0, 2.0, 3.0])
    
    def test_to_numpy_from_array(self):
        """Test conversión desde array."""
        data = np.array([1, 2, 3], dtype=np.int32)
        arr = to_numpy(data, dtype=np.float64)
        
        assert arr.dtype == np.float64
    
    def test_to_numpy_from_polars(self):
        """Test conversión desde Polars."""
        pl = pytest.importorskip("polars")
        
        series = pl.Series("prices", [1.0, 2.0, 3.0])
        arr = to_numpy(series)
        
        assert isinstance(arr, np.ndarray)
        assert_allclose(arr, [1.0, 2.0, 3.0])
    
    def test_to_polars_from_dict(self):
        """Test conversión a Polars desde dict."""
        pl = pytest.importorskip("polars")
        
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = to_polars(data)
        
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 2)


# Benchmark tests (requieren pytest-benchmark)
class TestPerformance:
    """Tests de rendimiento."""
    
    @pytest.mark.benchmark
    def test_detection_speed(self, benchmark):
        """Benchmark de velocidad de detección."""
        prices = np.random.randn(10000).cumsum() + 100
        prices = np.abs(prices)
        detector = DCDetector(theta=0.01)
        
        # Warm-up
        detector.detect(prices)
        
        # Benchmark
        result = benchmark(detector.detect, prices)
        
        assert result.n_events > 0