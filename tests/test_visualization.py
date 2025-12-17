"""
Tests para el módulo de visualización.
"""

import numpy as np
import pytest


# Solo ejecutar si matplotlib está disponible
matplotlib = pytest.importorskip("matplotlib")


class TestStaticPlots:
    """Tests para gráficos estáticos."""
    
    @pytest.fixture
    def sample_data(self):
        """Genera datos de prueba."""
        from intrinseca import DCDetector
        
        np.random.seed(42)
        prices = np.random.randn(500).cumsum() + 100
        prices = np.abs(prices)
        
        detector = DCDetector(theta=0.02)
        result = detector.detect(prices)
        
        return prices, result
    
    def test_plot_dc_events(self, sample_data):
        """Test generación de gráfico de eventos."""
        from intrinseca.visualization import plot_dc_events
        
        prices, result = sample_data
        fig = plot_dc_events(prices, result)
        
        assert fig is not None
        matplotlib.pyplot.close(fig)
    
    def test_plot_dc_summary(self, sample_data):
        """Test generación de panel resumen."""
        from intrinseca.visualization import plot_dc_summary
        
        prices, result = sample_data
        fig = plot_dc_summary(prices, result)
        
        assert fig is not None
        matplotlib.pyplot.close(fig)
    
    def test_plot_coastline(self, sample_data):
        """Test gráfico de coastline."""
        from intrinseca.visualization import plot_coastline
        from intrinseca import DCIndicators
        
        prices, _ = sample_data
        indicators = DCIndicators()
        coastline = indicators.compute_coastline(prices, [0.005, 0.01, 0.02, 0.05])
        
        fig = plot_coastline(coastline)
        
        assert fig is not None
        matplotlib.pyplot.close(fig)
    
    def test_plot_event_distribution(self, sample_data):
        """Test gráfico de distribución de eventos."""
        from intrinseca.visualization import plot_event_distribution
        
        _, result = sample_data
        fig = plot_event_distribution(result)
        
        assert fig is not None
        matplotlib.pyplot.close(fig)


# Tests de Plotly (opcional)
class TestInteractivePlots:
    """Tests para gráficos interactivos."""
    
    @pytest.fixture
    def sample_data(self):
        """Genera datos de prueba."""
        from intrinseca import DCDetector
        
        np.random.seed(42)
        prices = np.random.randn(500).cumsum() + 100
        prices = np.abs(prices)
        
        detector = DCDetector(theta=0.02)
        result = detector.detect(prices)
        
        return prices, result
    
    def test_create_interactive_chart(self, sample_data):
        """Test gráfico interactivo."""
        plotly = pytest.importorskip("plotly")
        from intrinseca.visualization.interactive import create_interactive_chart
        
        prices, result = sample_data
        fig = create_interactive_chart(prices, result)
        
        assert fig is not None