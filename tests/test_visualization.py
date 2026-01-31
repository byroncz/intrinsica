"""Tests para el módulo de visualización.

Estos tests usan datos mock del Silver Layer en vez de DCDetector legacy.
Marcados como @pytest.mark.slow porque involucran renderizado gráfico.
"""

import numpy as np
import polars as pl
import pytest

# Solo ejecutar si matplotlib está disponible
matplotlib = pytest.importorskip("matplotlib")


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_silver_df():
    """DataFrame Silver mock para visualización."""
    np.random.seed(42)
    n_events = 20

    # Generar eventos alternantes
    event_types = np.array([1 if i % 2 == 0 else -1 for i in range(n_events)], dtype=np.int8)

    # Precios base que oscilan
    base_price = 100.0
    reference_prices = []
    extreme_prices = []
    confirm_prices = []

    for i, et in enumerate(event_types):
        ref = base_price + i * 0.5
        if et == 1:  # Upturn
            ext = ref * 1.03
            conf = ref * 1.02
        else:  # Downturn
            ext = ref * 0.97
            conf = ref * 0.98
        reference_prices.append(ref)
        extreme_prices.append(ext)
        confirm_prices.append(conf)

    # Timestamps espaciados
    base_time = 1704067200_000_000_000  # 2024-01-01 00:00:00
    reference_times = [base_time + i * 3600_000_000_000 for i in range(n_events)]
    extreme_times = [t + 1800_000_000_000 for t in reference_times]
    confirm_times = [t + 2400_000_000_000 for t in reference_times]

    return pl.DataFrame(
        {
            "event_type": event_types,
            "reference_price": np.array(reference_prices, dtype=np.float64),
            "extreme_price": np.array(extreme_prices, dtype=np.float64),
            "confirm_price": np.array(confirm_prices, dtype=np.float64),
            "reference_time": np.array(reference_times, dtype=np.int64),
            "extreme_time": np.array(extreme_times, dtype=np.int64),
            "confirm_time": np.array(confirm_times, dtype=np.int64),
        }
    )


@pytest.fixture
def sample_ticks():
    """Datos de ticks para visualización."""
    np.random.seed(42)
    n = 500
    # Precios con tendencia y reversiones
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 5
    noise = np.random.randn(n) * 0.5
    prices = 100.0 + trend + noise.cumsum() * 0.05
    prices = np.abs(prices)  # Asegurar positivos
    return prices


@pytest.fixture
def sample_timestamps(sample_ticks):
    """Timestamps correspondientes a los ticks."""
    n = len(sample_ticks)
    base_time = 1704067200_000_000_000
    return np.array([base_time + i * 1_000_000_000 for i in range(n)], dtype=np.int64)


# =============================================================================
# TESTS: Gráficos Estáticos
# =============================================================================


@pytest.mark.slow
class TestStaticPlots:
    """Tests para gráficos estáticos con matplotlib."""

    def test_matplotlib_import(self):
        """Verifica que matplotlib está disponible."""
        import matplotlib.pyplot as plt

        assert plt is not None

    def test_basic_price_plot(self, sample_ticks):
        """Test que podemos crear un plot básico de precios."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(sample_ticks)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Price")

        assert fig is not None
        plt.close(fig)

    def test_silver_data_visualization(self, sample_silver_df, sample_ticks):
        """Test visualización de datos Silver Layer."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot de precios base
        ax.plot(sample_ticks, label="Price", alpha=0.7)

        # Marcar eventos del Silver Layer
        df = sample_silver_df
        upturns = df.filter(pl.col("event_type") == 1)
        downturns = df.filter(pl.col("event_type") == -1)

        # Solo verificamos que no lance error
        assert fig is not None
        assert len(upturns) > 0
        assert len(downturns) > 0

        plt.close(fig)

    def test_event_distribution_histogram(self, sample_silver_df):
        """Test histograma de distribución de eventos."""
        import matplotlib.pyplot as plt

        df = sample_silver_df

        # Calcular duraciones (extreme_time - reference_time)
        durations = (
            df.select(
                ((pl.col("extreme_time") - pl.col("reference_time")) / 1e9).alias("duration_s")
            )
            .get_column("duration_s")
            .to_numpy()
        )

        fig, ax = plt.subplots()
        ax.hist(durations, bins=10, edgecolor="black")
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Event Duration Distribution")

        assert fig is not None
        plt.close(fig)

    def test_price_returns_distribution(self, sample_silver_df):
        """Test distribución de retornos DC."""
        import matplotlib.pyplot as plt

        df = sample_silver_df

        # Calcular retornos DC
        returns = (
            df.select(
                (
                    (pl.col("extreme_price") - pl.col("reference_price"))
                    / pl.col("reference_price")
                    * 100
                ).alias("dc_return_pct")
            )
            .get_column("dc_return_pct")
            .to_numpy()
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histograma
        axes[0].hist(returns, bins=15, edgecolor="black")
        axes[0].set_xlabel("DC Return (%)")
        axes[0].set_title("Return Distribution")

        # Boxplot
        axes[1].boxplot(returns)
        axes[1].set_ylabel("DC Return (%)")
        axes[1].set_title("Return Boxplot")

        assert fig is not None
        plt.close(fig)


# =============================================================================
# TESTS: Gráficos Interactivos
# =============================================================================


@pytest.mark.slow
class TestInteractivePlots:
    """Tests para gráficos interactivos con Plotly."""

    def test_plotly_available(self):
        """Verifica que plotly está disponible."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        assert go is not None

    def test_basic_plotly_chart(self, sample_ticks):
        """Test creación de gráfico básico con Plotly."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sample_ticks, mode="lines", name="Price"))

        assert fig is not None
        assert len(fig.data) == 1

    def test_silver_layer_candlestick_style(self, sample_silver_df):
        """Test visualización estilo candlestick de eventos DC."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        df = sample_silver_df

        # Extraer datos para visualización
        reference_prices = df.get_column("reference_price").to_numpy()
        extreme_prices = df.get_column("extreme_price").to_numpy()
        confirm_prices = df.get_column("confirm_price").to_numpy()
        event_types = df.get_column("event_type").to_numpy()

        fig = go.Figure()

        # Agregar traces para cada tipo de evento
        colors = ["green" if et == 1 else "red" for et in event_types]

        for i in range(len(reference_prices)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i, i],
                    y=[reference_prices[i], extreme_prices[i], confirm_prices[i]],
                    mode="lines+markers",
                    line={"color": colors[i]},
                    showlegend=False,
                )
            )

        assert fig is not None
        assert len(fig.data) == len(reference_prices)


# =============================================================================
# TESTS: Utilidades de Visualización
# =============================================================================


@pytest.mark.slow
class TestVisualizationUtils:
    """Tests para utilidades de visualización."""

    def test_color_palette_generation(self):
        """Test generación de paleta de colores."""
        import matplotlib.pyplot as plt

        # Paleta para upturns/downturns
        upturn_color = plt.cm.Greens(0.7)
        downturn_color = plt.cm.Reds(0.7)

        assert len(upturn_color) == 4  # RGBA
        assert len(downturn_color) == 4

    def test_timestamp_formatting(self, sample_silver_df):
        """Test formateo de timestamps para visualización."""
        from datetime import datetime

        df = sample_silver_df
        first_time = df.get_column("reference_time")[0]

        # Convertir nanosegundos a datetime
        dt = datetime.fromtimestamp(first_time / 1e9)
        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")

        assert "2024-01-01" in formatted

    def test_figure_export_to_bytes(self, sample_ticks):
        """Test exportar figura a bytes (para integración web)."""
        from io import BytesIO

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(sample_ticks)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)

        # Verificar que se generaron bytes
        data = buf.read()
        assert len(data) > 0
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

        plt.close(fig)
