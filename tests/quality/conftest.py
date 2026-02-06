"""Fixtures de pytest para el framework de calidad.

IMPORTANTE: Este archivo define cómo se cargan los datos Silver.
NO usar datos dummy. SIEMPRE cargar datos reales.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from .config import CONFIG, QualityConfig


def build_silver_path(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    day: int,
    base_path: Path | None = None,
) -> Path:
    """Construye el path a un archivo Silver específico.

    Args:
        ticker: Símbolo del instrumento (e.g., "BTCUSDT")
        theta: Umbral DC
        year: Año
        month: Mes
        day: Día
        base_path: Path base (default: CONFIG.silver_base_path)

    Returns:
        Path al archivo data.parquet

    Example:
        >>> build_silver_path("BTCUSDT", 0.005, 2025, 11, 15)
        PosixPath('/Users/.../02_silver/BTCUSDT/theta=0.005/year=2025/month=11/day=15/data.parquet')
    """
    if base_path is None:
        base_path = CONFIG.silver_base_path

    theta_str = f"{theta:.6f}".rstrip("0").rstrip(".")

    return (
        base_path
        / ticker
        / f"theta={theta_str}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"day={day:02d}"
        / "data.parquet"
    )


def list_available_days(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    base_path: Path | None = None,
) -> list[date]:
    """Lista los días disponibles en Silver para un mes dado.

    Returns:
        Lista de fechas ordenadas ascendentemente
    """
    if base_path is None:
        base_path = CONFIG.silver_base_path

    theta_str = f"{theta:.6f}".rstrip("0").rstrip(".")
    month_path = (
        base_path / ticker / f"theta={theta_str}" / f"year={year}" / f"month={month:02d}"
    )

    if not month_path.exists():
        return []

    days = []
    for day_dir in month_path.iterdir():
        if day_dir.is_dir() and day_dir.name.startswith("day="):
            day_num = int(day_dir.name.split("=")[1])
            parquet_path = day_dir / "data.parquet"
            if parquet_path.exists():
                days.append(date(year, month, day_num))

    return sorted(days)


def load_silver_day(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    day: int,
    base_path: Path | None = None,
) -> pl.DataFrame | None:
    """Carga datos Silver de un día específico.

    Returns:
        DataFrame Polars o None si no existe
    """
    path = build_silver_path(ticker, theta, year, month, day, base_path)

    if not path.exists():
        return None

    return pl.read_parquet(path)


def load_silver_month(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    base_path: Path | None = None,
) -> dict[date, pl.DataFrame]:
    """Carga todos los días Silver de un mes.

    Returns:
        Dict {date: DataFrame} ordenado por fecha
    """
    days = list_available_days(ticker, theta, year, month, base_path)
    result = {}

    for d in days:
        df = load_silver_day(ticker, theta, year, month, d.day, base_path)
        if df is not None:
            result[d] = df

    return result


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def quality_config() -> QualityConfig:
    """Configuración del framework."""
    return CONFIG


@pytest.fixture(scope="session")
def silver_base_path() -> Path:
    """Path base de Silver."""
    return CONFIG.silver_base_path


@pytest.fixture
def silver_month_loader():
    """Factory fixture para cargar meses Silver."""

    def _loader(
        year: int, month: int, ticker: str = "BTCUSDT", theta: float = 0.005
    ):
        return load_silver_month(ticker, theta, year, month)

    return _loader


@pytest.fixture
def silver_day_loader():
    """Factory fixture para cargar días Silver."""

    def _loader(
        year: int,
        month: int,
        day: int,
        ticker: str = "BTCUSDT",
        theta: float = 0.005,
    ):
        return load_silver_day(ticker, theta, year, month, day)

    return _loader
