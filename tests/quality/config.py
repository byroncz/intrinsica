"""Configuración del framework de calidad de datos Silver.

NO MODIFICAR los paths sin actualizar QUALITY_TEST_PLAN.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """Severidad de las validaciones."""

    CRITICAL = "critical"  # Falla el test suite completo
    HIGH = "high"  # Falla el test pero continúa
    MEDIUM = "medium"  # Warning con impacto en score
    LOW = "low"  # Warning informativo
    INFO = "info"  # Solo para reporte


class ValidationLevel(Enum):
    """Nivel de exigencia del framework."""

    STRICT = "strict"  # Falla en cualquier warning
    NORMAL = "normal"  # Falla solo en CRITICAL/HIGH
    PERMISSIVE = "permissive"  # Solo reporta, nunca falla


@dataclass(frozen=True)
class QualityConfig:
    """Configuración inmutable del framework."""

    # Paths - NO CAMBIAR SIN ACTUALIZAR QUALITY_TEST_PLAN.md
    silver_base_path: Path = field(
        default_factory=lambda: Path(
            "/Users/byroncampo/My Drive/datalake/financial/02_silver"
        )
    )

    # Defaults
    default_ticker: str = "BTCUSDT"
    default_theta: float = 0.005

    # Tolerancias numéricas
    theta_tolerance: float = 1e-9  # Para comparaciones de umbral
    price_tolerance: float = 1e-12  # Para comparaciones de precio

    # Límites estadísticos
    max_zero_os_rate: float = 0.30  # 30% máximo de OS vacíos
    os_magnitude_factor_min: float = 0.5  # avg(OS) > 0.5 * theta
    os_magnitude_factor_max: float = 2.0  # avg(OS) < 2.0 * theta
    tmv_factor_min: float = 1.5  # avg(TMV) > 1.5 * theta
    tmv_factor_max: float = 3.0  # avg(TMV) < 3.0 * theta
    max_slippage_factor: float = 2.0  # slippage < 2 * theta

    # Columnas requeridas en Silver (orden exacto)
    required_columns: tuple[str, ...] = (
        "event_type",
        "reference_price",
        "reference_time",
        "extreme_price",
        "extreme_time",
        "confirm_price",
        "confirm_time",
        "price_dc",
        "price_os",
        "time_dc",
        "time_os",
        "qty_dc",
        "qty_os",
        "dir_dc",
        "dir_os",
    )

    # Tipos esperados (para validación)
    expected_types: dict[str, str] = field(
        default_factory=lambda: {
            "event_type": "Int8",
            "reference_price": "Float64",
            "reference_time": "Int64",
            "extreme_price": "Float64",
            "extreme_time": "Int64",
            "confirm_price": "Float64",
            "confirm_time": "Int64",
            "price_dc": "List(Float64)",
            "price_os": "List(Float64)",
            "time_dc": "List(Int64)",
            "time_os": "List(Int64)",
            "qty_dc": "List(Float64)",
            "qty_os": "List(Float64)",
            "dir_dc": "List(Int8)",
            "dir_os": "List(Int8)",
        }
    )


# Singleton de configuración
CONFIG = QualityConfig()
