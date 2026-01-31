"""Gestión de Estado para Silver Layer.

Manejo de estados transitorios y ticks huérfanos usando Apache Arrow IPC.
El formato Arrow garantiza isomorfismo de memoria (zero-copy) para máxima
eficiencia en el proceso de stitching entre días.

Arquitectura de Persistencia:
```
Silver Base Path
└── {ticker}/
    └── theta={theta}/
        └── year={YYYY}/
            └── month={MM}/
                └── day={DD}/
                    ├── data.parquet      # Eventos DC del día
                    └── state_{ticker}_{theta}.arrow  # Estado para stitching
```

El archivo state.arrow contiene:
- Columnas: price, time, quantity, direction (ticks huérfanos)
- Metadata: current_trend, last_os_ref, last_processed_date
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from numpy.typing import NDArray

# =============================================================================
# TYPE ALIASES
# =============================================================================

ArrayF64 = NDArray[np.float64]
ArrayI64 = NDArray[np.int64]
ArrayI8 = NDArray[np.int8]


# =============================================================================
# CONSTANTES
# =============================================================================

# Búsqueda de estado anterior
MAX_LOOKBACK_DAYS = 7

# Compresión Arrow IPC (uncompressed para máxima velocidad de lectura)
STATE_COMPRESSION = "uncompressed"

# Nombres de columnas en el archivo Arrow
STATE_COLUMNS = ("price", "time", "quantity", "direction")

# Claves de metadata
META_CURRENT_TREND = b"current_trend"
META_LAST_OS_REF = b"last_os_ref"
META_LAST_PROCESSED_DATE = b"last_processed_date"
META_REFERENCE_EXTREME_PRICE = b"reference_extreme_price"
META_REFERENCE_EXTREME_TIME = b"reference_extreme_time"

# Cache de arrays vacíos (singleton para evitar recreación)
_EMPTY_F64: ArrayF64 = np.array([], dtype=np.float64)
_EMPTY_I64: ArrayI64 = np.array([], dtype=np.int64)
_EMPTY_I8: ArrayI8 = np.array([], dtype=np.int8)


# =============================================================================
# UTILIDADES
# =============================================================================


def format_theta(theta: float) -> str:
    """Formatea theta como string sin ceros trailing.

    Examples:
    --------
        >>> format_theta(0.005)
        '0.005'
        >>> format_theta(0.010000)
        '0.01'

    """
    return f"{theta:.6f}".rstrip("0").rstrip(".")


# =============================================================================
# ESTADO DC
# =============================================================================


@dataclass
class DCState:
    """Estado del algoritmo DC para continuidad entre particiones.

    Contiene tanto los ticks huérfanos (posteriores al último evento confirmado)
    como el estado algorítmico necesario para reanudar el procesamiento.

    Los ticks huérfanos son aquellos que ocurrieron después del último evento
    confirmado pero antes del cierre del día. Son necesarios para el stitching
    porque el siguiente evento podría comenzar en ellos.

    ```
    Día N                          Día N+1
    ─────────────────────────────  ─────────────────────────
    [eventos confirmados][huérfanos][nuevos ticks...]
                         └─────────────────┘
                              stitching
    ```

    Attributes:
    ----------
        orphan_prices: Precios de los ticks huérfanos
        orphan_times: Timestamps de los ticks huérfanos (nanosegundos)
        orphan_quantities: Cantidades de los ticks huérfanos
        orphan_directions: Direcciones de los ticks huérfanos
        current_trend: Tendencia actual (1=upturn, -1=downturn, 0=indefinido)
        last_os_ref: Precio de referencia para conteo de OS runs
        last_processed_date: Fecha del último día procesado

    Example:
    -------
        >>> state = DCState(
        ...     orphan_prices=np.array([100.0, 101.0]),
        ...     orphan_times=np.array([1000, 2000]),
        ...     orphan_quantities=np.array([1.0, 1.0]),
        ...     orphan_directions=np.array([1, -1], dtype=np.int8),
        ...     current_trend=np.int8(1),
        ...     last_os_ref=np.float64(100.0),
        ...     last_processed_date=date(2025, 11, 28),
        ... )
        >>> state.n_orphans
        2

    """

    # Ticks huérfanos (arrays)
    orphan_prices: ArrayF64
    orphan_times: ArrayI64
    orphan_quantities: ArrayF64
    orphan_directions: ArrayI8

    # Estado algorítmico (escalares)
    current_trend: np.int8
    last_os_ref: np.float64

    # Metadatos (requerido)
    last_processed_date: date

    # Referencia del extremo para el próximo evento DC (con defaults)
    reference_extreme_price: np.float64 = np.float64(0.0)
    reference_extreme_time: np.int64 = np.int64(0)

    # Cache interno (no serializado)
    _extreme_cache: tuple[float, float] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validación y normalización post-inicialización."""
        # Asegurar tipos correctos
        if not isinstance(self.current_trend, np.int8):
            self.current_trend = np.int8(self.current_trend)
        if not isinstance(self.last_os_ref, np.float64):
            self.last_os_ref = np.float64(self.last_os_ref)

    @property
    def n_orphans(self) -> int:
        """Número de ticks huérfanos."""
        return len(self.orphan_prices)

    @property
    def has_orphans(self) -> bool:
        """Indica si hay ticks huérfanos pendientes."""
        return self.n_orphans > 0

    def get_extreme_prices(self) -> tuple[float, float]:
        """Calcula los precios extremos desde los huérfanos (cached).

        Returns:
        -------
            Tuple (ext_high_price, ext_low_price) derivados de los huérfanos.
            Si no hay huérfanos, retorna (0.0, 0.0).

        """
        if self._extreme_cache is not None:
            return self._extreme_cache

        if not self.has_orphans:
            result = (0.0, 0.0)
        else:
            result = (float(np.max(self.orphan_prices)), float(np.min(self.orphan_prices)))

        # Cache para llamadas subsecuentes
        object.__setattr__(self, "_extreme_cache", result)
        return result

    def memory_usage_bytes(self) -> int:
        """Estima el uso de memoria del estado en bytes.

        Returns:
        -------
            Tamaño aproximado en bytes

        """
        return (
            self.orphan_prices.nbytes
            + self.orphan_times.nbytes
            + self.orphan_quantities.nbytes
            + self.orphan_directions.nbytes
            + 8
            + 8
            + 8  # current_trend, last_os_ref, date overhead
        )

    def validate(self) -> tuple[bool, list[str]]:
        """Valida la consistencia interna del estado.

        Returns:
        -------
            Tuple (is_valid, errors) donde errors es lista de mensajes

        """
        errors = []
        n = self.n_orphans

        # Verificar longitudes consistentes
        if len(self.orphan_times) != n:
            errors.append(f"orphan_times length {len(self.orphan_times)} != {n}")
        if len(self.orphan_quantities) != n:
            errors.append(f"orphan_quantities length {len(self.orphan_quantities)} != {n}")
        if len(self.orphan_directions) != n:
            errors.append(f"orphan_directions length {len(self.orphan_directions)} != {n}")

        # Verificar tipos
        if self.orphan_prices.dtype != np.float64:
            errors.append(f"orphan_prices dtype {self.orphan_prices.dtype} != float64")
        if self.orphan_times.dtype != np.int64:
            errors.append(f"orphan_times dtype {self.orphan_times.dtype} != int64")
        if self.orphan_directions.dtype != np.int8:
            errors.append(f"orphan_directions dtype {self.orphan_directions.dtype} != int8")

        # Verificar trend válido
        if self.current_trend not in (-1, 0, 1):
            errors.append(f"current_trend {self.current_trend} not in (-1, 0, 1)")

        # Verificar timestamps ordenados (si hay huérfanos)
        if n > 1 and not np.all(np.diff(self.orphan_times) >= 0):
            errors.append("orphan_times not monotonically increasing")

        return (len(errors) == 0, errors)

    def to_dict(self) -> dict:
        """Convierte a diccionario (para debugging/logging)."""
        return {
            "n_orphans": self.n_orphans,
            "current_trend": int(self.current_trend),
            "last_os_ref": float(self.last_os_ref),
            "last_processed_date": self.last_processed_date.isoformat(),
            "memory_bytes": self.memory_usage_bytes(),
            "extreme_prices": self.get_extreme_prices(),
        }


# =============================================================================
# FUNCIONES DE PERSISTENCIA
# =============================================================================


def build_state_path(
    base_path: Path,
    ticker: str,
    theta: float,
    year: int,
    month: int,
    day: int,
    *,
    theta_str: str | None = None,
) -> Path:
    """Construye la ruta completa al archivo de estado.

    Args:
    ----
        base_path: Ruta base del datalake Silver
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC
        year: Año
        month: Mes
        day: Día
        theta_str: String pre-formateado de theta (para evitar recálculo)

    Returns:
    -------
        Path completo al archivo state.arrow

    Example:
    -------
        >>> build_state_path(Path("/silver"), "BTCUSDT", 0.005, 2025, 11, 28)
        PosixPath('/silver/BTCUSDT/theta=0.005/year=2025/month=11/day=28/state_BTCUSDT_0.005.arrow')

    """
    if theta_str is None:
        theta_str = format_theta(theta)

    return (
        base_path
        / ticker
        / f"theta={theta_str}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"day={day:02d}"
        / f"state_{ticker}_{theta_str}.arrow"
    )


def save_state(state: DCState, path: Path) -> int:
    """Persiste el estado como Arrow IPC (Feather v2).

    El archivo contiene:
    - Columnas con los ticks huérfanos (price, time, quantity, direction)
    - Metadata del schema con el estado algorítmico

    Args:
    ----
        state: Estado a persistir
        path: Ruta completa al archivo de destino

    Returns:
    -------
        Tamaño del archivo escrito en bytes

    Raises:
    ------
        ValueError: Si el estado es inválido

    """
    # Validar estado antes de guardar
    is_valid, errors = state.validate()
    if not is_valid:
        raise ValueError(f"Estado inválido: {errors}")

    # Crear directorio si no existe
    path.parent.mkdir(parents=True, exist_ok=True)

    # Construir tabla Arrow con los huérfanos
    table = pa.table(
        {
            "price": pa.array(state.orphan_prices, type=pa.float64()),
            "time": pa.array(state.orphan_times, type=pa.int64()),
            "quantity": pa.array(state.orphan_quantities, type=pa.float64()),
            "direction": pa.array(state.orphan_directions, type=pa.int8()),
        }
    )

    # Añadir metadata al schema
    metadata = {
        META_CURRENT_TREND: str(int(state.current_trend)).encode(),
        META_LAST_OS_REF: f"{state.last_os_ref:.15g}".encode(),
        META_LAST_PROCESSED_DATE: state.last_processed_date.isoformat().encode(),
        META_REFERENCE_EXTREME_PRICE: f"{state.reference_extreme_price:.15g}".encode(),
        META_REFERENCE_EXTREME_TIME: str(int(state.reference_extreme_time)).encode(),
    }

    new_schema = table.schema.with_metadata(metadata)
    table = table.cast(new_schema)

    # Escribir como Feather (Arrow IPC v2)
    feather.write_feather(table, path, compression=STATE_COMPRESSION)

    return path.stat().st_size


def load_state(path: Path) -> DCState | None:
    """Carga el estado desde Arrow IPC.

    Realiza memory-mapping para zero-copy donde sea posible.

    Args:
    ----
        path: Ruta al archivo de estado

    Returns:
    -------
        DCState si el archivo existe, None en caso contrario

    Raises:
    ------
        ValueError: Si el archivo existe pero tiene formato inválido

    """
    if not path.exists():
        return None

    # Leer con memory mapping
    table = feather.read_table(path, memory_map=True)

    # Verificar columnas requeridas
    missing_cols = set(STATE_COLUMNS) - set(table.column_names)
    if missing_cols:
        raise ValueError(f"Columnas faltantes en {path}: {missing_cols}")

    # Extraer metadata
    schema_metadata = table.schema.metadata or {}

    current_trend = np.int8(int(schema_metadata.get(META_CURRENT_TREND, b"0").decode()))
    last_os_ref = np.float64(float(schema_metadata.get(META_LAST_OS_REF, b"0").decode()))
    last_date_str = schema_metadata.get(META_LAST_PROCESSED_DATE, b"").decode()

    last_processed_date = date.fromisoformat(last_date_str) if last_date_str else date.today()

    # Extraer campos de referencia (nuevos)
    reference_extreme_price = np.float64(
        float(schema_metadata.get(META_REFERENCE_EXTREME_PRICE, b"0").decode())
    )
    reference_extreme_time = np.int64(
        int(schema_metadata.get(META_REFERENCE_EXTREME_TIME, b"0").decode())
    )

    # Extraer arrays con tipos correctos (evitar copia innecesaria)
    # PyArrow to_numpy() con zero_copy_only=False permite conversión eficiente
    directions_raw = table.column("direction").to_numpy()
    if directions_raw.dtype != np.int8:
        directions = directions_raw.astype(np.int8, copy=False)
    else:
        directions = directions_raw

    return DCState(
        orphan_prices=table.column("price").to_numpy(),
        orphan_times=table.column("time").to_numpy(),
        orphan_quantities=table.column("quantity").to_numpy(),
        orphan_directions=directions,
        current_trend=current_trend,
        last_os_ref=last_os_ref,
        reference_extreme_price=reference_extreme_price,
        reference_extreme_time=reference_extreme_time,
        last_processed_date=last_processed_date,
    )


def find_previous_state(
    base_path: Path,
    ticker: str,
    theta: float,
    current_date: date,
    *,
    max_lookback: int = MAX_LOOKBACK_DAYS,
) -> tuple[Path, DCState] | None:
    """Busca el archivo de estado del día anterior más reciente.

    Navega hacia atrás en el tiempo buscando el último estado disponible.

    Args:
    ----
        base_path: Ruta base del datalake Silver
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC
        current_date: Fecha actual siendo procesada
        max_lookback: Máximo de días hacia atrás a buscar (default: 7)

    Returns:
    -------
        Tuple (path, state) si se encuentra, None en caso contrario

    Example:
    -------
        >>> result = find_previous_state(Path("/silver"), "BTCUSDT", 0.005, date(2025, 11, 28))
        >>> if result:
        ...     path, state = result
        ...     print(f"Estado del {state.last_processed_date}: {state.n_orphans} huérfanos")

    """
    # Pre-calcular theta_str (evita recálculo en loop)
    theta_str = format_theta(theta)

    for days_back in range(1, max_lookback + 1):
        prev_date = current_date - timedelta(days=days_back)

        state_path = build_state_path(
            base_path,
            ticker,
            theta,
            prev_date.year,
            prev_date.month,
            prev_date.day,
            theta_str=theta_str,
        )

        state = load_state(state_path)
        if state is not None:
            return (state_path, state)

    return None


def create_empty_state(process_date: date) -> DCState:
    """Crea un estado vacío para iniciar el procesamiento.

    Usado cuando no existe estado previo (primer día de procesamiento).
    Reutiliza arrays vacíos singleton para evitar allocaciones.

    Args:
    ----
        process_date: Fecha del procesamiento

    Returns:
    -------
        DCState vacío con la fecha especificada

    """
    return DCState(
        orphan_prices=_EMPTY_F64,
        orphan_times=_EMPTY_I64,
        orphan_quantities=_EMPTY_F64,
        orphan_directions=_EMPTY_I8,
        current_trend=np.int8(0),
        last_os_ref=np.float64(0),
        reference_extreme_price=np.float64(0),
        reference_extreme_time=np.int64(0),
        last_processed_date=process_date,
    )


# =============================================================================
# FUNCIONES DE DIAGNÓSTICO
# =============================================================================


def list_available_states(
    base_path: Path,
    ticker: str,
    theta: float,
) -> Iterator[tuple[date, Path]]:
    """Lista todos los archivos de estado disponibles para un ticker/theta.

    Args:
    ----
        base_path: Ruta base del datalake Silver
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC

    Yields:
    ------
        Tuples (date, path) ordenados por fecha descendente

    Example:
    -------
        >>> for d, p in list_available_states(Path("/silver"), "BTCUSDT", 0.005):
        ...     print(f"{d}: {p}")

    """
    theta_str = format_theta(theta)
    ticker_path = base_path / ticker / f"theta={theta_str}"

    if not ticker_path.exists():
        return

    # Buscar todos los archivos state_*.arrow
    state_files = list(ticker_path.glob("**/state_*.arrow"))

    # Extraer fechas y ordenar
    dated_files: list[tuple[date, Path]] = []

    for path in state_files:
        try:
            # Extraer año/mes/día de la ruta
            parts = path.parts
            year = int([p for p in parts if p.startswith("year=")][0].split("=")[1])
            month = int([p for p in parts if p.startswith("month=")][0].split("=")[1])
            day = int([p for p in parts if p.startswith("day=")][0].split("=")[1])
            dated_files.append((date(year, month, day), path))
        except (IndexError, ValueError):
            continue

    # Ordenar por fecha descendente
    dated_files.sort(key=lambda x: x[0], reverse=True)

    yield from dated_files


def get_state_stats(
    base_path: Path,
    ticker: str,
    theta: float,
) -> dict:
    """Calcula estadísticas de los archivos de estado disponibles.

    Args:
    ----
        base_path: Ruta base del datalake Silver
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC

    Returns:
    -------
        Dict con estadísticas

    """
    states = list(list_available_states(base_path, ticker, theta))

    if not states:
        return {
            "n_states": 0,
            "total_bytes": 0,
            "date_range": None,
        }

    total_bytes = sum(p.stat().st_size for _, p in states)
    dates = [d for d, _ in states]

    return {
        "n_states": len(states),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "date_range": (min(dates), max(dates)),
        "oldest": min(dates),
        "newest": max(dates),
    }


def cleanup_old_states(
    base_path: Path,
    ticker: str,
    theta: float,
    keep_last_n: int = 1,
    dry_run: bool = True,
) -> list[Path]:
    """Elimina archivos de estado antiguos, conservando los más recientes.

    Args:
    ----
        base_path: Ruta base del datalake Silver
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC
        keep_last_n: Número de estados más recientes a conservar
        dry_run: Si True, solo lista archivos sin eliminar

    Returns:
    -------
        Lista de paths eliminados (o que serían eliminados si dry_run=True)

    """
    states = list(list_available_states(base_path, ticker, theta))

    if len(states) <= keep_last_n:
        return []

    # Estados a eliminar (los más antiguos)
    to_delete = [p for _, p in states[keep_last_n:]]

    if not dry_run:
        for path in to_delete:
            path.unlink()

    return to_delete
