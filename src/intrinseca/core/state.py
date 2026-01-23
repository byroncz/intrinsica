"""
Gestión de Estado para Silver Layer.

Manejo de estados transitorios y ticks huérfanos usando Apache Arrow IPC.
El formato Arrow garantiza isomorfismo de memoria (zero-copy) para máxima
eficiencia en el proceso de stitching entre días.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from numpy.typing import NDArray


@dataclass
class DCState:
    """
    Estado del algoritmo DC para continuidad entre particiones.
    
    Contiene tanto los ticks huérfanos (posteriores al último evento confirmado)
    como el estado algorítmico necesario para reanudar el procesamiento.
    
    Attributes:
        orphan_prices: Precios de los ticks huérfanos
        orphan_times: Timestamps de los ticks huérfanos (nanosegundos)
        orphan_quantities: Cantidades de los ticks huérfanos
        orphan_directions: Direcciones de los ticks huérfanos
        current_trend: Tendencia actual (1=upturn, -1=downturn, 0=indefinido)
        last_os_ref: Precio de referencia para conteo de OS runs
        last_processed_date: Fecha del último día procesado
    """
    # Ticks huérfanos (arrays)
    orphan_prices: NDArray[np.float64]
    orphan_times: NDArray[np.int64]
    orphan_quantities: NDArray[np.float64]
    orphan_directions: NDArray[np.int8]
    
    # Estado algorítmico (escalares)
    current_trend: np.int8
    last_os_ref: np.float64
    
    # Metadatos
    last_processed_date: date
    
    @property
    def n_orphans(self) -> int:
        """Número de ticks huérfanos."""
        return len(self.orphan_prices)
    
    @property
    def has_orphans(self) -> bool:
        """Indica si hay ticks huérfanos pendientes."""
        return self.n_orphans > 0
    
    def get_extreme_prices(self) -> tuple[float, float]:
        """
        Calcula los precios extremos desde los huérfanos.
        
        Returns:
            Tuple (ext_high_price, ext_low_price) derivados de los huérfanos.
            Si no hay huérfanos, retorna (0.0, 0.0).
        """
        if not self.has_orphans:
            return (0.0, 0.0)
        
        return (
            float(np.max(self.orphan_prices)),
            float(np.min(self.orphan_prices))
        )


def build_state_path(
    base_path: Path,
    ticker: str,
    theta: float,
    year: int,
    month: int,
    day: int
) -> Path:
    """
    Construye la ruta completa al archivo de estado.
    
    Estructura: base_path/ticker/theta={theta}/year={Y}/month={M}/day={D}/state_{ticker}_{theta}.arrow
    """
    theta_str = f"{theta:.6f}".rstrip('0').rstrip('.')
    state_filename = f"state_{ticker}_{theta_str}.arrow"
    
    return (
        base_path
        / ticker
        / f"theta={theta_str}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"day={day:02d}"
        / state_filename
    )


def save_state(state: DCState, path: Path) -> None:
    """
    Persiste el estado como Arrow IPC (Feather v2).
    
    El archivo contiene:
    - Columnas con los ticks huérfanos
    - Metadata del schema con el estado algorítmico
    
    Args:
        state: Estado a persistir
        path: Ruta completa al archivo de destino
    """
    # Crear directorio si no existe
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construir tabla Arrow con los huérfanos
    table = pa.table({
        "price": pa.array(state.orphan_prices, type=pa.float64()),
        "time": pa.array(state.orphan_times, type=pa.int64()),
        "quantity": pa.array(state.orphan_quantities, type=pa.float64()),
        "direction": pa.array(state.orphan_directions, type=pa.int8()),
    })
    
    # Añadir metadata al schema
    metadata = {
        b"current_trend": str(int(state.current_trend)).encode(),
        b"last_os_ref": f"{state.last_os_ref:.15g}".encode(),
        b"last_processed_date": state.last_processed_date.isoformat().encode(),
    }
    
    new_schema = table.schema.with_metadata(metadata)
    table = table.cast(new_schema)
    
    # Escribir como Feather (Arrow IPC v2)
    feather.write_feather(table, path, compression="uncompressed")


def load_state(path: Path) -> Optional[DCState]:
    """
    Carga el estado desde Arrow IPC.
    
    Realiza memory-mapping para zero-copy donde sea posible.
    
    Args:
        path: Ruta al archivo de estado
        
    Returns:
        DCState si el archivo existe, None en caso contrario
    """
    if not path.exists():
        return None
    
    # Leer con memory mapping
    table = feather.read_table(path, memory_map=True)
    
    # Extraer metadata
    schema_metadata = table.schema.metadata or {}
    
    current_trend = np.int8(int(schema_metadata.get(b"current_trend", b"0").decode()))
    last_os_ref = np.float64(float(schema_metadata.get(b"last_os_ref", b"0").decode()))
    last_date_str = schema_metadata.get(b"last_processed_date", b"").decode()
    
    last_processed_date = date.fromisoformat(last_date_str) if last_date_str else date.today()
    
    # Extraer arrays (zero-copy cuando es posible)
    return DCState(
        orphan_prices=table.column("price").to_numpy(),
        orphan_times=table.column("time").to_numpy(),
        orphan_quantities=table.column("quantity").to_numpy(),
        orphan_directions=table.column("direction").to_numpy().astype(np.int8),
        current_trend=current_trend,
        last_os_ref=last_os_ref,
        last_processed_date=last_processed_date,
    )


def find_previous_state(
    base_path: Path,
    ticker: str,
    theta: float,
    current_date: date
) -> Optional[tuple[Path, DCState]]:
    """
    Busca el archivo de estado del día anterior más reciente.
    
    Navega hacia atrás en el tiempo buscando el último estado disponible.
    Se limita a buscar hasta 7 días atrás para evitar búsquedas excesivas.
    
    Args:
        base_path: Ruta base del datalake Silver
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC
        current_date: Fecha actual siendo procesada
        
    Returns:
        Tuple (path, state) si se encuentra, None en caso contrario
    """
    from datetime import timedelta
    
    max_lookback_days = 7
    
    for days_back in range(1, max_lookback_days + 1):
        prev_date = current_date - timedelta(days=days_back)
        
        state_path = build_state_path(
            base_path, ticker, theta,
            prev_date.year, prev_date.month, prev_date.day
        )
        
        state = load_state(state_path)
        if state is not None:
            return (state_path, state)
    
    return None


def create_empty_state(process_date: date) -> DCState:
    """
    Crea un estado vacío para iniciar el procesamiento.
    
    Usado cuando no existe estado previo (primer día de procesamiento).
    """
    return DCState(
        orphan_prices=np.array([], dtype=np.float64),
        orphan_times=np.array([], dtype=np.int64),
        orphan_quantities=np.array([], dtype=np.float64),
        orphan_directions=np.array([], dtype=np.int8),
        current_trend=np.int8(0),
        last_os_ref=np.float64(0),
        last_processed_date=process_date,
    )
