"""
Detector de eventos Directional Change (DC) optimizado con Numba.

El paradigma DC transforma series temporales de precios en una secuencia
de eventos discretos basados en cambios porcentuales significativos.

Referencias:
    - Guillaume et al. (1997). From the bird's eye to the microscope.
    - Glattfelder et al. (2011). Patterns in high-frequency FX data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Sequence, Union

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from intrinseca.core.bridging import extract_price_column


class TrendState(IntEnum):
    """Estado de tendencia del autómata DC."""
    
    UNDEFINED = 0
    UPTREND = 1
    DOWNTREND = -1


@dataclass(frozen=True, slots=True)
class DCEvent:
    """
    Representa un evento de Directional Change.
    
    Attributes:
        index: Índice en la serie original donde ocurrió el evento.
        price: Precio en el momento del evento.
        timestamp: Timestamp opcional (si está disponible).
        event_type: Tipo de evento ('upturn' o 'downturn').
        extreme_index: Índice del extremo previo (máximo o mínimo local).
        extreme_price: Precio del extremo previo.
        dc_return: Retorno porcentual del movimiento DC.
        dc_duration: Duración en observaciones del evento DC.
    """
    
    index: int
    price: float
    event_type: str  # 'upturn' o 'downturn'
    extreme_index: int
    extreme_price: float
    dc_return: float
    dc_duration: int
    timestamp: Optional[np.datetime64] = None


@dataclass
class DCResult:
    """
    Resultado completo del análisis Directional Change.
    
    Attributes:
        events: Lista de eventos DC detectados.
        trends: Array con el estado de tendencia en cada punto.
        extremes_high: Índices de máximos locales (extremos en downtrend).
        extremes_low: Índices de mínimos locales (extremos en uptrend).
        overshoot_values: Valores de overshoot en cada punto.
    """
    
    events: list[DCEvent] = field(default_factory=list)
    trends: NDArray[np.int8] = field(default_factory=lambda: np.array([], dtype=np.int8))
    extremes_high: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))
    extremes_low: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))
    overshoot_values: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    @property
    def n_events(self) -> int:
        """Número total de eventos DC."""
        return len(self.events)
    
    @property
    def n_upturns(self) -> int:
        """Número de eventos upturn."""
        return sum(1 for e in self.events if e.event_type == "upturn")
    
    @property
    def n_downturns(self) -> int:
        """Número de eventos downturn."""
        return sum(1 for e in self.events if e.event_type == "downturn")


# =============================================================================
# KERNEL NUMBA: Algoritmo DC de alto rendimiento
# =============================================================================

@njit(cache=True, fastmath=True)
def _dc_core_algorithm(
    prices: NDArray[np.float64],
    theta: float
) -> tuple[
    NDArray[np.int64],    # event_indices
    NDArray[np.float64],  # event_prices
    NDArray[np.int8],     # event_types (1=upturn, -1=downturn)
    NDArray[np.int64],    # extreme_indices
    NDArray[np.float64],  # extreme_prices
    NDArray[np.int8],     # trend_states
    int                   # n_events
]:
    """
    Kernel Numba para detección de eventos DC.
    
    Este es un autómata de estados finitos que recorre la serie de precios
    y detecta cambios de tendencia cuando el movimiento supera el umbral theta.
    
    Args:
        prices: Array de precios (debe ser float64).
        theta: Umbral de cambio porcentual (ej: 0.01 = 1%).
    
    Returns:
        Tupla con arrays de eventos detectados y estados de tendencia.
    """
    n = len(prices)
    
    # Pre-allocación de arrays de salida (tamaño máximo teórico)
    max_events = n // 2 + 1
    event_indices = np.empty(max_events, dtype=np.int64)
    event_prices = np.empty(max_events, dtype=np.float64)
    event_types = np.empty(max_events, dtype=np.int8)
    extreme_indices = np.empty(max_events, dtype=np.int64)
    extreme_prices = np.empty(max_events, dtype=np.float64)
    
    # Estado de tendencia para cada observación
    trend_states = np.zeros(n, dtype=np.int8)
    
    # Inicialización del autómata
    current_trend = np.int8(0)  # 0=undefined, 1=uptrend, -1=downtrend
    extreme_high_price = prices[0]
    extreme_high_idx = 0
    extreme_low_price = prices[0]
    extreme_low_idx = 0
    
    n_events = 0
    
    # Bucle principal del autómata DC
    for i in range(1, n):
        p = prices[i]
        
        # Actualizar extremos
        if p > extreme_high_price:
            extreme_high_price = p
            extreme_high_idx = i
        if p < extreme_low_price:
            extreme_low_price = p
            extreme_low_idx = i
        
        if current_trend == 0:
            # Estado inicial: determinar primera tendencia
            up_change = (p - extreme_low_price) / extreme_low_price
            down_change = (extreme_high_price - p) / extreme_high_price
            
            if up_change >= theta:
                current_trend = 1  # Uptrend
                # Registrar evento upturn
                event_indices[n_events] = i
                event_prices[n_events] = p
                event_types[n_events] = 1
                extreme_indices[n_events] = extreme_low_idx
                extreme_prices[n_events] = extreme_low_price
                n_events += 1
                # Reset extremo alto
                extreme_high_price = p
                extreme_high_idx = i
                
            elif down_change >= theta:
                current_trend = -1  # Downtrend
                # Registrar evento downturn
                event_indices[n_events] = i
                event_prices[n_events] = p
                event_types[n_events] = -1
                extreme_indices[n_events] = extreme_high_idx
                extreme_prices[n_events] = extreme_high_price
                n_events += 1
                # Reset extremo bajo
                extreme_low_price = p
                extreme_low_idx = i
        
        elif current_trend == 1:
            # En uptrend: buscar downturn
            down_change = (extreme_high_price - p) / extreme_high_price
            
            if down_change >= theta:
                # Downturn confirmado
                current_trend = -1
                event_indices[n_events] = i
                event_prices[n_events] = p
                event_types[n_events] = -1
                extreme_indices[n_events] = extreme_high_idx
                extreme_prices[n_events] = extreme_high_price
                n_events += 1
                # Reset extremo bajo
                extreme_low_price = p
                extreme_low_idx = i
        
        else:  # current_trend == -1
            # En downtrend: buscar upturn
            up_change = (p - extreme_low_price) / extreme_low_price
            
            if up_change >= theta:
                # Upturn confirmado
                current_trend = 1
                event_indices[n_events] = i
                event_prices[n_events] = p
                event_types[n_events] = 1
                extreme_indices[n_events] = extreme_low_idx
                extreme_prices[n_events] = extreme_low_price
                n_events += 1
                # Reset extremo alto
                extreme_high_price = p
                extreme_high_idx = i
        
        trend_states[i] = current_trend
    
    return (
        event_indices[:n_events],
        event_prices[:n_events],
        event_types[:n_events],
        extreme_indices[:n_events],
        extreme_prices[:n_events],
        trend_states,
        n_events
    )


@njit(cache=True, fastmath=True, parallel=True)
def _compute_overshoot_batch(
    prices: NDArray[np.float64],
    event_indices: NDArray[np.int64],
    extreme_indices: NDArray[np.int64],
    event_types: NDArray[np.int8],
    theta: float
) -> NDArray[np.float64]:
    """
    Calcula el overshoot para cada evento DC en paralelo.
    
    Overshoot mide cuánto se extiende el movimiento más allá del umbral theta
    antes de que ocurra el siguiente evento DC.
    """
    n_events = len(event_indices)
    n_prices = len(prices)
    overshoots = np.zeros(n_events, dtype=np.float64)
    
    for i in prange(n_events):
        event_idx = event_indices[i]
        event_type = event_types[i]
        
        # Determinar el rango de búsqueda para el overshoot
        if i < n_events - 1:
            end_idx = event_indices[i + 1]
        else:
            end_idx = n_prices
        
        if event_type == 1:  # Upturn -> buscar máximo después
            max_price = prices[event_idx]
            for j in range(event_idx, end_idx):
                if prices[j] > max_price:
                    max_price = prices[j]
            # Overshoot = movimiento adicional después del evento
            overshoots[i] = (max_price - prices[event_idx]) / prices[event_idx]
        else:  # Downturn -> buscar mínimo después
            min_price = prices[event_idx]
            for j in range(event_idx, end_idx):
                if prices[j] < min_price:
                    min_price = prices[j]
            overshoots[i] = (prices[event_idx] - min_price) / prices[event_idx]
    
    return overshoots


# =============================================================================
# CLASE PRINCIPAL: DCDetector
# =============================================================================

class DCDetector:
    """
    Detector de eventos Directional Change de alto rendimiento.
    
    El paradigma DC transforma series de precios continuos en eventos discretos,
    capturando la microestructura del mercado de manera invariante al tiempo.
    
    Args:
        theta: Umbral de cambio porcentual para detectar eventos.
               Valores típicos: 0.001 (0.1%), 0.01 (1%), 0.02 (2%).
        compute_overshoot: Si calcular overshoot para cada evento.
    
    Example:
        >>> detector = DCDetector(theta=0.01)
        >>> result = detector.detect(prices)
        >>> print(f"Detectados {result.n_events} eventos DC")
        >>> for event in result.events[:5]:
        ...     print(f"{event.event_type} en índice {event.index}")
    """
    
    def __init__(self, theta: float = 0.01, compute_overshoot: bool = True):
        if not 0 < theta < 1:
            raise ValueError(f"theta debe estar en (0, 1), recibido: {theta}")
        
        self.theta = theta
        self.compute_overshoot = compute_overshoot
        self._is_compiled = False
    
    def _ensure_compiled(self) -> None:
        """Fuerza la compilación JIT de Numba en el primer uso."""
        if not self._is_compiled:
            # Warm-up con array pequeño
            dummy = np.array([1.0, 1.01, 0.99, 1.02], dtype=np.float64)
            _dc_core_algorithm(dummy, self.theta)
            self._is_compiled = True
    
    def detect(
        self,
        prices: Union[NDArray, Sequence[float], "pl.Series", "pl.DataFrame"],
        timestamps: Optional[NDArray] = None,
        price_column: str = "close"
    ) -> DCResult:
        """
        Detecta eventos DC en una serie de precios.
        
        Args:
            prices: Serie de precios. Acepta:
                - numpy array
                - lista de Python
                - Polars Series
                - Polars DataFrame (extrae columna especificada)
            timestamps: Array opcional de timestamps para asociar a eventos.
            price_column: Nombre de columna si prices es DataFrame.
        
        Returns:
            DCResult con eventos detectados y métricas asociadas.
        
        Raises:
            ValueError: Si la serie tiene menos de 2 elementos.
        """
        # Convertir a numpy
        prices_np = extract_price_column(prices, price_column)
        
        if len(prices_np) < 2:
            raise ValueError("La serie de precios debe tener al menos 2 elementos")
        
        # Asegurar tipo float64 para Numba
        if prices_np.dtype != np.float64:
            prices_np = prices_np.astype(np.float64)
        
        # Compilación JIT si es primera ejecución
        self._ensure_compiled()
        
        # Ejecutar kernel Numba
        (
            event_indices,
            event_prices,
            event_types,
            extreme_indices,
            extreme_prices,
            trend_states,
            n_events
        ) = _dc_core_algorithm(prices_np, self.theta)
        
        # Calcular overshoot si se solicita
        if self.compute_overshoot and n_events > 0:
            overshoots = _compute_overshoot_batch(
                prices_np, event_indices, extreme_indices, event_types, self.theta
            )
        else:
            overshoots = np.zeros(n_events, dtype=np.float64)
        
        # Construir lista de eventos
        events = []
        for i in range(n_events):
            idx = int(event_indices[i])
            ext_idx = int(extreme_indices[i])
            
            # Calcular retorno DC
            dc_return = (event_prices[i] - extreme_prices[i]) / extreme_prices[i]
            dc_duration = idx - ext_idx
            
            # Timestamp si está disponible
            ts = timestamps[idx] if timestamps is not None else None
            
            event = DCEvent(
                index=idx,
                price=float(event_prices[i]),
                event_type="upturn" if event_types[i] == 1 else "downturn",
                extreme_index=ext_idx,
                extreme_price=float(extreme_prices[i]),
                dc_return=float(dc_return),
                dc_duration=dc_duration,
                timestamp=ts
            )
            events.append(event)
        
        # Separar extremos por tipo
        extremes_high = extreme_indices[event_types == -1]
        extremes_low = extreme_indices[event_types == 1]
        
        return DCResult(
            events=events,
            trends=trend_states,
            extremes_high=extremes_high,
            extremes_low=extremes_low,
            overshoot_values=overshoots
        )
    
    def detect_streaming(
        self,
        new_price: float,
        state: Optional[dict] = None
    ) -> tuple[Optional[DCEvent], dict]:
        """
        Detección incremental para live-trading.
        
        Procesa un precio a la vez, manteniendo estado entre llamadas.
        Ideal para feeds en tiempo real.
        
        Args:
            new_price: Nuevo precio a procesar.
            state: Estado previo del detector (None para inicializar).
        
        Returns:
            Tupla (evento_detectado, nuevo_estado).
            evento_detectado es None si no hubo evento.
        
        Example:
            >>> detector = DCDetector(theta=0.01)
            >>> state = None
            >>> for price in live_feed:
            ...     event, state = detector.detect_streaming(price, state)
            ...     if event:
            ...         print(f"Evento: {event.event_type}")
        """
        if state is None:
            # Inicializar estado
            state = {
                "index": 0,
                "trend": 0,  # 0=undefined, 1=up, -1=down
                "extreme_high": new_price,
                "extreme_high_idx": 0,
                "extreme_low": new_price,
                "extreme_low_idx": 0,
                "prices": [new_price]
            }
            return None, state
        
        idx = state["index"] + 1
        state["index"] = idx
        state["prices"].append(new_price)
        
        # Actualizar extremos
        if new_price > state["extreme_high"]:
            state["extreme_high"] = new_price
            state["extreme_high_idx"] = idx
        if new_price < state["extreme_low"]:
            state["extreme_low"] = new_price
            state["extreme_low_idx"] = idx
        
        event = None
        
        if state["trend"] == 0:
            # Determinar tendencia inicial
            up_change = (new_price - state["extreme_low"]) / state["extreme_low"]
            down_change = (state["extreme_high"] - new_price) / state["extreme_high"]
            
            if up_change >= self.theta:
                state["trend"] = 1
                event = DCEvent(
                    index=idx,
                    price=new_price,
                    event_type="upturn",
                    extreme_index=state["extreme_low_idx"],
                    extreme_price=state["extreme_low"],
                    dc_return=up_change,
                    dc_duration=idx - state["extreme_low_idx"]
                )
                state["extreme_high"] = new_price
                state["extreme_high_idx"] = idx
                
            elif down_change >= self.theta:
                state["trend"] = -1
                event = DCEvent(
                    index=idx,
                    price=new_price,
                    event_type="downturn",
                    extreme_index=state["extreme_high_idx"],
                    extreme_price=state["extreme_high"],
                    dc_return=-down_change,
                    dc_duration=idx - state["extreme_high_idx"]
                )
                state["extreme_low"] = new_price
                state["extreme_low_idx"] = idx
        
        elif state["trend"] == 1:
            # En uptrend, buscar downturn
            down_change = (state["extreme_high"] - new_price) / state["extreme_high"]
            
            if down_change >= self.theta:
                state["trend"] = -1
                event = DCEvent(
                    index=idx,
                    price=new_price,
                    event_type="downturn",
                    extreme_index=state["extreme_high_idx"],
                    extreme_price=state["extreme_high"],
                    dc_return=-down_change,
                    dc_duration=idx - state["extreme_high_idx"]
                )
                state["extreme_low"] = new_price
                state["extreme_low_idx"] = idx
        
        else:  # trend == -1
            # En downtrend, buscar upturn
            up_change = (new_price - state["extreme_low"]) / state["extreme_low"]
            
            if up_change >= self.theta:
                state["trend"] = 1
                event = DCEvent(
                    index=idx,
                    price=new_price,
                    event_type="upturn",
                    extreme_index=state["extreme_low_idx"],
                    extreme_price=state["extreme_low"],
                    dc_return=up_change,
                    dc_duration=idx - state["extreme_low_idx"]
                )
                state["extreme_high"] = new_price
                state["extreme_high_idx"] = idx
        
        return event, state
    
    def __repr__(self) -> str:
        return f"DCDetector(theta={self.theta}, compute_overshoot={self.compute_overshoot})"