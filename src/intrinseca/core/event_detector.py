"""
Motor de Detección Directional Change (DC) de Alta Frecuencia.

Refactorizado para integración nativa con el stack moderno de datos (Polars/Arrow).
Implementa detección de Multi-Runs (Overshoots dinámicos) y métricas de velocidad.

Principios de Diseño:
    - Columnar-First: Los datos se mantienen en arrays contiguos el mayor tiempo posible.
    - Zero-Copy: Minimización de transferencias de memoria entre Python y C (Numba).
    - Type-Strict: Manejo riguroso de tipos int64 para timestamps (microsegundos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Any

import numpy as np
import polars as pl
from numba import njit, int8, int64, float64


# =============================================================================
# 1. KERNEL NUMBA OPTIMIZADO (CORE)
# =============================================================================

@njit(cache=True, fastmath=True, nogil=True)
def _dc_core_algorithm_optimized(
    prices: NDArray[np.float64],      # float64
    timestamps: NDArray[np.int64],  # int64 (microsegundos desde epoch)
    theta: float
) -> tuple[
    NDArray[np.int64],    # 0: event_indices
    NDArray[np.int64],    # 1: event_timestamps
    NDArray[np.float64],  # 2: event_prices
    NDArray[np.int8],    # 3: event_types
    NDArray[np.int64],    # 4: event_durations (tiempo)
    NDArray[np.int64],    # 5: extreme_indices
    NDArray[np.float64],  # 6: extreme_prices
    NDArray[np.int64],    # 7: extreme_timestamps
    NDArray[np.float64],  # 8: overshoots
    NDArray[np.float64],  # 9: returns_price
    NDArray[np.float64],  # 10: returns_speed
    NDArray[np.int64],    # 11: os_event_counts (runs totales por evento)
    NDArray[np.int8],     # 12: trend_states (tick-by-tick)
    NDArray[np.int8],     # 13: os_event_flags (tick-by-tick, magnitud del run)
    int                   # 14: n_events
]:
    """
    Kernel computacional compilado JIT.
    Implementa máquina de estados finitos con detección de bucles iterativos para 
    rupturas múltiples (Flash Events).
    """
    n = len(prices)
    
    # Constantes pre-calculadas para evitar divisiones en el bucle
    threshold_mult_up = 1.0 + theta
    threshold_mult_down = 1.0 - theta
    
    # Asignación de memoria (Allocación única)
    # Arrays de Eventos (Sparse) - Tamaño máximo N
    event_indices = np.empty(n, dtype=np.int64)
    event_timestamps = np.empty(n, dtype=np.int64)
    event_prices = np.empty(n, dtype=np.float64)
    event_types = np.empty(n, dtype=np.int8)
    event_durations = np.empty(n, dtype=np.int64)
    
    extreme_indices = np.empty(n, dtype=np.int64)
    extreme_prices = np.empty(n, dtype=np.float64)
    extreme_timestamps = np.empty(n, dtype=np.int64)
    
    overshoots = np.empty(n, dtype=np.float64)
    returns_price = np.empty(n, dtype=np.float64)
    returns_speed = np.empty(n, dtype=np.float64)
    os_event_counts = np.zeros(n, dtype=np.int64)
    
    # Arrays de Ticks (Dense) - Tamaño N exacto
    trend_states = np.zeros(n, dtype=np.int8)
    os_event_flags = np.zeros(n, dtype=np.int8)
    
    # Estado inicial
    current_trend = 0 # 0=Indefinido, 1=Alza, -1=Baja
    
    ext_high_price = prices[0]
    ext_high_idx = 0
    ext_low_price = prices[0]
    ext_low_idx = 0
    
    # Estado para lógica de Runs (Overshoots dinámicos)
    last_os_ref_price = prices[0]
    current_event_run_count = 0
    
    # Inicialización del primer evento (ficticio/ancla)
    tao = 0  # Es la variable de tao
    event_prices[0] = prices[0]
    event_indices[0] = 0
    event_timestamps[0] = timestamps[0]
    event_types[0] = 0
    
    # --- Bucle Principal (Tick a Tick) ---
    for t in range(1, n):
        p = prices[t]
        ts_val = timestamps[t]
        
        # 1. Actualización de Extremos
        if p > ext_high_price:
            ext_high_price = p
            ext_high_idx = t
        if p < ext_low_price:
            ext_low_price = p
            ext_low_idx = t
            
        new_trend = 0 
        
        # 2. Detección de Cambio de Tendencia (DC)
        if current_trend == 0:
            if p >= ext_low_price * threshold_mult_up:
                new_trend = 1
            elif p <= ext_high_price * threshold_mult_down:
                new_trend = -1
        elif current_trend == 1:
            if p <= ext_high_price * threshold_mult_down:
                new_trend = -1
        else: # current_trend == -1
            if p >= ext_low_price * threshold_mult_up:
                new_trend = 1

        # 3. Procesamiento
        if new_trend != 0:
            # --- Evento Confirmado ---
            
            # Resolver extremo previo según dirección
            if new_trend == 1: # Reversión a Alza -> El evento previo fue Baja (o inicio) -> Usamos Low
                prev_ext_idx = ext_low_idx
                prev_ext_price = ext_low_price
                prev_ext_time = timestamps[ext_low_idx]
                # Reset opuesto
                ext_high_price = p
                ext_high_idx = t
            else: # Reversión a Baja -> El evento previo fue Alza -> Usamos High
                prev_ext_idx = ext_high_idx
                prev_ext_price = ext_high_price
                prev_ext_time = timestamps[ext_high_idx]
                # Reset opuesto
                ext_low_price = p
                ext_low_idx = t
            
            # Cálculos de cierre del evento anterior
            price_move = prev_ext_price - event_prices[tao]
            prev_evt_p = event_prices[tao]
            
            # Métricas almacenadas en índice ev_k (el evento que "generó" este extremo)
            overshoots[tao] = price_move # Overshoot absoluto
            returns_price[tao] = (price_move / prev_evt_p) if prev_evt_p != 0 else 0.0
            os_event_counts[tao] = current_event_run_count
            
            # Registrar NUEVO evento
            tao += 1
            
            event_indices[tao] = t
            event_prices[tao] = p
            event_timestamps[tao] = ts_val
            event_types[tao] = new_trend
            
            extreme_indices[tao] = prev_ext_idx
            extreme_prices[tao] = prev_ext_price
            extreme_timestamps[tao] = prev_ext_time
            
            # Duración y Velocidad (del movimiento de ruptura actual)
            dt = ts_val - prev_ext_time
            event_durations[tao] = dt
            
            # Velocidad: Unidades de precio por segundo
            if dt > 0:
                returns_speed[tao] = (p - prev_ext_price) / (dt / 1_000_000_000.0)
            else:
                returns_speed[tao] = 0.0

            # Actualizar estado global
            current_trend = new_trend
            last_os_ref_price = p # Reset referencia runs
            current_event_run_count = 0
            os_event_flags[t] = 0 # El tick de evento DC no cuenta como run extra
            
        else:
            # --- Lógica de Multi-Runs (Overshoots dentro de tendencia) ---
            run_increments = 0
            
            if current_trend == 1: # Alza
                next_thresh = last_os_ref_price * threshold_mult_up
                if p >= next_thresh:
                    # Bucle iterativo eficiente (evita logs)
                    temp_ref = last_os_ref_price
                    while p >= next_thresh:
                        run_increments += 1
                        temp_ref = next_thresh
                        next_thresh = temp_ref * threshold_mult_up
                    
                    last_os_ref_price = temp_ref
                    current_event_run_count += run_increments
                    os_event_flags[t] = int8(run_increments) # Positivo
                    
            elif current_trend == -1: # Baja
                next_thresh = last_os_ref_price * threshold_mult_down
                if p <= next_thresh:
                    temp_ref = last_os_ref_price
                    while p <= next_thresh:
                        run_increments += 1
                        temp_ref = next_thresh
                        next_thresh = temp_ref * threshold_mult_down
                        
                    last_os_ref_price = temp_ref
                    current_event_run_count += run_increments
                    os_event_flags[t] = int8(-run_increments) # Negativo

        trend_states[t] = current_trend

    # Retornar vistas recortadas hasta el último evento válido
    return (
        event_indices[:tao+1],
        event_timestamps[:tao+1],
        event_prices[:tao+1],
        event_types[:tao+1],
        event_durations[:tao+1],
        extreme_indices[:tao+1],
        extreme_prices[:tao+1],
        extreme_timestamps[:tao+1],
        overshoots[:tao+1],
        returns_price[:tao+1],
        returns_speed[:tao+1],
        os_event_counts[:tao+1],
        trend_states,     # Array completo N
        os_event_flags,   # Array completo N
        tao
    )


# =============================================================================
# 2. ESTRUCTURA DE RESULTADOS (POLAR FRIENDLY)
# =============================================================================

@dataclass
class DCResult:
    """
    Contenedor de resultados optimizado para acceso columnar.
    Mantiene los datos en numpy arrays para evitar overhead de objetos.
    """
    # Métricas de Eventos (Longitud = n_events)
    event_indices: NDArray[np.int64]       # 0: event_indices
    event_timestamps: NDArray[np.int64]    # 1: event_timestamps
    event_prices: NDArray[np.float64]      # 2: event_prices
    event_types: NDArray[np.int8]          # 3: event_types
    event_durations: NDArray[np.int64]     # 4: event_durations (tiempo)
    extreme_indices: NDArray[np.int64]     # 5: extreme_indices
    extreme_prices: NDArray[np.float64]    # 6: extreme_prices
    extreme_timestamps: NDArray[np.int64]  # 7: extreme_timestamps
    overshoots: NDArray[np.float64]        # 8: overshoots
    returns_price: NDArray[np.float64]     # 9: returns_price
    returns_speed: NDArray[np.float64]     # 10: returns_speed
    os_event_counts: NDArray[np.int64]     # 11: os_event_counts (runs totales por evento)

    # Métricas de Ticks (Longitud = n_ticks original)
    trend_states: NDArray[np.int8]         # 12: trend_states (tick-by-tick)
    os_event_flags: NDArray[np.int8]       # 13: os_event_flags (tick-by-tick, magnitud del run)

    @property
    def n_events(self) -> int:
        return len(self.event_indices)

    def events_to_polars(self) -> pl.DataFrame:
        """
        Genera un DataFrame de Polars con el resumen de EVENTOS.
        Ideal para análisis estadístico de la distribución de eventos.
        """
        return pl.DataFrame({
            "event_idx": self.event_indices,
            "time": self.event_timestamps,
            "price": self.event_prices,
            "type": self.event_types,
            "duration_ns": self.event_durations,
            "ext_time": self.extreme_timestamps,
            "ext_price": self.extreme_prices,
            "ext_idx": self.extreme_indices,
            "overshoot": self.overshoots,
            "return": self.returns_price,
            "velocity": self.returns_speed,
            "runs_count": self.os_event_counts
        }).with_columns([
            pl.col("time").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"),
            pl.col("type").map_elements(lambda x: "upturn" if x == 1 else "downturn", return_dtype=pl.String).alias("type_desc")
        ])

    def ticks_to_polars(self, original_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Genera un DataFrame de Polars tick-a-tick.
        Si se pasa el DF original, hace un join horizontal (muy eficiente en Polars).
        """
        dc_cols = pl.DataFrame({
            "dc_trend": self.trend_states,
            "dc_run_flag": self.os_event_flags
        })
        
        if original_df is not None:
            if len(original_df) != len(dc_cols):
                raise ValueError("Longitud del DF original no coincide con resultados DC")
            original_df = original_df.with_columns([
                pl.col("time").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
            ])
            return pl.concat([original_df, dc_cols], how="horizontal")
        return dc_cols


# =============================================================================
# 3. DETECTOR (CLASE PRINCIPAL)
# =============================================================================

class DCDetector:
    """
    Detector DC compatible con el ecosistema Apache Arrow / Polars.
    """
    
    def __init__(self, theta: float):
        self.theta = float(theta)
        self._compiled = False
        
    def _warmup(self):
        """Compilación forzada con datos dummy."""
        if not self._compiled:
            p = np.array([100.0, 101.0, 99.0], dtype=np.float64)
            t = np.array([1000, 2000, 3000], dtype=np.int64)
            _dc_core_algorithm_optimized(p, t, self.theta)
            self._compiled = True

    def detect(self, 
               df: pl.DataFrame, 
               price_col: str = "price", 
               time_col: str = "timestamp") -> DCResult:
        """
        Ejecuta la detección sobre un Polars DataFrame.
        
        Requisitos:
            - time_col debe ser convertible a Int64 (microsegundos).
              Idealmente ya debe ser Datetime[μs].
        """
        self._warmup()
        
        # 1. Extracción Zero-Copy (o Low-Copy) hacia Numpy
        # Aseguramos que el tiempo esté en microsegundos y casteado a i64
        # Polars maneja datetimes internamente como i64
        
        try:
            # Verificamos tipo de tiempo
            dtype = df.schema[time_col]
            
            if isinstance(dtype, (pl.Datetime, pl.Duration)):
                 # Extraer buffer subyacente directamente si es posible
                 # cast(pl.Int64) en Polars es muy rápido (reinterpretación de bits para datetime)
                 ts_series = df.get_column(time_col).cast(pl.Int64)
            else:
                # Si es string u otro, intentar conversión (más lento)
                raise TypeError(f"Columna {time_col} debe ser Datetime o Duration, recibido: {dtype}")

            prices_np = df.get_column(price_col).cast(pl.Float64).to_numpy()
            timestamps_np = ts_series.to_numpy()
            
        except Exception as e:
            raise ValueError(f"Error preparando datos para Numba: {e}")

        # 2. Ejecución del Kernel
        (
            ev_idx, ev_ts, ev_pr, ev_ty, ev_dur,
            ex_idx, ex_pr, ex_ts,
            os_val, ret_pr, ret_spd, os_cnt,
            trend_st, os_flags,
            n_ev
        ) = _dc_core_algorithm_optimized(prices_np, timestamps_np, self.theta)
        
        # 3. Empaquetado de Resultados
        return DCResult(
            event_indices=ev_idx,
            event_timestamps=ev_ts,
            event_prices=ev_pr,
            event_types=ev_ty,
            event_durations=ev_dur,
            extreme_indices=ex_idx,
            extreme_prices=ex_pr,
            extreme_timestamps=ex_ts,
            overshoots=os_val,
            returns_price=ret_pr,
            returns_speed=ret_spd,
            os_event_counts=os_cnt,
            trend_states=trend_st,
            os_event_flags=os_flags
        )