"""
Motor de Detección Directional Change (DC) de Alta Frecuencia.
Versión: Marcadores Puntuales + Polars Fill Strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray
from numba import njit, int8, int64, float64


# =============================================================================
# 1. KERNEL NUMBA OPTIMIZADO (CORE)
# =============================================================================

@njit(cache=True, fastmath=True, nogil=True)
def _dc_core_algorithm_optimized(
    prices: NDArray[np.float64],      # float64
    timestamps: NDArray[np.int64],    # int64 (microsegundos desde epoch)
    theta: float
) -> tuple[
    NDArray[np.int64],    # 0: event_indices
    NDArray[np.int64],    # 1: event_timestamps
    NDArray[np.float64],  # 2: event_prices
    NDArray[np.int8],     # 3: event_types
    NDArray[np.int64],    # 4: event_durations
    NDArray[np.int64],    # 5: extreme_indices
    NDArray[np.float64],  # 6: extreme_prices
    NDArray[np.int64],    # 7: extreme_timestamps
    NDArray[np.float64],  # 8: overshoots
    NDArray[np.float64],  # 9: returns_price
    NDArray[np.float64],  # 10: returns_speed
    NDArray[np.int64],    # 11: os_event_counts
    NDArray[np.int8],     # 12: dc_markers (NUEVO: Solo puntos clave)
    int                   # 13: n_events
]:
    """
    Kernel JIT. Modificado para marcar únicamente los 4 puntos críticos del DC.
    No realiza propagación de estado (eso se delega a Polars).
    """
    n = len(prices)
    
    threshold_mult_up = 1.0 + theta
    threshold_mult_down = 1.0 - theta
    
    # --- Allocación de Memoria ---
    # Eventos (Sparse)
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
    
    # Ticks (Dense) - Ahora solo guardamos marcadores puntuales
    # 0=Nada, 1=ExtHigh, 2=DCC, 3=ExtLow, 4=UCC
    dc_markers = np.zeros(n, dtype=np.int8)
    
    # --- Estado Inicial ---
    current_trend = 0 # 0=Indefinido, 1=Alza, -1=Baja
    
    ext_high_price = prices[0]
    ext_high_idx = 0
    ext_low_price = prices[0]
    ext_low_idx = 0
    
    # Variables auxiliares para métricas de Overshoot (conteo)
    last_os_ref_price = prices[0]
    current_event_run_count = 0
    
    # Inicialización del primer evento (ancla)
    tao = 0
    event_prices[0] = prices[0]
    event_indices[0] = 0
    event_timestamps[0] = timestamps[0]
    event_types[0] = 0
    
    # --- Bucle Principal ---
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
        
        # 2. Detección de Cambio (Confirmation Points)
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

        # 3. Procesamiento de Eventos y Marcado
        if new_trend != 0:
            # --- Se confirma un evento ---

            # --- REGLA CONSERVADORA: Buscar el mejor precio de confirmación ---
            # Cuando hay múltiples ticks con el mismo timestamp que cruzan el threshold,
            # seleccionar el más conservador:
            # - Upturn: precio MÍNIMO >= threshold (el que "apenas" confirma)
            # - Downturn: precio MÁXIMO <= threshold (el que "apenas" confirma)

            if new_trend == 1:
                threshold = ext_low_price * threshold_mult_up
            else:
                threshold = ext_high_price * threshold_mult_down

            conf_timestamp = ts_val
            best_price = p
            best_idx = t

            # Lookahead: buscar todos los ticks con el mismo timestamp
            for j in range(t + 1, n):
                if timestamps[j] != conf_timestamp:
                    break
                pj = prices[j]
                if new_trend == 1:
                    # Upturn: buscar el precio MÍNIMO que cruza el threshold
                    if pj >= threshold and pj < best_price:
                        best_price = pj
                        best_idx = j
                else:
                    # Downturn: buscar el precio MÁXIMO que cruza el threshold
                    if pj <= threshold and pj > best_price:
                        best_price = pj
                        best_idx = j

            # Usar el mejor candidato
            conf_price = best_price
            conf_idx = best_idx
            conf_time = timestamps[conf_idx]

            if new_trend == 1:
                # Reversión a Alza (Upturn Confirmed)
                # El extremo relevante fue el Mínimo (Low) anterior
                prev_ext_idx = ext_low_idx
                prev_ext_price = ext_low_price
                prev_ext_time = timestamps[ext_low_idx]

                # MARCADORES CLAVE:
                # 1. Marcamos dónde ocurrió el Extremo (Retroactivo) -> Inicio Upturn Event
                dc_markers[prev_ext_idx] = 3  # 3 = Extreme Low
                # 2. Marcamos dónde estamos ahora (Confirmación) -> Inicio Upward Overshoot
                dc_markers[conf_idx] = 4      # 4 = Upturn Confirmation (UCC)

                # Reset del opuesto para el nuevo ciclo
                ext_high_price = conf_price
                ext_high_idx = conf_idx

            else:
                # Reversión a Baja (Downturn Confirmed)
                # El extremo relevante fue el Máximo (High) anterior
                prev_ext_idx = ext_high_idx
                prev_ext_price = ext_high_price
                prev_ext_time = timestamps[ext_high_idx]

                # MARCADORES CLAVE:
                # 1. Marcamos dónde ocurrió el Extremo (Retroactivo) -> Inicio Downturn Event
                dc_markers[prev_ext_idx] = 1  # 1 = Extreme High
                # 2. Marcamos dónde estamos ahora (Confirmación) -> Inicio Downward Overshoot
                dc_markers[conf_idx] = 2      # 2 = Downturn Confirmation (DCC)

                # Reset del opuesto para el nuevo ciclo
                ext_low_price = conf_price
                ext_low_idx = conf_idx

            # --- Métricas del Evento Finalizado ---
            price_move = prev_ext_price - event_prices[tao]
            prev_evt_p = event_prices[tao]

            overshoots[tao] = price_move
            returns_price[tao] = (price_move / prev_evt_p) if prev_evt_p != 0 else 0.0
            os_event_counts[tao] = current_event_run_count

            # Registrar nuevo evento en arrays sparse
            tao += 1
            event_indices[tao] = conf_idx
            event_prices[tao] = conf_price
            event_timestamps[tao] = conf_time
            event_types[tao] = new_trend

            extreme_indices[tao] = prev_ext_idx
            extreme_prices[tao] = prev_ext_price
            extreme_timestamps[tao] = prev_ext_time

            dt = conf_time - prev_ext_time
            event_durations[tao] = dt

            if dt > 0:
                returns_speed[tao] = (conf_price - prev_ext_price) / (dt / 1_000_000_000.0)
            else:
                returns_speed[tao] = 0.0

            # Actualizar estado global
            current_trend = new_trend
            last_os_ref_price = conf_price
            current_event_run_count = 0
            
        else:
            # --- Lógica de Conteo de Overshoots (Solo métrica, sin flagging denso) ---
            run_increments = 0
            if current_trend == 1:
                next_thresh = last_os_ref_price * threshold_mult_up
                if p >= next_thresh:
                    temp_ref = last_os_ref_price
                    while p >= next_thresh:
                        run_increments += 1
                        temp_ref = next_thresh
                        next_thresh = temp_ref * threshold_mult_up
                    last_os_ref_price = temp_ref
                    current_event_run_count += run_increments
                    
            elif current_trend == -1:
                next_thresh = last_os_ref_price * threshold_mult_down
                if p <= next_thresh:
                    temp_ref = last_os_ref_price
                    while p <= next_thresh:
                        run_increments += 1
                        temp_ref = next_thresh
                        next_thresh = temp_ref * threshold_mult_down
                    last_os_ref_price = temp_ref
                    current_event_run_count += run_increments

    # Retorno recortado
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
        dc_markers,    # Array completo N (con ceros y marcadores 1-4)
        tao
    )


# =============================================================================
# 2. ESTRUCTURA DE RESULTADOS
# =============================================================================

@dataclass
class DCResult:
    # Métricas de Eventos (Sparse)
    event_indices: NDArray[np.int64]
    event_timestamps: NDArray[np.int64]
    event_prices: NDArray[np.float64]
    event_types: NDArray[np.int8]
    event_durations: NDArray[np.int64]
    extreme_indices: NDArray[np.int64]
    extreme_prices: NDArray[np.float64]
    extreme_timestamps: NDArray[np.int64]
    overshoots: NDArray[np.float64]
    returns_price: NDArray[np.float64]
    returns_speed: NDArray[np.float64]
    os_event_counts: NDArray[np.int64]

    # Métricas de Ticks (Dense)
    dc_markers: NDArray[np.int8]  # La columna "semilla" para Polars

    @property
    def n_events(self) -> int:
        return len(self.event_indices)

    def events_to_polars(self) -> pl.DataFrame:
        """Resumen de eventos (sin cambios mayores)."""
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
            pl.col("ext_time").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"),
            pl.col("type").map_elements(lambda x: "upturn" if x == 1 else "downturn", return_dtype=pl.String).alias("type_desc")
        ])

    def ticks_to_polars(self, original_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Genera el DataFrame tick-a-tick implementando la lógica de llenado (Fill)
        para definir los intervalos de Directional Change.
        """
        # 1. Crear DF base con la columna de marcadores
        dc_cols = pl.DataFrame({
            "dc_marker": self.dc_markers
        })
        
        # 2. Lógica de Mapeo de Intervalos usando Polars Expressions
        #    1=ExtHigh, 2=DCC, 3=ExtLow, 4=UCC
        
        # Paso A: Convertir 0 a null para poder hacer forward_fill
        # Paso B: Forward Fill para propagar el último marcador conocido
        # Paso C: Mapear el marcador propagado al nombre del intervalo
        
        interval_map = {
            1: "Downturn Event",       # Desde ExtHigh hasta DCC
            2: "Downward Overshoot",   # Desde DCC hasta ExtLow
            3: "Upturn Event",         # Desde ExtLow hasta UCC
            4: "Upward Overshoot"      # Desde UCC hasta ExtHigh
        }
        
        dc_filled = dc_cols.select([
            pl.col("dc_marker"), # Conservamos la original por si acaso (debug)
            
            pl.col("dc_marker")
                .replace(0, None)                 # Ceros a Nulos
                .forward_fill()                   # Relleno eficiente (C++)
                .replace_strict(interval_map, default=None) # Mapeo rápido
                .alias("dc_interval_type")
        ])
        
        # 3. Join si existe DF original
        if original_df is not None:
            if len(original_df) != len(dc_cols):
                raise ValueError("Longitud del DF original no coincide con resultados DC")
                
            # Convertir timestamp original para consistencia visual
            original_df = original_df.with_columns([
                pl.col("time").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
            ])
            
            # Concatenación horizontal (Zero-Copy logic where possible)
            return pl.concat([original_df, dc_filled], how="horizontal")
            
        return dc_filled


# =============================================================================
# 3. DETECTOR (CLASE PRINCIPAL)
# =============================================================================

class DCDetector:
    def __init__(self, theta: float):
        self.theta = float(theta)
        self._compiled = False
        
    def _warmup(self):
        if not self._compiled:
            p = np.array([100.0, 101.0, 99.0], dtype=np.float64)
            t = np.array([1000, 2000, 3000], dtype=np.int64)
            _dc_core_algorithm_optimized(p, t, self.theta)
            self._compiled = True

    def detect(self, df: pl.DataFrame, price_col: str = "price", time_col: str = "timestamp") -> DCResult:
        self._warmup()
        
        try:
            dtype = df.schema[time_col]
            if isinstance(dtype, (pl.Datetime, pl.Duration)):
                 ts_series = df.get_column(time_col).cast(pl.Int64)
            else:
                raise TypeError(f"Columna {time_col} debe ser Datetime o Duration.")

            prices_np = df.get_column(price_col).cast(pl.Float64).to_numpy()
            timestamps_np = ts_series.to_numpy()
            
        except Exception as e:
            raise ValueError(f"Error preparando datos: {e}")

        # Ejecución del Kernel (sin trend_states ni os_event_flags)
        (
            ev_idx, ev_ts, ev_pr, ev_ty, ev_dur,
            ex_idx, ex_pr, ex_ts,
            os_val, ret_pr, ret_spd, os_cnt,
            dc_markers,  # <--- Solo recibimos los marcadores
            n_ev
        ) = _dc_core_algorithm_optimized(prices_np, timestamps_np, self.theta)
        
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
            dc_markers=dc_markers # <--- Pasamos marcadores
        )