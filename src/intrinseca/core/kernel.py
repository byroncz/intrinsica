"""
Kernel Numba para Segmentación de Eventos DC → Silver Layer.

Patrón: Búferes Paralelos Sincronizados (Separados DC/OS)
- Entrada: Arrays planos NumPy (Bronze)
- Salida: Búferes de valores separados para DC y OS + offsets para Arrow zero-copy

Este módulo NO crea objetos Python dentro del bucle JIT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit, int8, int64, float64


@njit(cache=True, fastmath=True, nogil=True)
def segment_events_kernel(
    # --- Inputs: Arrays planos Bronze ---
    prices: NDArray[np.float64],
    timestamps: NDArray[np.int64],
    quantities: NDArray[np.float64],
    directions: NDArray[np.int8],
    theta: float,
    # --- Estado inicial (para stitching) ---
    init_trend: int8,
    init_ext_high_price: float64,
    init_ext_low_price: float64,
    init_last_os_ref: float64
) -> tuple[
    # --- Búferes DC (separados) ---
    NDArray[np.float64],  # 0: dc_prices
    NDArray[np.int64],    # 1: dc_times
    NDArray[np.float64],  # 2: dc_quantities
    NDArray[np.int8],     # 3: dc_directions
    
    # --- Búferes OS (separados) ---
    NDArray[np.float64],  # 4: os_prices
    NDArray[np.int64],    # 5: os_times
    NDArray[np.float64],  # 6: os_quantities
    NDArray[np.int8],     # 7: os_directions
    
    # --- Metadatos por evento ---
    NDArray[np.int8],     # 8: event_types (1 por evento confirmado)
    
    # --- Búferes de OFFSETS ---
    NDArray[np.int64],    # 9: dc_offsets (len = n_events + 1)
    NDArray[np.int64],    # 10: os_offsets (len = n_events + 1)
    
    # --- Metadatos de Estado Final ---
    int64,                # 11: n_events confirmados
    int8,                 # 12: final_trend
    float64,              # 13: final_ext_high_price
    float64,              # 14: final_ext_low_price
    float64,              # 15: final_last_os_ref
    int64,                # 16: orphan_start_idx (primer tick huérfano)
]:
    """
    Kernel JIT que segmenta ticks en fases DC (Directional Change) y OS (Overshoot).
    
    El kernel opera bajo el paradigma de Búferes Paralelos Sincronizados:
    - Búferes DC y OS están SEPARADOS para permitir construcción Arrow directa
    - Los offsets permiten reconstruir las listas anidadas sin copias
    
    Semántica de un Evento DC:
    - Fase DC: Desde el extremo anterior hasta el punto de confirmación (exclusive)
    - Fase OS: Desde el punto de confirmación hasta el próximo extremo (exclusive)
    
    El último evento puede quedar incompleto (ticks huérfanos) si no hay confirmación
    antes del final del array de entrada.
    """
    n = len(prices)
    
    if n == 0:
        # Edge case: array vacío
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_i8 = np.empty(0, dtype=np.int8)
        empty_offsets = np.array([0], dtype=np.int64)
        return (
            empty_f64, empty_i64, empty_f64, empty_i8,  # DC
            empty_f64, empty_i64, empty_f64, empty_i8,  # OS
            empty_i8,  # event_types
            empty_offsets, empty_offsets,  # offsets
            0, init_trend, init_ext_high_price, init_ext_low_price, 
            init_last_os_ref, 0
        )
    
    threshold_mult_up = 1.0 + theta
    threshold_mult_down = 1.0 - theta
    
    # --- Allocación de Búferes SEPARADOS para DC y OS ---
    # DC buffers
    dc_prices = np.empty(n, dtype=np.float64)
    dc_times = np.empty(n, dtype=np.int64)
    dc_quantities = np.empty(n, dtype=np.float64)
    dc_directions = np.empty(n, dtype=np.int8)
    dc_ptr = 0  # Puntero para DC
    
    # OS buffers
    os_prices = np.empty(n, dtype=np.float64)
    os_times = np.empty(n, dtype=np.int64)
    os_quantities = np.empty(n, dtype=np.float64)
    os_directions = np.empty(n, dtype=np.int8)
    os_ptr = 0  # Puntero para OS
    
    # Event types (máximo N eventos teóricos)
    max_events = n
    event_types = np.empty(max_events, dtype=np.int8)
    
    # Offsets para DC y OS
    dc_offsets = np.empty(max_events + 1, dtype=np.int64)
    os_offsets = np.empty(max_events + 1, dtype=np.int64)
    dc_offsets[0] = 0
    os_offsets[0] = 0
    
    # --- Inicialización de Estado ---
    current_trend = init_trend
    ext_high_price = init_ext_high_price if init_ext_high_price > 0 else prices[0]
    ext_high_idx = 0
    ext_low_price = init_ext_low_price if init_ext_low_price > 0 else prices[0]
    ext_low_idx = 0
    last_os_ref = init_last_os_ref if init_last_os_ref > 0 else prices[0]
    
    # --- Contadores ---
    n_events = 0
    
    # Tracking del OS del evento anterior
    prev_os_start = -1  # -1 = no hay OS pendiente
    
    # Para identificar huérfanos
    last_event_confirmation_idx = 0
    
    # --- Bucle Principal ---
    for t in range(n):
        p = prices[t]
        
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
        else:  # current_trend == -1
            if p >= ext_low_price * threshold_mult_up:
                new_trend = 1
        
        # 3. Procesamiento de Confirmación de Evento
        if new_trend != 0:
            # --- Determinar el extremo relevante ---
            if new_trend == 1:
                prev_ext_idx = ext_low_idx
                threshold = ext_low_price * threshold_mult_up
            else:
                prev_ext_idx = ext_high_idx
                threshold = ext_high_price * threshold_mult_down

            # --- REGLA CONSERVADORA: Buscar el mejor precio de confirmación ---
            # Cuando hay múltiples ticks con el mismo timestamp que cruzan el threshold,
            # seleccionar el más conservador:
            # - Upturn: precio MÍNIMO >= threshold (el que "apenas" confirma)
            # - Downturn: precio MÁXIMO <= threshold (el que "apenas" confirma)

            conf_timestamp = timestamps[t]
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

            # Usar el mejor candidato como precio de confirmación
            conf_price = best_price
            conf_idx = best_idx

            # --- Validación semántica: DC debe tener al menos 1 tick ---
            # DC va de prev_ext_idx (inclusive) a conf_idx (exclusive)
            # Si prev_ext_idx >= conf_idx, no hay ticks válidos para DC
            if prev_ext_idx >= conf_idx:
                # Actualizar extremos para mantener consistencia algorítmica
                if new_trend == 1:
                    ext_high_price = conf_price
                    ext_high_idx = conf_idx
                else:
                    ext_low_price = conf_price
                    ext_low_idx = conf_idx
                # Actualizar tendencia (el cambio de dirección es real)
                current_trend = new_trend
                last_os_ref = conf_price
                # NO incrementar n_events, NO escribir DC/OS
                # Continuar al siguiente tick
                continue

            # --- Actualizar extremos (solo si el evento es válido) ---
            if new_trend == 1:
                ext_high_price = conf_price
                ext_high_idx = conf_idx
            else:
                ext_low_price = conf_price
                ext_low_idx = conf_idx

            # --- Cerrar OS del evento ANTERIOR (si existe) ---
            if n_events > 0 and prev_os_start >= 0:
                # El OS del evento anterior va desde prev_os_start hasta prev_ext_idx (exclusive)
                for i in range(prev_os_start, prev_ext_idx):
                    os_prices[os_ptr] = prices[i]
                    os_times[os_ptr] = timestamps[i]
                    os_quantities[os_ptr] = quantities[i]
                    os_directions[os_ptr] = directions[i]
                    os_ptr += 1
                os_offsets[n_events] = os_ptr

            # --- Escribir DC del evento ACTUAL ---
            # DC va desde prev_ext_idx (inclusive) hasta conf_idx (exclusive)
            for i in range(prev_ext_idx, conf_idx):
                dc_prices[dc_ptr] = prices[i]
                dc_times[dc_ptr] = timestamps[i]
                dc_quantities[dc_ptr] = quantities[i]
                dc_directions[dc_ptr] = directions[i]
                dc_ptr += 1

            # Registrar offset de DC para este evento
            dc_offsets[n_events + 1] = dc_ptr

            # Registrar tipo de evento
            event_types[n_events] = new_trend

            # El OS de ESTE evento empezará en conf_idx (se escribirá cuando se confirme el siguiente)
            prev_os_start = conf_idx

            # Actualizar estado
            current_trend = new_trend
            last_os_ref = conf_price
            n_events += 1
            last_event_confirmation_idx = conf_idx
    
    # --- Finalización ---
    
    # Cerrar el último OS (si existe un evento confirmado)
    if n_events > 0 and prev_os_start >= 0:
        # El OS del último evento va desde prev_os_start hasta n
        for i in range(prev_os_start, n):
            os_prices[os_ptr] = prices[i]
            os_times[os_ptr] = timestamps[i]
            os_quantities[os_ptr] = quantities[i]
            os_directions[os_ptr] = directions[i]
            os_ptr += 1
        os_offsets[n_events] = os_ptr
    
    # --- Identificar Ticks Huérfanos ---
    # Huérfanos = ticks que no pertenecen a ningún evento confirmado
    # Son los ticks desde el último extremo trackado (que podría ser el inicio de un futuro DC)
    if n_events == 0:
        orphan_start_idx = 0
    else:
        # Los huérfanos son todo lo que está después del último evento
        # El último evento confirmado terminó en last_event_confirmation_idx
        # Su OS abarca hasta el final, así que técnicamente no hay huérfanos "puros"
        # PERO: los ticks desde el último extremo hacia adelante son los que
        # formarían la fase DC del PRÓXIMO evento
        if current_trend == 1:
            # Tendencia alcista: el extremo high es el potencial inicio del próximo DC
            orphan_start_idx = ext_high_idx
        else:
            # Tendencia bajista: el extremo low es el potencial inicio del próximo DC
            orphan_start_idx = ext_low_idx
    
    # --- Recortar búferes a tamaño real ---
    dc_prices = dc_prices[:dc_ptr]
    dc_times = dc_times[:dc_ptr]
    dc_quantities = dc_quantities[:dc_ptr]
    dc_directions = dc_directions[:dc_ptr]
    
    os_prices = os_prices[:os_ptr]
    os_times = os_times[:os_ptr]
    os_quantities = os_quantities[:os_ptr]
    os_directions = os_directions[:os_ptr]
    
    event_types = event_types[:n_events]
    dc_offsets = dc_offsets[:n_events + 1]
    os_offsets = os_offsets[:n_events + 1] if n_events > 0 else np.array([0], dtype=np.int64)
    
    return (
        dc_prices, dc_times, dc_quantities, dc_directions,
        os_prices, os_times, os_quantities, os_directions,
        event_types,
        dc_offsets, os_offsets,
        n_events,
        current_trend,
        ext_high_price,
        ext_low_price,
        last_os_ref,
        orphan_start_idx
    )


def warmup_kernel(theta: float = 0.01) -> None:
    """
    Pre-compila el kernel con datos dummy.
    Llamar una vez al inicio para evitar latencia en la primera ejecución real.
    """
    p = np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64)
    t = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    q = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    d = np.array([1, -1, 1, -1], dtype=np.int8)
    
    segment_events_kernel(
        p, t, q, d, theta,
        np.int8(0), np.float64(0), np.float64(0), np.float64(0)
    )
