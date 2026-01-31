"""
Kernel Numba para Segmentación de Eventos DC → Silver Layer.

Optimizado para HFT:
- Estimación inteligente de memoria basada en theta
- Zero-copy slicing donde es posible
- Cache-friendly memory access patterns
- Compilación nopython garantizada

Patrón: Búferes Paralelos Sincronizados (Separados DC/OS)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, int8, int64, float64, types
from numba.extending import overload


# =============================================================================
# TYPE ALIASES (para legibilidad)
# =============================================================================

ArrayF64 = NDArray[np.float64]
ArrayI64 = NDArray[np.int64]
ArrayI8 = NDArray[np.int8]


class KernelResult(NamedTuple):
    """Resultado estructurado del kernel (solo para documentación, Numba retorna tuple)."""
    # Búferes DC (listas anidadas de ticks)
    dc_prices: ArrayF64
    dc_times: ArrayI64
    dc_quantities: ArrayF64
    dc_directions: ArrayI8
    # Búferes OS (listas anidadas de ticks)
    os_prices: ArrayF64
    os_times: ArrayI64
    os_quantities: ArrayF64
    os_directions: ArrayI8
    # Metadatos eventos
    event_types: ArrayI8
    dc_offsets: ArrayI64
    os_offsets: ArrayI64
    # Atributos de evento (escalares por evento, zero indirection)
    reference_prices: ArrayF64    # Precio del extremo de referencia (del evento N-1)
    reference_times: ArrayI64     # Timestamp del extremo de referencia
    extreme_prices: ArrayF64      # Precio del extremo (último tick del OS)
    extreme_times: ArrayI64       # Timestamp del extremo
    confirm_prices: ArrayF64      # Precio de confirmación (último tick del DC)
    confirm_times: ArrayI64       # Timestamp de confirmación
    # Estado final
    n_events: int
    final_trend: int
    final_ext_high: float
    final_ext_low: float
    final_os_ref: float
    orphan_start_idx: int


# =============================================================================
# CONSTANTES
# =============================================================================

# Estimador de ratio eventos/ticks basado en theta típico
# Para theta=0.5%, ratio empírico es ~1:5000 a 1:10000
# Usamos 1:1000 como margen conservador
_EVENT_RATIO_ESTIMATE = 1000

# Mínimo de slots para eventos (evita edge cases)
_MIN_EVENT_SLOTS = 64


# =============================================================================
# KERNEL PRINCIPAL
# =============================================================================

@njit(cache=True, fastmath=True, nogil=True)
def segment_events_kernel(
    prices: ArrayF64,
    timestamps: ArrayI64,
    quantities: ArrayF64,
    directions: ArrayI8,
    theta: float64,
    init_trend: int8,
    init_ext_high_price: float64,
    init_ext_low_price: float64,
    init_last_os_ref: float64,
) -> tuple:
    """
    Kernel JIT para segmentación de eventos DC.

    Complejidad: O(n) donde n = número de ticks
    Memoria: O(n/k + e) donde k ≈ 1000 (ratio ticks/eventos), e = eventos detectados

    Args:
        prices: Precios de ticks (float64)
        timestamps: Timestamps en nanosegundos (int64)
        quantities: Cantidades/volúmenes (float64)
        directions: Direcciones 1=buy, -1=sell (int8)
        theta: Umbral DC (ej: 0.005 para 0.5%)
        init_trend: Tendencia inicial (0=indefinido, 1=up, -1=down)
        init_ext_high_price: Precio extremo alto inicial
        init_ext_low_price: Precio extremo bajo inicial
        init_last_os_ref: Precio referencia para OS runs

    Returns:
        Tuple con 21 elementos (ver KernelResult para documentación)
    """
    n = len(prices)

    # --- Early exit para array vacío ---
    if n == 0:
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_i8 = np.empty(0, dtype=np.int8)
        zero_offset = np.zeros(1, dtype=np.int64)
        return (
            empty_f64, empty_i64, empty_f64, empty_i8,  # DC buffers
            empty_f64, empty_i64, empty_f64, empty_i8,  # OS buffers
            empty_i8,                                    # event_types
            zero_offset, zero_offset,                    # offsets
            empty_f64, empty_i64,                        # reference price/time
            empty_f64, empty_i64, empty_f64, empty_i64,  # extreme/confirm price/time
            int64(0), init_trend, init_ext_high_price, init_ext_low_price,
            init_last_os_ref, int64(0)
        )

    # --- Pre-cálculo de thresholds (evita multiplicaciones repetidas) ---
    theta_up = 1.0 + theta
    theta_down = 1.0 - theta

    # --- Estimación inteligente de memoria con factor de seguridad 3x ---
    # Eventos esperados ≈ n / ratio, con mínimo garantizado y factor 3x
    estimated_events = max(n // _EVENT_RATIO_ESTIMATE, _MIN_EVENT_SLOTS) * 3

    # Búferes para datos de ticks (tamaño completo necesario para worst case)
    # Pero usamos n directamente ya que en el peor caso todos los ticks son eventos
    dc_prices = np.empty(n, dtype=np.float64)
    dc_times = np.empty(n, dtype=np.int64)
    dc_quantities = np.empty(n, dtype=np.float64)
    dc_directions = np.empty(n, dtype=np.int8)

    os_prices = np.empty(n, dtype=np.float64)
    os_times = np.empty(n, dtype=np.int64)
    os_quantities = np.empty(n, dtype=np.float64)
    os_directions = np.empty(n, dtype=np.int8)

    # Búferes para metadatos de eventos (tamaño estimado * 3x)
    event_types = np.empty(estimated_events, dtype=np.int8)
    dc_offsets = np.empty(estimated_events + 1, dtype=np.int64)
    os_offsets = np.empty(estimated_events + 1, dtype=np.int64)

    # Atributos de evento (escalares por evento, elimina indirección list.first())
    reference_prices = np.empty(estimated_events, dtype=np.float64)
    reference_times = np.empty(estimated_events, dtype=np.int64)
    extreme_prices = np.full(estimated_events, -1.0, dtype=np.float64)  # -1.0 = provisional
    extreme_times = np.full(estimated_events, int64(-1), dtype=np.int64)
    confirm_prices = np.empty(estimated_events, dtype=np.float64)
    confirm_times = np.empty(estimated_events, dtype=np.int64)

    # Inicializar offsets
    dc_offsets[0] = 0
    os_offsets[0] = 0

    # --- Punteros de escritura ---
    dc_ptr = 0
    os_ptr = 0
    n_events = 0

    # --- Estado del algoritmo ---
    current_trend = init_trend

    # Extremos: precio e índice
    p0 = prices[0]
    ext_high_price = init_ext_high_price if init_ext_high_price > 0.0 else p0
    ext_high_idx = 0
    ext_low_price = init_ext_low_price if init_ext_low_price > 0.0 else p0
    ext_low_idx = 0

    # Referencia para OS
    last_os_ref = init_last_os_ref if init_last_os_ref > 0.0 else p0

    # Tracking de OS pendiente
    prev_os_start = -1
    last_conf_idx = 0

    # --- Bucle Principal ---
    for t in range(n):
        p = prices[t]

        # 1. Actualización de extremos (branch-free comparison)
        if p > ext_high_price:
            ext_high_price = p
            ext_high_idx = t
        if p < ext_low_price:
            ext_low_price = p
            ext_low_idx = t

        # 2. Detección de cambio de tendencia
        new_trend = int8(0)

        if current_trend == 0:
            # Estado inicial: detectar primera tendencia
            if p >= ext_low_price * theta_up:
                new_trend = int8(1)
            elif p <= ext_high_price * theta_down:
                new_trend = int8(-1)
        elif current_trend == 1:
            # En tendencia alcista: buscar reversión bajista
            if p <= ext_high_price * theta_down:
                new_trend = int8(-1)
        else:
            # En tendencia bajista: buscar reversión alcista
            if p >= ext_low_price * theta_up:
                new_trend = int8(1)

        # 3. Procesar confirmación de evento
        if new_trend != 0:
            # Determinar extremo relevante y threshold
            if new_trend == 1:
                prev_ext_idx = ext_low_idx
                threshold = ext_low_price * theta_up
            else:
                prev_ext_idx = ext_high_idx
                threshold = ext_high_price * theta_down

            # --- Regla Conservadora: lookahead para mejor precio ---
            conf_ts = timestamps[t]
            best_price = p
            best_idx = t

            # Buscar en ticks con mismo timestamp
            j = t + 1
            while j < n and timestamps[j] == conf_ts:
                pj = prices[j]
                if new_trend == 1:
                    # Upturn: precio MÍNIMO >= threshold
                    if pj >= threshold and pj < best_price:
                        best_price = pj
                        best_idx = j
                else:
                    # Downturn: precio MÁXIMO <= threshold
                    if pj <= threshold and pj > best_price:
                        best_price = pj
                        best_idx = j
                j += 1

            conf_price = best_price
            conf_idx = best_idx

            # Validación: DC debe tener al menos 1 tick
            if prev_ext_idx >= conf_idx:
                # Evento inválido: actualizar estado y continuar
                if new_trend == 1:
                    ext_high_price = conf_price
                    ext_high_idx = conf_idx
                else:
                    ext_low_price = conf_price
                    ext_low_idx = conf_idx
                current_trend = new_trend
                last_os_ref = conf_price
                continue

            # --- Resize dinámico si es necesario ---
            if n_events >= estimated_events:
                # Duplicar capacidad
                new_size = estimated_events * 2

                new_event_types = np.empty(new_size, dtype=np.int8)
                new_event_types[:n_events] = event_types[:n_events]
                event_types = new_event_types

                new_dc_offsets = np.empty(new_size + 1, dtype=np.int64)
                new_dc_offsets[:n_events + 1] = dc_offsets[:n_events + 1]
                dc_offsets = new_dc_offsets

                new_os_offsets = np.empty(new_size + 1, dtype=np.int64)
                new_os_offsets[:n_events + 1] = os_offsets[:n_events + 1]
                os_offsets = new_os_offsets

                # Resize atributos de evento
                new_reference_prices = np.empty(new_size, dtype=np.float64)
                new_reference_prices[:n_events] = reference_prices[:n_events]
                reference_prices = new_reference_prices

                new_reference_times = np.empty(new_size, dtype=np.int64)
                new_reference_times[:n_events] = reference_times[:n_events]
                reference_times = new_reference_times

                new_extreme_prices = np.full(new_size, -1.0, dtype=np.float64)
                new_extreme_prices[:n_events] = extreme_prices[:n_events]
                extreme_prices = new_extreme_prices

                new_extreme_times = np.full(new_size, int64(-1), dtype=np.int64)
                new_extreme_times[:n_events] = extreme_times[:n_events]
                extreme_times = new_extreme_times

                new_confirm_prices = np.empty(new_size, dtype=np.float64)
                new_confirm_prices[:n_events] = confirm_prices[:n_events]
                confirm_prices = new_confirm_prices

                new_confirm_times = np.empty(new_size, dtype=np.int64)
                new_confirm_times[:n_events] = confirm_times[:n_events]
                confirm_times = new_confirm_times

                estimated_events = new_size

            # Actualizar extremos
            if new_trend == 1:
                ext_high_price = conf_price
                ext_high_idx = conf_idx
            else:
                ext_low_price = conf_price
                ext_low_idx = conf_idx

            # --- Cerrar OS del evento anterior (incluye el extremo) - VECTORIZADO ---
            if n_events > 0 and prev_os_start >= 0:
                os_length = prev_ext_idx + 1 - prev_os_start
                os_prices[os_ptr:os_ptr + os_length] = prices[prev_os_start:prev_ext_idx + 1]
                os_times[os_ptr:os_ptr + os_length] = timestamps[prev_os_start:prev_ext_idx + 1]
                os_quantities[os_ptr:os_ptr + os_length] = quantities[prev_os_start:prev_ext_idx + 1]
                os_directions[os_ptr:os_ptr + os_length] = directions[prev_os_start:prev_ext_idx + 1]
                os_ptr += os_length
                os_offsets[n_events] = os_ptr
                
                # Llenar retrospectivamente extreme_price del evento ANTERIOR
                extreme_prices[n_events - 1] = prices[prev_ext_idx]
                extreme_times[n_events - 1] = timestamps[prev_ext_idx]

            # --- Escribir DC del evento actual (excluye extremo, incluye DCC) - VECTORIZADO ---
            dc_start = prev_ext_idx + 1
            dc_end = conf_idx + 1
            dc_length = dc_end - dc_start
            dc_prices[dc_ptr:dc_ptr + dc_length] = prices[dc_start:dc_end]
            dc_times[dc_ptr:dc_ptr + dc_length] = timestamps[dc_start:dc_end]
            dc_quantities[dc_ptr:dc_ptr + dc_length] = quantities[dc_start:dc_end]
            dc_directions[dc_ptr:dc_ptr + dc_length] = directions[dc_start:dc_end]
            dc_ptr += dc_length

            # Registrar evento
            dc_offsets[n_events + 1] = dc_ptr
            event_types[n_events] = new_trend

            # Registrar atributos del evento (zero indirection)
            # reference = extremo que define este DC (pertenece al OS anterior)
            reference_prices[n_events] = prices[prev_ext_idx]
            reference_times[n_events] = timestamps[prev_ext_idx]
            # extreme se llenará retrospectivamente cuando se confirme el siguiente evento
            confirm_prices[n_events] = conf_price
            confirm_times[n_events] = timestamps[conf_idx]

            # Preparar siguiente OS (empieza DESPUÉS del DCC)
            prev_os_start = conf_idx + 1  # CAMBIO: +1 para excluir DCC
            current_trend = new_trend
            last_os_ref = conf_price
            n_events += 1
            last_conf_idx = conf_idx

    # --- Finalización: cerrar último OS - VECTORIZADO ---
    if n_events > 0 and prev_os_start >= 0:
        final_os_length = n - prev_os_start
        os_prices[os_ptr:os_ptr + final_os_length] = prices[prev_os_start:n]
        os_times[os_ptr:os_ptr + final_os_length] = timestamps[prev_os_start:n]
        os_quantities[os_ptr:os_ptr + final_os_length] = quantities[prev_os_start:n]
        os_directions[os_ptr:os_ptr + final_os_length] = directions[prev_os_start:n]
        os_ptr += final_os_length
        os_offsets[n_events] = os_ptr

    # --- Calcular índice de huérfanos (excluye el extremo, que pertenece al OS) ---
    if n_events == 0:
        orphan_start_idx = 0
    else:
        if current_trend == 1:
            orphan_start_idx = ext_high_idx + 1  # CAMBIO: +1 para excluir extremo
        else:
            orphan_start_idx = ext_low_idx + 1   # CAMBIO: +1 para excluir extremo

    # --- Recortar búferes al tamaño real ---
    return (
        dc_prices[:dc_ptr],
        dc_times[:dc_ptr],
        dc_quantities[:dc_ptr],
        dc_directions[:dc_ptr],
        os_prices[:os_ptr],
        os_times[:os_ptr],
        os_quantities[:os_ptr],
        os_directions[:os_ptr],
        event_types[:n_events],
        dc_offsets[:n_events + 1],
        os_offsets[:n_events + 1] if n_events > 0 else np.zeros(1, dtype=np.int64),
        # Atributos de evento (escalares)
        reference_prices[:n_events],
        reference_times[:n_events],
        extreme_prices[:n_events],
        extreme_times[:n_events],
        confirm_prices[:n_events],
        confirm_times[:n_events],
        # Estado final
        int64(n_events),
        current_trend,
        ext_high_price,
        ext_low_price,
        last_os_ref,
        int64(orphan_start_idx),
    )


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def warmup_kernel(theta: float = 0.01) -> None:
    """
    Pre-compila el kernel con datos dummy.

    Llamar una vez al inicio para evitar latencia JIT en la primera ejecución real.
    La compilación se cachea en disco, así que solo es lenta la primera vez.
    """
    p = np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64)
    t = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    q = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    d = np.array([1, -1, 1, -1], dtype=np.int8)

    segment_events_kernel(
        p, t, q, d, theta,
        np.int8(0), np.float64(0), np.float64(0), np.float64(0)
    )


def verify_nopython_mode() -> dict:
    """
    Verifica que el kernel esté compilado en modo nopython.

    Returns:
        Dict con información de compilación:
        - nopython: bool - True si está en modo nopython
        - signatures: list - Firmas compiladas
        - cache_hits: int - Número de hits de cache
    """
    from numba.core import types as numba_types

    func = segment_events_kernel

    # Obtener información del dispatcher
    signatures = list(func.signatures) if hasattr(func, 'signatures') else []

    # Verificar modo nopython
    nopython = True
    if hasattr(func, 'nopython_signatures'):
        nopython = len(func.nopython_signatures) > 0

    # Stats de cache
    stats = func.stats if hasattr(func, 'stats') else {}
    cache_hits = stats.get('cache_hits', 0) if isinstance(stats, dict) else 0

    return {
        'nopython': nopython,
        'signatures': [str(s) for s in signatures],
        'cache_hits': cache_hits,
        'cache_path': str(func._cache._cache_path) if hasattr(func, '_cache') else None,
    }


def estimate_memory_usage(n_ticks: int, theta: float = 0.005) -> dict:
    """
    Estima el uso de memoria del kernel para un número dado de ticks.

    Args:
        n_ticks: Número de ticks a procesar
        theta: Umbral DC (afecta ratio de eventos)

    Returns:
        Dict con estimaciones en bytes y MB
    """
    # Estimación de eventos basada en theta
    # Menor theta = más eventos
    event_ratio = max(100, int(1000 * theta / 0.005))
    estimated_events = max(n_ticks // event_ratio, _MIN_EVENT_SLOTS)

    # Búferes de ticks (worst case: todos van a DC o OS)
    tick_buffers = 8 * n_ticks * 8  # 8 arrays × n × 8 bytes promedio

    # Búferes de eventos
    event_buffers = (
        estimated_events * 1 +  # event_types (int8)
        (estimated_events + 1) * 8 * 2  # offsets (int64) × 2
    )

    # Atributos de evento: 4 arrays × n_events × 8 bytes
    attribute_buffers = estimated_events * 8 * 4

    total = tick_buffers + event_buffers + attribute_buffers

    return {
        'n_ticks': n_ticks,
        'estimated_events': estimated_events,
        'tick_buffers_bytes': tick_buffers,
        'event_buffers_bytes': event_buffers,
        'attribute_buffers_bytes': attribute_buffers,
        'total_bytes': total,
        'total_mb': total / (1024 * 1024),
    }


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def benchmark_kernel(
    n_ticks: int = 1_000_000,
    theta: float = 0.005,
    n_runs: int = 5,
    warmup_runs: int = 2,
) -> dict:
    """
    Benchmark del kernel con datos sintéticos.

    Args:
        n_ticks: Número de ticks sintéticos
        theta: Umbral DC
        n_runs: Número de ejecuciones para promediar
        warmup_runs: Ejecuciones de calentamiento (no contadas)

    Returns:
        Dict con métricas de rendimiento
    """
    import time

    # Generar datos sintéticos (random walk)
    np.random.seed(42)
    returns = np.random.randn(n_ticks) * 0.0001  # ~0.01% volatilidad por tick
    prices = 100.0 * np.exp(np.cumsum(returns)).astype(np.float64)
    timestamps = np.arange(n_ticks, dtype=np.int64) * 1_000_000  # 1ms entre ticks
    quantities = np.ones(n_ticks, dtype=np.float64)
    directions = np.random.choice(np.array([1, -1], dtype=np.int8), n_ticks)

    # Warmup
    for _ in range(warmup_runs):
        segment_events_kernel(
            prices, timestamps, quantities, directions, theta,
            np.int8(0), np.float64(0), np.float64(0), np.float64(0)
        )

    # Benchmark
    times = []
    n_events_list = []

    for _ in range(n_runs):
        start = time.perf_counter()
        result = segment_events_kernel(
            prices, timestamps, quantities, directions, theta,
            np.int8(0), np.float64(0), np.float64(0), np.float64(0)
        )
        end = time.perf_counter()

        times.append(end - start)
        n_events_list.append(result[15])  # n_events (índice 15 después de 4 atributos)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_events = np.mean(n_events_list)

    return {
        'n_ticks': n_ticks,
        'theta': theta,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'ticks_per_second': n_ticks / avg_time,
        'avg_events': int(avg_events),
        'event_ratio': n_ticks / avg_events if avg_events > 0 else float('inf'),
        'memory_estimate_mb': estimate_memory_usage(n_ticks, theta)['total_mb'],
    }
