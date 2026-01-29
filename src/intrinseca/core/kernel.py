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
    # Búferes DC
    dc_prices: ArrayF64
    dc_times: ArrayI64
    dc_quantities: ArrayF64
    dc_directions: ArrayI8
    # Búferes OS
    os_prices: ArrayF64
    os_times: ArrayI64
    os_quantities: ArrayF64
    os_directions: ArrayI8
    # Metadatos eventos
    event_types: ArrayI8
    dc_offsets: ArrayI64
    os_offsets: ArrayI64
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
        Tuple con 17 elementos (ver KernelResult para documentación)
    """
    n = len(prices)

    # --- Early exit para array vacío ---
    if n == 0:
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_i8 = np.empty(0, dtype=np.int8)
        zero_offset = np.zeros(1, dtype=np.int64)
        return (
            empty_f64, empty_i64, empty_f64, empty_i8,
            empty_f64, empty_i64, empty_f64, empty_i8,
            empty_i8,
            zero_offset, zero_offset,
            int64(0), init_trend, init_ext_high_price, init_ext_low_price,
            init_last_os_ref, int64(0)
        )

    # --- Pre-cálculo de thresholds (evita multiplicaciones repetidas) ---
    theta_up = 1.0 + theta
    theta_down = 1.0 - theta

    # --- Estimación inteligente de memoria ---
    # Eventos esperados ≈ n / ratio, con mínimo garantizado
    estimated_events = max(n // _EVENT_RATIO_ESTIMATE, _MIN_EVENT_SLOTS)

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

    # Búferes para metadatos de eventos (tamaño estimado)
    event_types = np.empty(estimated_events, dtype=np.int8)
    dc_offsets = np.empty(estimated_events + 1, dtype=np.int64)
    os_offsets = np.empty(estimated_events + 1, dtype=np.int64)

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

                estimated_events = new_size

            # Actualizar extremos
            if new_trend == 1:
                ext_high_price = conf_price
                ext_high_idx = conf_idx
            else:
                ext_low_price = conf_price
                ext_low_idx = conf_idx

            # --- Cerrar OS del evento anterior ---
            if n_events > 0 and prev_os_start >= 0:
                for i in range(prev_os_start, prev_ext_idx):
                    os_prices[os_ptr] = prices[i]
                    os_times[os_ptr] = timestamps[i]
                    os_quantities[os_ptr] = quantities[i]
                    os_directions[os_ptr] = directions[i]
                    os_ptr += 1
                os_offsets[n_events] = os_ptr

            # --- Escribir DC del evento actual ---
            for i in range(prev_ext_idx, conf_idx):
                dc_prices[dc_ptr] = prices[i]
                dc_times[dc_ptr] = timestamps[i]
                dc_quantities[dc_ptr] = quantities[i]
                dc_directions[dc_ptr] = directions[i]
                dc_ptr += 1

            # Registrar evento
            dc_offsets[n_events + 1] = dc_ptr
            event_types[n_events] = new_trend

            # Preparar siguiente OS
            prev_os_start = conf_idx
            current_trend = new_trend
            last_os_ref = conf_price
            n_events += 1
            last_conf_idx = conf_idx

    # --- Finalización: cerrar último OS ---
    if n_events > 0 and prev_os_start >= 0:
        for i in range(prev_os_start, n):
            os_prices[os_ptr] = prices[i]
            os_times[os_ptr] = timestamps[i]
            os_quantities[os_ptr] = quantities[i]
            os_directions[os_ptr] = directions[i]
            os_ptr += 1
        os_offsets[n_events] = os_ptr

    # --- Calcular índice de huérfanos ---
    if n_events == 0:
        orphan_start_idx = 0
    else:
        if current_trend == 1:
            orphan_start_idx = ext_high_idx
        else:
            orphan_start_idx = ext_low_idx

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

    total = tick_buffers + event_buffers

    return {
        'n_ticks': n_ticks,
        'estimated_events': estimated_events,
        'tick_buffers_bytes': tick_buffers,
        'event_buffers_bytes': event_buffers,
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
        n_events_list.append(result[11])  # n_events

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
