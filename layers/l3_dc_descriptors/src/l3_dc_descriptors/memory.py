"""Ajuste de los allocators para que el RSS de la unidad sea O(lote).

Es el de L2 (`l2_dc_events.memory`, medido en ITSC-275): el pool por defecto de Arrow
(`mimalloc`) retiene páginas por hilo y el umbral dinámico de `mmap` de glibc fragmenta el
heap de cada hilo; con el pool del sistema (`ARROW_DEFAULT_MEMORY_POOL=system`, que Arrow
lee al importarse y por eso va en el `ENV` de la imagen) y el umbral fijo en 16 KiB, la
memoria liberada vuelve al sistema.

El umbral se puede cambiar por entorno con `MALLOC_MMAP_THRESHOLD_` (ITSC-366): con 16 KiB
cada buffer grande es un `mmap`/`munmap` y se vuelve a tocar página por página, que son los
≈ 32 M de fallos de página de la unidad; un umbral mayor deja a glibc reutilizar el heap a
cambio de más RSS. `unit_timing` registra el que rigió (`mmap_threshold`).
"""

import ctypes
import logging
import os

import pyarrow as pa

logger = logging.getLogger(__name__)

# `mallopt(M_MMAP_THRESHOLD, ...)` de glibc. Fijarlo es lo que apaga el ajuste dinámico.
_M_MMAP_THRESHOLD = -3
_ENV = "MALLOC_MMAP_THRESHOLD_"
DEFAULT_MMAP_THRESHOLD = 16384
# Lo máximo que glibc acepta en 64 bits (`HEAP_MAX_SIZE / 2`): con más, `mallopt` falla.
MAX_MMAP_THRESHOLD = 32 << 20


def _valid(text: str | None) -> int | None:
    """El umbral de `text` si es un entero entre 0 y `MAX_MMAP_THRESHOLD`; si no, `None`."""
    # Solo dígitos decimales, por decisión propia: glibc también acepta `0x…` y octales al
    # arrancar, pero `mallopt` pisa lo que haya leído con el valor que sale de aquí.
    if text is None or not (text.isascii() and text.isdigit()):
        return None
    value = int(text)
    return value if value <= MAX_MMAP_THRESHOLD else None


def mmap_threshold() -> int | None:
    """El umbral de `mmap` con que corre el proceso, o `None` si quedó dinámico.

    La fuente es `MALLOC_MMAP_THRESHOLD_`: glibc la lee al arrancar y `tune_allocators` la
    deja igual a lo que fijó (o la quita si no pudo fijar nada). Así no hay estado de módulo.
    """
    return _valid(os.environ.get(_ENV))


def _requested() -> int:
    """El umbral que pide el entorno si es un entero válido; si no, el de por defecto."""
    text = os.environ.get(_ENV)
    value = _valid(text)
    if value is not None:
        return value
    if text is not None:
        logger.warning(
            "%s=%r no es un entero entre 0 y %d: se usa %d",
            _ENV,
            text,
            MAX_MMAP_THRESHOLD,
            DEFAULT_MMAP_THRESHOLD,
        )
    return DEFAULT_MMAP_THRESHOLD


def tune_allocators() -> None:
    """Fija el umbral de `mmap` de glibc y avisa si el pool de Arrow no es el del sistema.

    El umbral es `MALLOC_MMAP_THRESHOLD_` si está definida y es válida, y 16 KiB si no. Deja
    la variable con el valor fijado, para que `mmap_threshold()` diga con cuál corrió.
    """
    if pa.default_memory_pool().backend_name != "system":
        logger.warning(
            "el pool de Arrow es %s, no el del sistema: el RSS puede ser mayor; fija "
            "ARROW_DEFAULT_MEMORY_POOL=system antes de importar pyarrow",
            pa.default_memory_pool().backend_name,
        )
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except AttributeError, OSError:
        logger.warning("sin mallopt: el umbral de mmap queda dinámico")
        os.environ.pop(_ENV, None)
        return
    threshold = _requested()
    if not mallopt(_M_MMAP_THRESHOLD, threshold):
        logger.warning("mallopt rechazó el umbral de mmap: queda dinámico")
        os.environ.pop(_ENV, None)
        return
    os.environ[_ENV] = str(threshold)
