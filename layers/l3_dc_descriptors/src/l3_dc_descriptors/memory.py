"""Ajuste de los allocators para que el RSS de la unidad sea O(lote).

Es el de L2 (`l2_dc_events.memory`, medido en ITSC-275): el pool por defecto de Arrow
(`mimalloc`) retiene páginas por hilo y el umbral dinámico de `mmap` de glibc fragmenta el
heap de cada hilo; con el pool del sistema (`ARROW_DEFAULT_MEMORY_POOL=system`, que Arrow
lee al importarse y por eso va en el `ENV` de la imagen) y el umbral fijo en 16 KiB, la
memoria liberada vuelve al sistema. L3 no tiene hilos todavía; el ajuste queda igual
para que el RSS que mida ITSC-335 sea el de la imagen que irá a producción.
"""

import ctypes
import logging

import pyarrow as pa

logger = logging.getLogger(__name__)

# `mallopt(M_MMAP_THRESHOLD, ...)` de glibc. Fijarlo es lo que apaga el ajuste dinámico.
_M_MMAP_THRESHOLD = -3
_MMAP_THRESHOLD = 16384


def tune_allocators() -> None:
    """Fija el umbral de `mmap` de glibc y avisa si el pool de Arrow no es el del sistema."""
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
        return
    if not mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD):
        logger.warning("mallopt rechazó el umbral de mmap: queda dinámico")
