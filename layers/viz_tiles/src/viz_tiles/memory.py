"""Ajuste de los allocators para que el RSS de la unidad sea O(lote).

Igual que en L2 (`l2_dc_events/memory.py`, ITSC-275): el pool de Arrow por defecto
(`mimalloc`) y el umbral dinámico de `mmap` de glibc retienen memoria ya liberada
(row groups de L1 y de eventos de cientos de KB), y el RSS crece sin que haya más
datos vivos. Se corrige con dos ajustes:

- `ARROW_DEFAULT_MEMORY_POOL=system`, que Arrow lee al importarse: por eso
  `viz_tiles/__init__.py` lo fija antes de importar pyarrow, y el Dockerfile
  también lo declara como `ENV`.
- `mallopt(M_MMAP_THRESHOLD, 16 KiB)`, que apaga el ajuste dinámico de glibc: los
  bloques mayores vuelven a ser `mmap` y `munmap` los devuelve. La imagen lo
  declara como `MALLOC_MMAP_THRESHOLD_`; esta función cubre la ejecución local.
"""

import ctypes
import logging

import pyarrow as pa

logger = logging.getLogger(__name__)

# `mallopt(M_MMAP_THRESHOLD, ...)` de glibc.
_M_MMAP_THRESHOLD = -3
_MMAP_THRESHOLD = 16384


def tune_allocators() -> None:
    """Fija el umbral de `mmap` y avisa si el pool de Arrow no es el del sistema."""
    if pa.default_memory_pool().backend_name != "system":
        logger.warning(
            "el pool de Arrow es %s, no el del sistema: el RSS puede crecer; "
            "fija ARROW_DEFAULT_MEMORY_POOL=system antes de importar pyarrow",
            pa.default_memory_pool().backend_name,
        )
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except AttributeError, OSError:
        logger.warning("sin mallopt: el umbral de mmap queda dinámico")
        return
    if not mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD):
        logger.warning("mallopt rechazó el umbral de mmap: queda dinámico")
