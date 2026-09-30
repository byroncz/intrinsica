"""Ajuste de los allocators para que el RSS de la unidad sea O(lote).

Con los escritores en varios hilos (`pipeline._Writes`), el pico de RSS de
2020-01 pasó de 261 a ~580 MiB sin que Arrow tuviera más de ~70 MiB vivos: era
memoria liberada que el allocator no devolvía al sistema. Dos causas, medidas
(ITSC-275):

- El pool por defecto de Arrow (`mimalloc`) retiene páginas por hilo; con 10
  hilos escribiendo, cada uno guarda las suyas. El pool del sistema (`malloc`
  de glibc) las devuelve. Se elige con `ARROW_DEFAULT_MEMORY_POOL=system`, que
  Arrow lee al importarse: por eso `l2_dc_events/__init__.py` la fija antes de
  importar pyarrow. `pa.set_memory_pool` no sirve: no alcanza a los pools que
  el código C++ de Parquet toma por su cuenta.
- glibc sube dinámicamente su umbral de `mmap` cada vez que se libera un bloque
  grande, y los buffers de cada tramo (cientos de KB) terminan en el heap de
  cada hilo, fragmentado. Fijar el umbral lo desactiva: los bloques de más de
  16 KiB vuelven a ser `mmap` y `munmap` los devuelve al liberarse. Con 128 KiB
  (el valor inicial de glibc) el pico queda ~20 MiB más alto; con 16 KiB el
  costo en tiempo de pared no se nota.

Con ambos, el mismo mes queda en ~250 MiB (base, con un solo hilo: 261). Cada
uno por separado no alcanza (414 y ~580 MiB). En la imagen (E4a) conviene fijar
las dos como `ENV` (`ARROW_DEFAULT_MEMORY_POOL=system`,
`MALLOC_MMAP_THRESHOLD_=16384`); este módulo cubre la ejecución local.
"""

import ctypes
import logging

import pyarrow as pa

logger = logging.getLogger(__name__)

# `mallopt(M_MMAP_THRESHOLD, ...)` de glibc. Fijarlo es lo que apaga el ajuste
# dinámico.
_M_MMAP_THRESHOLD = -3
_MMAP_THRESHOLD = 16384


def tune_allocators() -> None:
    """Fija el umbral de `mmap` de glibc y avisa si el pool de Arrow no es el del
    sistema. Llamar antes de crear los hilos y los escritores.
    """
    if pa.default_memory_pool().backend_name != "system":
        logger.warning(
            "el pool de Arrow es %s, no el del sistema: con varios hilos el RSS "
            "puede duplicarse; fija ARROW_DEFAULT_MEMORY_POOL=system antes de "
            "importar pyarrow",
            pa.default_memory_pool().backend_name,
        )
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except AttributeError, OSError:
        logger.warning("sin mallopt: el umbral de mmap queda dinámico")
        return
    if not mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD):
        logger.warning("mallopt rechazó el umbral de mmap: queda dinámico")
