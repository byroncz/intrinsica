"""Ajuste de los allocators para que el RSS de la unidad sea O(lote).

Parte del de L2 (`l2_dc_events.memory`, medido en ITSC-275): el pool por defecto de Arrow
(`mimalloc`) retiene páginas por hilo y el umbral dinámico de `mmap` de glibc fragmenta el
heap de cada hilo; con el pool del sistema (`ARROW_DEFAULT_MEMORY_POOL=system`, que Arrow
lee al importarse y por eso va en el `ENV` de la imagen) la memoria liberada vuelve al
sistema. Eso sigue vigente en L3. Lo que L2 hace además, fijar el umbral de `mmap` en 16 KiB,
L3 ya no lo usa (ITSC-367). Con ese umbral cada buffer grande es un
`mmap`/`munmap` y se vuelve a tocar página por página: ≈ 32 M de fallos de página y ≈ 100 s
de CPU de sistema por unidad (ITSC-366). Con el umbral en 32 MiB (el máximo de glibc) y sin
recorte del heap (`MALLOC_TRIM_THRESHOLD_` en 1 GiB, que fija la imagen y glibc lee al
arrancar) glibc reutiliza el heap: −31 % de CPU a cambio de ≈ 1,1 GiB más de RSS (de 2,0 a
3,1 GiB), todavía acotado por K, R, W y los sellados pendientes dentro de una unidad; entre
unidades no se cumplió y `release_to_kernel` lo corrige al cerrar cada una (ITSC-371).
Decisión y cifras: TRD-L3 ADR-L3-16. `MALLOC_MMAP_THRESHOLD_` se puede cambiar por entorno; `unit_timing` registra lo
que rigió (`mmap_threshold` y `trim_threshold`).
"""

import ctypes
import functools
import gc
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pyarrow as pa

logger = logging.getLogger(__name__)

# `mallopt(M_MMAP_THRESHOLD, ...)` de glibc. Fijarlo es lo que apaga el ajuste dinámico.
_M_MMAP_THRESHOLD = -3
_ENV = "MALLOC_MMAP_THRESHOLD_"
_TRIM_ENV = "MALLOC_TRIM_THRESHOLD_"
# Lo máximo que glibc acepta en 64 bits (`HEAP_MAX_SIZE / 2`): con más, `mallopt` falla.
MAX_MMAP_THRESHOLD = 32 << 20
# Lo que rige sin la variable (en local y en pruebas); la imagen fija el mismo en su `ENV`.
DEFAULT_MMAP_THRESHOLD = MAX_MMAP_THRESHOLD


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


def trim_threshold() -> int | None:
    """El umbral de recorte del heap con que corre el proceso, o `None` si no está fijado.

    `MALLOC_TRIM_THRESHOLD_` la fija la imagen y glibc la lee al arrancar: no hay `mallopt`
    aquí, esto solo dice con cuál corrió la unidad. Sin ella, glibc recorta con su umbral
    dinámico.
    """
    text = os.environ.get(_TRIM_ENV)
    return int(text) if text is not None and text.isascii() and text.isdigit() else None


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


def rss_bytes() -> int | None:
    """El RSS del proceso ahora (`VmRSS`), o `None` si no se puede leer.

    No es `ru_maxrss`: ese es un pico y no baja, y lo que importa tras devolver memoria al
    kernel es cuánto ocupa el proceso en este momento.
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _malloc_trim() -> Callable[[int], int] | None:
    """`malloc_trim` del libc, o `None` si no lo tiene (otro libc)."""
    try:
        return ctypes.CDLL(None).malloc_trim
    except AttributeError, OSError:
        return None


def release_to_kernel() -> tuple[int | None, int | None]:
    """Devuelve al kernel la memoria libre de la unidad y dice el RSS antes y después.

    Con el umbral de recorte de la imagen glibc no devuelve al kernel lo liberado en sus
    arenas y el RSS crece unidad a unidad hasta el OOM (ITSC-371). La causa exacta es una
    hipótesis: glibc reutiliza las arenas de los hilos terminados, y el trim reduce la deriva
    a la mitad, no la elimina (TRD-L3 ADR-L3-16, «El RSS crece entre unidades»).
    `gc.collect()` suelta primero lo que solo retienen ciclos (buffers de Arrow anclados por
    ellos) y `malloc_trim(0)` recorre todas las arenas y devuelve las páginas libres; cuesta
    milisegundos y no toca datos. Sin `malloc_trim` (otro libc) avisa y sigue: el RSS no
    baja.
    """
    before = rss_bytes()
    gc.collect()
    malloc_trim = _malloc_trim()
    if malloc_trim is None:
        logger.warning(
            "sin malloc_trim: la memoria libre no vuelve al kernel entre unidades"
        )
    else:
        malloc_trim(0)
    return before, rss_bytes()


def tune_allocators() -> None:
    """Fija el umbral de `mmap` de glibc y avisa si el pool de Arrow no es el del sistema.

    El umbral es `MALLOC_MMAP_THRESHOLD_` si está definida y es válida, y 32 MiB si no. Deja
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


# ITSC-375: qué hay dentro del RSS en el pico y al cierre de la unidad.
_SAMPLE_ENV = "L3_MEMORY_SAMPLE_S"
DEFAULT_SAMPLE_S = 0.25
# ITSC-376: el trim por marca de agua del muestreador. Con 1 024 MiB los meses chicos (pico bajo
# 1 GiB) no pagan ningún trim; en los densos el pico queda acotado por la marca más lo que crece
# en `DEFAULT_TRIM_MIN_S`. Una marca de 0 lo desactiva.
_TRIM_RSS_ENV = "L3_TRIM_RSS_MIB"
_TRIM_MIN_ENV = "L3_TRIM_MIN_S"
DEFAULT_TRIM_RSS_MIB = 1024
DEFAULT_TRIM_MIN_S = 0.5
_MIB = 1 << 20


class _Mallinfo2(ctypes.Structure):
    """`struct mallinfo2` de glibc ≥ 2.33: diez `size_t`, en este orden."""

    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


@functools.cache
def _mallinfo2_function() -> Callable[[], _Mallinfo2] | None:
    try:
        function = ctypes.CDLL(None).mallinfo2
    except AttributeError, OSError:
        return None
    function.restype = _Mallinfo2
    return function


def _mallinfo2() -> _Mallinfo2 | None:
    """La contabilidad de glibc de todas las arenas, o `None` sin `mallinfo2` en el libc."""
    function = _mallinfo2_function()
    return None if function is None else function()


def sample_period(env: Mapping[str, str]) -> float | None:
    """`L3_MEMORY_SAMPLE_S` como segundos (> 0), o `None` si falta o está vacía.

    Un valor que no es un número finito mayor que 0 lanza `ValueError` con el motivo: la CLI
    lo convierte en código 2.
    """
    raw = env.get(_SAMPLE_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{_SAMPLE_ENV}={raw!r} no es un número") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{_SAMPLE_ENV}={raw!r} debe ser un número mayor que 0")
    return value


def trim_rss_mib(env: Mapping[str, str]) -> int | None:
    """`L3_TRIM_RSS_MIB` como MiB (entero ≥ 0; 0 desactiva el trim periódico), o `None` si
    falta o está vacía.

    Un valor que no es un entero ≥ 0 lanza `ValueError` con el motivo: la CLI lo convierte en
    código 2.
    """
    raw = env.get(_TRIM_RSS_ENV)
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    if not (text.isascii() and text.isdigit()):
        raise ValueError(f"{_TRIM_RSS_ENV}={raw!r} debe ser un entero de MiB, 0 o más")
    return int(text)


def trim_min_s(env: Mapping[str, str]) -> float | None:
    """`L3_TRIM_MIN_S` como segundos (número finito ≥ 0), o `None` si falta o está vacía.

    Un valor inválido lanza `ValueError` con el motivo: la CLI lo convierte en código 2.
    """
    raw = env.get(_TRIM_MIN_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{_TRIM_MIN_ENV}={raw!r} no es un número") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{_TRIM_MIN_ENV}={raw!r} debe ser un número de 0 o más")
    return value


@dataclass(frozen=True)
class MemSample:
    """Una foto del proceso: su RSS y cómo lo cuenta glibc en ese instante.

    `uordblks` es la memoria en uso del heap (sin las regiones `mmap`, que van en `hblkhd`) y
    `fordblks`, la que el programa ya liberó y glibc retiene; `arena` es su suma, lo que las
    arenas piden al sistema. `arrow` son los bytes vivos de buffers de Arrow, `engine_inflight`
    los tramos que el motor tiene anclados (`None` si no hay motor) y `elapsed_s` los segundos
    desde el inicio de la unidad. Los campos de glibc son `None` sin `mallinfo2`.
    """

    rss: int | None
    arena: int | None
    hblkhd: int | None
    uordblks: int | None
    fordblks: int | None
    keepcost: int | None
    arrow: int
    engine_inflight: int | None
    elapsed_s: float

    def details(self) -> dict:
        def mib(size: int | None) -> float | None:
            return None if size is None else round(size / _MIB, 1)

        return {
            "rss_mib": mib(self.rss),
            "arena_mib": mib(self.arena),
            "hblkhd_mib": mib(self.hblkhd),
            "uordblks_mib": mib(self.uordblks),
            "fordblks_mib": mib(self.fordblks),
            "keepcost_mib": mib(self.keepcost),
            "arrow_mib": mib(self.arrow),
            "engine_inflight": self.engine_inflight,
            "elapsed_s": round(self.elapsed_s, 3),
        }


def take_sample(
    started: float,
    inflight: Callable[[], int] | None = None,
    rss: int | None = None,
) -> MemSample:
    """Toma una muestra ahora. `started` es el `perf_counter` del inicio de la unidad; `rss`,
    el RSS ya leído (el del tick del muestreador), y si falta se lee aquí.

    Sin `mallinfo2` deja en `None` los campos de glibc (el aviso lo da `MemorySampler.start`);
    nunca lanza.
    """
    info = _mallinfo2()
    engine = None
    if inflight is not None:
        try:
            engine = inflight()
        except Exception:  # noqa: BLE001 - el motor ya cerró: sin dato, no un error
            engine = None
    return MemSample(
        rss=rss_bytes() if rss is None else rss,
        arena=None if info is None else info.arena,
        hblkhd=None if info is None else info.hblkhd,
        uordblks=None if info is None else info.uordblks,
        fordblks=None if info is None else info.fordblks,
        keepcost=None if info is None else info.keepcost,
        arrow=pa.total_allocated_bytes(),
        engine_inflight=engine,
        elapsed_s=time.perf_counter() - started,
    )


class MemorySampler:
    """Hilo `l3-memory-sampler`: guarda la muestra del RSS más alto que vio la unidad y, por
    marca de agua, devuelve al kernel la memoria libre mientras la unidad corre (ITSC-376).

    En cada tick (`period_s`) lee `VmRSS`, que es barato; solo si supera el máximo visto toma
    una muestra completa, porque `mallinfo2` recorre las listas libres con el candado de cada
    arena. Así son a lo más unas pocas llamadas por segundo mientras el RSS crece y ninguna
    cuando ya no sube. Si además `VmRSS` llega a `trim_rss_mib` y pasaron `trim_min_s` desde el
    último trim, llama a `malloc_trim(0)`, que también recorre las arenas con sus candados: por
    eso no va tras cada sellado ni cada N segundos, y bajo la marca cuesta cero. Una marca de 0
    lo desactiva. `trims` y `trim_s` dicen cuántos hizo y cuánto tardaron en total. `inflight`
    lo asigna quien crea el motor (y lo quita al cerrarlo).
    """

    def __init__(
        self,
        period_s: float,
        started: float | None = None,
        trim_rss_mib: int = DEFAULT_TRIM_RSS_MIB,
        trim_min_s: float = DEFAULT_TRIM_MIN_S,
    ) -> None:
        self.period_s = period_s
        self.started = time.perf_counter() if started is None else started
        self.trim_rss_mib = trim_rss_mib
        self.trim_min_s = trim_min_s
        self.inflight: Callable[[], int] | None = None
        self.peak: MemSample | None = None
        self.ticks = 0
        self.trims = 0
        self.trim_s = 0.0
        self._trim: Callable[[int], int] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="l3-memory-sampler", daemon=True
        )

    def start(self) -> None:
        if _mallinfo2_function() is None:
            logger.warning(
                "sin mallinfo2: las muestras de memoria no traen la contabilidad de glibc"
            )
        if self.trim_rss_mib > 0:
            self._trim = _malloc_trim()
            if self._trim is None:
                logger.warning(
                    "sin malloc_trim: la memoria libre no vuelve al kernel durante la unidad"
                )
        self._thread.start()

    def stop(self) -> None:
        """Para el hilo y espera a que termine; las muestras quedan en `peak` y `ticks`."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        top = 0
        mark = self.trim_rss_mib << 20
        last_trim = -math.inf
        while True:
            rss = rss_bytes()
            self.ticks += 1
            if rss is not None and rss > top:
                top = rss
                self.peak = take_sample(self.started, self.inflight, rss)
            if self._trim is not None and rss is not None and rss >= mark:
                now = time.perf_counter()
                if now - last_trim >= self.trim_min_s:
                    self._trim(0)
                    last_trim = time.perf_counter()
                    self.trims += 1
                    self.trim_s += last_trim - now
            if self._stop.wait(self.period_s):
                return
