"""El resumidor: de L1 y L2 de un mes a `summaries.parquet` de cada θ (TRD-L3 §8.1).

Una pasada por mes y por lista de θ, sin estado entre meses (ADR-L3-12). Este módulo es el
orquestador de la pasada (ITSC-363, ADR-L3-16): Python lee, escribe y mide; **no calcula**.
La aritmética, las posiciones de las fronteras y los siete invariantes de §7.2 viven en el
motor nativo (`dc_pyo3.Engine`, crate `dc_descriptors`), que tiene sus propios hilos.

```text
  L2 (50 events.parquet) ─ pool de E/S ─► BoundaryFeed ─┐ fronteras, por delante de cada row group
  L1 (consolidated) ─ read_row_groups(R) ─► push ───────┤
                                                        ▼
                                                  motor nativo (hilos, ítems (row group, θ))
                                                        │ pop_sealed
                                            Collector ──┴─► SealedWriters (W hilos, cola por θ)
                                                                   ▼
                                                        FamilyWriter de cada θ ─► commit
```

Un row group de L1 se empuja cuando hay lugar: a lo más `inflight` (K) row groups anclan sus
ticks en el motor, y los que esperan escribirse no pasan de un tope; el lector trae `readers`
(R) row groups por delante. Antes de cada `push`, `BoundaryFeed.cover(high)` entrega al motor
las fronteras de L2 que ese row group toca, y no más: el motor suelta cada lote cuando sus
eventos cierran, así que lo retenido es O(lote + eventos abiertos), no O(mes). El orden de
los θ y de los row groups no cambia un bit de la salida: el motor deja cada fila en la casilla
de su evento y sella los row groups de salida de cada θ en orden.

Atómico por archivo: nada se publica hasta que todo el mes se resumió y concilió. Con
`threads=1` no hay pool de E/S: la compuerta, la apertura, la conciliación y la publicación
van en serie en el hilo principal (los hilos del motor y del lector no se apagan).
"""

import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, ExitStack, closing, nullcontext
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import dc_pyo3
from dc_frames import (
    FamilySkeletonMismatch,
    FamilyWriter,
    FrameBoundaryError,
    FramesInputError,
    family_path,
    l2_path,
)
from dc_frames.lake import L1, EventFile, Month, month_of, month_ordinal

from l3_dc_descriptors import native
from l3_dc_descriptors.boundaries import BatchStream
from l3_dc_descriptors.cpu import cpu_limit
from l3_dc_descriptors.errors import L2StateInconsistent, SkeletonMismatch, month_text
from l3_dc_descriptors.event import EVENT_COLUMNS
from l3_dc_descriptors.l2 import EVENTS, CarryOver, read_carry_over, require
from l3_dc_descriptors.outputs import POLL_S, Collector, SealedWriters
from l3_dc_descriptors.reconcile import L1Month, Reconciliation, reconcile
from l3_dc_descriptors.schema import FAMILY, STATE_VERSION, SUMMARIES_SCHEMA
from l3_dc_descriptors.timing import Phases

DEFAULT_KEY = ("binance", "spot", "BTCUSDT")
_THETA_SCALE = Decimal("1e-8")
# Hilos del pool de E/S por vCPU efectivo: la espera de red de cada θ (compuerta, apertura,
# lectura de L2, conciliación, publicación y linaje) se solapa con la de otros (ITSC-360).
THREADS_PER_CORE = 3
# Lectores de L1 (R), row groups en vuelo en el motor (K) y escritores (W). Ley de Little
# (ITSC-363): con el motor al 95 % la unidad consume ≈ 1 row group por segundo y con
# `pre_buffer` una lectura tarda ≈ 0,6 s, así que dos lectores bastan; K = 4 deja uno en
# cómputo, dos por terminar y uno por empezar.
DEFAULT_READERS = 2
DEFAULT_INFLIGHT = 4
DEFAULT_WRITERS = 4
# Row groups de salida sellados que esperan escribirse, por escritor, antes de dejar de
# empujar row groups nuevos: acota lo sellado que se acumula en RAM si escribir se atrasa.
PENDING_PER_WRITER = 8


@dataclass(frozen=True, slots=True)
class ThetaResult:
    """Lo que la unidad dejó de un θ: su archivo, sus filas, su hash y su conciliación."""

    path: str
    rows: int
    content_hash: str
    reconciliation: Reconciliation


def ordered[T, R](
    pool: ThreadPoolExecutor | None, fn: Callable[[T], R], items: Iterable[T]
) -> list[R]:
    """`fn` de cada item, en el pool y devuelto en el orden de `items`.

    Sin `pool`, en serie en este hilo y se detiene en el primer error. Con `pool`, espera
    a todas las tareas y relanza el error del primer item que falló en el orden de
    `items`: el mismo que lanzaría el bucle en serie.
    """
    if pool is None:
        return [fn(item) for item in items]
    futures = [pool.submit(fn, item) for item in items]
    wait(futures)
    return [future.result() for future in futures]


def _workers(requested: int | None, thetas: int) -> int:
    """Hilos del pool de E/S: lo pedido o, por defecto, `THREADS_PER_CORE` por vCPU efectivo
    (redondeado hacia arriba); nunca más que θ.
    """
    if requested is None:
        requested = max(1, math.ceil(THREADS_PER_CORE * cpu_limit().cores))
    return min(requested, thetas)


def _engine_threads(requested: int | None) -> int:
    """Hilos del motor: lo pedido o, por defecto, los vCPU efectivos (redondeado hacia arriba)."""
    if requested is None:
        requested = max(1, math.ceil(cpu_limit().cores))
    return requested


def theta_pool(threads: int) -> AbstractContextManager[ThreadPoolExecutor | None]:
    """El pool de E/S de `threads` hilos para un `with`, o `None` (todo en serie) con uno."""
    if threads <= 1:
        return nullcontext()
    return ThreadPoolExecutor(threads, thread_name_prefix="l3-theta")


def _gate_theta(
    l2_root: str,
    key: tuple[str, str, str],
    month: Month,
    series_start: Month,
    text: str,
) -> tuple[CarryOver, int]:
    """La compuerta de entrada de un θ (§9.2): su carry-over y la dirección del mes anterior."""
    require(l2_path(l2_root, key, text, month), text, month, EVENTS)
    carry = read_carry_over(l2_root, key, text, month)
    if month == series_start:
        return carry, 0
    before = month_of(month_ordinal(month) - 1)
    return carry, read_carry_over(l2_root, key, text, before).direction


def _theta(value: Decimal | str) -> Decimal:
    theta = Decimal(value).quantize(_THETA_SCALE)
    if theta != Decimal(value):
        raise ValueError(f"θ {value} no cabe en 8 decimales, que es la partición de L2")
    return theta


def _inconsistent(error: FrameBoundaryError, month: Month) -> L2StateInconsistent:
    """El `FrameBoundaryError` del lector como `l2_state_inconsistent` (§9.2).

    El θ, el campo y los dos valores salen de los atributos del error; el mensaje
    completo viaja en `details["message"]`.
    """
    result = L2StateInconsistent.mismatch(
        error.theta, month, error.field, error.got, error.wanted
    )
    result.details["message"] = str(error)
    return result


def _open_all[T: AbstractContextManager](
    pool: ThreadPoolExecutor | None,
    stack: ExitStack,
    make: Callable[[str], T],
    texts: Sequence[str],
) -> list[T]:
    """Abre `make(texto)` de cada θ en el pool y registra el `__exit__` de cada uno en
    `stack`, para que se cierren con la excepción que los alcance.

    `make` devuelve el objeto ya abierto. Si alguno falla, los que sí se abrieron se
    cierran antes de relanzar el primer error.
    """
    opened: list[T] = []

    def one(text: str) -> T:
        item = make(text)
        opened.append(item)
        return item

    try:
        items = ordered(pool, one, texts)
    except BaseException as error:
        for item in reversed(opened):
            item.__exit__(type(error), error, error.__traceback__)
        raise
    for item in items:
        stack.push(item)
    return items


def _open_writer(l2_root: str, l3_root: str, key, month: Month, text: str):
    """El `FamilyWriter` del θ, ya abierto (su temporal creado)."""
    writer = FamilyWriter(
        family_path(l3_root, key, text, month, FAMILY),
        SUMMARIES_SCHEMA,
        STATE_VERSION,
        l2_path(l2_root, key, text, month),
    )
    try:
        return writer.__enter__()
    except BaseException as error:
        writer.__exit__(type(error), error, error.__traceback__)
        raise


def _require_statistics(file: EventFile, name: str) -> tuple[int, int] | None:
    """El rango de `name` del archivo, o `None` si no tiene filas; sin estadísticas falla."""
    if not any(file.rows):
        return None
    found = file.id_range(name)
    if found is None:
        raise FramesInputError(
            f"{file.path}: el archivo no trae estadísticas de {name}"
        )
    return found


def _check(collector: Collector, sink: SealedWriters) -> None:
    """Relanza el error de quien recoge o escribe lo sellado, si lo hubo."""
    for error in (collector.error, sink.error):
        if error is not None:
            raise error


def _push_all(
    engine: dc_pyo3.Engine,
    row_groups,
    feeds: Sequence[native.BoundaryFeed],
    collector: Collector,
    sink: SealedWriters,
    month: Month,
    phases: Phases,
) -> int:
    """Empuja al motor los row groups de L1, cada uno con las fronteras que toca.

    Un row group se pide al lector solo cuando hay menos de `inflight` en el motor y pocos
    sellados sin escribir (contrapresión): el lector trae por delante a lo más `readers`,
    así que en RAM hay a lo más `inflight + readers + 1` row groups de L1. `read_s` suma lo
    que este hilo esperó al lector y a las fronteras de L2. Devuelve cuántos empujó.
    """
    limit = PENDING_PER_WRITER * phases.writers
    source = iter(row_groups)
    pushed = 0
    while True:
        while engine.inflight() >= phases.inflight or sink.pending >= limit:
            _check(collector, sink)
            time.sleep(POLL_S)
        _check(collector, sink)
        started = time.perf_counter()
        rg = next(source, None)
        if rg is not None:
            high = rg.high
            for feed in feeds:
                feed.cover(high)
        phases.read_s += time.perf_counter() - started
        if rg is None:
            return pushed
        native.push(engine, pushed, rg.ids, rg.columns, rg.month < month)
        del rg  # el motor retiene sus buffers hasta terminarlo
        pushed += 1
        phases.inflight_max = max(phases.inflight_max, engine.inflight())


def summarize_month(
    l1_root: str | Path,
    l2_root: str | Path,
    l3_root: str | Path,
    month: Month,
    thetas: Iterable[Decimal | str],
    *,
    series_start: Month,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
    phases: Phases | None = None,
    threads: int | None = None,
    readers: int | None = None,
    inflight: int | None = None,
    writers: int | None = None,
    engine_threads: int | None = None,
) -> dict[Decimal, ThetaResult]:
    """Escribe `summaries.parquet` del mes para cada θ y devuelve lo que dejó de cada uno.

    Los eventos son los que cierran en `month` (la partición de L2); sus ticks pueden
    estar en meses anteriores de L1, que el lector relee. `series_start` es el primer mes
    de la serie: el único cuyo mes anterior no tiene carry-over de L2.

    Los hilos y los topes son del orquestador y no cambian el resultado: `threads` es el
    pool de E/S (compuerta, apertura, lectura de L2, conciliación y publicación; por defecto
    `THREADS_PER_CORE` × vCPU efectivos, nunca más que θ; con 1 no hay pool y esos pasos
    van en serie), `readers` los row groups de L1 que se leen a la vez (R), `inflight` los
    que pueden anclar sus ticks en el motor (K), `writers` los escritores (W) y
    `engine_threads` los hilos del motor (por defecto los vCPU efectivos). Ver el módulo
    para la cuenta de memoria.

    Atómico por archivo: la unidad comprueba la entrada de L2, resume todos los θ y concilia
    los ticks de cada uno antes de publicar el primero; un error de entrada, de resumen o de
    conciliación (todos son `SummarizeError` o `FramesError`) no deja ningún archivo y borra
    los temporales. Un error del motor en cualquier ítem se convierte en la misma excepción
    tipada que lanzaba el resumidor de Python (`native.to_summarize_error`). Si la
    publicación misma falla en un θ (por ejemplo, el `move` a GCS), los θ ya publicados
    quedan: re-ejecutar el mes los reescribe con el mismo contenido.

    Sin estado: todo lo que la unidad acumula (el motor, los escritores, los tiempos) es
    suyo y muere con ella. `phases`, si se pasa, recibe los segundos de cada fase y la
    configuración con que corrió (ver `timing`).
    """
    phases = phases if phases is not None else Phases()
    started = time.perf_counter()
    l1_root, l2_root, l3_root = str(l1_root), str(l2_root), str(l3_root)
    key = (provider, market, asset)
    values = [_theta(t) for t in thetas]
    if not values or len(set(values)) != len(values):
        raise ValueError("thetas debe traer al menos un θ y sin repetidos")
    if month < series_start:
        raise ValueError(
            f"el mes {month_text(month)} es anterior al inicio de la serie "
            f"{month_text(series_start)}"
        )
    for name, value in (
        ("threads", threads),
        ("readers", readers),
        ("inflight", inflight),
        ("writers", writers),
        ("engine_threads", engine_threads),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} {value} debe ser al menos 1")
    threads = _workers(threads, len(values))
    phases.threads = threads
    phases.readers = DEFAULT_READERS if readers is None else readers
    phases.inflight = DEFAULT_INFLIGHT if inflight is None else inflight
    phases.writers = DEFAULT_WRITERS if writers is None else writers
    phases.engine_threads = _engine_threads(engine_threads)
    texts = [f"{theta:.8f}" for theta in values]

    # Compuerta de entrada (§9.2), antes de abrir ningún escritor: los θ se comprueban en
    # el pool y, si alguno falla, se lanza el primero en el orden de `texts`.
    with theta_pool(threads) as pool:
        gated = ordered(
            pool,
            lambda text: _gate_theta(l2_root, key, month, series_start, text),
            texts,
        )
    phases.gate_s += time.perf_counter() - started

    began = time.perf_counter()
    with ExitStack() as stack:
        pool = stack.enter_context(theta_pool(threads))
        # Abrir los 50 escritores (cada uno lee el pie de su esqueleto) y los 50 archivos
        # de L2 en el pool, no uno tras otro. El orden de entrada es el inverso del cierre:
        # lo último que se abre (hilos, motor) se cierra primero.
        files = _open_all(
            pool,
            stack,
            lambda text: EventFile(
                l2_path(l2_root, key, text, month),
                f"L2 del θ {text} en {month_text(month)}",
                EVENT_COLUMNS,
            ),
            texts,
        )
        family = _open_all(
            pool,
            stack,
            lambda text: _open_writer(l2_root, l3_root, key, month, text),
            texts,
        )
        l1 = stack.enter_context(closing(L1Month(l1_root, key, month)))
        engine = native.new_engine(
            values, [w.row_groups for w in family], phases.engine_threads
        )
        stack.callback(engine.close)

        # Dónde empiezan y terminan los eventos de todos los θ, con las estadísticas del
        # pie de L2 (sin leer datos): el primer row group de L1 que se lee es el que
        # contiene la primera referencia y el último, el que contiene el último extremo.
        references = [_require_statistics(f, "reference_agg_trade_id") for f in files]
        extremes = [_require_statistics(f, "extreme_agg_trade_id") for f in files]
        streams = [BatchStream(file, pool, phases.read) for file in files]
        for stream in reversed(streams):
            stack.callback(stream.close)
        feeds = [
            native.BoundaryFeed(engine, position, stream)
            for position, stream in enumerate(streams)
        ]
        sink = SealedWriters(
            stack.enter_context(
                ThreadPoolExecutor(phases.writers, thread_name_prefix="l3-write")
            ),
            family,
            values,
            texts,
            month,
            phases.write,
        )
        stack.callback(sink.stop)
        collector = stack.enter_context(Collector(engine, sink))
        phases.setup_s += time.perf_counter() - began

        begun = time.perf_counter()
        try:
            if any(references):
                first = min(r[0] for r in references if r)
                last = max(e[1] for e in extremes if e)
                row_groups = stack.enter_context(
                    closing(
                        L1(l1_root, key).row_groups(
                            month,
                            month,
                            first,
                            lambda low, high: high >= first and low <= last,
                            readers=phases.readers,
                        )
                    )
                )
                pushed = _push_all(
                    engine, row_groups, feeds, collector, sink, month, phases
                )
                phases.items = pushed * sum(1 for f in files if any(f.rows))
            stats = engine.finish()
            phases.aggregate_s += time.perf_counter() - begun
            phases.engine_busy = engine.busy()
            collector.stop()
            _check(collector, sink)
            sink.join()
        except dc_pyo3.DescriptorError as error:
            converted = native.to_summarize_error(error, texts, month)
            if converted is error:
                raise
            raise converted from error

        begun = time.perf_counter()
        reconciled = ordered(
            pool,
            lambda i: reconcile(
                l1,
                theta=texts[i],
                month=month,
                closed=stats[i][1],
                before_month=stats[i][2],
                last_extreme=stats[i][4],
                first_reference=stats[i][3],
                previous_direction=gated[i][1],
                carry=gated[i][0],
            ),
            range(len(values)),
        )
        phases.reconcile_s += time.perf_counter() - begun

        begun = time.perf_counter()
        digests = ordered(
            pool, lambda i: _commit(family[i], texts[i], month), range(len(values))
        )
        phases.publish_s += time.perf_counter() - begun
        return {
            theta: ThetaResult(family[i].path, stats[i][0], digests[i], reconciled[i])
            for i, theta in enumerate(values)
        }


def _commit(writer: FamilyWriter, text: str, month: Month) -> str:
    """Publica el archivo del θ y devuelve su `content_hash`."""
    try:
        return writer.commit()
    except FamilySkeletonMismatch as error:
        raise SkeletonMismatch(text, month, FAMILY, error.detail) from error
