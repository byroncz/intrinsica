"""El resumidor: de L1 y L2 de un mes a `summaries.parquet` de cada θ (TRD-L3 §8.1).

Una pasada por mes y por lista de θ, sin estado entre meses (ADR-L3-12). Este módulo es el
orquestador de la pasada (ITSC-363, ADR-L3-16): Python lee, escribe y mide; **no calcula**.
La aritmética, las posiciones de las fronteras y los siete invariantes de §7.2 viven en el
motor nativo (`dc_pyo3.Engine`, crate `dc_descriptors`), que tiene sus propios hilos.

```text
  L2 (50 events.parquet) ─ pool de E/S ─► BoundaryFeed ─┐ fronteras, por delante de cada row group
       └─ tap: hash de las 11 columnas y huella de las claves (linaje y esqueleto)
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

**Entrada de los θ en dos tramos (ITSC-365).** El primer tramo, una tarea por θ en el pool de
E/S, abre `events.parquet` (si falta, `l2_missing`), comprueba sus tipos y arranca la lectura
anticipada de sus fronteras; de ahí salen el pie del archivo y, con él, los tamaños de row
group que el constructor del motor pide de los 50 θ. El segundo tramo (la compuerta de
carry-over del mes y del anterior, con el hash del mes, y la apertura del `FamilyWriter` con
el esqueleto en RAM que deja `L2Events`) corre en el mismo pool mientras el hilo principal
construye el motor y espera el pie de L1. El orquestador espera los carry-overs antes del
primer `push` (la conciliación anota con ellos los ids que va a contar) y los escritores solo
cuando necesita el primero: los row groups sellados los esperan (`_LateWriter`) y el
orquestador los reúne todos antes de conciliar y publicar. Un fallo al abrir un escritor
detiene la unidad en el siguiente `push`, no al final.

Cada archivo de L2 se lee una vez: los mismos lotes alimentan al motor, al
`l2_events_content_hash` y a la comprobación de las claves del esqueleto. El carry-over se
lee una vez y de ahí salen su contenido y su hash. Nada se relee para el linaje.

Atómico por archivo: nada se publica hasta que todo el mes se resumió y concilió. Con
`threads=1` no hay pool de E/S: la compuerta, la apertura, la conciliación y la publicación
van en serie en el hilo principal (los hilos del motor y del lector no se apagan).
"""

import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
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
from l3_dc_descriptors.boundaries import BatchStream, L2Events, check_types
from l3_dc_descriptors.cpu import cpu_limit
from l3_dc_descriptors.errors import L2StateInconsistent, SkeletonMismatch, month_text
from l3_dc_descriptors.l2 import (
    EVENTS,
    read_carry_over,
    read_carry_over_hashed,
    require,
)
from l3_dc_descriptors.outputs import POLL_S, Collector, SealedWriters
from l3_dc_descriptors.reconcile import (
    L1Month,
    Reconciliation,
    first_reference_bound,
    reconcile,
)
from l3_dc_descriptors.schema import FAMILY, STATE_VERSION, SUMMARIES_SCHEMA
from l3_dc_descriptors.timing import Phases

DEFAULT_KEY = ("binance", "spot", "BTCUSDT")
_THETA_SCALE = Decimal("1e-8")
# Hilos del pool de E/S por vCPU efectivo: la espera de red de cada θ (compuerta y apertura,
# lectura de L2, conciliación y publicación) se solapa con la de otros (ITSC-360).
THREADS_PER_CORE = 3
# Lectores de L1 (R), row groups en vuelo en el motor (K) y escritores (W). Ley de Little
# (ITSC-363): con el motor al 95 % la unidad consume ≈ 1 row group por segundo y con
# `pre_buffer` una lectura tarda ≈ 0,6 s, así que dos lectores bastan; K = 4 deja uno en
# cómputo, dos por terminar y uno por empezar.
DEFAULT_READERS = 2
DEFAULT_INFLIGHT = 4
DEFAULT_WRITERS = 12
# Row groups de salida sellados que esperan escribirse, por escritor, antes de dejar de
# empujar row groups nuevos: acota lo sellado que se acumula en RAM si escribir se atrasa. El
# tope total es `PENDING_PER_WRITER × W` y cada row group sellado pesa ≈ 8 MiB (256 B por
# evento × ≈ 34 000 eventos): con W = 12 y 3 son 36 ≈ 290 MiB, lo mismo que 8 con W = 4 (32 ≈
# 260 MiB). Con 8 y W = 12 serían 96 ≈ 770 MiB, y T4b (W = 12) ya pagó +467 MiB sobre T4
# (2 331 contra 1 864).
PENDING_PER_WRITER = 3


@dataclass(frozen=True, slots=True)
class ThetaResult:
    """Lo que la unidad dejó de un θ: su archivo, sus filas, su hash, su conciliación y el
    `content_hash` de las dos entradas de L2 que usó (el linaje de `summaries_summary`).
    """

    path: str
    rows: int
    content_hash: str
    reconciliation: Reconciliation
    l2_events_content_hash: str
    l2_carry_over_content_hash: str


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
    """Hilos del motor: lo pedido o, por defecto, los vCPU efectivos (redondeado al entero
    más cercano: con 3,95 vCPU medidos son 4, no un hilo más que cores).
    """
    if requested is None:
        requested = max(1, round(cpu_limit().cores))
    return requested


def theta_pool(threads: int) -> AbstractContextManager[ThreadPoolExecutor | None]:
    """El pool de E/S de `threads` hilos para un `with`, o `None` (todo en serie) con uno."""
    if threads <= 1:
        return nullcontext()
    return ThreadPoolExecutor(threads, thread_name_prefix="l3-theta")


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
    make: Callable[[int], T],
    positions: Sequence[int],
) -> list[T]:
    """Abre `make(posición)` de cada θ en el pool y registra el `__exit__` de cada uno en
    `stack`, para que se cierren con la excepción que los alcance.

    `make` devuelve el objeto ya abierto. Si alguno falla, los que sí se abrieron se
    cierran antes de relanzar el primer error.
    """
    opened: list[T] = []

    def one(position: int) -> T:
        item = make(position)
        opened.append(item)
        return item

    try:
        items = ordered(pool, one, positions)
    except BaseException as error:
        for item in reversed(opened):
            item.__exit__(type(error), error, error.__traceback__)
        raise
    for item in items:
        stack.push(item)
    return items


def _submit[T](
    pool: ThreadPoolExecutor | None, fn: Callable[..., T], *args
) -> Future[T]:
    """`fn(*args)` en el pool; sin pool, ahora mismo y en este hilo, ya resuelto."""
    if pool is not None:
        return pool.submit(fn, *args)
    future: Future[T] = Future()
    try:
        future.set_result(fn(*args))
    except BaseException as error:  # noqa: BLE001 - sale en `result()`, como con pool
        future.set_exception(error)
    return future


class _Input:
    """Lo que la entrada de un θ deja abierto y leído.

    `file` es su `events.parquet`; `events`, lo que se sacó de él al leerlo (linaje y
    esqueleto); `stream`, sus lotes de fronteras, ya en lectura anticipada. Esos tres salen
    del primer tramo. El segundo los deja en dos futuros: `gate`, la compuerta de entrada
    (§9.2), cuyo resultado el orquestador vuelca en `carry`, `carry_hash` y
    `previous_direction`, y `writer`, el `FamilyWriter` abierto. Como context manager cierra,
    en este orden, la lectura anticipada, el escritor (que descarta el temporal si no hubo
    `commit`) y el archivo.
    """

    __slots__ = (
        "carry",
        "carry_hash",
        "events",
        "file",
        "gate",
        "previous_direction",
        "stream",
        "writer",
    )

    def __init__(self) -> None:
        self.gate: Future | None = None
        self.writer: Future[FamilyWriter] | None = None

    def __exit__(self, *exc) -> None:
        try:
            self.stream.close()
        finally:
            try:
                _close_writer(self.writer, exc)
            finally:
                self.file.close()


def _close_writer(opening: Future[FamilyWriter] | None, exc: tuple) -> None:
    """Cierra el `FamilyWriter` que se abre en el pool: lo cancela si no empezó, y si no, espera
    a que termine y lo cierra (si no se abrió, `_open_writer` ya limpió).
    """
    if opening is None or opening.cancel():
        return
    if opening.exception() is None:
        opening.result().__exit__(*exc)


class _LateWriter:
    """El escritor de un θ cuya apertura puede no haber terminado: espera a que termine.

    `SealedWriters` solo pide `write_row_group`; el primer row group sellado de un θ llega
    mucho después de que su archivo se abrió, así que la espera casi nunca existe.
    """

    __slots__ = ("_opening",)

    def __init__(self, opening: Future[FamilyWriter]) -> None:
        self._opening = opening

    def write_row_group(self, batch) -> None:
        self._opening.result().write_row_group(batch)


def _open_events(l2_root: str, key, month: Month, text: str) -> EventFile:
    """`events.parquet` del θ con todas sus columnas; si falta, `l2_missing` (§9.2)."""
    path = l2_path(l2_root, key, text, month)
    try:
        return EventFile(
            path, f"L2 del θ {text} en {month_text(month)}", all_columns=True
        )
    except FramesInputError:
        require(path, text, month, EVENTS)  # lanza `L2InputMissing` si el archivo falta
        raise


def _open_writer(l3_root: str, key, month: Month, text: str, skeleton: L2Events):
    """El `FamilyWriter` del θ, ya abierto (su temporal creado)."""
    writer = FamilyWriter(
        family_path(l3_root, key, text, month, FAMILY),
        SUMMARIES_SCHEMA,
        STATE_VERSION,
        skeleton,
    )
    try:
        return writer.__enter__()
    except BaseException as error:
        writer.__exit__(type(error), error, error.__traceback__)
        raise


def _enter_theta(
    pool: ThreadPoolExecutor | None,
    phases: Phases,
    l2_root: str,
    key: tuple[str, str, str],
    month: Month,
    theta: Decimal,
    text: str,
) -> _Input:
    """El primer tramo de la entrada de un θ: su `events.parquet`, abierto y en lectura.

    En orden: abrir `events.parquet` (si falta, `l2_missing`), comprobar los tipos de L2,
    preparar lo que se saca de él al leerlo (`L2Events`) y arrancar la lectura anticipada de
    las fronteras. Un fallo cierra el archivo y se relanza. La compuerta de carry-over y el
    escritor son el segundo tramo (`_read_gate`, `_open_writer`).
    """
    item = _Input()
    item.file = _open_events(l2_root, key, month, text)
    try:
        check_types(item.file)
        item.events = L2Events(item.file, theta)
        item.stream = BatchStream(item.file, pool, phases.read, tap=item.events.tap)
    except BaseException:
        item.file.close()
        raise
    return item


def _read_gate(
    l2_root: str,
    key: tuple[str, str, str],
    month: Month,
    series_start: Month,
    text: str,
):
    """La compuerta de carry-over de un θ (§9.2): el del mes con su hash y la dirección del
    anterior (0 en el primer mes de la serie, que no tiene anterior).
    """
    carry, carry_hash = read_carry_over_hashed(l2_root, key, text, month)
    previous = (
        0
        if month == series_start
        else read_carry_over(
            l2_root, key, text, month_of(month_ordinal(month) - 1)
        ).direction
    )
    return carry, carry_hash, previous


def _await_gates(inputs: Sequence[_Input]) -> None:
    """Espera la compuerta de todos los θ y deja su resultado en cada uno.

    Si alguno falló, relanza el error del primero en el orden de `inputs`, como `ordered`.
    """
    wait([item.gate for item in inputs])
    for item in inputs:
        item.carry, item.carry_hash, item.previous_direction = item.gate.result()
        item.gate = None


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


def _check(collector: Collector, sink: SealedWriters, inputs: Sequence[_Input]) -> None:
    """Relanza el error de quien recoge o escribe lo sellado, o de un escritor que no logró
    abrirse, si lo hubo.
    """
    for error in (collector.error, sink.error):
        if error is not None:
            raise error
    for item in inputs:
        if item.writer.done() and (error := item.writer.exception()) is not None:
            raise error


def _push_all(
    engine: dc_pyo3.Engine,
    row_groups,
    feeds: Sequence[native.BoundaryFeed],
    collector: Collector,
    sink: SealedWriters,
    inputs: Sequence[_Input],
    month: Month,
    phases: Phases,
    l1: L1Month,
) -> int:
    """Empuja al motor los row groups de L1, cada uno con las fronteras que toca.

    Un row group se pide al lector solo cuando hay menos de `inflight` en el motor y pocos
    sellados sin escribir (contrapresión): el lector trae por delante a lo más `readers`,
    así que en RAM hay a lo más `inflight + readers + 1` row groups de L1. `read_s` suma lo
    que este hilo esperó al lector y a las fronteras de L2. Cada row group del mes pasa por
    `l1.observe` antes de empujarse: la conciliación cuenta de ahí los ticks hasta sus extremos
    sin decodificar de nuevo. Devuelve cuántos empujó.
    """
    limit = PENDING_PER_WRITER * phases.writers
    source = iter(row_groups)
    pushed = 0
    while True:
        while engine.inflight() >= phases.inflight or sink.pending >= limit:
            _check(collector, sink, inputs)
            time.sleep(POLL_S)
        _check(collector, sink, inputs)
        started = time.perf_counter()
        rg = next(source, None)
        if rg is not None:
            high = rg.high
            for feed in feeds:
                feed.cover(high)
        phases.read_s += time.perf_counter() - started
        if rg is None:
            return pushed
        if rg.month == month:
            l1.observe(rg.ids)
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

    with ExitStack() as stack:
        pool = stack.enter_context(theta_pool(threads))
        # El pie de L1 se lee mientras los θ pasan su entrada. Si algún θ falla, el error
        # de L2 sale primero, como cuando la compuerta iba antes que L1: su resultado
        # se pide después de esperar la compuerta de carry-over.
        pending = pool.submit(L1Month, l1_root, key, month) if pool else None
        if pending is not None:
            stack.callback(_close_l1, pending)

        # Primer tramo de la entrada: el `events.parquet` de los 50 θ abiertos y en lectura,
        # todos en el pool a la vez; si alguno falla se lanza el primero en el orden de
        # `texts`. El orden de entrada es el inverso del cierre: lo último que se abre
        # (hilos, motor) se cierra primero.
        inputs = _open_all(
            pool,
            stack,
            lambda i: _enter_theta(
                pool, phases, l2_root, key, month, values[i], texts[i]
            ),
            range(len(texts)),
        )
        phases.gate_s += time.perf_counter() - started
        began = time.perf_counter()
        # Segundo tramo, en el pool mientras este hilo construye el motor: la compuerta de
        # carry-over de todos primero, que se espera antes del primer `push`, y después los
        # escritores, que se esperan al escribir el primer row group sellado de cada θ.
        for item, text in zip(inputs, texts, strict=True):
            item.gate = _submit(
                pool, _read_gate, l2_root, key, month, series_start, text
            )
        for item, text in zip(inputs, texts, strict=True):
            item.writer = _submit(
                pool, _open_writer, l3_root, key, month, text, item.events
            )
        # El motor pide los tamaños de row group de los 50 θ, que están en el pie de sus
        # `events.parquet`: no necesita los escritores abiertos.
        engine = native.new_engine(
            values, [item.events.row_groups for item in inputs], phases.engine_threads
        )
        stack.callback(engine.close)
        # La compuerta de carry-over se espera antes que el pie de L1: con L1 ilegible y un
        # carry-over ausente sale el `l2_missing` de la compuerta, no el error de L1.
        _await_gates(inputs)
        if pending is not None:
            l1 = pending.result()
        else:
            l1 = stack.enter_context(closing(L1Month(l1_root, key, month)))

        # Dónde empiezan y terminan los eventos de todos los θ, con las estadísticas del
        # pie de L2 (sin leer datos): el primer row group de L1 que se lee es el que
        # contiene la primera referencia y el último, el que contiene el último extremo.
        files = [item.file for item in inputs]
        references = [_require_statistics(f, "reference_agg_trade_id") for f in files]
        extremes = [_require_statistics(f, "extreme_agg_trade_id") for f in files]
        # Los ids hasta los que la conciliación cuenta ticks, con lo que se sabe antes de
        # leer L1; si el motor encuentra otros, `count_up_to` decodifica como siempre.
        l1.watch(_reconciliation_targets(inputs, references, extremes))
        feeds = [
            native.BoundaryFeed(engine, position, item.stream, phases.retained)
            for position, item in enumerate(inputs)
        ]
        sink = SealedWriters(
            stack.enter_context(
                ThreadPoolExecutor(phases.writers, thread_name_prefix="l3-write")
            ),
            [_LateWriter(item.writer) for item in inputs],
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
                        L1(l1_root, key, {month: l1.footer}).row_groups(
                            month,
                            month,
                            first,
                            lambda low, high: high >= first and low <= last,
                            readers=phases.readers,
                        )
                    )
                )
                pushed = _push_all(
                    engine,
                    row_groups,
                    feeds,
                    collector,
                    sink,
                    inputs,
                    month,
                    phases,
                    l1,
                )
                phases.items = pushed * sum(1 for f in files if any(f.rows))
            # Completa la lectura de L2: el hash de cada archivo cubre todos sus lotes. En
            # este hilo y no en el pool: una tarea que espera a `BatchStream` ocuparía el
            # hilo que la lectura necesita.
            for item in inputs:
                _drain(item.stream)
            stats = engine.finish()
            phases.aggregate_s += time.perf_counter() - begun
            phases.engine_busy = engine.busy()
            collector.stop()
            _check(collector, sink, inputs)
            sink.join()
        except dc_pyo3.DescriptorError as error:
            converted = native.to_summarize_error(error, texts, month)
            if converted is error:
                raise
            raise converted from error

        wait([item.writer for item in inputs])
        family = [item.writer.result() for item in inputs]

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
                previous_direction=inputs[i].previous_direction,
                carry=inputs[i].carry,
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
            theta: ThetaResult(
                family[i].path,
                stats[i][0],
                digests[i],
                reconciled[i],
                inputs[i].events.hexdigest(),
                inputs[i].carry_hash,
            )
            for i, theta in enumerate(values)
        }


def _close_l1(pending: Future[L1Month]) -> None:
    """Cierra el `L1Month` que se abrió en el pool, si llegó a abrirse."""
    if pending.exception() is None:  # espera a que termine
        pending.result().close()


def _drain(stream: BatchStream) -> None:
    """Lee los lotes de L2 que el motor no pidió (los que no traen eventos que cierren)."""
    for _ in stream:
        pass


def _reconciliation_targets(inputs, references, extremes) -> list[int]:
    """Los ids hasta los que `reconcile` va a contar ticks: `τ` y la frontera de cada θ.

    Se calculan con las estadísticas del pie de L2 (primera referencia y mayor extremo) en
    lugar de las del motor, que aún no existen. Son los mismos si los extremos de L2 crecen
    con los eventos, como el detector los escribe.
    """
    targets = []
    for item, reference, extreme in zip(inputs, references, extremes, strict=True):
        tau = first_reference_bound(
            item.previous_direction, item.carry, reference[0] if reference else None
        )
        targets.append(tau)
        targets.append(tau if extreme is None else max(tau, extreme[1]))
    return targets


def _commit(writer: FamilyWriter, text: str, month: Month) -> str:
    """Publica el archivo del θ y devuelve su `content_hash`."""
    try:
        return writer.commit()
    except FamilySkeletonMismatch as error:
        raise SkeletonMismatch(text, month, FAMILY, error.detail) from error
