"""El resumidor: de L1 y L2 de un mes a `summaries.parquet` de cada θ (TRD-L3 §8.1).

Una pasada por mes y por lista de θ, sin estado entre meses (ADR-L3-12). El lector
entrega cada row group de L1 una vez, con las fronteras de los eventos de todos los θ como
posiciones (ADR-L3-14): `Prefixes` calcula lo común a los θ y, por θ, `ThetaAggregator`
agrega todos los eventos del row group en llamadas vectorizadas y completa el evento
abierto con su acumulador parcial. El row group se suelta antes de pedir el siguiente.
Cada θ arma su row group de salida (el mismo número de filas que el de `events.parquet`)
en un buffer por columna y lo entrega al escritor de familias.

Dos tipos de hilo, sin tocar la aritmética (ADR-L3-15): un lector que trae y decodifica
el siguiente row group mientras se procesa el actual (`_ReadAhead`, `prefetch` row groups
de ventaja) y un pool que reparte los θ de cada row group, cada uno escribiendo su
archivo. El hilo principal espera los θ del row group antes de abrir el siguiente: el
estado de un θ es secuencial entre row groups (el evento abierto en la frontera), así que
cada `_Lane` corre en un solo hilo a la vez y en orden de row group. Con `threads=1` y
`prefetch=0` el flujo es el secuencial de siempre.
"""

import contextvars
import queue
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Self

from dc_frames import (
    FamilySkeletonMismatch,
    FamilyWriter,
    FrameBoundaryError,
    RowGroupFrames,
    ThetaEvents,
    family_path,
    l2_path,
    read_row_groups,
)
from dc_frames.lake import Month, month_of, month_ordinal

from l3_dc_descriptors.cpu import cpu_limit
from l3_dc_descriptors.errors import L2StateInconsistent, SkeletonMismatch, month_text
from l3_dc_descriptors.event import EVENT_COLUMNS
from l3_dc_descriptors.l2 import EVENTS, CarryOver, read_carry_over, require
from l3_dc_descriptors.reconcile import L1Month, Reconciliation, reconcile
from l3_dc_descriptors.rowgroup import Closed, Prefixes, ThetaAggregator
from l3_dc_descriptors.schema import (
    COLUMN,
    FAMILY,
    STATE_VERSION,
    SUMMARIES_SCHEMA,
    RowGroupBuffer,
)
from l3_dc_descriptors.timing import Phases

DEFAULT_KEY = ("binance", "spot", "BTCUSDT")
C_TICKS, OS_TICKS = COLUMN["c_ticks"], COLUMN["os_ticks"]
_THETA_SCALE = Decimal("1e-8")
# Row groups que el lector trae de ventaja: uno en proceso y uno listo.
DEFAULT_PREFETCH = 1


@dataclass(frozen=True, slots=True)
class ThetaResult:
    """Lo que la unidad dejó de un θ: su archivo, sus filas, su hash y su conciliación."""

    path: str
    rows: int
    content_hash: str
    reconciliation: Reconciliation


class _Lane:
    """El avance de un θ en la pasada: su agregador, su escritor, su row group de salida y
    sus cuentas.

    Es lo único que `add` escribe: recibe `Prefixes` y `ThetaEvents` solo para leerlos, así
    que los θ de un row group se pueden agregar en hilos distintos sin coordinarse. Un
    solo hilo a la vez corre `add` de un mismo θ, y en orden de row group.
    `aggregate_s` y `write_s` son los segundos que este θ gastó agregando y armando y
    escribiendo sus row groups: se suman sobre los θ, así que con hilos pasan de la pared.
    """

    def __init__(
        self,
        theta: Decimal,
        text: str,
        month: Month,
        writer: FamilyWriter,
        carry: CarryOver,
        previous_direction: int,
    ) -> None:
        self.text = text
        self.month = month
        self.writer = writer
        self.carry = carry
        self.previous_direction = previous_direction
        self.aggregator = ThetaAggregator(text, month)
        self._theta = int(theta.scaleb(8))
        self._sizes = iter(writer.row_groups)
        self._rows: RowGroupBuffer | None = None
        self.rows = 0
        self.aggregate_s = 0.0
        self.write_s = 0.0
        self.closed_ticks = 0
        self.before_month = 0
        self.first_reference: int | None = None
        self.last_extreme: int | None = None

    def _mismatch(self, detail: str) -> SkeletonMismatch:
        return SkeletonMismatch(self.text, self.month, FAMILY, detail)

    def add(self, agg: Prefixes, events: ThetaEvents) -> None:
        started = time.perf_counter()
        closed = self.aggregator.feed(agg, events)
        self.aggregate_s += time.perf_counter() - started
        if closed is None:
            return
        started = time.perf_counter()
        try:
            self._write(closed)
        finally:
            self.write_s += time.perf_counter() - started
        self.rows += len(closed.rows)
        self.closed_ticks += int(
            closed.rows[:, C_TICKS].sum() + closed.rows[:, OS_TICKS].sum()
        )
        self.before_month += closed.before_month
        if self.first_reference is None:
            self.first_reference = closed.first_reference
        self.last_extreme = closed.last_extreme

    def _write(self, closed: Closed) -> None:
        """Pasa las filas al row group de salida; lo escribe cada vez que se llena."""
        done = 0
        while done < len(closed.rows):
            if self._rows is None:
                size = next(self._sizes, None)
                if size is None:
                    raise self._mismatch(
                        f"el esqueleto tiene {len(self.writer.row_groups)} row groups y "
                        f"se escribe otro ({self.writer.path})"
                    )
                self._rows = RowGroupBuffer(size)
            take = min(self._rows.size - self._rows.fill, len(closed.rows) - done)
            self._rows.extend(closed.rows[done : done + take])
            done += take
            if self._rows.fill == self._rows.size:
                batch = self._rows.to_batch(self._theta)
                try:
                    self.writer.write_row_group(batch)
                except FamilySkeletonMismatch as error:
                    raise self._mismatch(error.detail) from error
                self._rows = None

    def commit(self) -> str:
        """Publica el archivo del θ y devuelve su `content_hash`."""
        started = time.perf_counter()
        try:
            return self.writer.commit()
        except FamilySkeletonMismatch as error:
            raise self._mismatch(error.detail) from error
        finally:
            self.write_s += time.perf_counter() - started


def _consume(
    rg: RowGroupFrames,
    lanes: dict[Decimal, _Lane],
    pool: ThreadPoolExecutor | None,
    phases: Phases,
) -> None:
    """Agrega un row group en todos los θ que lo tocan; al volver no queda nada de él.

    Sin `pool`, los θ van en serie en este hilo. Con `pool`, cada θ es una tarea y se
    espera a todas antes de volver: los 50 pares (row group, θ) terminan antes de abrir
    los del siguiente. Si fallan varios θ, se lanza el primero en el orden de `rg.thetas`,
    el mismo que lanzaría el bucle en serie.
    """
    started = time.perf_counter()
    agg = Prefixes(rg.frame)
    phases.aggregate_s += time.perf_counter() - started
    if pool is None:
        for events in rg.thetas:
            lanes[events.theta].add(agg, events)
        return
    futures = [pool.submit(lanes[e.theta].add, agg, e) for e in rg.thetas]
    wait(futures)
    for future in futures:
        future.result()


class _Failed:
    """Un error del hilo lector, para relanzarlo en el hilo principal."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


_END = object()


class _ReadAhead:
    """Un hilo que consume el lector de row groups y deja a lo más `depth` de ventaja.

    Es un iterador sobre los row groups de `source`, para usarse en un `with`. El hilo
    pide el siguiente row group mientras el principal procesa el actual; `depth` es
    cuántos puede tener decodificados además del que está en proceso (1: uno en proceso y
    uno listo). Los permisos acotan lo decodificado, no solo lo encolado: el hilo no pide
    el siguiente al lector sin un permiso, así que nunca conviven más de `depth + 1`.

    Corre en una copia del contexto de quien lo crea: el contador de row groups de
    `unit_timing` viaja en una `ContextVar` y un hilo nuevo no la hereda. Un error del
    lector (`FrameBoundaryError`, de entrada) se relanza en `__next__`. Al salir del `with`
    el hilo termina y suelta lo que tuviera; hay que salir antes de cerrar el lector, que
    mientras el hilo lo ejecuta no se puede cerrar.
    """

    def __init__(self, source: Iterator[RowGroupFrames], depth: int) -> None:
        self._source = source
        self._permits = threading.Semaphore(depth)
        self._ready: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._finished = False
        self._thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(self._fill,),
            name="l3-read-ahead",
            daemon=True,
        )

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._permits.release()
        self._thread.join()
        while not self._ready.empty():  # soltar los row groups que no se usaron
            self._ready.get_nowait()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> RowGroupFrames:
        if self._finished:
            raise StopIteration
        item = self._ready.get()
        self._permits.release()
        if item is _END:
            self._finished = True
            raise StopIteration
        if isinstance(item, _Failed):
            self._finished = True
            raise item.error
        return item

    def _fill(self) -> None:
        while True:
            self._permits.acquire()
            if self._stop.is_set():
                return
            try:
                item = next(self._source, _END)
            except BaseException as error:  # noqa: BLE001 - se relanza en __next__
                self._ready.put(_Failed(error))
                return
            self._ready.put(item)
            if item is _END:
                return


def _timed_next(row_groups: Iterator[RowGroupFrames], phases: Phases):
    """El siguiente row group (o `None`), sumando a `read_s` lo que el hilo principal
    esperó por él: sin lector anticipado, todo lo que tardó el lector en leerlo,
    decodificarlo y entregarlo con las posiciones de todos los θ; con él, solo lo que
    faltó de esa espera cuando el hilo principal terminó el row group anterior.
    """
    started = time.perf_counter()
    try:
        return next(row_groups, None)
    finally:
        phases.read_s += time.perf_counter() - started


def _workers(requested: int | None, thetas: int) -> int:
    """Hilos del pool: lo pedido o, por defecto, los vCPU efectivos; nunca más que θ."""
    if requested is None:
        requested = max(1, int(cpu_limit().cores))
    return min(requested, thetas)


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
    prefetch: int | None = None,
) -> dict[Decimal, ThetaResult]:
    """Escribe `summaries.parquet` del mes para cada θ y devuelve lo que dejó de cada uno.

    Los eventos son los que cierran en `month` (la partición de L2); sus ticks pueden
    estar en meses anteriores de L1, que el lector relee. `series_start` es el primer mes
    de la serie: el único cuyo mes anterior no tiene carry-over de L2.

    `threads` es el tamaño del pool que agrega los θ de cada row group (por defecto los
    vCPU efectivos, nunca más que θ) y `prefetch`, cuántos row groups trae de ventaja el
    lector (por defecto `DEFAULT_PREFETCH`; 0 lo deja en el hilo principal). Con `threads=1`
    y `prefetch=0` el flujo es el secuencial; el resultado no depende de ellos.

    Atómico por archivo: la unidad comprueba la entrada de L2, resume
    todos los θ y concilia los ticks de cada uno antes de publicar el primero; un error
    de entrada, de resumen o de conciliación (todos son `SummarizeError` o `FramesError`)
    no deja ningún archivo y borra los temporales. Si la publicación misma falla en un θ
    (por ejemplo, el `move` a GCS), los θ ya publicados quedan: re-ejecutar el mes los
    reescribe con el mismo contenido.

    Sin estado: todo lo que la unidad acumula (los `_Lane`, los tiempos) es suyo y muere
    con ella. `phases`, si se pasa, recibe los segundos de la compuerta, la lectura, la
    agregación, la escritura y la conciliación (ver `timing`).
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
    if threads is not None and threads < 1:
        raise ValueError(f"threads {threads} debe ser al menos 1")
    if prefetch is not None and prefetch < 0:
        raise ValueError(f"prefetch {prefetch} no puede ser negativo")
    threads = _workers(threads, len(values))
    prefetch = DEFAULT_PREFETCH if prefetch is None else prefetch
    phases.threads, phases.prefetch = threads, prefetch
    texts = {theta: f"{theta:.8f}" for theta in values}

    # Compuerta de entrada (§9.2), antes de abrir ningún escritor.
    carries: dict[Decimal, tuple[CarryOver, int]] = {}
    for theta, text in texts.items():
        require(l2_path(l2_root, key, text, month), text, month, EVENTS)
        carry = read_carry_over(l2_root, key, text, month)
        if month == series_start:
            previous = 0
        else:
            before = month_of(month_ordinal(month) - 1)
            previous = read_carry_over(l2_root, key, text, before).direction
        carries[theta] = (carry, previous)
    phases.gate_s += time.perf_counter() - started

    with ExitStack() as stack:
        lanes: dict[Decimal, _Lane] = {}
        for theta, text in texts.items():
            writer = FamilyWriter(
                family_path(l3_root, key, text, month, FAMILY),
                SUMMARIES_SCHEMA,
                STATE_VERSION,
                l2_path(l2_root, key, text, month),
            )
            stack.enter_context(writer)
            lanes[theta] = _Lane(theta, text, month, writer, *carries[theta])
        l1 = stack.enter_context(closing(L1Month(l1_root, key, month)))
        row_groups = stack.enter_context(
            closing(
                read_row_groups(
                    values,
                    l1_root,
                    l2_root,
                    month,
                    month,
                    *key,
                    columns=EVENT_COLUMNS,
                )
            )
        )
        try:
            # Los hilos viven solo durante el recorrido: al salir (también por un error)
            # el pool termina sus tareas y el lector suelta lo que traía.
            with ExitStack() as workers:
                source = (
                    workers.enter_context(_ReadAhead(row_groups, prefetch))
                    if prefetch
                    else row_groups
                )
                pool = (
                    workers.enter_context(
                        ThreadPoolExecutor(threads, thread_name_prefix="l3-theta")
                    )
                    if threads > 1
                    else None
                )
                while (rg := _timed_next(source, phases)) is not None:
                    _consume(rg, lanes, pool, phases)
                    del rg  # el siguiente row group no convive con este
        except FrameBoundaryError as error:
            raise _inconsistent(error, month) from error
        phases.aggregate_s += sum(lane.aggregate_s for lane in lanes.values())

        begun = time.perf_counter()
        reconciled = {
            theta: reconcile(
                l1,
                theta=lane.text,
                month=month,
                closed=lane.closed_ticks,
                before_month=lane.before_month,
                last_extreme=lane.last_extreme,
                first_reference=lane.first_reference,
                previous_direction=lane.previous_direction,
                carry=lane.carry,
            )
            for theta, lane in lanes.items()
        }
        phases.reconcile_s += time.perf_counter() - begun
        results = {}
        for theta, lane in lanes.items():
            digest = lane.commit()
            results[theta] = ThetaResult(
                lane.writer.path, lane.rows, digest, reconciled[theta]
            )
        phases.write_s += sum(lane.write_s for lane in lanes.values())
    return results
