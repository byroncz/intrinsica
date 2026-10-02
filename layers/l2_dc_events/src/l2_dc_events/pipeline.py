"""Núcleo de una unidad de L2: un mes de un activo para los θ que le tocan (§8.1 del TRD-L2).

Carga el carry-over del mes anterior (o arranca en frío si es el primero de la
serie), lee la landing y alimenta el fan-out lote a lote, escribe los eventos de
cada θ a medida que cierran y, al final, publica `events.parquet` y
`carry_over.parquet` de cada θ y emite los hallazgos de DQ.
"""

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass

import dc_pyo3
import pyarrow as pa

from l2_dc_events import cpu, findings
from l2_dc_events.carry import (
    CarryOverError,
    Coordinates,
    read_carry_over,
    write_carry_over,
)
from l2_dc_events.context import RunContext, Unit
from l2_dc_events.events import EventWriter
from l2_dc_events.landing import (
    LandingError,
    consolidated_path,
    open_consolidated,
    read_batches,
    ticks_of,
)
from l2_dc_events.parallel import ParallelWriters
from l2_dc_events.thetas import load_thetas
from l2_dc_events.timing import Phases, Timing
from l2_dc_events.write import CARRY_OVER, EVENTS, partition_path

__all__ = [
    "FEED_TICKS",
    "Result",
    "RunContext",
    "Unit",
    "process_unit",
]

logger = logging.getLogger(__name__)

# Ticks por llamada al fan-out. Los eventos que cada llamada devuelve son
# buffers (~113 B por evento; con θ = 0,01 % puede ser un evento cada pocos
# ticks), así que un row group entero de 1 M de ticks multiplicaría por 50 θ un
# pico de cientos de MiB. Con tramos de 65 536 el pico de eventos es O(tramo), y
# el tiempo de crear los hilos sigue siendo una fracción pequeña.
FEED_TICKS = 65_536


@dataclass(frozen=True)
class Result:
    """Lo que dejó la unidad, por θ: eventos escritos y `content_hash` de cada archivo."""

    n_ticks: int
    n_row_groups: int
    events_per_theta: list[int]
    events_hashes: list[str]
    carry_over_hashes: list[str]
    timing: Timing


def _feed(
    fanout: dc_pyo3.FanOut,
    batch: pa.RecordBatch,
    writes: ParallelWriters,
    phases: Phases,
) -> int:
    """Alimenta el lote al fan-out por tramos y pasa sus eventos a los escritores.

    Es una función aparte para que las vistas del lote (que retienen su
    memoria) mueran al volver, no cuando el bucle de `process_unit` las
    reasigne: si no, el row group anterior seguiría vivo al leer el siguiente.
    """
    ticks = ticks_of(batch)
    for chunk in ticks.chunks(FEED_TICKS):
        started, cpu_started = time.perf_counter(), time.thread_time()
        closed = fanout.feed_batch_columns(
            chunk.prices, chunk.times, chunk.agg_trade_ids
        )
        detected, cpu_detected = time.perf_counter(), time.thread_time()
        # `submit` se bloquea si los escritores van atrasados: esa es la espera.
        writes.submit(closed)
        phases.detect_s += detected - started
        phases.detect_cpu_s += cpu_detected - cpu_started
        phases.wait_s += time.perf_counter() - detected
    return len(ticks)


def _path(ctx: RunContext, unit: Unit, theta: int, filename: str) -> str:
    return partition_path(
        ctx.events_root,
        unit.provider,
        unit.market,
        unit.asset,
        theta,
        unit.year,
        unit.month,
        filename,
    )


def _start(
    unit: Unit, ctx: RunContext, threads: int, thetas: list[int]
) -> dc_pyo3.FanOut:
    """El fan-out del mes: en frío si es el primero de la serie, si no con el
    carry-over del mes anterior de cada θ.

    Compuerta fail-closed (ADR-L2-08, §9.2): si al menos un θ no tiene
    carry-over utilizable, emite un hallazgo por cada uno y aborta la unidad
    entera con el primer `CarryOverError`. Los θ del mes comparten la lectura,
    así que uno atrasado detiene a todos hasta que se resuelva.
    """
    if ctx.series_start == (unit.year, unit.month):
        return dc_pyo3.FanOut(thetas, threads)

    previous = unit.previous()
    carries, errors = [], []
    for theta in thetas:
        try:
            carries.append(
                read_carry_over(_path(ctx, previous, theta, CARRY_OVER), theta)
            )
        except CarryOverError as exc:
            errors.append(exc)
    if errors:
        findings.emit(
            ctx, [findings.failure(ctx, unit, e.check_type, e.details) for e in errors]
        )
        logger.error(
            "unidad %s: %d de %d θ sin carry-over de %s utilizable",
            unit,
            len(errors),
            len(thetas),
            previous,
        )
        raise errors[0]
    return dc_pyo3.FanOut.from_carry_over(thetas, carries, threads)


def process_unit(
    unit: Unit,
    ctx: RunContext,
    fanout: dc_pyo3.FanOut | None = None,
    thetas: Sequence[int] | None = None,
) -> Result:
    """Procesa el mes: eventos y carry-over de los θ pedidos a partir del consolidado.

    `thetas` son los θ que el mes lleva en el fan-out (por defecto, los de la
    semilla de `config/thetas.yaml`); el llamador decide cuáles, según la
    frontera de cada uno (`frontier.py`).

    Cada lote se conforma como buffers sin copia, se alimenta a los θ del mes y se
    suelta antes de pedir el siguiente; los eventos salen del binding ya en
    columnas y los escritores de cada θ los codifican a `events.parquet` en paralelo
    (`parallel.py`), a medida que cierran. La RAM de la unidad es O(row group). Al final se cierra
    el grupo de empate abierto (RF-L2-12) y se publica todo: primero los
    eventos de cada θ y luego su carry-over, así un carry-over presente
    significa que el mes de ese θ quedó completo. Re-ejecutar el mes da los
    mismos archivos (RF-L2-08): la entrada es la misma y nada de la salida
    depende de la ejecución.

    `fanout` permite que el llamador aporte uno ya construido, y entonces no
    se consulta el carry-over: es quien lo llama el que responde por el estado.
    """
    started = time.perf_counter()
    throttled_before = cpu.throttled_s()
    # Un solo lugar decide el paralelismo (`cpu.py`): hilos del fan-out y
    # escritores salen de la cuota del cgroup, no de los cores visibles.
    limit = cpu.cpu_limit()
    phases = Phases()
    path = consolidated_path(
        str(ctx.landing_root),
        unit.provider,
        unit.market,
        unit.asset,
        unit.year,
        unit.month,
    )
    try:
        parquet = open_consolidated(path)
        # Existencia y footer del archivo: también es espera de I/O.
        phases.read_s += time.perf_counter() - started
    except LandingError as exc:
        if exc.check_type:
            findings.emit(
                ctx, [findings.failure(ctx, unit, exc.check_type, exc.details)]
            )
        raise

    where = Coordinates(unit.provider, unit.market, unit.asset, unit.year, unit.month)
    # `with`: el archivo se cierra al terminar la unidad, también si falla.
    with parquet, ExitStack() as stack:
        if fanout is None:
            carry_started = time.perf_counter()
            fanout = _start(
                unit,
                ctx,
                limit.fanout_threads,
                load_thetas() if thetas is None else list(thetas),
            )
            phases.carry_s = time.perf_counter() - carry_started
        thetas = fanout.thetas
        writers = [
            stack.enter_context(EventWriter(_path(ctx, unit, theta, EVENTS), theta))
            for theta in thetas
        ]
        # Después de los escritores: al salir, primero se cancela y se espera
        # al pool y solo entonces se cierran (o se borran) los temporales.
        pool = ThreadPoolExecutor(max_workers=limit.writers)
        stack.callback(pool.shutdown, wait=True, cancel_futures=True)
        writes = ParallelWriters(pool, writers)
        n_ticks = 0
        n_row_groups = parquet.num_row_groups
        # `closing`: el hilo lector termina antes de que `with parquet` cierre el archivo.
        with closing(read_batches(parquet, phases, ctx.read_ahead)) as batches:
            for batch in batches:
                n_ticks += _feed(fanout, batch, writes, phases)
                # Soltar el lote antes de pedir el siguiente row group.
                del batch
        started_tail, cpu_tail = time.perf_counter(), time.thread_time()
        tail = fanout.finish_columns()
        closed, cpu_closed = time.perf_counter(), time.thread_time()
        writes.submit(tail)
        writes.wait()
        phases.detect_s += closed - started_tail
        phases.detect_cpu_s += cpu_closed - cpu_tail
        phases.wait_s += time.perf_counter() - closed

        discarded = fanout.discarded()
        carries = fanout.carry_overs()

        def publish(theta: int, writer: EventWriter, carry: dc_pyo3.CarryOver):
            # Por θ, primero los eventos y luego su carry-over.
            events_hash = writer.commit()
            carry_started = time.perf_counter()
            _, carry_hash = write_carry_over(
                carry, where, _path(ctx, unit, theta, CARRY_OVER)
            )
            return events_hash, carry_hash, time.perf_counter() - carry_started

        published = time.perf_counter()
        hashes = list(pool.map(publish, thetas, writers, carries))
        phases.wait_s += time.perf_counter() - published
        events_hashes = [events_hash for events_hash, _, _ in hashes]
        carry_hashes = [carry_hash for _, carry_hash, _ in hashes]
        phases.write_s = sum(writer.write_s for writer in writers) + sum(
            carry_write_s for _, _, carry_write_s in hashes
        )

    events = [writer.n_events for writer in writers]
    throttled_after = cpu.throttled_s()
    timing = Timing(
        wall_s=time.perf_counter() - started,
        read_s=phases.read_s,
        decode_s=phases.decode_s,
        detect_s=phases.detect_s,
        write_s=phases.write_s,
        carry_s=phases.carry_s,
        wait_s=phases.wait_s,
        row_groups=phases.row_groups,
        bytes_in=phases.bytes_in,
        cores=limit.cores,
        cores_visible=limit.visible,
        cores_source=limit.source,
        fanout_threads=limit.fanout_threads,
        write_workers=limit.writers,
        detect_cpu_s=phases.detect_cpu_s,
        cpu_throttled_s=(
            None
            if throttled_before is None or throttled_after is None
            else throttled_after - throttled_before
        ),
    )
    _report(
        ctx,
        unit,
        thetas,
        events,
        carries,
        events_hashes,
        carry_hashes,
        discarded,
        timing,
    )
    logger.info("unidad %s: %d ticks, %d row groups", unit, n_ticks, n_row_groups)
    return Result(n_ticks, n_row_groups, events, events_hashes, carry_hashes, timing)


def _report(
    ctx: RunContext,
    unit: Unit,
    thetas: list[int],
    events: list[int],
    carries: list[dc_pyo3.CarryOver],
    events_hashes: list[str],
    carry_hashes: list[str],
    discarded: list[int],
    timing: Timing,
) -> None:
    """Emite el resumen por θ, el tiempo por fase de la unidad y, si la guarda
    de §9.1 descartó algo, su hallazgo."""
    out = [findings.unit_timing(ctx, unit, timing)]
    rows = zip(thetas, events, carries, events_hashes, carry_hashes, discarded)
    for theta, n, carry, events_hash, carry_hash, dropped in rows:
        out.append(
            findings.events_summary(
                ctx,
                unit,
                theta=theta,
                events=n,
                has_pending_event=carry.pending is not None,
                events_content_hash=events_hash,
                carry_over_content_hash=carry_hash,
            )
        )
        if dropped:
            out.append(
                findings.zero_tick_discarded(ctx, unit, theta=theta, discarded=dropped)
            )
        logger.info(
            "θ=%d eventos=%d events_content_hash=%s carry_over_content_hash=%s",
            theta,
            n,
            events_hash,
            carry_hash,
        )
    findings.emit(ctx, out)
