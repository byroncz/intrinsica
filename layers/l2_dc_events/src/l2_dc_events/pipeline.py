"""Núcleo de una unidad de L2: un mes de un activo para los 50 θ (§8.1 del TRD-L2).

Carga el carry-over del mes anterior (o arranca en frío si es el primero de la
serie), lee la landing y alimenta el fan-out lote a lote, escribe los eventos de
cada θ a medida que cierran y, al final, publica `events.parquet` y
`carry_over.parquet` de cada θ y emite los hallazgos de DQ.
"""

import logging
from contextlib import ExitStack
from dataclasses import dataclass

import dc_pyo3
import pyarrow as pa

from l2_dc_events import findings
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
from l2_dc_events.thetas import load_thetas
from l2_dc_events.write import CARRY_OVER, EVENTS, partition_path

__all__ = ["FEED_TICKS", "Result", "RunContext", "Unit", "process_unit"]

logger = logging.getLogger(__name__)

# Ticks por llamada al fan-out. Los eventos que cada llamada devuelve son
# objetos de Python (~100 B cada uno; con θ = 0,01 % puede ser un evento cada
# pocos ticks), así que un row group entero de 1 M de ticks multiplicaría por 50
# θ un pico de cientos de MiB. Con tramos de 65 536 el pico de eventos es
# O(tramo), y el tiempo de crear los hilos sigue siendo una fracción pequeña.
FEED_TICKS = 65_536


@dataclass(frozen=True)
class Result:
    """Lo que dejó la unidad, por θ: eventos escritos y `content_hash` de cada archivo."""

    n_ticks: int
    n_row_groups: int
    events_per_theta: list[int]
    events_hashes: list[str]
    carry_over_hashes: list[str]


def _feed(
    fanout: dc_pyo3.FanOut, batch: pa.RecordBatch, writers: list[EventWriter]
) -> int:
    """Alimenta el lote al fan-out por tramos y pasa sus eventos a los escritores.

    Es una función aparte para que las vistas del lote (que retienen su
    memoria) mueran al volver, no cuando el bucle de `process_unit` las
    reasigne: si no, el row group anterior seguiría vivo al leer el siguiente.
    """
    ticks = ticks_of(batch)
    for chunk in ticks.chunks(FEED_TICKS):
        closed = fanout.feed_batch(chunk.prices, chunk.times, chunk.agg_trade_ids)
        for writer, new in zip(writers, closed, strict=True):
            writer.add(new)
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


def _start(unit: Unit, ctx: RunContext) -> dc_pyo3.FanOut:
    """El fan-out del mes: en frío si es el primero de la serie, si no con el
    carry-over del mes anterior de cada θ.

    Compuerta fail-closed (ADR-L2-08, §9.2): si al menos un θ no tiene
    carry-over utilizable, emite un hallazgo por cada uno y aborta la unidad
    entera con el primer `CarryOverError`. Los 50 θ comparten la lectura del
    mes, así que uno atrasado detiene a todos hasta que se resuelva.
    """
    thetas = load_thetas()
    if ctx.series_start == (unit.year, unit.month):
        return dc_pyo3.FanOut(thetas)

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
    return dc_pyo3.FanOut.from_carry_over(thetas, carries)


def process_unit(
    unit: Unit,
    ctx: RunContext,
    fanout: dc_pyo3.FanOut | None = None,
) -> Result:
    """Procesa el mes: eventos y carry-over de los 50 θ a partir del consolidado.

    Cada lote se conforma como buffers sin copia, se alimenta a los 50 θ y se
    suelta antes de pedir el siguiente; los eventos van a `events.parquet` a
    medida que cierran. La RAM de la unidad es O(row group). Al final se cierra
    el grupo de empate abierto (RF-L2-12) y se publica todo: primero los
    eventos de cada θ y luego su carry-over, así un carry-over presente
    significa que el mes de ese θ quedó completo. Re-ejecutar el mes da los
    mismos archivos (RF-L2-08): la entrada es la misma y nada de la salida
    depende de la ejecución.

    `fanout` permite que el llamador aporte uno ya construido, y entonces no
    se consulta el carry-over: es quien lo llama el que responde por el estado.
    """
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
            fanout = _start(unit, ctx)
        thetas = fanout.thetas
        writers = [
            stack.enter_context(EventWriter(_path(ctx, unit, theta, EVENTS), theta))
            for theta in thetas
        ]
        n_ticks = 0
        n_row_groups = parquet.num_row_groups
        for batch in read_batches(parquet):
            n_ticks += _feed(fanout, batch, writers)
            # Soltar el lote antes de pedir el siguiente row group.
            del batch
        for writer, last in zip(writers, fanout.finish(), strict=True):
            if last is not None:
                writer.add([last])

        discarded = fanout.discarded()
        carries = fanout.carry_overs()
        events_hashes, carry_hashes = [], []
        for theta, writer, carry in zip(thetas, writers, carries, strict=True):
            events_hashes.append(writer.commit())
            _, carry_hash = write_carry_over(
                carry, where, _path(ctx, unit, theta, CARRY_OVER)
            )
            carry_hashes.append(carry_hash)

    events = [writer.n_events for writer in writers]
    _report(ctx, unit, thetas, events, carries, events_hashes, carry_hashes, discarded)
    logger.info("unidad %s: %d ticks, %d row groups", unit, n_ticks, n_row_groups)
    return Result(n_ticks, n_row_groups, events, events_hashes, carry_hashes)


def _report(
    ctx: RunContext,
    unit: Unit,
    thetas: list[int],
    events: list[int],
    carries: list[dc_pyo3.CarryOver],
    events_hashes: list[str],
    carry_hashes: list[str],
    discarded: list[int],
) -> None:
    """Emite el resumen por θ y, si la guarda de §9.1 descartó algo, su hallazgo."""
    out = []
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
