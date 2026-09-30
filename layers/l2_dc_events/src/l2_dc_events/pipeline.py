"""Núcleo de una unidad de L2: un mes de un activo para los 50 θ (§8.1 del TRD-L2).

Esta versión lee la landing y alimenta el fan-out lote a lote; guardar los
eventos y el carry-over es de la hija siguiente de la Épica.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import dc_pyo3
import pyarrow as pa

from l2_dc_events.landing import (
    consolidated_path,
    open_consolidated,
    read_batches,
    ticks_of,
)
from l2_dc_events.thetas import load_thetas

logger = logging.getLogger(__name__)

# Ticks por llamada al fan-out. Los eventos que cada llamada devuelve son
# objetos de Python (~100 B cada uno; con θ = 0,01 % puede ser un evento cada
# pocos ticks), así que un row group entero de 1 M de ticks multiplicaría por 50
# θ un pico de cientos de MiB. Con tramos de 65 536 el pico de eventos es
# O(tramo), y el tiempo de crear los hilos sigue siendo una fracción pequeña.
FEED_TICKS = 65_536


@dataclass(frozen=True)
class Unit:
    """Un mes de un activo."""

    year: int
    month: int
    provider: str = "binance"
    market: str = "spot"
    asset: str = "BTCUSDT"

    def __str__(self) -> str:
        return (
            f"{self.provider}/{self.market}/{self.asset}/"
            f"{self.year:04d}-{self.month:02d}"
        )


@dataclass(frozen=True)
class RunContext:
    mode: str
    run_id: str
    image_version: str
    landing_root: str | Path
    events_root: str | Path
    dq_root: str | Path


@dataclass(frozen=True)
class Result:
    """Lo que dejó la unidad: cuántos ticks vio y cuántos eventos cerró cada θ."""

    n_ticks: int
    n_row_groups: int
    events_per_theta: list[int]


def _feed(fanout: dc_pyo3.FanOut, batch: pa.RecordBatch, events: list[int]) -> int:
    """Alimenta el lote al fan-out por tramos y suma sus eventos a `events`.

    Es una función aparte para que las vistas del lote (que retienen su
    memoria) mueran al volver, no cuando el bucle de `process_unit` las
    reasigne: si no, el row group anterior seguiría vivo al leer el siguiente.
    """
    ticks = ticks_of(batch)
    for chunk in ticks.chunks(FEED_TICKS):
        closed = fanout.feed_batch(chunk.prices, chunk.times, chunk.agg_trade_ids)
        for i, new in enumerate(closed):
            events[i] += len(new)
    return len(ticks)


def process_unit(
    unit: Unit,
    ctx: RunContext,
    fanout: dc_pyo3.FanOut | None = None,
) -> Result:
    """Alimenta al fan-out con los ticks del `consolidated.parquet` del mes.

    Cada lote se conforma como buffers sin copia, se alimenta a los 50 θ y se
    suelta antes de pedir el siguiente: la RAM de la unidad es O(row group).
    Al final se cierra el grupo de empate abierto (RF-L2-12).

    `fanout` permite que el llamador aporte uno ya construido (con el
    carry-over del mes anterior); sin él arranca en frío, con los θ de
    `config/thetas.yaml`.
    """
    path = consolidated_path(
        str(ctx.landing_root),
        unit.provider,
        unit.market,
        unit.asset,
        unit.year,
        unit.month,
    )
    if fanout is None:
        fanout = dc_pyo3.FanOut(load_thetas())

    events = [0] * len(fanout)
    n_ticks = 0
    # `with`: el archivo se cierra al terminar la unidad, también si falla.
    with open_consolidated(path) as parquet:
        n_row_groups = parquet.num_row_groups
        for batch in read_batches(parquet):
            n_ticks += _feed(fanout, batch, events)
            # Soltar el lote antes de pedir el siguiente row group.
            del batch
    events = [
        total + (last is not None) for total, last in zip(events, fanout.finish())
    ]

    logger.info("unidad %s: %d ticks, %d row groups", unit, n_ticks, n_row_groups)
    return Result(n_ticks, n_row_groups, events)
