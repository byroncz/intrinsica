"""El camino por evento: el oráculo del resumidor por row group (TRD-L3 §8.1, ADR-L3-14).

Es el resumidor de ITSC-332 tal como era: el lector en modo por chunks entrega un
`EventFrames` por evento y Arrow compute agrega cada trozo de cada fase. Ya no es el
camino de producción; vive aquí para comprobar que el resumidor por row group da la misma
tabla y el mismo `content_hash` (criterio 7 de §13). Comparte con el resumidor solo el
cierre escalar de un evento (`build_row`: cantidades, invariantes y VWAP), no la agregación.
"""

from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from dc_frames import (
    EventFrames,
    FamilyWriter,
    Frame,
    family_path,
    l2_path,
    read_frames,
)
from dc_frames.lake import Month
from l3_dc_descriptors.event import EVENT_COLUMNS, EventRow, build_row
from l3_dc_descriptors.phase import scaled
from l3_dc_descriptors.schema import (
    COLUMN,
    FAMILY,
    STATE_VERSION,
    SUMMARIES_SCHEMA,
    RowGroupBuffer,
)

C_TICKS, OS_TICKS = COLUMN["c_ticks"], COLUMN["os_ticks"]


def _sum(array: pa.Array) -> int:
    """La suma exacta de un array decimal, sin escalar; 0 si está vacío."""
    total = pc.sum(array, min_count=0)
    return scaled(total.as_py(), total.type.scale)


def _side_sum(quantity: pa.Array, mask: pa.Array) -> int:
    """La suma de `quantity` donde `mask` es verdadero."""
    return _sum(quantity.filter(mask))


def _min_max(prices: pa.Array) -> tuple[int, int]:
    both = pc.min_max(prices)
    return scaled(both["min"].as_py()), scaled(both["max"].as_py())


class Instant:
    """Los ticks de la confirmación con `transact_time = confirm_time` (ADR-L3-01)."""

    def __init__(self, time: int) -> None:
        self.time = time
        self.ticks = 0
        self.high: int | None = None
        self.low: int | None = None

    def add(self, frame: Frame) -> None:
        mask = pc.equal(frame.transact_time, self.time)
        ticks = pc.sum(mask, min_count=0).as_py()
        if not ticks:
            return
        low, high = _min_max(frame.price.filter(mask))
        self.ticks += ticks
        self.low = low if self.low is None else min(self.low, low)
        self.high = high if self.high is None else max(self.high, high)


class PhaseAcc:
    """El acumulador de una fase con Arrow compute, un trozo (row group) a la vez."""

    def __init__(self, instant_time: int | None = None) -> None:
        self.ticks = 0
        self.buy = 0
        self.sell = 0
        self.total = 0
        self.notional = 0  # Σ price·quantity, escala 10⁻¹⁶
        self.high: int | None = None
        self.low: int | None = None
        self.first_time: int | None = None
        self.last_time: int | None = None
        self.instant = Instant(instant_time) if instant_time is not None else None

    def add(self, frame: Frame) -> None:
        count = len(frame)
        if not count:
            return
        quantity, sold = frame.quantity, frame.is_buyer_maker
        self.ticks += count
        self.total += _sum(quantity)
        self.sell += _side_sum(quantity, sold)
        self.buy += _side_sum(quantity, pc.invert(sold))
        self.notional += _sum(pc.multiply(frame.price, quantity))
        low, high = _min_max(frame.price)
        self.low = low if self.low is None else min(self.low, low)
        self.high = high if self.high is None else max(self.high, high)
        if self.first_time is None:
            self.first_time = frame.transact_time[0].as_py()
        self.last_time = frame.transact_time[count - 1].as_py()
        if self.instant is not None:
            self.instant.add(frame)

    def consume(self, frames) -> None:
        for frame in frames:
            self.add(frame)


def summarize_event(ev: EventFrames, theta: str, month: Month) -> EventRow:
    """Agrega las dos fases del evento (consume sus iteradores) y arma su fila."""
    l2 = ev.event.select(EVENT_COLUMNS).to_pylist()[0]
    confirmation = PhaseAcc(l2["confirm_time"])
    confirmation.consume(ev.confirmation)
    overshoot = PhaseAcc()
    overshoot.consume(ev.overshoot)
    return build_row(theta, month, l2, confirmation, overshoot, confirmation.instant)


class Lane:
    """El avance de un θ en el camino por evento: escritor, row group de salida y cuentas."""

    def __init__(self, theta: Decimal, text: str, month: Month, writer: FamilyWriter):
        self.text = text
        self.month = month
        self.writer = writer
        self._theta = int(theta.scaleb(8))
        self._sizes = iter(writer.row_groups)
        self._rows: RowGroupBuffer | None = None
        self.rows = 0
        self.closed_ticks = 0
        self.before_month = 0

    def add(self, ev: EventFrames) -> None:
        event = summarize_event(ev, self.text, self.month)
        if self._rows is None:
            self._rows = RowGroupBuffer(next(self._sizes))
        self._rows.append(event.values)
        self.rows += 1
        self.closed_ticks += event.values[C_TICKS] + event.values[OS_TICKS]
        self.before_month += ev.ticks_before_month
        if self._rows.fill == self._rows.size:
            self.writer.write_row_group(self._rows.to_batch(self._theta))
            self._rows = None


def summarize_month_by_event(
    l1_root: str | Path,
    l2_root: str | Path,
    l3_root: str | Path,
    month: Month,
    thetas: list[str],
    key: tuple[str, str, str] = ("binance", "spot", "BTCUSDT"),
) -> dict[Decimal, dict]:
    """`summaries.parquet` del mes por el camino por evento, sin compuerta ni conciliación.

    Devuelve, por θ, `{"path", "rows", "content_hash", "closed_ticks", "before_month"}`.
    """
    values = [Decimal(t) for t in thetas]
    with ExitStack() as stack:
        lanes: dict[Decimal, Lane] = {}
        for theta, text in zip(values, thetas, strict=True):
            writer = FamilyWriter(
                family_path(str(l3_root), key, text, month, FAMILY),
                SUMMARIES_SCHEMA,
                STATE_VERSION,
                l2_path(str(l2_root), key, text, month),
            )
            lanes[theta] = Lane(theta, text, month, stack.enter_context(writer))
        for ev in read_frames(
            values,
            l1_root,
            l2_root,
            month,
            month,
            *key,
            chunks=True,
            columns=EVENT_COLUMNS,
        ):
            lanes[ev.theta].add(ev)
        return {
            theta: {
                "path": lane.writer.path,
                "rows": lane.rows,
                "content_hash": lane.writer.commit(),
                "closed_ticks": lane.closed_ticks,
                "before_month": lane.before_month,
            }
            for theta, lane in lanes.items()
        }
