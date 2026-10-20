"""La entrega por row group: cada row group de L1 una vez, con las fronteras de los eventos
de todos los θ como posiciones (TRD-L3 §7.4, ADR-L3-14).

No construye objetos por evento ni retiene ticks: el consumidor completa los eventos que
cruzan row groups con su propio acumulador parcial, así que nada se relee.
"""

from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa

from dc_frames.lake import L1, EventGroup, Month, RowGroup, event_groups
from dc_frames.track import Track, locate, unclosed_error
from dc_frames.types import Frame, RowGroupFrames, ThetaEvents

# Las siete columnas de `events.parquet` que usa L3 (TRD-L3 §7.1): las cuatro que el
# lector siempre lee más `direction` y los precios de confirmación y extremo.
DELIVERY_COLUMNS = (
    "reference_agg_trade_id",
    "confirm_agg_trade_id",
    "confirm_time",
    "extreme_agg_trade_id",
    "direction",
    "confirm_price",
    "extreme_price",
)


class RowGroupCursor(Track):
    """El avance de un θ por row group: qué eventos toca cada uno y qué lleva el abierto.

    Como mucho el último evento que toca un row group sigue abierto al salir de él
    (`E_i ≤ R_{i+1}`); lo único que se lleva al siguiente es si sus fronteras se vieron
    y cuántos ticks anteriores al mes ha contado. Ningún tick.
    """

    def feed(self, rg: RowGroup) -> ThetaEvents | None:
        """Los eventos que tocan `rg`, o `None` si ninguno lo toca."""
        pieces: list[ThetaEvents] = []
        while self._group is not None:
            group = self._group
            touched = int(np.searchsorted(group.bounds[:, 0], rg.high, "right"))
            if touched > self._k:
                pieces.append(self._process(rg, group, touched))
            if self._k < len(group):
                break
            self._next_group()
        return _join(self.theta, pieces)

    def _process(self, rg: RowGroup, group: EventGroup, touched: int) -> ThetaEvents:
        k = self._k
        bounds = group.bounds[k:touched]
        right, seen = locate(
            self.theta, rg, bounds, group.confirm_times[k:touched], self._seen
        )
        closed = int(np.count_nonzero(bounds[:, 2] <= rg.high))
        # Los ticks de (R, E] de este row group son `e - r`; cuentan como anteriores al
        # mes si el row group es de un mes previo al de la partición del evento.
        before = np.zeros(len(bounds), np.int64)
        if rg.month < group.month:
            before = (right[:, 2] - right[:, 0]).astype(np.int64)
        before[0] += self._before
        self._k = k + closed
        if closed < len(bounds):
            # El evento abierto lleva su cuenta al siguiente row group y sale con 0.
            self._before = int(before[closed])
            before[closed] = 0
            self._seen = seen[closed]
        else:
            self._before = 0
            self._seen = np.zeros(3, bool)
        return ThetaEvents(
            self.theta,
            group.batch.slice(k, touched - k),
            right.astype(np.int64),
            bounds[:, 0] < rg.low,
            bounds[:, 2] > rg.high,
            before,
        )


def _join(theta: Decimal, pieces: list[ThetaEvents]) -> ThetaEvents | None:
    """Une los tramos de un θ que vienen de varios row groups de eventos de L2.

    Con uno solo no copia nada: las filas son un slice del row group de eventos.
    """
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return ThetaEvents(
        theta,
        pa.concat_batches([p.events for p in pieces]),
        np.concatenate([p.positions for p in pieces]),
        np.concatenate([p.starts_before for p in pieces]),
        np.concatenate([p.ends_after for p in pieces]),
        np.concatenate([p.ticks_before_month for p in pieces]),
    )


def deliver(
    thetas: list[Decimal],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    key: tuple[str, str, str],
    columns: Sequence[str] | None,
    readers: int = 1,
) -> Iterator[RowGroupFrames]:
    l1 = L1(l1_root, key)
    wanted = DELIVERY_COLUMNS if columns is None else columns
    cursors = [
        RowGroupCursor(t, event_groups(l2_root, key, f"{t:.8f}", start, end, wanted))
        for t in thetas
    ]
    live = [c for c in cursors if not c.done]
    if not live:
        return
    reference = min(c.reference for c in live)

    def needed(_low: int, high: int) -> bool:
        return any(not c.done and c.reference <= high for c in live)

    rows = l1.row_groups(start, end, reference, needed, readers)
    try:
        # Se suelta todo lo del row group antes de pedir el siguiente: aquí no queda
        # ninguna referencia a sus buffers mientras el consumidor trabaja con el siguiente.
        rg = element = None
        while True:
            rg = element = None
            rg = next(rows, None)
            if rg is None:
                break
            touching = (c.feed(rg) for c in live)
            element = RowGroupFrames(
                rg.month,
                rg.month < start,
                rg.ids,
                Frame(*rg.columns),
                tuple(t for t in touching if t is not None),
            )
            live = [c for c in live if not c.done]
            yield element
            if not live:
                return
    finally:
        rows.close()
    raise unclosed_error(live)
