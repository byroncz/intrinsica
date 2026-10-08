"""El lector de tramas: recorta L1 con las fronteras de L2 por `agg_trade_id` (TRD-L3 §7.4).

La pertenencia de un tick a una fase se decide por **valor** de `agg_trade_id`, nunca
por posición ni por tiempo: el proveedor deja huecos de ids, y restar posiciones
correría la frontera. Cada frontera se busca con `searchsorted` sobre los ids del row
group, para todos los eventos del row group a la vez; el único bucle de Python es uno
por evento, que arma sus slices.
"""

from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa

from dc_frames.lake import (
    L1,
    EventGroup,
    Month,
    RowGroup,
    event_group,
    event_groups,
)
from dc_frames.types import EventFrames, Frame, FrameBoundaryError, FramesInputError

DEFAULT_KEY = ("binance", "spot", "BTCUSDT")
_THETA_SCALE = Decimal("1e-8")
_NAMES = ("referencia", "confirmación", "extremo")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class _Cursor:
    """El avance de un θ sobre sus eventos: cuál es el que sigue y qué ticks lleva ya.

    Solo el primer evento sin entregar puede estar a medias (`E_i ≤ R_{i+1}`): lo que ya
    se leyó de él se guarda **copiado** en `_conf` y `_over` (una lista de arrays por
    columna), para no anclar el row group del que salió.
    """

    def __init__(self, theta: Decimal, groups: Iterator[EventGroup]) -> None:
        self.theta = theta
        self._groups = groups
        self._group = next(groups, None)
        self._k = 0
        # Qué fronteras del evento `k` ya se vieron como ticks de L1: (R, C, E).
        self._seen = np.zeros(3, bool)
        self._conf: list[list[pa.Array]] = [[], [], [], []]
        self._over: list[list[pa.Array]] = [[], [], [], []]

    @property
    def done(self) -> bool:
        return self._group is None

    @property
    def reference(self) -> int:
        """Referencia del evento en curso. Solo si no está `done`."""
        return int(self._group.bounds[self._k, 0])

    def describe(self) -> str:
        r, c, e = (int(x) for x in self._group.bounds[self._k])
        return (
            f"θ {self.theta}, evento con referencia {r}, confirmación {c} y extremo {e}"
        )

    def feed(self, rg: RowGroup) -> Iterator[EventFrames]:
        """Entrega los eventos que cierran en este row group y deja a medias el que no."""
        while self._group is not None:
            group = self._group
            touched = int(np.searchsorted(group.bounds[:, 0], rg.high, "right"))
            if touched > self._k:
                yield from self._process(rg, group, touched)
            if self._k < len(group):
                return
            self._next_group()

    def _next_group(self) -> None:
        last_extreme = int(self._group.bounds[-1, 2])
        self._group = next(self._groups, None)
        self._k = 0
        self._seen = np.zeros(3, bool)
        if self._group is not None and self._group.bounds[0, 0] < last_extreme:
            raise FramesInputError(
                f"θ {self.theta}: un row group de eventos empieza en la referencia "
                f"{self._group.bounds[0, 0]}, antes del extremo {last_extreme} del anterior"
            )

    def _process(
        self, rg: RowGroup, group: EventGroup, touched: int
    ) -> Iterator[EventFrames]:
        k = self._k
        bounds = group.bounds[k:touched]
        left = np.searchsorted(rg.ids, bounds, "left")
        right = np.searchsorted(rg.ids, bounds, "right")
        inside = (bounds >= rg.low) & (bounds <= rg.high)
        seen = np.zeros(bounds.shape, bool)
        seen[0] = self._seen
        # Una frontera dentro del rango del row group debe ser un tick; una anterior a
        # él debe haberse visto en un row group previo, o cae en un hueco entre ambos.
        bad = (inside & (left == right)) | ((bounds < rg.low) & ~seen)
        if bad.any():
            raise self._boundary_error(bounds, bad, rg)
        self._check_confirm_times(rg, group, bounds, left, inside[:, 1])
        seen |= inside

        closed = int(np.count_nonzero(bounds[:, 2] <= rg.high))
        # `right` es la posición siguiente a la frontera: las fases son
        # confirmación [R+, C+) y overshoot [C+, E+), y una frontera fuera del row
        # group queda en 0 o en su largo.
        edges = right.tolist()
        for i in range(closed):
            r, c, e = edges[i]
            if i == 0:
                confirmation = self._take(self._conf, rg, r, c)
                overshoot = self._take(self._over, rg, c, e)
            else:
                confirmation, overshoot = rg.frame(r, c), rg.frame(c, e)
            yield EventFrames(
                self.theta, group.batch.slice(k + i, 1), confirmation, overshoot
            )
        self._k = k + closed
        if closed < len(bounds):
            r, c, e = edges[closed]
            self._keep(self._conf, rg, r, c)
            self._keep(self._over, rg, c, e)
            self._seen = seen[closed]
        else:
            self._seen = np.zeros(3, bool)

    @staticmethod
    def _keep(parts: list[list[pa.Array]], rg: RowGroup, start: int, stop: int) -> None:
        """Guarda una copia de los ticks `[start, stop)`: el row group se suelta."""
        if stop > start:
            for pieces, column in zip(parts, rg.columns, strict=True):
                pieces.append(pa.concat_arrays([column.slice(start, stop - start)]))

    @staticmethod
    def _take(
        parts: list[list[pa.Array]], rg: RowGroup, start: int, stop: int
    ) -> Frame:
        """La trama de `[start, stop)` más lo que se guardó antes, que se vacía.

        Sin nada guardado son slices sin copia. Con piezas guardadas se concatena una
        columna a la vez: el pico es el evento más una columna, no dos eventos.
        """
        if not any(parts):
            return rg.frame(start, stop)
        columns = []
        for pieces, column in zip(parts, rg.columns, strict=True):
            pieces.append(column.slice(start, stop - start))
            columns.append(pa.concat_arrays(pieces))
            pieces.clear()
        return Frame(*columns)

    def _check_confirm_times(
        self,
        rg: RowGroup,
        group: EventGroup,
        bounds: np.ndarray,
        left: np.ndarray,
        inside: np.ndarray,
    ) -> None:
        """El tick `C` de cada evento cuya confirmación cae en este row group debe tener
        `transact_time = confirm_time` (TRD-L3 §7.4, garantía 1): si no, L1 y L2 no cuadran.
        """
        rows = np.flatnonzero(inside)
        if not rows.size:
            return
        found = rg.columns[0].to_numpy(zero_copy_only=False)[left[rows, 1]]
        wanted = group.confirm_times[self._k + rows]
        wrong = np.flatnonzero(found != wanted)
        if wrong.size:
            i = int(rows[wrong[0]])
            raise FrameBoundaryError(
                f"θ {self.theta}: el tick de confirmación {bounds[i, 1]} (evento con "
                f"referencia {bounds[i, 0]}) tiene transact_time {found[wrong[0]]} y L2 "
                f"dice confirm_time {wanted[wrong[0]]}: L1 y L2 no cuadran"
            )

    def _boundary_error(
        self, bounds: np.ndarray, bad: np.ndarray, rg: RowGroup
    ) -> FrameBoundaryError:
        i, j = (int(x) for x in np.argwhere(bad)[0])
        event = bounds[i]
        return FrameBoundaryError(
            f"θ {self.theta}: el id de {_NAMES[j]} {event[j]} del evento "
            f"(referencia {event[0]}, confirmación {event[1]}, extremo {event[2]}) "
            f"no es un tick de L1: cae en un hueco del proveedor o L1 y L2 no cuadran "
            f"(row group de L1 con ids {rg.low}..{rg.high})"
        )


def _drive(
    l1: L1, cursors: list[_Cursor], first: Month, last: Month
) -> Iterator[EventFrames]:
    live = [c for c in cursors if not c.done]
    if not live:
        return
    reference = min(c.reference for c in live)

    def needed(high: int) -> bool:
        return any(not c.done and c.reference <= high for c in live)

    rows = l1.row_groups(first, last, reference, needed)
    try:
        # Se suelta el row group antes de pedir el siguiente: nunca hay dos en vuelo.
        rg = None
        while True:
            rg = None
            rg = next(rows, None)
            if rg is None:
                break
            for cursor in live:
                yield from cursor.feed(rg)
            live = [c for c in live if not c.done]
            if not live:
                return
    finally:
        rows.close()
    raise FrameBoundaryError(
        "L1 terminó antes de cerrar todos los eventos; el extremo no existe como tick. "
        + "; ".join(c.describe() for c in live)
    )


def _theta(value: Decimal | str) -> Decimal:
    if not isinstance(value, Decimal | str):
        raise TypeError(f"θ debe ser Decimal o str, no {type(value).__name__}")
    return Decimal(value).quantize(_THETA_SCALE)


def read_frames(
    thetas: Decimal | str | Iterable[Decimal | str],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
) -> Iterator[EventFrames]:
    """Los eventos cerrados de los θ en los meses `start`..`end`, con sus tramas.

    `start` y `end` son `(año, mes)` de las particiones de L2, ambos inclusivos; un
    evento pertenece al mes que lo cierra y sus ticks pueden estar en meses anteriores
    a `start`, que el lector también lee. Dentro de cada θ salen por
    `confirm_agg_trade_id` ascendente; entre θ no hay orden garantizado. Con varios θ
    cada row group de L1 se decodifica una sola vez.

    Memoria: un row group de L1, un row group de eventos por θ y el evento en curso de
    cada θ. Quien conserva los arrays de un evento que cabía en un row group lo ancla.

    Errores: `FramesInputError` si falta un archivo o no cumple su contrato;
    `FrameBoundaryError` si una frontera de L2 no es un tick de L1.
    """
    if isinstance(thetas, Decimal | str):
        thetas = [thetas]
    values = [_theta(t) for t in thetas]
    if not values or len(set(values)) != len(values):
        raise ValueError("thetas debe traer al menos un θ y sin repetidos")
    if start > end:
        raise ValueError(f"start {start} es posterior a end {end}")
    key = (provider, market, asset)
    return _read(values, l1_root, l2_root, start, end, key)


def _read(
    thetas: list[Decimal],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    key: tuple[str, str, str],
) -> Iterator[EventFrames]:
    cursors = [
        _Cursor(t, event_groups(l2_root, key, f"{t:.8f}", start, end)) for t in thetas
    ]
    yield from _drive(L1(l1_root, key), cursors, start, end)


def _month_of_time(microseconds: int) -> Month:
    moment = _EPOCH + timedelta(microseconds=microseconds)
    return moment.year, moment.month


def frames_of(
    event: pa.RecordBatch | pa.Table | Mapping[str, object],
    l1_root: str | Path,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
) -> EventFrames:
    """Las tramas de un evento cuya fila de L2 el llamador ya tiene.

    `event` es esa fila (un `RecordBatch` o `Table` de una fila, o un mapa con sus 11
    columnas). Los meses de L1 salen de `reference_time` y `extreme_time`; de ahí solo
    se decodifican los row groups que las estadísticas de `agg_trade_id` piden.
    """
    if isinstance(event, Mapping):
        batch = pa.RecordBatch.from_pylist([dict(event)])
    elif isinstance(event, pa.Table):
        batch = event.combine_chunks().to_batches()[0]
    else:
        batch = event
    if batch.num_rows != 1:
        raise ValueError(f"frames_of recibe una fila de L2, no {batch.num_rows}")
    row = batch.to_pylist()[0]
    cursor = _Cursor(
        Decimal(row["theta"]).quantize(_THETA_SCALE),
        iter([event_group(batch, "frames_of")]),
    )
    first = _month_of_time(row["reference_time"])
    last = _month_of_time(row["extreme_time"])
    key = (provider, market, asset)
    return next(_drive(L1(l1_root, key), [cursor], first, last))
