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
    L1_COLUMNS,
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
# Tipos de las cuatro columnas de una trama, en el orden de `Frame`.
L1_TYPES = tuple(kind for name, kind in L1_COLUMNS.items() if name != "agg_trade_id")


class _Cursor:
    """El avance de un θ sobre sus eventos: cuál es el que sigue y qué ticks lleva ya.

    Solo el primer evento sin entregar puede estar a medias (`E_i ≤ R_{i+1}`): en modo
    por evento lo que ya se leyó de él se guarda **copiado** en `_conf` y `_over` (una
    lista de arrays por columna), para no anclar el row group del que salió. En modo
    por chunks no se guarda ningún tick: solo cuántos van (`_conf_n`, `_over_n`), para
    saber si hay que releerlos al entregar el evento.
    """

    def __init__(
        self,
        theta: Decimal,
        groups: Iterator[EventGroup],
        l1: L1,
        chunks: bool = False,
        max_event_ticks: int | None = None,
    ) -> None:
        self.theta = theta
        self._groups = groups
        self._l1 = l1
        self._chunks = chunks
        self._budget = max_event_ticks
        self._group = next(groups, None)
        self._k = 0
        # Qué fronteras del evento `k` ya se vieron como ticks de L1: (R, C, E).
        self._seen = np.zeros(3, bool)
        self._conf: list[list[pa.Array]] = [[], [], [], []]
        self._over: list[list[pa.Array]] = [[], [], [], []]
        # Ticks de cada fase y ticks anteriores al mes del evento `k` en row groups ya
        # soltados.
        self._conf_n = 0
        self._over_n = 0
        self._before = 0

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
        # group queda en 0 o en su largo. Los ticks de (R, E] de este row group son
        # `e - r`; cuentan como anteriores al mes si el row group es de un mes previo.
        edges = right.tolist()
        before = np.zeros(len(bounds), np.int64)
        if rg.month < group.month:
            before = right[:, 2] - right[:, 0]
        before[0] += self._before
        over = self._over_budget(bounds)
        for i in range(closed):
            r, c, e = edges[i]
            yield self._deliver(
                rg, group, k + i, (r, c, e), int(before[i]), bool(over[i])
            )
        self._k = k + closed
        if closed < len(bounds):
            r, c, e = edges[closed]
            carried = closed == 0
            self._before = int(before[closed])
            self._conf_n = (self._conf_n if carried else 0) + c - r
            self._over_n = (self._over_n if carried else 0) + e - c
            # En modo por chunks solo se cuentan los ticks: se releen al entregar.
            if not self._chunks and not over[closed]:
                self._keep(self._conf, rg, r, c)
                self._keep(self._over, rg, c, e)
            self._seen = seen[closed]
        else:
            self._before = self._conf_n = self._over_n = 0
            self._seen = np.zeros(3, bool)

    def _over_budget(self, bounds: np.ndarray) -> np.ndarray:
        """Qué eventos superan `max_event_ticks`, por `E - R` de su fila de L2 (ADR-L3-13)."""
        if self._budget is None:
            return np.zeros(len(bounds), bool)
        return bounds[:, 2] - bounds[:, 0] > self._budget

    def _deliver(
        self,
        rg: RowGroup,
        group: EventGroup,
        index: int,
        edges: tuple[int, int, int],
        before: int,
        over: bool,
    ) -> EventFrames:
        """El evento `index` del grupo, que cierra en `rg`, con las posiciones de sus
        fronteras en el row group. Solo el primer evento sin entregar trae ticks de row
        groups previos (`E_0 > ` máximo del anterior y `R_i ≥ E_0` para los demás).
        """
        r, c, e = edges
        first = index == self._k
        row = group.batch.slice(index, 1)
        if over:
            return EventFrames(self.theta, row, _empty(), _empty(), before, True)
        if self._chunks:
            return EventFrames(
                self.theta,
                row,
                _phase(
                    self._l1,
                    rg.frame(r, c),
                    rg.month,
                    self._conf_n if first else 0,
                    (int(group.bounds[index, 0]), int(group.bounds[index, 1])),
                    rg.low,
                ),
                _phase(
                    self._l1,
                    rg.frame(c, e),
                    rg.month,
                    self._over_n if first else 0,
                    (int(group.bounds[index, 1]), int(group.bounds[index, 2])),
                    rg.low,
                ),
                before,
            )
        if first:
            confirmation = self._take(self._conf, rg, r, c)
            overshoot = self._take(self._over, rg, c, e)
        else:
            confirmation, overshoot = rg.frame(r, c), rg.frame(c, e)
        return EventFrames(self.theta, row, confirmation, overshoot, before)

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


def _empty() -> Frame:
    """Una trama de largo 0 que no ancla ningún row group."""
    return Frame(*(pa.array([], kind) for kind in L1_TYPES))


def _phase(
    l1: L1,
    tail: Frame,
    month: Month,
    passed: int,
    span: tuple[int, int],
    limit: int,
) -> Iterator[Frame]:
    """Los trozos de una fase `(after, until]`, uno por row group, en orden de la serie.

    `tail` es lo que cae en el row group en curso (el primero con id `limit`). Si la fase
    ya pasó `passed` ticks por row groups anteriores, se releen de L1 de uno en uno
    (a lo más un row group en memoria) y se entregan antes de `tail`. Un row group sin
    ticks de la fase no entrega trozo. No toca el estado del lector: se puede consumir
    después de pedir el siguiente evento.
    """
    after, until = span
    if passed:
        rows = l1.row_groups(
            month,
            month,
            after,
            lambda low, high: high > after and low <= until and high < limit,
        )
        try:
            for rg in rows:
                start = int(np.searchsorted(rg.ids, after, "right"))
                stop = int(np.searchsorted(rg.ids, until, "right"))
                if stop > start:
                    yield rg.frame(start, stop)
                del rg
        finally:
            rows.close()
    if len(tail):
        yield tail


def _drive(
    l1: L1, cursors: list[_Cursor], first: Month, last: Month
) -> Iterator[EventFrames]:
    live = [c for c in cursors if not c.done]
    if not live:
        return
    reference = min(c.reference for c in live)

    def needed(_low: int, high: int) -> bool:
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
    theta = Decimal(value)
    scaled = theta.quantize(_THETA_SCALE)
    if scaled != theta:
        raise ValueError(f"θ {value} no cabe en 8 decimales, que es la partición de L2")
    return scaled


def _check_month(name: str, month: Month) -> None:
    if not 1 <= month[1] <= 12:
        raise ValueError(f"{name} {month}: el mes debe estar entre 1 y 12")


def read_frames(
    thetas: Decimal | str | Iterable[Decimal | str],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
    chunks: bool = False,
    max_event_ticks: int | None = None,
) -> Iterator[EventFrames]:
    """Los eventos cerrados de los θ en los meses `start`..`end`, con sus tramas.

    `start` y `end` son `(año, mes)` de las particiones de L2, ambos inclusivos; un
    evento pertenece al mes que lo cierra y sus ticks pueden estar en meses anteriores
    a `start`, que el lector también lee. Dentro de cada θ salen por
    `confirm_agg_trade_id` ascendente; entre θ no hay orden garantizado. Con varios θ
    cada row group de L1 se decodifica una sola vez.

    Modo por evento (por defecto): cada fase es un `Frame` con todos sus ticks.
    Memoria: un row group de L1, un row group de eventos por θ y el evento en curso de
    cada θ. Quien conserva los arrays de un evento que cabía en un row group lo ancla.
    `max_event_ticks` es el presupuesto de ticks por evento: uno con
    `extreme_agg_trade_id - reference_agg_trade_id` mayor se entrega con las dos tramas
    vacías y `over_budget = True`, sin copiar ni retener un solo tick (el lector sí
    recorre sus row groups para validar las fronteras y contar `ticks_before_month`).

    Modo por chunks (`chunks=True`): cada fase es un iterador de `Frame`, uno por row
    group (ver `EventFrames`). El lector no retiene ningún evento: el pico es un row
    group de L1 en el pase, más el que relee un evento que cruza row groups, y un row
    group de eventos por θ, con cualquier número de θ y de eventos. No admite
    `max_event_ticks`: no tiene presupuesto.

    Errores: `FramesInputError` si falta un archivo o no cumple su contrato;
    `FrameBoundaryError` si una frontera de L2 no es un tick de L1 o el tick de
    confirmación no tiene `transact_time = confirm_time`.
    """
    if chunks and max_event_ticks is not None:
        raise ValueError(
            "max_event_ticks es del modo por evento; chunks no tiene presupuesto"
        )
    if max_event_ticks is not None and max_event_ticks < 0:
        raise ValueError(f"max_event_ticks {max_event_ticks} no puede ser negativo")
    if isinstance(thetas, Decimal | str):
        thetas = [thetas]
    values = [_theta(t) for t in thetas]
    if not values or len(set(values)) != len(values):
        raise ValueError("thetas debe traer al menos un θ y sin repetidos")
    _check_month("start", start)
    _check_month("end", end)
    if start > end:
        raise ValueError(f"start {start} es posterior a end {end}")
    key = (provider, market, asset)
    return _read(values, l1_root, l2_root, start, end, key, chunks, max_event_ticks)


def _read(
    thetas: list[Decimal],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    key: tuple[str, str, str],
    chunks: bool,
    max_event_ticks: int | None,
) -> Iterator[EventFrames]:
    l1 = L1(l1_root, key)
    cursors = [
        _Cursor(
            t,
            event_groups(l2_root, key, f"{t:.8f}", start, end),
            l1,
            chunks,
            max_event_ticks,
        )
        for t in thetas
    ]
    yield from _drive(l1, cursors, start, end)


def _month_of_time(microseconds: int) -> Month:
    moment = _EPOCH + timedelta(microseconds=microseconds)
    return moment.year, moment.month


def frames_of(
    event: pa.RecordBatch | pa.Table | Mapping[str, object],
    l1_root: str | Path,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
    month: Month | None = None,
) -> EventFrames:
    """Las tramas de un evento cuya fila de L2 el llamador ya tiene.

    `event` es esa fila (un `RecordBatch` o `Table` de una fila, o un mapa con sus 11
    columnas). Los meses de L1 salen de `reference_time` y `extreme_time`; de ahí solo
    se decodifican los row groups que las estadísticas de `agg_trade_id` piden.
    `month` es la partición de L2 que cierra el evento, con la que se cuenta
    `ticks_before_month`; la fila no la trae, así que por defecto es el mes de
    `extreme_time`. Siempre es el modo por evento y sin presupuesto.

    Con un mapa, Arrow infiere los tipos (`direction` sale `int64` y `theta` con la
    precisión de su valor), así que `EventFrames.event` no es la fila de L2 sin
    transformar: solo un `RecordBatch` o una `Table` de L2 la conservan tal cual.
    """
    rows = 1 if isinstance(event, Mapping) else event.num_rows
    if rows != 1:
        raise ValueError(f"frames_of recibe una fila de L2, no {rows}")
    if isinstance(event, Mapping):
        batch = pa.RecordBatch.from_pylist([dict(event)])
    elif isinstance(event, pa.Table):
        batch = event.combine_chunks().to_batches()[0]
    else:
        batch = event
    row = batch.to_pylist()[0]
    first = _month_of_time(row["reference_time"])
    last = _month_of_time(row["extreme_time"])
    if month is not None:
        _check_month("month", month)
    group = event_group(batch, "frames_of", month or last)
    key = (provider, market, asset)
    l1 = L1(l1_root, key)
    cursor = _Cursor(Decimal(row["theta"]).quantize(_THETA_SCALE), iter([group]), l1)
    return next(_drive(l1, [cursor], first, last))
