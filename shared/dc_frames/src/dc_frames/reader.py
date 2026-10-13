"""El lector de tramas: recorta L1 con las fronteras de L2 por `agg_trade_id` (TRD-L3 §7.4).

La pertenencia de un tick a una fase se decide por **valor** de `agg_trade_id`, nunca
por posición ni por tiempo: el proveedor deja huecos de ids, y restar posiciones
correría la frontera. Cada frontera se busca con `searchsorted` sobre los ids del row
group, para todos los eventos del row group a la vez; el único bucle de Python es uno
por evento, que arma sus slices.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa

from dc_frames.delivery import deliver
from dc_frames.lake import (
    L1,
    L1_COLUMNS,
    EventGroup,
    Month,
    RowGroup,
    event_group,
    event_groups,
)
from dc_frames.track import Track, locate, unclosed_error
from dc_frames.types import (
    EventFrames,
    Frame,
    RowGroupFrames,
)

DEFAULT_KEY = ("binance", "spot", "BTCUSDT")
_THETA_SCALE = Decimal("1e-8")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# Tipos de las cuatro columnas de una trama, en el orden de `Frame`.
L1_TYPES = tuple(kind for name, kind in L1_COLUMNS.items() if name != "agg_trade_id")


class _Previous:
    """El row group anterior al del pase: una sola copia que comparten todos los θ.

    Solo el modo por chunks lo usa. El pase lo deja aquí mientras algún evento abierto
    lo necesite (ver `_Cursor.holds`); así, al entregar un evento que cruza el borde, su
    fase toma los ticks de este row group sin releerlos de L1, sea cual sea el número de θ.
    """

    def __init__(self) -> None:
        self.rg: RowGroup | None = None


class _Cursor(Track):
    """El avance de un θ sobre sus eventos en los modos por evento y por chunks.

    En modo por evento lo que ya se leyó del evento a medias se guarda **copiado** en
    `_conf` y `_over` (una lista de arrays por columna), para no anclar el row group del
    que salió. En modo por chunks no se guarda ningún tick: solo cuántos van (`_conf_n`,
    `_over_n`), para saber si quedan ticks por tomar del row group anterior (`previous`)
    o por releer de L1.
    """

    def __init__(
        self,
        theta: Decimal,
        groups: Iterator[EventGroup],
        l1: L1,
        chunks: bool = False,
        max_event_ticks: int | None = None,
        previous: _Previous | None = None,
    ) -> None:
        super().__init__(theta, groups)
        self._l1 = l1
        self._chunks = chunks
        self._previous = previous
        self._budget = max_event_ticks
        self._conf: list[list[pa.Array]] = [[], [], [], []]
        self._over: list[list[pa.Array]] = [[], [], [], []]
        # Ticks de cada fase del evento a medias, en row groups ya soltados.
        self._conf_n = 0
        self._over_n = 0

    def holds(self, rg: RowGroup) -> bool:
        """Si su evento en curso ya empezó en `rg` o antes y sigue abierto: necesitará `rg`."""
        return self._group is not None and self.reference <= rg.high

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

    def _process(
        self, rg: RowGroup, group: EventGroup, touched: int
    ) -> Iterator[EventFrames]:
        k = self._k
        bounds = group.bounds[k:touched]
        right, seen = locate(
            self.theta, rg, bounds, group.confirm_times[k:touched], self._seen
        )

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
            ref, conf, ext = (int(x) for x in group.bounds[index])
            return EventFrames(
                self.theta,
                row,
                self._chunked(rg, (r, c), (ref, conf), self._conf_n if first else 0),
                self._chunked(rg, (c, e), (conf, ext), self._over_n if first else 0),
                before,
            )
        if first:
            confirmation = self._take(self._conf, rg, r, c)
            overshoot = self._take(self._over, rg, c, e)
        else:
            confirmation, overshoot = rg.frame(r, c), rg.frame(c, e)
        return EventFrames(self.theta, row, confirmation, overshoot, before)

    def _chunked(
        self, rg: RowGroup, edges: tuple[int, int], span: tuple[int, int], passed: int
    ) -> Iterator[Frame]:
        """El iterador de una fase: `edges` son sus posiciones en `rg` y `span`, sus ids.

        De los `passed` ticks que la fase tiene en row groups anteriores, los del row
        group anterior se toman ya, como slice sin copia (así el iterador no depende del
        estado del lector); el resto se relee de L1 al consumirlo.
        """
        previous = self._previous.rg if self._previous is not None else None
        head = None
        if previous is not None and passed:
            after, until = span
            start = int(np.searchsorted(previous.ids, after, "right"))
            stop = int(np.searchsorted(previous.ids, until, "right"))
            if stop > start:
                head = previous.frame(start, stop)
        return _phase(
            self._l1,
            head,
            rg.frame(*edges),
            rg.month,
            passed,
            span,
            (rg if previous is None else previous).low,
        )

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


def _empty() -> Frame:
    """Una trama de largo 0 que no ancla ningún row group."""
    return Frame(*(pa.array([], kind) for kind in L1_TYPES))


def _phase(
    l1: L1,
    head: Frame | None,
    tail: Frame,
    month: Month,
    passed: int,
    span: tuple[int, int],
    limit: int,
) -> Iterator[Frame]:
    """Los trozos de una fase `(after, until]`, uno por row group, en orden de la serie.

    `tail` es lo que cae en el row group en curso y `head`, lo que cae en el anterior
    (ambos slices sin copia, tomados al entregar el evento). Si la fase ya pasó `passed`
    ticks por row groups anteriores y `head` no los cubre, el resto está en row groups de
    ids menores que `limit` (el primero de `head`): se releen de L1 de uno en uno (a lo
    más un row group en memoria) y se entregan antes de `head` y `tail`. Un row group sin
    ticks de la fase no entrega trozo. No toca el estado del lector: se puede consumir
    después de pedir el siguiente evento, pero hasta entonces ancla los row groups de
    `head` y `tail`.
    """
    after, until = span
    if passed > (len(head) if head is not None else 0):
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
    if head is not None:
        yield head
    if len(tail):
        yield tail


def _drive(
    l1: L1,
    cursors: list[_Cursor],
    first: Month,
    last: Month,
    previous: _Previous | None = None,
) -> Iterator[EventFrames]:
    live = [c for c in cursors if not c.done]
    if not live:
        return
    reference = min(c.reference for c in live)

    def needed(_low: int, high: int) -> bool:
        return any(not c.done and c.reference <= high for c in live)

    rows = l1.row_groups(first, last, reference, needed)
    try:
        # Se suelta el row group antes de pedir el siguiente. Solo el modo por chunks
        # conserva uno más, el anterior (`previous`), mientras un evento abierto lo use.
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
            if previous is not None:
                previous.rg = rg if any(c.holds(rg) for c in live) else None
    finally:
        rows.close()
    raise unclosed_error(live)


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


def _check_request(
    thetas: Decimal | str | Iterable[Decimal | str], start: Month, end: Month
) -> list[Decimal]:
    """Valida θ y meses de una lectura y devuelve los θ ya normalizados."""
    if isinstance(thetas, Decimal | str):
        thetas = [thetas]
    values = [_theta(t) for t in thetas]
    if not values or len(set(values)) != len(values):
        raise ValueError("thetas debe traer al menos un θ y sin repetidos")
    _check_month("start", start)
    _check_month("end", end)
    if start > end:
        raise ValueError(f"start {start} es posterior a end {end}")
    return values


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
    columns: Sequence[str] | None = None,
) -> Iterator[EventFrames]:
    """Los eventos cerrados de los θ en los meses `start`..`end`, con sus tramas.

    `start` y `end` son `(año, mes)` de las particiones de L2, ambos inclusivos; un
    evento pertenece al mes que lo cierra y sus ticks pueden estar en meses anteriores
    a `start`, que el lector también lee. Dentro de cada θ salen por
    `confirm_agg_trade_id` ascendente; entre θ no hay orden garantizado. Con varios θ
    cada row group de L1 se decodifica una sola vez, salvo en el modo por chunks por los
    eventos que abarcan más de dos row groups (ver abajo).

    Modo por evento (por defecto): cada fase es un `Frame` con todos sus ticks.
    Memoria: un row group de L1, un row group de eventos por θ y el evento en curso de
    cada θ. Quien conserva los arrays de un evento que cabía en un row group lo ancla.
    `max_event_ticks` es el presupuesto de ticks por evento: uno con
    `extreme_agg_trade_id - reference_agg_trade_id` mayor se entrega con las dos tramas
    vacías y `over_budget = True`, sin copiar ni retener un solo tick (el lector sí
    recorre sus row groups para validar las fronteras y contar `ticks_before_month`).

    Modo por chunks (`chunks=True`): cada fase es un iterador de `Frame`, uno por row
    group (ver `EventFrames`). El lector no retiene ningún evento. Conserva una sola
    copia del row group anterior, compartida por todos los θ, mientras algún evento
    abierto la necesite: el pico es de dos row groups de L1 (el del pase y ese) y un row
    group de eventos por θ, con cualquier número de θ y de eventos. Un evento que abarca
    más de dos row groups relee de L1 solo los que sobran, de uno en uno, y suma ese al
    pico. Los iteradores de fase de un evento se consumen antes de pedir el siguiente:
    consumirlos después ancla los row groups de sus trozos. No admite
    `max_event_ticks`: no tiene presupuesto.

    `columns` proyecta las columnas de `events.parquet` que se decodifican: `EventFrames.event`
    trae esas más las tres fronteras y `confirm_time`, que el lector siempre lee. Sin él
    trae las 11.

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
    values = _check_request(thetas, start, end)
    key = (provider, market, asset)
    return _read(
        values, l1_root, l2_root, start, end, key, chunks, max_event_ticks, columns
    )


def _read(
    thetas: list[Decimal],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    key: tuple[str, str, str],
    chunks: bool,
    max_event_ticks: int | None,
    columns: Sequence[str] | None,
) -> Iterator[EventFrames]:
    l1 = L1(l1_root, key)
    previous = _Previous() if chunks else None
    cursors = [
        _Cursor(
            t,
            event_groups(l2_root, key, f"{t:.8f}", start, end, columns),
            l1,
            chunks,
            max_event_ticks,
            previous,
        )
        for t in thetas
    ]
    yield from _drive(l1, cursors, start, end, previous)


def read_row_groups(
    thetas: Decimal | str | Iterable[Decimal | str],
    l1_root: str | Path,
    l2_root: str | Path,
    start: Month,
    end: Month,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
    columns: Sequence[str] | None = None,
) -> Iterator[RowGroupFrames]:
    """Los row groups de L1 del pase, uno por elemento, con los eventos de todos los θ.

    Es la entrega por row group de TRD-L3 §7.4 (ADR-L3-14), la que usan el resumidor y
    las familias por chunks. Los parámetros son los de `read_frames`, sin `chunks` ni
    `max_event_ticks`. El pase retrocede al row group más antiguo que necesita algún
    evento de `start` y recorre desde ahí cada row group **una sola vez**, en orden, para
    todos los θ a la vez: los eventos que cruzan row groups o meses no releen nada,
    porque el consumidor los completa con su acumulador parcial.

    Cada `RowGroupFrames` trae los cuatro arrays del row group como vistas sin copia y,
    por θ, las filas de L2 de los eventos que lo tocan con sus posiciones `R`, `C` y `E`
    (ver `ThetaEvents`). Un evento sale en cada row group desde el de su referencia hasta
    el de su extremo, y su `ticks_before_month` sale en el que lo cierra.

    Memoria: un row group de L1 en vuelo y ninguno retenido, más un row group de
    eventos de L2 por θ. El consumidor debe soltar el elemento antes de pedir el
    siguiente (`del`): el `for` conserva el anterior hasta recibir el nuevo y, sin el
    `del`, conviven dos row groups.

    `columns` proyecta las columnas de `events.parquet` que se entregan, más las cuatro
    que el lector siempre lee (las tres fronteras y `confirm_time`). Sin él trae las
    siete de TRD-L3 §7.1, no las 11 de `read_frames`.

    Errores: los de `read_frames`. Una frontera de un evento que cae dentro de un row
    group y no es un tick, o un tick de confirmación con otro `transact_time`, lanza
    `FrameBoundaryError` al entregar el row group.
    """
    values = _check_request(thetas, start, end)
    key = (provider, market, asset)
    return deliver(values, l1_root, l2_root, start, end, key, columns)


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
