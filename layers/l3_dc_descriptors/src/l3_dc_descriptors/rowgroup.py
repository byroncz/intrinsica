"""La agregación por row group del resumidor (TRD-L3 §8.1, ADR-L3-14).

Por cada row group de L1, una vez para todos los θ, `Prefixes` calcula las sumas
acumuladas que no dependen del θ. Por θ, `Segments` toma las posiciones de las fronteras
que entrega el lector y saca de ahí, para todas las fases de todos los eventos a la vez,
los conteos y las sumas (diferencia de prefijos en los cortes), el máximo y el mínimo de
`price` (`reduceat`), el primer y el último tiempo y el instante de confirmación. El
evento que sigue abierto al cierre del row group deja un acumulador parcial (`Open`):
conteos, sumas, extremos y tiempos, nunca ticks.

**Sumas exactas.** `quantity` se suma en `int64`: un row group cabe porque la suma de
`quantity` de sus ticks queda muy por debajo de 2⁶³ (si no, `Prefixes.estimate` lo
detecta). `price·quantity` no cabe ni en un tick, así que cada operando se descompone en
tres palabras de 21 bits y se acumulan los 9 productos parciales en `uint64`. Cada
producto es menor que 2⁴² y un row group tiene a lo sumo 2²⁰ ticks, así que cada prefijo
queda por debajo de 2⁶² y cualquier diferencia cabe en 63 bits. Las nueve diferencias se
agrupan por peso (`i + j`) en 5 palabras y solo se recomponen en entero de Python por
segmento, `Σ p·q = Σ_k W_k · 2^(21 k)`. Nunca se multiplica `price` por `quantity` en
`int64` ni se reduce antes de multiplicar.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pyarrow as pa
from dc_frames import Frame, FramesInputError, ThetaEvents
from dc_frames.lake import Month

from l3_dc_descriptors.event import EVENT_COLUMNS, build_row
from l3_dc_descriptors.phase import divide_half_even
from l3_dc_descriptors.schema import COLUMN, LIMIT, WIDTH

WORD_BITS = 21
WORD_MASK = np.uint64((1 << WORD_BITS) - 1)
OPERAND_WORDS = 3  # 3 × 21 = 63 bits por operando
SUM_WORDS = 2 * OPERAND_WORDS - 1  # pesos 2^(21 k), k = 0..4
# Con 2²⁰ filas, cada producto parcial (< 2⁴²) sumado cabe en 62 bits.
MAX_ROWS = 1 << 20
# Por encima de esta suma estimada una fase no cabe en DECIMAL(18,8) (10¹⁸ < 2⁶²).
_FLOAT_LIMIT = float(1 << 62)


def unscaled(array: pa.Array) -> np.ndarray:
    """El valor sin escalar de un `decimal128` como vista `int64`, sin copia.

    L1 y L2 garantizan que los valores caben en `int64`: la palabra baja de cada decimal
    de 128 bits es el valor, y está cada 2 posiciones del buffer.
    """
    raw = np.frombuffer(array.buffers()[1], dtype="<i8")
    return raw[2 * array.offset : 2 * (array.offset + len(array)) : 2]


def _prefix(values: np.ndarray, dtype) -> np.ndarray:
    """La suma acumulada con un 0 al frente: `prefix[i]` suma los primeros `i` valores."""
    out = np.zeros(len(values) + 1, dtype)
    np.cumsum(values, dtype=dtype, out=out[1:])
    return out


class Prefixes:
    """Lo que se calcula una vez por row group y comparten todos los θ.

    `time` y `price` son vistas del buffer de Arrow. `total`, `buy` y `sell` son los
    prefijos de `quantity` (todos los ticks, `is_buyer_maker = false` y `true`);
    `words[k]` son los prefijos de los productos parciales de peso `2^(21 k)`: los 9
    de `price_i · quantity_j` agrupados por `i + j`. `estimate`, solo si la suma del row
    group se acerca al tope de `int64`, es el prefijo de `quantity` en `float64`: sirve
    para detectar las fases que no caben, no para sumar.
    """

    __slots__ = ("buy", "estimate", "price", "sell", "time", "total", "words")

    def __init__(self, frame: Frame) -> None:
        n = len(frame)
        if n > MAX_ROWS:
            raise FramesInputError(
                f"un row group de L1 trae {n} ticks y la suma exacta de price·quantity "
                f"admite hasta {MAX_ROWS}"
            )
        price, quantity = unscaled(frame.price), unscaled(frame.quantity)
        if price.min() < 0 or quantity.min() < 0:
            raise FramesInputError("un row group de L1 trae price o quantity negativos")
        sold = frame.is_buyer_maker.to_numpy(zero_copy_only=False)
        self.time = frame.transact_time.to_numpy(zero_copy_only=True)
        self.price = price
        self.total = _prefix(quantity, np.int64)
        self.sell = _prefix(np.where(sold, quantity, 0), np.int64)
        self.buy = _prefix(np.where(sold, 0, quantity), np.int64)
        self.estimate = None
        if quantity.sum(dtype=np.float64) >= _FLOAT_LIMIT:
            self.estimate = _prefix(quantity, np.float64)
        q = [
            (quantity.view(np.uint64) >> np.uint64(WORD_BITS * j)) & WORD_MASK
            for j in range(OPERAND_WORDS)
        ]
        self.words: list[list[np.ndarray]] = [[] for _ in range(SUM_WORDS)]
        for i in range(OPERAND_WORDS):
            p = (price.view(np.uint64) >> np.uint64(WORD_BITS * i)) & WORD_MASK
            for j in range(OPERAND_WORDS):
                self.words[i + j].append(_prefix(p * q[j], np.uint64))


class Phase:
    """Los agregados de una fase, o de un tramo de ella: escalares de Python, exactos.

    `words` son las 5 palabras de `Σ price·quantity`, que se recomponen en `notional`.
    Los nombres son los que lee `close_phase`.
    """

    __slots__ = (
        "buy",
        "first_time",
        "high",
        "last_time",
        "low",
        "sell",
        "ticks",
        "total",
        "words",
    )

    def __init__(self) -> None:
        self.ticks = 0
        self.total = 0
        self.buy = 0
        self.sell = 0
        self.words = [0] * SUM_WORDS
        self.high: int | None = None
        self.low: int | None = None
        self.first_time: int | None = None
        self.last_time: int | None = None

    @property
    def notional(self) -> int:
        return recompose(self.words)

    def merge(self, piece: Phase) -> None:
        """Suma el tramo siguiente de la fase: conteos y sumas se suman, el mínimo y el
        máximo se comparan, el primer tiempo se conserva y el último se reemplaza.
        """
        if not piece.ticks:
            return
        self.ticks += piece.ticks
        self.total += piece.total
        self.buy += piece.buy
        self.sell += piece.sell
        for k, word in enumerate(piece.words):
            self.words[k] += word
        self.high = piece.high if self.high is None else max(self.high, piece.high)
        self.low = piece.low if self.low is None else min(self.low, piece.low)
        if self.first_time is None:
            self.first_time = piece.first_time
        self.last_time = piece.last_time


def recompose(words: Sequence[int]) -> int:
    """`Σ_k words[k] · 2^(21 k)`: el `Σ price·quantity` en entero de Python."""
    return sum(word << (WORD_BITS * k) for k, word in enumerate(words))


class Instant:
    """Los ticks de la confirmación con `transact_time = confirm_time` (ADR-L3-01)."""

    __slots__ = ("high", "low", "ticks")

    def __init__(self, ticks: int = 0, high: int | None = None, low: int | None = None):
        self.ticks = ticks
        self.high = high
        self.low = low

    def merge(self, piece: Instant) -> None:
        if not piece.ticks:
            return
        self.ticks += piece.ticks
        self.high = piece.high if self.high is None else max(self.high, piece.high)
        self.low = piece.low if self.low is None else min(self.low, piece.low)


class Open:
    """El acumulador parcial de un θ: el evento abierto al cierre de un row group.

    Estado O(1): las dos fases y el instante de confirmación. Al llegar el siguiente row
    group se fusiona con el tramo que le toca, sin releer ningún tick.
    """

    __slots__ = ("confirmation", "instant", "overshoot")

    def __init__(self, confirmation: Phase, overshoot: Phase, instant: Instant) -> None:
        self.confirmation = confirmation
        self.overshoot = overshoot
        self.instant = instant

    def merge(self, confirmation: Phase, overshoot: Phase, instant: Instant) -> None:
        self.confirmation.merge(confirmation)
        self.overshoot.merge(overshoot)
        self.instant.merge(instant)


class L2Columns:
    """Las columnas de L2 de los eventos de un θ en un row group, como arreglos."""

    def __init__(self, events: pa.RecordBatch) -> None:
        self.batch = events
        self.reference = events.column("reference_agg_trade_id").to_numpy()
        self.confirm = events.column("confirm_agg_trade_id").to_numpy()
        self.confirm_time = events.column("confirm_time").to_numpy()
        self.extreme = events.column("extreme_agg_trade_id").to_numpy()
        self.direction = events.column("direction").to_numpy().astype(np.int64)
        self.confirm_price = unscaled(events.column("confirm_price"))
        self.extreme_price = unscaled(events.column("extreme_price"))

    def row(self, i: int) -> dict:
        """La fila `i` de L2 como diccionario de Python (los precios, `Decimal`)."""
        return self.batch.slice(i, 1).select(EVENT_COLUMNS).to_pylist()[0]


class Segments:
    """Los agregados de las fases de todos los eventos de un θ en un row group.

    Una posición `2 i` es la confirmación del evento `i` y `2 i + 1` su overshoot; cada
    fase es el tramo `[inicio, fin)` del row group. Las fases de un θ son contiguas y no
    se solapan, así que los cortes están ordenados. Las fases sin ticks quedan en 0 (los
    precios y los tiempos no significan nada ahí).
    """

    def __init__(
        self, agg: Prefixes, positions: np.ndarray, confirm_times: np.ndarray
    ) -> None:
        starts = np.ascontiguousarray(positions[:, :2]).reshape(-1)
        stops = np.ascontiguousarray(positions[:, 1:]).reshape(-1)
        self.ticks = stops - starts
        self.total = agg.total[stops] - agg.total[starts]
        self.buy = agg.buy[stops] - agg.buy[starts]
        self.sell = agg.sell[stops] - agg.sell[starts]
        self.overflow = None
        if agg.estimate is not None:
            estimated = agg.estimate[stops] - agg.estimate[starts]
            self.overflow = estimated >= _FLOAT_LIMIT
        # Las 9 diferencias, sumadas por peso en `uint64` (< 3·2⁶² < 2⁶⁴), y a Python.
        self.words: list[list[int]] = []
        for terms in agg.words:
            word = terms[0][stops] - terms[0][starts]
            for prefix in terms[1:]:
                word += prefix[stops] - prefix[starts]
            self.words.append(word.tolist())
        full = self.ticks > 0
        self.high = np.zeros(len(starts), np.int64)
        self.low = np.zeros(len(starts), np.int64)
        self.first = np.zeros(len(starts), np.int64)
        self.last = np.zeros(len(starts), np.int64)
        if full.any():
            first, end = starts[full], stops[full]
            # `reduceat` no admite segmentos vacíos: solo van los que tienen ticks. Entre
            # un corte y el siguiente quedan tramos que no se piden y se descartan.
            cuts = np.empty(2 * len(first), np.int64)
            cuts[0::2], cuts[1::2] = first, end
            if cuts[-1] == len(agg.price):
                cuts = cuts[:-1]  # el último tramo ya llega al final del row group
            self.high[full] = np.maximum.reduceat(agg.price, cuts)[0::2]
            self.low[full] = np.minimum.reduceat(agg.price, cuts)[0::2]
            self.first[full] = agg.time[first]
            self.last[full] = agg.time[end - 1]
        self._instant(agg, positions, confirm_times)

    def _instant(
        self, agg: Prefixes, positions: np.ndarray, confirm_times: np.ndarray
    ) -> None:
        """El instante de confirmación: una máscara `transact_time == confirm_time`
        restringida a la fase de confirmación de cada evento.
        """
        n = len(positions)
        reference, confirm = positions[:, 0], positions[:, 1]
        edges = np.empty(2 * n + 2, np.int64)
        edges[0], edges[-1] = 0, len(agg.time)
        edges[1:-1:2], edges[2:-1:2] = reference, confirm
        # Cada tick lleva el `confirm_time` de su evento si es de su confirmación y -1
        # (ningún tiempo lo iguala) si no.
        wanted = np.full(2 * n + 1, -1, np.int64)
        wanted[1::2] = confirm_times
        hits = np.flatnonzero(agg.time == np.repeat(wanted, np.diff(edges)))
        first = np.searchsorted(hits, reference, "left")
        self.instant_ticks = np.searchsorted(hits, confirm, "left") - first
        self.instant_high = np.zeros(n, np.int64)
        self.instant_low = np.zeros(n, np.int64)
        found = self.instant_ticks > 0
        if found.any():
            prices = agg.price[hits]
            self.instant_high[found] = np.maximum.reduceat(prices, first[found])
            self.instant_low[found] = np.minimum.reduceat(prices, first[found])

    def phase(self, slot: int) -> Phase:
        """La fase `slot` como escalares de Python."""
        phase = Phase()
        phase.ticks = int(self.ticks[slot])
        if not phase.ticks:
            return phase
        phase.total = int(self.total[slot])
        phase.buy = int(self.buy[slot])
        phase.sell = int(self.sell[slot])
        if self.overflow is not None and self.overflow[slot]:
            # No cabe en DECIMAL(18,8): el valor exacto no importa, `close_phase` lo rechaza.
            phase.total = phase.buy = LIMIT
            phase.sell = 0
        phase.words = [word[slot] for word in self.words]
        phase.high, phase.low = int(self.high[slot]), int(self.low[slot])
        phase.first_time, phase.last_time = int(self.first[slot]), int(self.last[slot])
        return phase

    def instant(self, event: int) -> Instant:
        ticks = int(self.instant_ticks[event])
        if not ticks:
            return Instant()
        return Instant(
            ticks, int(self.instant_high[event]), int(self.instant_low[event])
        )

    def pieces(self, event: int) -> tuple[Phase, Phase, Instant]:
        return self.phase(2 * event), self.phase(2 * event + 1), self.instant(event)


def _vwaps(words: list[list[int]], total: np.ndarray) -> np.ndarray:
    """El VWAP (10⁻⁸) de cada fase: `Σ price·quantity / Σ quantity`, half-even; 0 si no
    hay cantidad. La suma exacta se recompone en entero de Python por segmento.
    """
    a, b, c, d, e = words
    out = [
        divide_half_even(w0 + (w1 << 21) + (w2 << 42) + (w3 << 63) + (w4 << 84), q)
        if q
        else 0
        for w0, w1, w2, w3, w4, q in zip(a, b, c, d, e, total.tolist(), strict=True)
    ]
    return np.array(out, np.int64)


def _close_batch(seg: Segments, l2: L2Columns, lo: int, hi: int) -> np.ndarray | None:
    """Las filas de los eventos `[lo, hi)`, todos dentro del row group, en bloque.

    Aplica a todo el lote las mismas comprobaciones que `build_row` aplica a un evento
    (cantidades de las fases, invariantes 1 a 7 de §7.2 contra L2) y devuelve `None` si
    alguna falla: el lote entero se repite entonces evento por evento, que es quien
    explica el primer fallo.
    """
    c, o, e = slice(2 * lo, 2 * hi, 2), slice(2 * lo + 1, 2 * hi, 2), slice(lo, hi)
    if seg.overflow is not None and seg.overflow[2 * lo : 2 * hi].any():
        return None
    direction = l2.direction[e]
    if not ((direction == 1) | (direction == -1)).all():
        return None
    phases = []
    for part in (c, o):
        ticks, total = seg.ticks[part], seg.total[part]
        buy, sell = seg.buy[part], seg.sell[part]
        if (buy + sell != total).any():
            return None
        if ((ticks > 0) & ((buy >= LIMIT) | (sell >= LIMIT) | (total <= 0))).any():
            return None
        words = [word[part] for word in seg.words]
        phases.append((ticks, total, buy, sell, _vwaps(words, total), part))
    (c_ticks, _, c_buy, c_sell, c_vwap, _), (o_ticks, _, o_buy, o_sell, o_vwap, _) = (
        phases
    )
    c_high, c_low = seg.high[c], seg.low[c]
    o_high, o_low = seg.high[o], seg.low[o]
    c_duration = seg.last[c] - seg.first[c]
    reference, confirm, extreme = l2.reference[e], l2.confirm[e], l2.extreme[e]
    confirm_price, extreme_price = l2.confirm_price[e], l2.extreme_price[e]
    instant = seg.instant_ticks[e]
    up = direction == 1
    best = np.where(up, seg.instant_high[e], seg.instant_low[e])
    worst = np.where(up, seg.instant_low[e], seg.instant_high[e])
    top = np.where(up, c_high, c_low)
    edge = np.where(up, o_high, o_low)
    checks = (
        (instant >= 1) & (instant <= c_ticks),  # 1
        np.where(
            up,
            (worst <= confirm_price) & (confirm_price <= best),
            (best <= confirm_price) & (confirm_price <= worst),
        ),  # 2
        top == best,  # 3
        np.where(
            o_ticks > 0, edge == extreme_price, extreme_price == confirm_price
        ),  # 4
        (o_ticks == 0) == (extreme == confirm),  # 5
        (c_ticks <= confirm - reference) & (o_ticks <= extreme - confirm),  # 6
        (c_duration == 0) == (c_ticks == instant),  # 7
        (c_low <= c_vwap) & (c_vwap <= c_high),
        (o_ticks == 0) | ((o_low <= o_vwap) & (o_vwap <= o_high)),
    )
    if not all(check.all() for check in checks):
        return None

    rows = np.zeros((hi - lo, WIDTH), np.int64)
    rows[:, COLUMN["confirm_agg_trade_id"]] = confirm
    rows[:, COLUMN["c_ticks"]] = c_ticks
    rows[:, COLUMN["c_quantity_buy"]] = c_buy
    rows[:, COLUMN["c_quantity_sell"]] = c_sell
    rows[:, COLUMN["c_price_max"]] = c_high
    rows[:, COLUMN["c_price_min"]] = c_low
    rows[:, COLUMN["c_duration"]] = c_duration
    rows[:, COLUMN["c_vwap"]] = c_vwap
    rows[:, COLUMN["os_ticks"]] = o_ticks
    rows[:, COLUMN["os_quantity_buy"]] = o_buy
    rows[:, COLUMN["os_quantity_sell"]] = o_sell
    rows[:, COLUMN["os_price_max"]] = o_high
    rows[:, COLUMN["os_price_min"]] = o_low
    rows[:, COLUMN["os_duration"]] = seg.last[o] - seg.first[o]
    rows[:, COLUMN["os_vwap"]] = o_vwap
    rows[:, COLUMN["confirm_instant_ticks"]] = instant
    rows[:, COLUMN["confirm_instant_price_best"]] = best
    rows[:, COLUMN["confirm_instant_price_worst"]] = worst
    return rows


class Closed(NamedTuple):
    """Los eventos que un row group dejó cerrados en un θ, en el orden de L2."""

    rows: np.ndarray  # (k, WIDTH) enteros, los decimales sin escalar
    first_reference: int  # `reference_agg_trade_id` del primero
    last_extreme: int  # `extreme_agg_trade_id` del último
    before_month: int  # Σ de `ticks_before_month`


class ThetaAggregator:
    """La agregación de un θ a lo largo de la pasada: el evento abierto entre row groups."""

    def __init__(self, text: str, month: Month) -> None:
        self.text = text
        self.month = month
        self.open: Open | None = None

    def feed(self, agg: Prefixes, events: ThetaEvents) -> Closed | None:
        """Agrega los eventos de `events` en el row group de `agg`.

        Devuelve los que cierran aquí (en orden) o `None` si ninguno. Como mucho el
        primero viene de un row group anterior (se completa con `self.open`) y como
        mucho el último sigue abierto al salir (queda en `self.open`).
        """
        n = len(events)
        l2 = L2Columns(events.events)
        seg = Segments(agg, events.positions, l2.confirm_time)
        stays = bool(events.ends_after[n - 1])
        closed = n - stays
        rows = []
        lo = 0
        if events.starts_before[0]:
            if self.open is None:
                raise RuntimeError(
                    f"θ {self.text}: un evento llega abierto de un row group anterior "
                    "y no hay acumulador parcial"
                )
            self.open.merge(*seg.pieces(0))
            lo = 1
            if closed:
                rows.append(self._row(l2, 0, self.open))
                self.open = None
        if stays and n - 1 >= lo:
            self.open = Open(*seg.pieces(n - 1))
        if closed > lo:
            rows.append(self._batch(seg, l2, lo, closed))
        if not closed:
            return None
        return Closed(
            np.concatenate(rows) if len(rows) > 1 else rows[0],
            int(l2.reference[0]),
            int(l2.extreme[closed - 1]),
            int(events.ticks_before_month[:closed].sum()),
        )

    def _row(self, l2: L2Columns, event: int, open_: Open) -> np.ndarray:
        """La fila de un evento completado con su acumulador parcial."""
        row = build_row(
            self.text,
            self.month,
            l2.row(event),
            open_.confirmation,
            open_.overshoot,
            open_.instant,
        )
        return np.array([row.values], np.int64)

    def _batch(self, seg: Segments, l2: L2Columns, lo: int, hi: int) -> np.ndarray:
        rows = _close_batch(seg, l2, lo, hi)
        if rows is not None:
            return rows
        # Algo no cuadra: se repite evento por evento hasta dar con el que falla.
        return np.array(
            [
                build_row(self.text, self.month, l2.row(i), *seg.pieces(i)).values
                for i in range(lo, hi)
            ],
            np.int64,
        )
