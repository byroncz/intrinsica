"""Reducción M4 de un día de ticks a los tiles de precio y volumen.

TRD-viz §7.3 y ADR-VZ-08. Los ticks se leen una sola vez, lote a lote, y solo
se guardan acumuladores por columna del nivel más fino (4 096 columnas, unos
cientos de KB): la RAM es O(lote) más O(columnas), nunca O(ticks). Los niveles
gruesos se derivan del fino sin releer nada, porque M4 es componible.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pyarrow as pa

from viz_tiles.contract import (
    DAY_MS,
    DAY_US,
    EMPTY_PRICE,
    FINEST,
    INT32_MAX,
    L1_SCALE,
    LEVELS,
)

_EMPTY = -1


class PriceUnrepresentable(ValueError):
    """El precio máximo del día no cabe en `int32` con el `price_scale` del activo.

    Guarda teórica (TRD-viz §9.3): con 100 el máximo es 21 474 836,47 USDT.
    Quien llama no escribe el día y emite el hallazgo `price_unrepresentable`.
    """

    def __init__(self, price_scale: int, max_price_int: int) -> None:
        super().__init__(
            f"precio máximo {max_price_int} (×10⁻⁸) no cabe en int32 "
            f"con price_scale {price_scale}"
        )
        self.price_scale = price_scale
        self.max_price_int = max_price_int


def _factor(price_scale: int) -> int:
    """Enteros de L1 (×10⁸) por unidad de precio del tile."""
    if price_scale < 1 or L1_SCALE % price_scale:
        raise ValueError(f"price_scale debe dividir 10⁸, llegó {price_scale}")
    return L1_SCALE // price_scale


def _round_to_tick(price_int: np.ndarray, factor: int) -> np.ndarray:
    """`price_int / factor` redondeado al entero más cercano, mitad al par."""
    quotient, rest = np.divmod(price_int, factor)
    up = (2 * rest > factor) | ((2 * rest == factor) & (quotient % 2 == 1))
    return quotient + up


def day_start_us(day: date) -> int:
    """Inicio del día UTC en µs desde la época: el origen del tiempo de los tiles."""
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return int(start.timestamp()) * 1_000_000


def _scaled_ints(column: pa.Array) -> np.ndarray:
    """Enteros exactos (valor × 10⁸) de una columna `decimal128(18, 8)`, sin copiar.

    Arrow guarda cada decimal128 como un entero de 16 bytes little-endian. Con
    18 dígitos el valor cabe en int64 y su palabra baja es el entero completo.
    """
    kind = column.type
    if not pa.types.is_decimal128(kind) or kind.scale != 8 or kind.precision > 18:
        raise ValueError(f"se esperaba decimal128(18, 8), llegó {kind}")
    if column.null_count:
        raise ValueError("price y quantity no admiten nulos")
    words = np.frombuffer(column.buffers()[1], dtype="<i8")
    return words[2 * column.offset : 2 * (column.offset + len(column)) : 2]


@dataclass
class _Level:
    """Acumuladores M4 por columna de un nivel. Tiempos en µs relativos al día."""

    present: np.ndarray
    first_t: np.ndarray
    first_p: np.ndarray
    min_t: np.ndarray
    min_p: np.ndarray
    max_t: np.ndarray
    max_p: np.ndarray
    last_t: np.ndarray
    last_p: np.ndarray
    last_id: np.ndarray
    volume: np.ndarray

    @classmethod
    def empty(cls, w: int) -> _Level:
        fields = {
            name: np.zeros(w, dtype=np.int64)
            for name in cls.__dataclass_fields__
            if name not in ("present", "last_id")
        }
        return cls(
            present=np.zeros(w, dtype=bool),
            last_id=np.full(w, _EMPTY, np.int64),
            **fields,
        )

    def __len__(self) -> int:
        return len(self.present)

    def coarsen(self) -> _Level:
        """El nivel de la mitad de columnas: M4 de dos columnas contiguas.

        El primero es el de la primera columna no vacía, el último el de la
        última no vacía, y el mínimo y el máximo ignoran las columnas vacías.
        En un empate de precio gana la columna izquierda (el tick anterior).
        """
        left = {n: a[0::2] for n, a in vars(self).items()}
        right = {n: a[1::2] for n, a in vars(self).items()}
        lp, rp = left["present"], right["present"]
        pick: dict[str, np.ndarray] = {"present": lp | rp}
        for name in ("t", "p"):
            pick[f"first_{name}"] = np.where(
                lp, left[f"first_{name}"], right[f"first_{name}"]
            )
            pick[f"last_{name}"] = np.where(
                rp, right[f"last_{name}"], left[f"last_{name}"]
            )
        take_min = rp & (~lp | (right["min_p"] < left["min_p"]))
        take_max = rp & (~lp | (right["max_p"] > left["max_p"]))
        for name in ("t", "p"):
            pick[f"min_{name}"] = np.where(
                take_min, right[f"min_{name}"], left[f"min_{name}"]
            )
            pick[f"max_{name}"] = np.where(
                take_max, right[f"max_{name}"], left[f"max_{name}"]
            )
        pick["last_id"] = np.where(rp, right["last_id"], left["last_id"])
        pick["volume"] = left["volume"] + right["volume"]
        return _Level(**pick)


class M4Accumulator:
    """Consume lotes de ticks en orden y retiene solo los acumuladores del día.

    Cada lote es un `RecordBatch` con `agg_trade_id`, `price`, `quantity` y
    `transact_time`, con los tipos del contrato de L1, ordenado como el
    Parquet de L1 (por `transact_time`). Los ticks fuera del día se ignoran,
    así que se pueden pasar los row groups completos de un mes.
    """

    def __init__(self, day: date, price_scale: int) -> None:
        self.day_start_us = day_start_us(day)
        self.price_scale = price_scale
        self._factor = _factor(price_scale)
        self.ticks = 0
        # Ticks cuyo precio no cae en el tick, y la mayor distancia al tick más
        # cercano entre ellos (en enteros de L1, ×10⁻⁸): el hallazgo `price_rounded`.
        self.rounded = 0
        self.max_abs_delta_int = 0
        self._level = _Level.empty(FINEST)
        self._last_time = -(2**63)

    def update(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        times = batch.column("transact_time").to_numpy(zero_copy_only=True)
        if times[0] < self._last_time or np.any(times[1:] < times[:-1]):
            raise ValueError("los ticks deben llegar ordenados por transact_time")
        self._last_time = times[-1]
        rel = times - self.day_start_us
        inside = np.flatnonzero((rel >= 0) & (rel < DAY_US))
        if not len(inside):
            return
        # Los ticks del día son un tramo contiguo: el lote está ordenado.
        keep = slice(inside[0], inside[-1] + 1)
        price = _scaled_ints(batch.column("price"))[keep]
        quantity = _scaled_ints(batch.column("quantity"))[keep]
        ids = batch.column("agg_trade_id").to_numpy(zero_copy_only=True)[keep]
        self._add(rel[keep], price, quantity, ids)
        self.ticks += len(price)
        rest = price % self._factor
        off = rest != 0
        if off.any():
            self.rounded += int(off.sum())
            delta = np.minimum(rest, self._factor - rest)[off]
            self.max_abs_delta_int = max(self.max_abs_delta_int, int(delta.max()))

    def _add(self, rel, price, quantity, ids) -> None:
        col = rel * FINEST // DAY_US
        starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
        ends = np.r_[starts[1:], len(col)] - 1
        cols = col[starts]
        segment = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(col)]))

        def extreme(reduce):
            """Posición del primer tick que alcanza el extremo de cada tramo."""
            best = reduce.reduceat(price, starts)
            hits = np.flatnonzero(price == best[segment])
            first = np.unique(segment[hits], return_index=True)[1]
            return hits[first]

        lo, hi = extreme(np.minimum), extreme(np.maximum)
        acc = self._level
        new = ~acc.present[cols]
        acc.first_t[cols[new]] = rel[starts[new]]
        acc.first_p[cols[new]] = price[starts[new]]
        better_lo = new | (price[lo] < acc.min_p[cols])
        acc.min_t[cols[better_lo]] = rel[lo[better_lo]]
        acc.min_p[cols[better_lo]] = price[lo[better_lo]]
        better_hi = new | (price[hi] > acc.max_p[cols])
        acc.max_t[cols[better_hi]] = rel[hi[better_hi]]
        acc.max_p[cols[better_hi]] = price[hi[better_hi]]
        acc.last_t[cols] = rel[ends]
        acc.last_p[cols] = price[ends]
        acc.last_id[cols] = ids[ends]
        acc.volume[cols] += np.add.reduceat(quantity, starts)
        acc.present[cols] = True

    def finish(self) -> DayReduction:
        """Los tiles de todos los niveles. Suelta los acumuladores."""
        level, self._level = self._level, _Level.empty(FINEST)
        levels = {}
        for w in reversed(LEVELS):
            levels[w] = level
            if w != LEVELS[0]:
                level = level.coarsen()
        finest = levels[FINEST]
        if finest.present.any():
            max_price_int = int(finest.max_p[finest.present].max())
            rounded_max = int(_round_to_tick(np.int64(max_price_int), self._factor))
            if rounded_max > INT32_MAX:
                raise PriceUnrepresentable(self.price_scale, max_price_int)
        return DayReduction(
            ticks=self.ticks,
            price_scale=self.price_scale,
            rounded=self.rounded,
            max_abs_delta_int=self.max_abs_delta_int,
            price={w: _price_tile(lv, self._factor) for w, lv in levels.items()},
            volume={w: _volume_tile(lv) for w, lv in levels.items()},
            last_ids={w: lv.last_id for w, lv in levels.items()},
        )


@dataclass(frozen=True)
class DayReduction:
    """Tiles de precio y volumen de un día, por nivel.

    `price[w]` es el arreglo de `8w` int32 del archivo (bloque de tiempos en ms
    y bloque de precios en unidades de `1/price_scale`, `EMPTY_PRICE` en una
    columna vacía); `volume[w]`, de `w` float32. `last_ids[w]` es el
    `agg_trade_id` del último tick de cada columna (-1 si está vacía): con él
    se resuelve el estado de dirección (`direction.direction_tile`).
    `rounded` y `max_abs_delta_int` alimentan el hallazgo `price_rounded`
    (ticks fuera del tick y su mayor distancia al tick más cercano, ×10⁻⁸).
    """

    ticks: int
    price_scale: int
    rounded: int
    max_abs_delta_int: int
    price: dict[int, np.ndarray]
    volume: dict[int, np.ndarray]
    last_ids: dict[int, np.ndarray]


def reduce_day(
    batches: Iterable[pa.RecordBatch], day: date, price_scale: int
) -> DayReduction:
    """Reduce los lotes de ticks del día `day` a los tiles de precio y volumen.

    `price_scale` es el del activo (`contract.price_scale`). Lanza
    `PriceUnrepresentable` si el precio máximo no cabe en `int32`.
    """
    acc = M4Accumulator(day, price_scale)
    for batch in batches:
        acc.update(batch)
    return acc.finish()


def _price_tile(level: _Level, factor: int) -> np.ndarray:
    """Arma el arreglo `[t0..t(4w-1), p0..p(4w-1)]` de los puntos M4 en orden de tiempo.

    Los acumuladores traen el precio crudo (×10⁸); aquí se redondea una sola
    vez al tick, mitad al par.
    """
    w = len(level)
    min_first = level.min_t <= level.max_t
    points_t = np.stack(
        [
            level.first_t,
            np.where(min_first, level.min_t, level.max_t),
            np.where(min_first, level.max_t, level.min_t),
            level.last_t,
        ],
        axis=1,
    )
    points_p = np.stack(
        [
            level.first_p,
            np.where(min_first, level.min_p, level.max_p),
            np.where(min_first, level.max_p, level.min_p),
            level.last_p,
        ],
        axis=1,
    )
    empty = ~level.present
    # Columna vacía: t es el inicio de la columna (no decreciente, lo que exige
    # el eje X de uPlot) y p es el centinela.
    starts = np.arange(w, dtype=np.int64) * DAY_MS // w
    millis = np.where(empty[:, None], starts[:, None], points_t // 1000)
    prices = np.where(empty[:, None], EMPTY_PRICE, _round_to_tick(points_p, factor))
    return np.concatenate([millis.ravel(), prices.ravel()]).astype("<i4")


def _volume_tile(level: _Level) -> np.ndarray:
    """Suma de `quantity` por columna: se suma en enteros y se convierte al final."""
    return (level.volume / L1_SCALE).astype("<f4")
