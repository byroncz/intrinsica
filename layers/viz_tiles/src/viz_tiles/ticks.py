"""Los ticks de un día en `ticks.bin`: deltas en varint, tal como los lee el navegador.

TRD-viz §7.3 y ADR-VZ-14. La vista dibuja lo que mide L1, a cualquier zoom: lo
que se dibuja es un tick o la envolvente exacta de los ticks de un píxel. Por eso
la página lleva todos los ticks del día, sin reducir, y el navegador deriva precio,
volumen y confirmaciones por píxel al dibujar.

Memoria: los ticks se leen una sola vez, lote a lote, y cada lote se codifica y se
suelta. Lo único que se retiene del día son los bytes codificados (≈ 5 B por tick,
el propio `ticks.bin`): nunca conviven los ticks y su codificación.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pyarrow as pa

from viz_tiles.contract import DAY_US, INT32_MAX, L1_SCALE, TICK_SECTIONS

# Un varint de uint64 ocupa a lo sumo 10 bytes.
_MAX_VARINT_BYTES = 10


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
    """Enteros de L1 (×10⁸) por unidad de precio del archivo."""
    if price_scale < 1 or L1_SCALE % price_scale:
        raise ValueError(f"price_scale debe dividir 10⁸, llegó {price_scale}")
    return L1_SCALE // price_scale


def _round_to_tick(price_int: np.ndarray, factor: int) -> np.ndarray:
    """`price_int / factor` redondeado al entero más cercano, mitad al par."""
    quotient, rest = np.divmod(price_int, factor)
    up = (2 * rest > factor) | ((2 * rest == factor) & (quotient % 2 == 1))
    return quotient + up


def day_start_us(day: date) -> int:
    """Inicio del día UTC en µs desde la época: el origen del tiempo del día."""
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


def encode_varint(values: np.ndarray) -> np.ndarray:
    """Los enteros sin signo `values` como varint (LEB128), uno tras otro, en `uint8`.

    Vectorizado: cuenta los bytes de cada valor y escribe el byte `k` de todos los
    valores que lo tienen, de una pasada por `k`.
    """
    v = np.asarray(values, dtype=np.uint64)
    count = np.ones(len(v), dtype=np.int64)
    for k in range(1, _MAX_VARINT_BYTES):
        more = (v >> np.uint64(7 * k)) != 0
        if not more.any():
            break
        count += more
    ends = np.cumsum(count)
    out = np.empty(int(ends[-1]) if len(v) else 0, dtype=np.uint8)
    starts = ends - count
    for k in range(int(count.max()) if len(v) else 0):
        has = count > k
        byte = ((v[has] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        out[starts[has] + k] = byte | (np.uint8(0x80) * (count[has] > k + 1))
    return out


def decode_varint(data: np.ndarray, count: int) -> tuple[np.ndarray, int]:
    """Los primeros `count` enteros varint de `data` (`uint8`) y los bytes que ocuparon."""
    last = np.flatnonzero(data < 0x80)[:count]
    if len(last) < count:
        raise ValueError(f"se esperaban {count} enteros varint y hay {len(last)}")
    size = int(last[-1]) + 1 if count else 0
    head = data[:size]
    starts = np.r_[0, last[:-1] + 1] if count else np.empty(0, dtype=np.int64)
    length = last - starts + 1
    if count and length.max() > _MAX_VARINT_BYTES:
        raise ValueError("un varint de más de 10 bytes no cabe en un uint64")
    shift = (np.arange(size) - np.repeat(starts, length)).astype(np.uint64) * np.uint64(
        7
    )
    parts = (head & 0x7F).astype(np.uint64) << shift
    return (np.add.reduceat(parts, starts) if count else parts), size


def zigzag(delta: np.ndarray) -> np.ndarray:
    """Enteros con signo a sin signo: 0, -1, 1, -2… van a 0, 1, 2, 3…"""
    d = np.asarray(delta, dtype=np.int64)
    return ((d << 1) ^ (d >> 63)).astype(np.uint64)


def unzigzag(value: np.ndarray) -> np.ndarray:
    z = np.asarray(value, dtype=np.uint64)
    return (z >> np.uint64(1)).astype(np.int64) ^ -(z & np.uint64(1)).astype(np.int64)


@dataclass(frozen=True)
class DecodedTicks:
    """Los ticks de `ticks.bin` ya decodificados, para las pruebas y el sondeo.

    `time_ms` son milisegundos desde el inicio del día, `price` unidades de
    `1/price_scale` y `quantity` unidades de 10⁻⁸.
    """

    time_ms: np.ndarray
    price: np.ndarray
    quantity: np.ndarray


def decode_ticks(data: bytes | memoryview, ticks: int) -> DecodedTicks:
    """Decodifica `ticks.bin` de `ticks` ticks. Lanza `ValueError` si no cuadra."""
    raw = np.frombuffer(data, dtype=np.uint8)
    pos = 0
    sections = []
    for _ in TICK_SECTIONS:
        values, size = decode_varint(raw[pos:], ticks)
        pos += size
        sections.append(values)
    if pos != len(raw):
        raise ValueError(f"sobran {len(raw) - pos} bytes después de la tercera sección")
    dt, dprice, quantity = sections
    return DecodedTicks(
        time_ms=np.cumsum(dt.astype(np.int64)),
        price=np.cumsum(unzigzag(dprice)),
        quantity=quantity.astype(np.int64),
    )


def encode_ticks(
    time_ms: np.ndarray, price: np.ndarray, quantity: np.ndarray
) -> list[np.ndarray]:
    """Las tres secciones de `ticks.bin` (en `uint8`) de ticks completos, de una vez.

    Es lo que `TicksAccumulator` hace por lotes; sirve para comprobar que el
    resultado no depende de cómo se parta el día.
    """
    acc = _Sections()
    acc.add(
        np.asarray(time_ms, np.int64),
        np.asarray(price, np.int64),
        np.asarray(quantity, np.int64),
    )
    return [np.frombuffer(s, np.uint8) for s in acc.buffers]


class _Sections:
    """Las tres secciones en construcción y el último valor de cada una."""

    def __init__(self) -> None:
        self.buffers = tuple(bytearray() for _ in TICK_SECTIONS)
        self._time = 0
        self._price = 0

    def add(self, time_ms: np.ndarray, price: np.ndarray, quantity: np.ndarray) -> None:
        if not len(time_ms):
            return
        dt = np.diff(time_ms, prepend=self._time)
        if (dt < 0).any():
            raise ValueError("los ticks deben llegar ordenados por transact_time")
        if (quantity < 0).any():
            raise ValueError("quantity no admite valores negativos")
        dprice = np.diff(price, prepend=self._price)
        self._time, self._price = int(time_ms[-1]), int(price[-1])
        for buffer, values in zip(
            self.buffers, (dt, zigzag(dprice), quantity), strict=True
        ):
            buffer += encode_varint(values).tobytes()


@dataclass(frozen=True)
class DayTicks:
    """Los ticks de un día, codificados, y lo que el índice y los hallazgos declaran.

    `sections` son las tres secciones de `ticks.bin` en orden; juntas son el
    archivo. `first_agg_trade_id` y `last_agg_trade_id` son los de su primer y
    último tick. `rounded` y `max_abs_delta_int` alimentan el hallazgo
    `price_rounded` (ticks fuera del tick y su mayor distancia al tick más cercano,
    ×10⁻⁸).
    """

    ticks: int
    price_scale: int
    first_agg_trade_id: int
    last_agg_trade_id: int
    rounded: int
    max_abs_delta_int: int
    sections: tuple[bytearray, ...]

    @property
    def nbytes(self) -> int:
        return sum(len(s) for s in self.sections)

    def parts(self) -> list[memoryview]:
        """Los bytes de `ticks.bin`, sin copiarlos: una vista por sección."""
        return [memoryview(s) for s in self.sections]

    def to_bytes(self) -> bytes:
        """`ticks.bin` entero en un `bytes` (copia: para las pruebas, no para el job)."""
        return b"".join(self.sections)


class TicksAccumulator:
    """Consume lotes de ticks en orden y retiene solo `ticks.bin` del día.

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
        self._first_id: int | None = None
        self._last_id = 0
        # Ticks cuyo precio no cae en el tick, y la mayor distancia al tick más
        # cercano entre ellos (en enteros de L1, ×10⁻⁸): el hallazgo `price_rounded`.
        self.rounded = 0
        self.max_abs_delta_int = 0
        self._max_price_int = 0
        self._sections = _Sections()
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
        self._sections.add(
            rel[keep] // 1000, _round_to_tick(price, self._factor), quantity
        )
        if self._first_id is None:
            self._first_id = int(ids[0])
        self._last_id = int(ids[-1])
        self.ticks += len(price)
        self._max_price_int = max(self._max_price_int, int(price.max()))
        rest = price % self._factor
        off = rest != 0
        if off.any():
            self.rounded += int(off.sum())
            delta = np.minimum(rest, self._factor - rest)[off]
            self.max_abs_delta_int = max(self.max_abs_delta_int, int(delta.max()))

    def finish(self) -> DayTicks:
        """El día codificado. Lanza `PriceUnrepresentable` si algún precio no cabe en `int32`."""
        if self.ticks:
            rounded_max = int(
                _round_to_tick(np.int64(self._max_price_int), self._factor)
            )
            if rounded_max > INT32_MAX:
                raise PriceUnrepresentable(self.price_scale, self._max_price_int)
        return DayTicks(
            ticks=self.ticks,
            price_scale=self.price_scale,
            first_agg_trade_id=self._first_id if self._first_id is not None else -1,
            last_agg_trade_id=self._last_id if self.ticks else -1,
            rounded=self.rounded,
            max_abs_delta_int=self.max_abs_delta_int,
            sections=self._sections.buffers,
        )


def encode_day(
    batches: Iterable[pa.RecordBatch], day: date, price_scale: int
) -> DayTicks:
    """Los ticks del día `day` de esos lotes, codificados como `ticks.bin`."""
    acc = TicksAccumulator(day, price_scale)
    for batch in batches:
        acc.update(batch)
    return acc.finish()
