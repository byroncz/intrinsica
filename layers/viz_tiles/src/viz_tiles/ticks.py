"""Los ticks de un día en `ticks.bin`: deltas en varint por tramos, tal como los lee el navegador.

TRD-viz §7.3 y ADR-VZ-14. La vista dibuja lo que mide L1, a cualquier zoom: lo
que se dibuja es un tick o la envolvente exacta de los ticks de un píxel. Por eso
la página lleva todos los ticks del día, sin reducir, y el navegador deriva precio,
volumen y confirmaciones por píxel al dibujar.

Memoria: los ticks se leen una sola vez, lote a lote. Cada lote se codifica en el
tramo en curso y, cuando el tramo se llena (`TICKS_CHUNK` ticks, ≈ 330 KB), se
escribe al objeto y se suelta. El pico es O(lote) más un tramo, nunca O(día).
"""

import io
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import BinaryIO

import numpy as np
import pyarrow as pa

from viz_tiles.contract import (
    DAY_US,
    INT32_MAX,
    L1_SCALE,
    TICK_OUTSIDE,
    TICK_SECTIONS,
    TICKS_CHUNK,
    TICKS_CHUNK_HEADER,
)

# Un varint de uint64 ocupa a lo sumo 10 bytes.
_MAX_VARINT_BYTES = 10
_HEADER_BYTES = struct.calcsize(TICKS_CHUNK_HEADER)


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


def scaled_ints(column: pa.Array) -> np.ndarray:
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


def decode_ticks(
    data: bytes | memoryview, ticks: int, chunk: int = TICKS_CHUNK
) -> DecodedTicks:
    """Decodifica `ticks.bin` de `ticks` ticks en tramos de hasta `chunk`.

    Recorre los tramos en orden; el Δtiempo y el Δprecio del primer tick de cada uno
    son relativos al último del anterior. Lanza `ValueError` si no cuadra.
    """
    raw = np.frombuffer(data, dtype=np.uint8)
    pos = 0
    time_ms: list[np.ndarray] = []
    price: list[np.ndarray] = []
    quantity: list[np.ndarray] = []
    last_time = last_price = total = 0
    while pos < len(raw):
        if len(raw) - pos < _HEADER_BYTES:
            raise ValueError("cabecera de tramo truncada")
        count, *sizes = struct.unpack_from(TICKS_CHUNK_HEADER, raw, pos)
        pos += _HEADER_BYTES
        if not 0 < count <= chunk:
            raise ValueError(f"un tramo de {count} ticks no cabe en {chunk}")
        if sum(sizes) > len(raw) - pos:
            raise ValueError("tramo truncado")
        sections = []
        for size in sizes:
            values, used = decode_varint(raw[pos : pos + size], count)
            if used != size:
                raise ValueError(f"una sección ocupa {used} bytes y declara {size}")
            pos += size
            sections.append(values)
        dt, dprice, qty = sections
        time_ms.append(last_time + np.cumsum(dt.astype(np.int64)))
        price.append(last_price + np.cumsum(unzigzag(dprice)))
        quantity.append(qty.astype(np.int64))
        last_time, last_price = int(time_ms[-1][-1]), int(price[-1][-1])
        total += count
    if total != ticks:
        raise ValueError(f"se esperaban {ticks} ticks y hay {total}")
    empty = np.empty(0, dtype=np.int64)
    return DecodedTicks(
        time_ms=np.concatenate(time_ms) if time_ms else empty,
        price=np.concatenate(price) if price else empty,
        quantity=np.concatenate(quantity) if quantity else empty,
    )


def encode_ticks(
    time_ms: np.ndarray,
    price: np.ndarray,
    quantity: np.ndarray,
    chunk: int = TICKS_CHUNK,
) -> bytes:
    """`ticks.bin` de ticks completos, de una vez (para las pruebas y el sondeo).

    Es lo que `TicksAccumulator` hace por lotes: sirve para comprobar que el
    resultado no depende de cómo se parta el día.
    """
    out = io.BytesIO()
    writer = _ChunkWriter(out, chunk)
    writer.add(
        np.asarray(time_ms, np.int64),
        np.asarray(price, np.int64),
        np.asarray(quantity, np.int64),
    )
    writer.flush()
    return out.getvalue()


class _ChunkWriter:
    """Codifica los ticks en tramos y escribe cada tramo en cuanto se llena.

    Retiene a lo sumo un tramo (`chunk` ticks, ≈ 5 B por tick) y el último tiempo y
    precio, que son la referencia del siguiente tick. De cada tramo anota dónde
    empieza y con qué tiempo y precio arranca (`chunks`): son cuatro enteros por tramo y
    permiten releer uno suelto sin decodificar los anteriores (`TicksReader`).
    """

    def __init__(self, out: BinaryIO, chunk: int) -> None:
        if chunk < 1 or chunk > 2**32 - 1:
            raise ValueError(f"el tramo debe caber en un uint32, llegó {chunk}")
        self._out = out
        self.chunk = chunk
        self._sections = tuple(bytearray() for _ in TICK_SECTIONS)
        self._count = 0
        self._time = 0
        self._price = 0
        self.nbytes = 0
        self.chunks: list[tuple[int, int, int, int]] = []
        self._written = 0  # ticks de los tramos ya escritos
        self._start = (0, 0)  # tiempo y precio con que arranca el tramo en curso
        self._tail = (0, 0)  # tiempo y precio del último tick recibido

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
        values = (dt, zigzag(dprice), quantity)
        pos = 0
        while pos < len(dt):
            take = min(self.chunk - self._count, len(dt) - pos)
            for section, column in zip(self._sections, values, strict=True):
                section += encode_varint(column[pos : pos + take]).tobytes()
            self._count += take
            pos += take
            self._tail = (int(time_ms[pos - 1]), int(price[pos - 1]))
            if self._count == self.chunk:
                self.flush()

    def flush(self) -> None:
        """Escribe el tramo en curso (si lo hay) y lo suelta."""
        if not self._count:
            return
        self.chunks.append((self._written, self.nbytes, *self._start))
        header = struct.pack(
            TICKS_CHUNK_HEADER, self._count, *(len(s) for s in self._sections)
        )
        self._out.write(header)
        self.nbytes += len(header)
        for section in self._sections:
            self._out.write(section)
            self.nbytes += len(section)
            section.clear()
        self._written += self._count
        self._start = self._tail
        self._count = 0


@dataclass(frozen=True)
class DayTicks:
    """Lo que el índice y los hallazgos declaran de los ticks de un día.

    Los bytes de `ticks.bin` no viven aquí: salieron al objeto tramo a tramo.
    `nbytes` es su tamaño, `chunk` el tamaño máximo de un tramo, y
    `first_agg_trade_id` y `last_agg_trade_id` los de su primer y último tick.
    `rounded` y `max_abs_delta_int` alimentan el hallazgo `price_rounded` (ticks
    fuera del tick y su mayor distancia al tick más cercano, ×10⁻⁸).

    `id_runs` dice dónde está cada `agg_trade_id` sin guardar uno por tick: una
    tupla `(posición, id)` por cada tramo de ids consecutivos (un tramo si el día no
    tiene huecos; un tramo más por cada hueco del proveedor). Vacía, el día se
    toma como un solo tramo desde `first_agg_trade_id`.

    `chunks` dice dónde está cada tramo de `ticks.bin`: una tupla `(posición del
    primer tick, byte donde empieza, tiempo en ms y precio con que arranca)` por
    tramo. El tiempo y el precio de arranque son el último tick del tramo anterior
    (0 y 0 en el primero), de los que el tramo guarda solo el delta.
    """

    ticks: int
    chunk: int
    price_scale: int
    first_agg_trade_id: int
    last_agg_trade_id: int
    rounded: int
    max_abs_delta_int: int
    nbytes: int
    id_runs: tuple[tuple[int, int], ...] = ()
    chunks: tuple[tuple[int, int, int, int], ...] = ()

    def tick_positions(self, ids: np.ndarray) -> np.ndarray:
        """Posición en `ticks.bin` (`uint32`) de cada `agg_trade_id` de `ids`.

        Un id fuera del día, o sea anterior al primer tick o posterior al último,
        da `TICK_OUTSIDE`. Lanza `ValueError` si el id cae dentro del día pero en un
        hueco: ningún tick lo tiene, y un evento de L2 no puede apuntar ahí.
        """
        ids = np.asarray(ids, dtype=np.int64)
        out = np.full(len(ids), TICK_OUTSIDE, dtype="<u4")
        if not self.ticks:
            return out
        if self.ticks >= TICK_OUTSIDE:
            raise ValueError(f"{self.ticks} ticks no caben en una posición uint32")
        runs = self.id_runs or ((0, self.first_agg_trade_id),)
        starts = np.array([start for start, _ in runs], dtype=np.int64)
        first_ids = np.array([first for _, first in runs], dtype=np.int64)
        ends = np.append(starts[1:], self.ticks)
        inside = (ids >= self.first_agg_trade_id) & (ids <= self.last_agg_trade_id)
        run = np.searchsorted(first_ids, ids[inside], side="right") - 1
        position = starts[run] + (ids[inside] - first_ids[run])
        if np.any(position >= ends[run]):
            raise ValueError(
                "un evento apunta a un agg_trade_id que no es un tick del día"
            )
        out[inside] = position
        return out


class TicksAccumulator:
    """Consume lotes de ticks en orden y escribe `ticks.bin` del día, tramo a tramo.

    Cada lote es un `RecordBatch` con `agg_trade_id`, `price`, `quantity` y
    `transact_time`, con los tipos del contrato de L1, ordenado como el
    Parquet de L1 (por `transact_time`). Los ticks fuera del día se ignoran,
    así que se pueden pasar los row groups completos de un mes. `out` recibe los
    bytes del archivo; quien llama lo cierra después de `finish()`.
    """

    def __init__(
        self, day: date, price_scale: int, out: BinaryIO, chunk: int = TICKS_CHUNK
    ) -> None:
        self.day_start_us = day_start_us(day)
        self.price_scale = price_scale
        self._factor = _factor(price_scale)
        self.ticks = 0
        self._first_id: int | None = None
        self._last_id = 0
        self._id_runs: list[tuple[int, int]] = []
        # Ticks cuyo precio no cae en el tick, y la mayor distancia al tick más
        # cercano entre ellos (en enteros de L1, ×10⁻⁸): el hallazgo `price_rounded`.
        self.rounded = 0
        self.max_abs_delta_int = 0
        self._max_price_int = 0
        self._writer = _ChunkWriter(out, chunk)
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
        price = scaled_ints(batch.column("price"))[keep]
        quantity = scaled_ints(batch.column("quantity"))[keep]
        ids = batch.column("agg_trade_id").to_numpy(zero_copy_only=True)[keep]
        self._writer.add(
            rel[keep] // 1000, _round_to_tick(price, self._factor), quantity
        )
        self._track_ids(ids)
        self.ticks += len(price)
        self._max_price_int = max(self._max_price_int, int(price.max()))
        rest = price % self._factor
        off = rest != 0
        if off.any():
            self.rounded += int(off.sum())
            delta = np.minimum(rest, self._factor - rest)[off]
            self.max_abs_delta_int = max(self.max_abs_delta_int, int(delta.max()))

    def _track_ids(self, ids: np.ndarray) -> None:
        """Anota dónde empieza cada tramo de ids consecutivos (`DayTicks.id_runs`).

        Los ids de un día crecen estrictamente (TRD-viz §7.5): uno que no crece
        rompería la búsqueda de posiciones, y se rechaza. El tramo que arranca en
        el primer tick del lote se anota aparte, midiéndolo contra el último id
        del lote anterior.
        """
        steps = np.diff(ids)
        previous = self._last_id
        if self._first_id is not None and ids[0] <= previous:
            raise ValueError(
                f"agg_trade_id {int(ids[0])} no supera al anterior {previous}"
            )
        if (steps <= 0).any():
            raise ValueError("los agg_trade_id de un día deben crecer estrictamente")
        if self._first_id is None:
            self._first_id = int(ids[0])
            self._id_runs.append((self.ticks, int(ids[0])))
        elif ids[0] != previous + 1:
            self._id_runs.append((self.ticks, int(ids[0])))
        for b in np.flatnonzero(steps != 1) + 1:
            self._id_runs.append((self.ticks + int(b), int(ids[b])))
        self._last_id = int(ids[-1])

    def finish(self) -> DayTicks:
        """Escribe el último tramo y devuelve el día. Lanza `PriceUnrepresentable` si algún precio no cabe en `int32`."""
        self._writer.flush()
        if self.ticks:
            rounded_max = int(
                _round_to_tick(np.int64(self._max_price_int), self._factor)
            )
            if rounded_max > INT32_MAX:
                raise PriceUnrepresentable(self.price_scale, self._max_price_int)
        return DayTicks(
            ticks=self.ticks,
            chunk=self._writer.chunk,
            price_scale=self.price_scale,
            first_agg_trade_id=self._first_id if self._first_id is not None else -1,
            last_agg_trade_id=self._last_id if self.ticks else -1,
            rounded=self.rounded,
            max_abs_delta_int=self.max_abs_delta_int,
            nbytes=self._writer.nbytes,
            id_runs=tuple(self._id_runs),
            chunks=tuple(self._writer.chunks),
        )


def encode_day(
    batches: Iterable[pa.RecordBatch],
    day: date,
    price_scale: int,
    out: BinaryIO,
    chunk: int = TICKS_CHUNK,
) -> DayTicks:
    """Los ticks del día `day` de esos lotes, escritos en `out` como `ticks.bin`."""
    acc = TicksAccumulator(day, price_scale, out, chunk)
    for batch in batches:
        acc.update(batch)
    return acc.finish()


class TicksReader:
    """Relee tramos sueltos de un `ticks.bin` ya escrito, sin retener el día.

    `read(offset, size)` devuelve esos bytes del archivo. Un tramo se decodifica con
    el tiempo y el precio de arranque que `DayTicks.chunks` anotó al escribirlo, así
    que no hace falta recorrer los anteriores. En RAM nunca hay más de un tramo
    decodificado (≈ 1 MB), salvo el caso rarísimo de un instante que cruza tramos.
    """

    def __init__(self, ticks: DayTicks, read: Callable[[int, int], bytes]) -> None:
        self._ticks = ticks
        self._read = read
        self._firsts = np.array([c[0] for c in ticks.chunks], dtype=np.int64)
        self._factor = _factor(ticks.price_scale)

    def _decode(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """`(tiempo en ms, precio)` de los ticks del tramo `index`, como `int64`."""
        chunks = self._ticks.chunks
        _, start, time0, price0 = chunks[index]
        end = chunks[index + 1][1] if index + 1 < len(chunks) else self._ticks.nbytes
        raw = np.frombuffer(self._read(start, end - start), dtype=np.uint8)
        count, size_time, size_price, _ = struct.unpack_from(TICKS_CHUNK_HEADER, raw, 0)
        pos = _HEADER_BYTES
        dt, _ = decode_varint(raw[pos : pos + size_time], count)
        pos += size_time
        dprice, _ = decode_varint(raw[pos : pos + size_price], count)
        return (
            time0 + np.cumsum(dt.astype(np.int64)),
            price0 + np.cumsum(unzigzag(dprice)),
        )

    def first_at_price(
        self, positions: np.ndarray, price_int: np.ndarray, after: np.ndarray
    ) -> np.ndarray:
        """Para cada posición, el primer tick de su instante que tiene ese precio.

        `positions` son posiciones en `ticks.bin` (`uint32`, `TICK_OUTSIDE` si el
        punto cae fuera del día: se deja igual) y `price_int` los precios de L2 en
        enteros de 10⁻⁸. El instante de una posición son los ticks con su mismo
        milisegundo hasta ella: L2 cierra el grupo de empate en su último tick
        (ADR-L2-03) y el precio de la confirmación es el de uno de ellos. Con varios
        ticks al mismo precio gana el primero, o sea el de menor `agg_trade_id`; si
        es el único, la posición no cambia. Lanza `ValueError` si ningún
        tick del instante lo tiene: L2 y L1 no cuadran.

        `after` es la posición de la referencia de cada evento (`TICK_OUTSIDE` si cae
        fuera del día). El milisegundo puede traer ticks anteriores a la referencia, de
        otro µs, con el mismo precio; la búsqueda empieza en el tick siguiente a ella.
        Tras la referencia el umbral es fijo, así que el primer tick con ese precio en
        ese tramo es del grupo de L2.
        """
        out = np.array(positions, dtype="<u4")
        inside = np.flatnonzero(out != TICK_OUTSIDE)
        if not len(inside):
            return out
        at = out[inside].astype(np.int64)
        target = _round_to_tick(np.asarray(price_int, dtype=np.int64), self._factor)[
            inside
        ]
        reference = np.asarray(after, dtype=np.int64)[inside]
        floor = np.where(reference == TICK_OUTSIDE, 0, reference + 1)
        chunk_of = np.searchsorted(self._firsts, at, side="right") - 1
        for index in np.unique(chunk_of):
            mine = np.flatnonzero(chunk_of == index)
            time_ms, price = self._decode(int(index))
            local = at[mine] - self._firsts[index]
            group_start = np.searchsorted(time_ms, time_ms[local], side="left")
            # Solo se busca hacia atrás donde puede haber otro tick del instante o donde el
            # último no tiene el precio (el caso común, un tick solo en su instante, no pasa).
            walk = (
                (group_start < local)
                | (price[local] != target[mine])
                | ((group_start == 0) & (index > 0))
            )
            for k in mine[walk]:
                out[inside[k]] = self._walk_back(
                    int(index),
                    time_ms,
                    price,
                    int(at[k]),
                    int(target[k]),
                    int(floor[k]),
                )
        return out

    def _walk_back(
        self,
        index: int,
        time_ms: np.ndarray,
        price: np.ndarray,
        position: int,
        target: int,
        floor: int,
    ) -> int:
        """La posición del primer tick del instante de `position` con precio `target`, desde `floor`."""
        first = int(self._firsts[index])
        instant = time_ms[position - first]
        hi = position - first + 1
        found = None
        while True:
            lo = int(np.searchsorted(time_ms[:hi], instant, side="left"))
            start = max(lo, floor - first)
            hit = np.flatnonzero(price[start:hi] == target)
            if len(hit):
                found = first + start + int(hit[0])
            if lo > 0 or index == 0 or floor >= first:
                break
            # El instante sigue en el tramo anterior (un instante con decenas de miles de ticks).
            index -= 1
            time_ms, price = self._decode(index)
            if time_ms[-1] != instant:
                break
            first, hi = int(self._firsts[index]), len(time_ms)
        if found is None:
            raise ValueError(
                f"ningún tick del instante de la posición {position} tiene el precio "
                f"{target}: el evento de L2 no cuadra con L1"
            )
        return found
