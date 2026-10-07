"""`ticks.bin`: deltas en varint (TRD-viz §7.3). Cada prueba compara con enteros de Python."""

import io
import struct
from datetime import date
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest
from viz_helpers import TICKS_SCHEMA, read_ticks_csv, ticks_batch
from viz_tiles.contract import (
    INT32_MAX,
    L1_SCALE,
    TICK_OUTSIDE,
    TICKS_CHUNK,
    TICKS_CHUNK_HEADER,
    price_scale,
)
from viz_tiles.ticks import (
    PriceUnrepresentable,
    TicksAccumulator,
    TicksReader,
    day_start_us,
    decode_ticks,
    decode_varint,
    encode_day,
    encode_ticks,
    encode_varint,
    unzigzag,
    zigzag,
)

DAY = date(2026, 8, 31)
SCALE = price_scale("BTCUSDT")

# (id, segundos desde el inicio del día, precio, cantidad)
ROWS = [
    (10, 1, 100, 1),
    (11, 2, 105, 2),
    (12, 3, 95, "0.5"),
    (13, 4, 101, "0.25"),
    (14, 30, 102, 1),
    (15, 70, 110, 1),
    (16, 80, 110, 1),
]


class Encoded:
    """Un día codificado en memoria: su resumen (`DayTicks`) y los bytes de `ticks.bin`."""

    def __init__(self, day_ticks, data):
        self.day_ticks = day_ticks
        self.data = data

    def __getattr__(self, name):
        return getattr(self.day_ticks, name)

    def to_bytes(self):
        return self.data


def encode(batches, day=DAY, scale=SCALE, chunk=TICKS_CHUNK):
    out = io.BytesIO()
    return Encoded(encode_day(batches, day, scale, out, chunk), out.getvalue())


def chunks_of(data: bytes) -> list[tuple[int, list[int]]]:
    """Los tramos de `ticks.bin`: `(ticks, bytes de cada sección)`, sin decodificar."""
    size = struct.calcsize(TICKS_CHUNK_HEADER)
    out, pos = [], 0
    while pos < len(data):
        count, *sections = struct.unpack_from(TICKS_CHUNK_HEADER, data, pos)
        out.append((count, sections))
        pos += size + sum(sections)
    return out


def decoded(day_ticks):
    return decode_ticks(day_ticks.to_bytes(), day_ticks.ticks)


def test_varint_matches_the_leb128_reference_bytes():
    values = np.array([0, 1, 127, 128, 300, 16383, 16384, 2**32, 2**63], np.uint64)
    expected = bytes(
        [0, 1, 127, 0x80, 1, 0xAC, 2, 0xFF, 0x7F, 0x80, 0x80, 1]
        + [0x80, 0x80, 0x80, 0x80, 0x10]
        + [0x80] * 9
        + [1]
    )
    assert encode_varint(values).tobytes() == expected
    back, size = decode_varint(np.frombuffer(expected, np.uint8), len(values))
    assert size == len(expected) and back.tolist() == values.tolist()


def test_varint_round_trip_of_random_values_of_every_width():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 64, 5000)
    values = rng.integers(0, 2**63, 5000, dtype=np.uint64) >> bits.astype(np.uint64)
    back, size = decode_varint(encode_varint(values), len(values))
    assert back.tolist() == values.tolist()
    assert size == len(encode_varint(values))


def test_varint_of_nothing_is_nothing():
    assert len(encode_varint(np.empty(0, np.uint64))) == 0
    assert decode_varint(np.empty(0, np.uint8), 0)[1] == 0


def test_truncated_varint_is_rejected():
    with pytest.raises(ValueError, match="varint"):
        decode_varint(np.array([0x80, 0x80], np.uint8), 1)


def test_zigzag_orders_small_magnitudes_first():
    deltas = np.array([0, -1, 1, -2, 2, -(2**31), 2**31 - 1])
    assert zigzag(deltas).tolist() == [0, 1, 2, 3, 4, 2**32 - 1, 2**32 - 2]
    assert unzigzag(zigzag(deltas)).tolist() == deltas.tolist()


def test_hand_computed_day():
    out = encode([ticks_batch(DAY, ROWS)])
    assert out.ticks == 7 and out.price_scale == SCALE
    assert (out.first_agg_trade_id, out.last_agg_trade_id) == (10, 16)
    time, price, quantity = (a.tolist() for a in vars(decoded(out)).values())
    # Tiempo en ms desde el inicio del día; precio en unidades de tick (1/100 USDT);
    # cantidad en 10⁻⁸.
    assert time == [1000, 2000, 3000, 4000, 30_000, 70_000, 80_000]
    assert price == [10_000, 10_500, 9_500, 10_100, 10_200, 11_000, 11_000]
    assert quantity == [
        100_000_000,
        200_000_000,
        50_000_000,
        25_000_000,
        100_000_000,
        100_000_000,
        100_000_000,
    ]
    # Un solo tramo: cabecera y tres secciones de varint de los deltas, el primero
    # desde el inicio del día.
    dt = encode_varint(
        np.array([1000, 1000, 1000, 1000, 26_000, 40_000, 10_000], np.uint64)
    )
    dp = encode_varint(zigzag(np.array([10_000, 500, -1000, 600, 100, 800, 0])))
    qty = encode_varint(
        np.array(
            [100_000_000, 200_000_000, 50_000_000, 25_000_000] + [100_000_000] * 3,
            np.uint64,
        )
    )
    header = struct.pack(TICKS_CHUNK_HEADER, 7, len(dt), len(dp), len(qty))
    assert out.to_bytes() == header + dt.tobytes() + dp.tobytes() + qty.tobytes()
    assert out.chunk == TICKS_CHUNK


def test_time_truncates_microseconds_to_the_millisecond():
    out = encode([ticks_batch(DAY, [(1, 1.000999, 100, 1)])])
    assert decoded(out).time_ms.tolist() == [1000]


def test_ticks_of_the_same_instant_keep_their_order_and_a_zero_delta():
    rows = [
        (1, 5.0, 100, 1),
        (2, 5.0004, 101, 2),
        (3, 5.0009, 99, 3),
        (4, 5.001, 100, 1),
    ]
    out = encode([ticks_batch(DAY, rows)])
    got = decoded(out)
    assert got.time_ms.tolist() == [5000, 5000, 5000, 5001]
    assert got.price.tolist() == [10_000, 10_100, 9_900, 10_000]


def test_batching_does_not_change_the_bytes():
    whole = encode([ticks_batch(DAY, ROWS)])
    # Un tick por lote: cada delta cruza la frontera de un lote.
    split = encode([ticks_batch(DAY, [r]) for r in ROWS])
    assert split.to_bytes() == whole.to_bytes()


def test_sliced_batch_is_read_with_its_offset():
    batch = ticks_batch(DAY, ROWS)
    sliced = encode([batch.slice(2, 3)])
    expected = encode([ticks_batch(DAY, ROWS[2:5])])
    assert sliced.to_bytes() == expected.to_bytes()
    assert (sliced.first_agg_trade_id, sliced.last_agg_trade_id) == (12, 14)


def test_ticks_outside_the_day_are_ignored():
    before = (1, -5, 999, 7)
    after = (99, 86_400, 999, 7)
    out = encode([ticks_batch(DAY, [before, *ROWS, after])])
    expected = encode([ticks_batch(DAY, ROWS)])
    assert out.ticks == len(ROWS)
    assert out.to_bytes() == expected.to_bytes()
    assert (out.first_agg_trade_id, out.last_agg_trade_id) == (10, 16)


def test_day_without_ticks_is_empty():
    out = encode([])
    assert out.ticks == 0 and out.nbytes == 0
    assert (out.first_agg_trade_id, out.last_agg_trade_id) == (-1, -1)


def test_price_is_the_exact_integer_of_the_decimal():
    out = encode([ticks_batch(DAY, [(1, 1, "4285.08", 1)])])
    assert decoded(out).price.tolist() == [428_508]
    assert out.rounded == 0


def test_quantity_above_32_bits_is_exact():
    # 1 000 BTC son 10¹¹ unidades de 10⁻⁸: no caben en 32 bits.
    out = encode([ticks_batch(DAY, [(1, 1, 100, 1000)])])
    assert decoded(out).quantity.tolist() == [100_000_000_000]


def test_price_off_the_tick_rounds_to_the_nearest_tick_half_to_even():
    # Distancias al tick de 0,01 en enteros de L1 (×10⁻⁸): 0,004 → 400 000.
    prices = ["100.004", "100.006", "100.015", "100.025", "100.00500001", "100.01"]
    expected = [10_000, 10_001, 10_002, 10_002, 10_001, 10_001]
    for price, tick in zip(prices, expected, strict=True):
        out = encode([ticks_batch(DAY, [(1, 1, price, 1)])])
        assert decoded(out).price.tolist() == [tick], price


def test_off_tick_prices_are_counted_and_the_day_is_still_written():
    rows = [(1, 1, "100.00", 1), (2, 2, "100.004", 1), (3, 3, "100.0049", 1)]
    out = encode([ticks_batch(DAY, rows)])
    assert out.rounded == 2 and out.ticks == 3
    # 100,0049 está a 490 000 (×10⁻⁸) del tick de 100,00: la mayor distancia.
    assert out.max_abs_delta_int == 490_000
    # El conteo no depende de cómo se parten los lotes.
    split = encode([ticks_batch(DAY, [r]) for r in rows])
    assert (split.rounded, split.max_abs_delta_int) == (2, 490_000)


def test_price_above_int32_is_unrepresentable():
    top = INT32_MAX // SCALE  # 21 474 836 USDT: el último entero que cabe
    ok = encode([ticks_batch(DAY, [(1, 1, top, 1)])])
    assert decoded(ok).price.tolist() == [top * SCALE]
    with pytest.raises(PriceUnrepresentable) as error:
        encode([ticks_batch(DAY, [(1, 1, top + 1, 1)])])
    assert error.value.price_scale == SCALE
    assert error.value.max_price_int == (top + 1) * L1_SCALE


def test_price_scale_must_divide_the_l1_scale():
    with pytest.raises(ValueError, match="price_scale"):
        TicksAccumulator(DAY, 30, io.BytesIO())


def test_unsorted_ticks_are_rejected():
    with pytest.raises(ValueError, match="ordenados"):
        encode([ticks_batch(DAY, [(2, 5, 100, 1), (1, 3, 100, 1)])])
    with pytest.raises(ValueError, match="ordenados"):
        encode(
            [ticks_batch(DAY, [(2, 5, 100, 1)]), ticks_batch(DAY, [(1, 3, 100, 1)])],
            DAY,
            SCALE,
        )


def test_other_types_are_rejected():
    batch = pa.record_batch(
        {
            "agg_trade_id": [1],
            "price": [100.0],
            "quantity": pa.array([Decimal(1)], pa.decimal128(18, 8)),
            "transact_time": [day_start_us(DAY)],
        }
    )
    with pytest.raises(ValueError, match="decimal128"):
        encode([batch])


def test_extra_or_missing_bytes_are_rejected():
    out = encode([ticks_batch(DAY, ROWS)])
    raw = out.to_bytes()
    with pytest.raises(ValueError, match="cabecera"):
        decode_ticks(raw + b"\x00", out.ticks)
    with pytest.raises(ValueError, match="truncado"):
        decode_ticks(raw[:-1], out.ticks)
    with pytest.raises(ValueError, match="se esperaban"):
        decode_ticks(raw, out.ticks + 1)
    # La cabecera declara más bytes de los que ocupan sus valores.
    count, dt, dprice, qty = struct.unpack_from(TICKS_CHUNK_HEADER, raw)
    wide = struct.pack(TICKS_CHUNK_HEADER, count, dt + 1, dprice, qty)
    padded = wide + raw[16 : 16 + dt] + b"\x00" + raw[16 + dt :]
    with pytest.raises(ValueError, match="ocupa"):
        decode_ticks(padded, out.ticks)
    with pytest.raises(ValueError, match="no cabe"):
        decode_ticks(raw, out.ticks, chunk=count - 1)


def real_day_batches(size=1000):
    """Los ticks de `ticks.csv` en lotes, con cantidad sintética y determinista."""
    rows = read_ticks_csv()
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        yield pa.record_batch(
            [
                pa.array([int(r["agg_trade_id"]) for r in chunk], pa.int64()),
                pa.array([Decimal(r["price"]) for r in chunk], pa.decimal128(18, 8)),
                pa.array(
                    [Decimal(int(r["agg_trade_id"]) % 7 + 1) / 100 for r in chunk],
                    pa.decimal128(18, 8),
                ),
                pa.array([int(r["transact_time"]) for r in chunk], pa.int64()),
            ],
            schema=TICKS_SCHEMA,
        )


def test_real_day_decodes_to_the_ticks_of_l1_and_re_encodes_to_the_same_bytes():
    """Criterio de aceptación: los ticks decodificados son los de `ticks.csv` y la ida y vuelta es exacta."""
    day = date(2017, 8, 18)
    t0 = day_start_us(day)
    in_day = [
        r
        for r in read_ticks_csv()
        if 0 <= int(r["transact_time"]) - t0 < 86_400_000_000
    ]
    out = encode(real_day_batches(), day, SCALE)
    assert out.ticks == len(in_day) == 4735
    assert out.rounded == 0
    ids = [int(r["agg_trade_id"]) for r in in_day]
    assert (out.first_agg_trade_id, out.last_agg_trade_id) == (ids[0], ids[-1])

    got = decoded(out)
    assert got.time_ms.tolist() == [
        (int(r["transact_time"]) - t0) // 1000 for r in in_day
    ]
    assert got.price.tolist() == [int(Decimal(r["price"]) * SCALE) for r in in_day]
    assert got.quantity.tolist() == [
        (int(r["agg_trade_id"]) % 7 + 1) * 1_000_000 for r in in_day
    ]
    # Volver a codificar lo decodificado da los mismos bytes.
    again = encode_ticks(got.time_ms, got.price, got.quantity)
    assert again == out.to_bytes()
    # Y cabe en pocos bytes por tick: el del humo no llega a 8.
    assert out.nbytes / out.ticks < 8


def many_ticks(count: int):
    """`count` ticks sintéticos: tiempo creciente con repetidos, precio que sube y baja."""
    rng = np.random.default_rng(317)
    time_ms = np.sort(rng.integers(0, 86_400_000, count))
    price = 6_000_000 + np.cumsum(rng.integers(-50, 51, count))
    quantity = rng.integers(1, 400_000_000, count)
    return time_ms, price, quantity


def batches_of(time_ms, price, quantity, size):
    """Los ticks como lotes de L1 de `size` filas (precio en USDT con 2 decimales)."""
    t0 = day_start_us(DAY)
    for start in range(0, len(time_ms), size):
        end = start + size
        yield pa.record_batch(
            [
                pa.array(np.arange(start + 1, min(end, len(time_ms)) + 1), pa.int64()),
                pa.array(
                    [Decimal(int(p)) / SCALE for p in price[start:end]],
                    pa.decimal128(18, 8),
                ),
                pa.array(
                    [Decimal(int(q)) / L1_SCALE for q in quantity[start:end]],
                    pa.decimal128(18, 8),
                ),
                pa.array(t0 + time_ms[start:end] * 1000, pa.int64()),
            ],
            schema=TICKS_SCHEMA,
        )


def test_a_day_of_more_than_one_chunk_round_trips_through_every_chunk_boundary():
    """Un día de más de un tramo: el primer tick de cada tramo es relativo al último del anterior."""
    count = 2 * TICKS_CHUNK + 1234
    time_ms, price, quantity = many_ticks(count)
    out = encode(batches_of(time_ms, price, quantity, 50_000))
    # Los tramos salen llenos, salvo el último.
    assert [n for n, _ in chunks_of(out.to_bytes())] == [
        TICKS_CHUNK,
        TICKS_CHUNK,
        1234,
    ]
    got = decoded(out)
    assert got.time_ms.tolist() == time_ms.tolist()
    assert got.price.tolist() == price.tolist()
    assert got.quantity.tolist() == quantity.tolist()
    # Ida y vuelta: volver a codificar lo decodificado da los mismos bytes.
    assert encode_ticks(got.time_ms, got.price, got.quantity) == out.to_bytes()


def test_the_first_tick_of_a_chunk_is_relative_to_the_last_of_the_previous_one():
    time_ms, price, quantity = many_ticks(10)
    out = encode(batches_of(time_ms, price, quantity, 10), chunk=4)
    sizes = chunks_of(out.to_bytes())
    assert [n for n, _ in sizes] == [4, 4, 2]
    # Los deltas son los mismos que sin partir: solo cambia dónde cae cada cabecera.
    whole = np.diff(time_ms, prepend=0)
    second = struct.calcsize(TICKS_CHUNK_HEADER) + sum(sizes[0][1])
    start = second + struct.calcsize(TICKS_CHUNK_HEADER)
    raw = np.frombuffer(out.to_bytes(), np.uint8)
    first_dt, _ = decode_varint(raw[start : start + sizes[1][1][0]], 4)
    assert first_dt.tolist() == whole[4:8].tolist()
    assert out.chunk == 4


def test_chunks_do_not_depend_on_how_the_batches_are_cut():
    time_ms, price, quantity = many_ticks(25)
    reference = encode(batches_of(time_ms, price, quantity, 25), chunk=8).to_bytes()
    for size in (1, 3, 8, 9):
        got = encode(batches_of(time_ms, price, quantity, size), chunk=8).to_bytes()
        assert got == reference, size


def test_a_chunk_is_written_as_soon_as_it_fills_up_and_released():
    """El acumulador no retiene el día: tras cada lote solo queda el tramo en curso."""
    time_ms, price, quantity = many_ticks(30)
    out = io.BytesIO()
    acc = TicksAccumulator(DAY, SCALE, out, chunk=8)
    written = []
    for batch in batches_of(time_ms, price, quantity, 5):
        acc.update(batch)
        written.append(len(out.getvalue()))
    # 30 ticks en tramos de 8: tres tramos escritos antes de `finish`, el cuarto (6) después.
    assert [n for n, _ in chunks_of(out.getvalue())] == [8, 8, 8]
    assert written[0] == 0 and written == sorted(written)
    acc.finish()
    assert [n for n, _ in chunks_of(out.getvalue())] == [8, 8, 8, 6]


def test_chunk_size_must_fit_a_uint32():
    with pytest.raises(ValueError, match="uint32"):
        TicksAccumulator(DAY, SCALE, io.BytesIO(), chunk=0)


# -- posición de un tick por su agg_trade_id (ITSC-319) ---------------------------------


def _ids_day(ids, split=None):
    """El `DayTicks` de ticks de un por segundo con esos ids, partidos en lotes de `split`."""
    rows = [(i, k + 1, 100 + k, 1) for k, i in enumerate(ids)]
    step = split or len(rows)
    return encode(
        [ticks_batch(DAY, rows[a : a + step]) for a in range(0, len(rows), step)]
    )


def test_ids_without_gaps_are_one_run_and_the_position_is_the_offset_from_the_first():
    out = _ids_day([30, 31, 32, 33, 34])
    assert out.id_runs == ((0, 30),)
    got = out.tick_positions(np.array([30, 32, 34]))
    assert got.tolist() == [0, 2, 4] and got.dtype == np.dtype("<u4")


def test_a_gap_of_the_provider_starts_a_run_and_positions_skip_it():
    out = _ids_day([10, 11, 12, 20, 21, 30])
    assert out.id_runs == ((0, 10), (3, 20), (5, 30))
    ids = np.array([10, 12, 20, 21, 30])
    assert out.tick_positions(ids).tolist() == [0, 2, 3, 4, 5]
    # Un id dentro del día que ningún tick tiene (cae en un hueco) no se adivina.
    for hole in (13, 19, 22, 29):
        with pytest.raises(ValueError, match="no es un tick"):
            out.tick_positions(np.array([hole]))


def test_ids_outside_the_day_get_the_sentinel():
    out = _ids_day([10, 11, 12])
    got = out.tick_positions(np.array([9, 10, 12, 13, 2**40]))
    assert got.tolist() == [TICK_OUTSIDE, 0, 2, TICK_OUTSIDE, TICK_OUTSIDE]
    assert encode([]).tick_positions(np.array([1])).tolist() == [TICK_OUTSIDE]


@pytest.mark.parametrize("split", [1, 2, 3, 4])
def test_the_runs_do_not_depend_on_how_the_batches_are_split(split):
    ids = [10, 11, 12, 20, 21, 30, 31, 32, 40]
    assert _ids_day(ids, split).id_runs == _ids_day(ids).id_runs


def test_ids_that_do_not_grow_are_rejected():
    with pytest.raises(ValueError, match="crecer"):
        _ids_day([10, 11, 11])
    with pytest.raises(ValueError, match="no supera"):
        _ids_day([10, 12, 9], split=2)


# -- relectura de tramos sueltos (TicksReader) --------------------------------


def reader_of(batches, chunk: int):
    """`(DayTicks, TicksReader, bytes)` de un día escrito con tramos de `chunk` ticks."""
    out = io.BytesIO()
    day_ticks = encode_day(batches, DAY, SCALE, out, chunk)
    data = out.getvalue()
    reads = []

    def read(offset: int, size: int) -> bytes:
        reads.append((offset, size))
        return data[offset : offset + size]

    return day_ticks, TicksReader(day_ticks, read), data, reads


def test_the_writer_notes_where_every_chunk_starts_and_with_what_state():
    rows = [(i + 1, 1 + i // 4, 100 + (i * 7) % 13, 1) for i in range(23)]
    day_ticks, _, data, _ = reader_of([ticks_batch(DAY, rows)], chunk=5)
    decoded = decode_ticks(data, len(rows), chunk=5)
    assert [c[0] for c in day_ticks.chunks] == [0, 5, 10, 15, 20]
    for first, _, time0, price0 in day_ticks.chunks[1:]:
        # Arranca desde el último tick del tramo anterior.
        assert (time0, price0) == (decoded.time_ms[first - 1], decoded.price[first - 1])
    assert day_ticks.chunks[0][1:] == (0, 0, 0)
    assert day_ticks.chunks[-1][1] < day_ticks.nbytes


def test_a_chunk_is_reread_alone_and_decodes_like_the_whole_file():
    rows = [(i + 1, 1 + i // 3, 100 + (i * 7) % 13, 1) for i in range(40)]
    day_ticks, reader, data, reads = reader_of([ticks_batch(DAY, rows)], chunk=8)
    whole = decode_ticks(data, len(rows), chunk=8)
    # Cada tramo, de a uno y en desorden, sin tocar los demás.
    for index in (3, 0, 4, 2, 1):
        reads.clear()
        time_ms, price = reader._decode(index)
        first = day_ticks.chunks[index][0]
        assert time_ms.tolist() == whole.time_ms[first : first + 8].tolist()
        assert price.tolist() == whole.price[first : first + 8].tolist()
        assert len(reads) == 1


def test_first_at_price_picks_the_first_tick_of_the_instant_with_that_price():
    # Un instante (el segundo 5) con tres ticks: 100, 110, 100; ids 14, 15, 16.
    rows = [
        (11, 1, 90, 1),
        (12, 2, 95, 1),
        (13, 4, 96, 1),
        (14, 5, 100, 1),
        (15, 5, 110, 1),
        (16, 5, 100, 1),
        (17, 6, 100, 1),
    ]
    _, reader, _, _ = reader_of([ticks_batch(DAY, rows)], chunk=100)
    last = np.array([5], dtype="<u4")  # el último tick del instante, el id 16
    ref = np.array([0], dtype="<u4")  # la referencia, antes del instante
    price = lambda p: np.array([p * L1_SCALE], dtype=np.int64)
    # Con 110 la confirmación pasa al id 15, el único del instante con ese precio.
    assert reader.first_at_price(last, price(110), ref).tolist() == [4]
    # Con 100 gana el primero de los dos, el id 14, no el último.
    assert reader.first_at_price(last, price(100), ref).tolist() == [3]
    # Un tick solo en su instante que ya tiene el precio no se mueve, aunque otro lo repita después.
    assert reader.first_at_price(np.array([6], "<u4"), price(100), ref).tolist() == [6]
    # Fuera del día no hay tick: el centinela se deja igual.
    outside = np.array([TICK_OUTSIDE, 5], dtype="<u4")
    both = np.array([100 * L1_SCALE, 100 * L1_SCALE], dtype=np.int64)
    refs = np.array([0, 0], dtype="<u4")
    assert reader.first_at_price(outside, both, refs).tolist() == [TICK_OUTSIDE, 3]
    # Una referencia fuera del día no acota: se busca desde el inicio del instante.
    assert reader.first_at_price(
        last, price(100), np.array([TICK_OUTSIDE], "<u4")
    ).tolist() == [3]
    # Un precio que ningún tick del instante tiene es una entrada rota.
    with pytest.raises(ValueError, match="no cuadra"):
        reader.first_at_price(last, price(120), ref)


def test_first_at_price_never_lands_before_the_reference_in_a_shared_millisecond():
    """Caso A/B/C en un mismo ms: A a 95, B el máximo 110 (referencia), C a 95 que confirma.

    L1 está en µs y el grupo de empate de L2 es el de un mismo µs (ADR-L2-03): A es de
    otro µs, anterior a la referencia, y la confirmación no puede quedar en él.
    """
    rows = [
        (1, 1, 100, 1),
        (2, 5.0001, 95, 1),  # A
        (3, 5.0002, 110, 1),  # B, la referencia del evento bajista
        (4, 5.0003, 95, 1),  # C, el tick que confirma
        (5, 6, 100, 1),
    ]
    _, reader, data, _ = reader_of([ticks_batch(DAY, rows)], chunk=100)
    decoded = decode_ticks(data, len(rows))
    assert len(set(decoded.time_ms[1:4].tolist())) == 1  # los tres en el mismo ms
    confirm = np.array([3], dtype="<u4")  # C, el `confirm_agg_trade_id` de L2
    reference = np.array([2], dtype="<u4")  # B
    got = reader.first_at_price(confirm, np.array([95 * L1_SCALE]), reference)
    assert got.tolist() == [3]
    assert reference[0] < got[0] <= confirm[0]


def test_first_at_price_stops_at_the_reference_in_an_earlier_chunk():
    # Seis ticks en el segundo 3, de a tres por tramo; 50 está antes y después de la referencia.
    rows = [(1, 1, 10, 1)]
    rows += [(2 + k, 3, (50, 70, 50, 60, 61, 62)[k], 1) for k in range(6)]
    _, reader, _, _ = reader_of([ticks_batch(DAY, rows)], chunk=3)
    last = np.array([6], dtype="<u4")
    # La referencia es la posición 2 (precio 70): el 50 de la posición 1 queda fuera.
    got = reader.first_at_price(last, np.array([50 * L1_SCALE]), np.array([2], "<u4"))
    assert got.tolist() == [3]


def test_first_at_price_follows_an_instant_that_crosses_chunks():
    # Nueve ticks en el segundo 3, de a tres por tramo; el precio 50 solo está en el primero.
    rows = [(1, 1, 10, 1), (2, 2, 20, 1)]
    rows += [(3 + k, 3, 50 if k == 0 else 60 + k, 1) for k in range(9)]
    rows += [(12, 4, 70, 1)]
    day_ticks, reader, _, _ = reader_of([ticks_batch(DAY, rows)], chunk=3)
    assert len(day_ticks.chunks) == 4
    last = np.array([10], dtype="<u4")  # el último tick del segundo 3 (id 11)
    ref = np.array([0], dtype="<u4")
    assert reader.first_at_price(last, np.array([50 * L1_SCALE]), ref).tolist() == [2]
    # Un precio que está solo en el último tramo no se busca más atrás.
    assert reader.first_at_price(last, np.array([68 * L1_SCALE]), ref).tolist() == [10]
