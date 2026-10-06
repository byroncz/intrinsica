"""`ticks.bin`: deltas en varint (TRD-viz §7.3). Cada prueba compara con enteros de Python."""

from datetime import date
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest
from viz_helpers import TICKS_SCHEMA, read_ticks_csv, ticks_batch
from viz_tiles.contract import INT32_MAX, L1_SCALE, price_scale
from viz_tiles.ticks import (
    PriceUnrepresentable,
    TicksAccumulator,
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
    out = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
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
    # Las tres secciones son varint de los deltas: el primero desde el inicio del día.
    sections = [bytes(s) for s in out.sections]
    dt = [1000, 1000, 1000, 1000, 26_000, 40_000, 10_000]
    assert sections[0] == encode_varint(np.array(dt, np.uint64)).tobytes()
    dp = [10_000, 500, -1000, 600, 100, 800, 0]
    assert sections[1] == encode_varint(zigzag(np.array(dp))).tobytes()


def test_time_truncates_microseconds_to_the_millisecond():
    out = encode_day([ticks_batch(DAY, [(1, 1.000999, 100, 1)])], DAY, SCALE)
    assert decoded(out).time_ms.tolist() == [1000]


def test_ticks_of_the_same_instant_keep_their_order_and_a_zero_delta():
    rows = [
        (1, 5.0, 100, 1),
        (2, 5.0004, 101, 2),
        (3, 5.0009, 99, 3),
        (4, 5.001, 100, 1),
    ]
    out = encode_day([ticks_batch(DAY, rows)], DAY, SCALE)
    got = decoded(out)
    assert got.time_ms.tolist() == [5000, 5000, 5000, 5001]
    assert got.price.tolist() == [10_000, 10_100, 9_900, 10_000]


def test_batching_does_not_change_the_bytes():
    whole = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    # Un tick por lote: cada delta cruza la frontera de un lote.
    split = encode_day([ticks_batch(DAY, [r]) for r in ROWS], DAY, SCALE)
    assert split.to_bytes() == whole.to_bytes()
    assert [bytes(s) for s in split.sections] == [bytes(s) for s in whole.sections]


def test_sliced_batch_is_read_with_its_offset():
    batch = ticks_batch(DAY, ROWS)
    sliced = encode_day([batch.slice(2, 3)], DAY, SCALE)
    expected = encode_day([ticks_batch(DAY, ROWS[2:5])], DAY, SCALE)
    assert sliced.to_bytes() == expected.to_bytes()
    assert (sliced.first_agg_trade_id, sliced.last_agg_trade_id) == (12, 14)


def test_ticks_outside_the_day_are_ignored():
    before = (1, -5, 999, 7)
    after = (99, 86_400, 999, 7)
    out = encode_day([ticks_batch(DAY, [before, *ROWS, after])], DAY, SCALE)
    expected = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    assert out.ticks == len(ROWS)
    assert out.to_bytes() == expected.to_bytes()
    assert (out.first_agg_trade_id, out.last_agg_trade_id) == (10, 16)


def test_day_without_ticks_is_empty():
    out = encode_day([], DAY, SCALE)
    assert out.ticks == 0 and out.nbytes == 0
    assert (out.first_agg_trade_id, out.last_agg_trade_id) == (-1, -1)


def test_price_is_the_exact_integer_of_the_decimal():
    out = encode_day([ticks_batch(DAY, [(1, 1, "4285.08", 1)])], DAY, SCALE)
    assert decoded(out).price.tolist() == [428_508]
    assert out.rounded == 0


def test_quantity_above_32_bits_is_exact():
    # 1 000 BTC son 10¹¹ unidades de 10⁻⁸: no caben en 32 bits.
    out = encode_day([ticks_batch(DAY, [(1, 1, 100, 1000)])], DAY, SCALE)
    assert decoded(out).quantity.tolist() == [100_000_000_000]


def test_price_off_the_tick_rounds_to_the_nearest_tick_half_to_even():
    # Distancias al tick de 0,01 en enteros de L1 (×10⁻⁸): 0,004 → 400 000.
    prices = ["100.004", "100.006", "100.015", "100.025", "100.00500001", "100.01"]
    expected = [10_000, 10_001, 10_002, 10_002, 10_001, 10_001]
    for price, tick in zip(prices, expected, strict=True):
        out = encode_day([ticks_batch(DAY, [(1, 1, price, 1)])], DAY, SCALE)
        assert decoded(out).price.tolist() == [tick], price


def test_off_tick_prices_are_counted_and_the_day_is_still_written():
    rows = [(1, 1, "100.00", 1), (2, 2, "100.004", 1), (3, 3, "100.0049", 1)]
    out = encode_day([ticks_batch(DAY, rows)], DAY, SCALE)
    assert out.rounded == 2 and out.ticks == 3
    # 100,0049 está a 490 000 (×10⁻⁸) del tick de 100,00: la mayor distancia.
    assert out.max_abs_delta_int == 490_000
    # El conteo no depende de cómo se parten los lotes.
    split = encode_day([ticks_batch(DAY, [r]) for r in rows], DAY, SCALE)
    assert (split.rounded, split.max_abs_delta_int) == (2, 490_000)


def test_price_above_int32_is_unrepresentable():
    top = INT32_MAX // SCALE  # 21 474 836 USDT: el último entero que cabe
    ok = encode_day([ticks_batch(DAY, [(1, 1, top, 1)])], DAY, SCALE)
    assert decoded(ok).price.tolist() == [top * SCALE]
    with pytest.raises(PriceUnrepresentable) as error:
        encode_day([ticks_batch(DAY, [(1, 1, top + 1, 1)])], DAY, SCALE)
    assert error.value.price_scale == SCALE
    assert error.value.max_price_int == (top + 1) * L1_SCALE


def test_price_scale_must_divide_the_l1_scale():
    with pytest.raises(ValueError, match="price_scale"):
        TicksAccumulator(DAY, 30)


def test_unsorted_ticks_are_rejected():
    with pytest.raises(ValueError, match="ordenados"):
        encode_day([ticks_batch(DAY, [(2, 5, 100, 1), (1, 3, 100, 1)])], DAY, SCALE)
    with pytest.raises(ValueError, match="ordenados"):
        encode_day(
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
        encode_day([batch], DAY, SCALE)


def test_extra_or_missing_bytes_are_rejected():
    out = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    raw = out.to_bytes()
    with pytest.raises(ValueError, match="sobran"):
        decode_ticks(raw + b"\x00", out.ticks)
    with pytest.raises(ValueError, match="varint"):
        decode_ticks(raw[:-1], out.ticks)


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
    out = encode_day(real_day_batches(), day, SCALE)
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
    # Volver a codificar lo decodificado da los mismos bytes, sección por sección.
    again = encode_ticks(got.time_ms, got.price, got.quantity)
    assert [a.tobytes() for a in again] == [bytes(s) for s in out.sections]
    # Y cabe en pocos bytes por tick: el del humo no llega a 8.
    assert out.nbytes / out.ticks < 8
