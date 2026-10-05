from datetime import date
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest
from viz_helpers import TICKS_SCHEMA, read_ticks_csv, ticks_batch
from viz_tiles.contract import LEVELS
from viz_tiles.reduce import day_start_us, reduce_day

DAY = date(2026, 8, 31)

# (id, segundos desde el inicio del día, precio, cantidad). A w = 4096 la
# columna dura 21,09375 s: los ticks caen en las columnas 0, 1 y 3.
ROWS = [
    (10, 1, 100, 1),
    (11, 2, 105, 2),
    (12, 3, 95, "0.5"),
    (13, 4, 101, "0.25"),
    (14, 30, 102, 1),
    (15, 70, 110, 1),
    (16, 80, 110, 1),
]
NAN = np.nan


def points(tile: np.ndarray, col: int) -> tuple[list[float], list[float]]:
    """Tiempos y precios de los cuatro puntos M4 de la columna `col`."""
    n = len(tile) // 2
    return (
        tile[4 * col : 4 * col + 4].tolist(),
        tile[n + 4 * col : n + 4 * col + 4].tolist(),
    )


def test_hand_computed_day_at_finest_level():
    out = reduce_day([ticks_batch(DAY, ROWS)], DAY)
    price = out.price[4096]
    assert price.dtype == np.dtype("<f4") and price.shape == (8 * 4096,)
    # Columna 0: primero (1, 100), máximo (2, 105), mínimo (3, 95), último (4, 101).
    assert points(price, 0) == ([1, 2, 3, 4], [100, 105, 95, 101])
    # Columna 1: un solo tick, el punto se repite cuatro veces.
    assert points(price, 1) == ([30] * 4, [102] * 4)
    # Columna 2: vacía. t es el inicio de la columna y el precio es NaN.
    t, p = points(price, 2)
    assert t == [2 * 21.09375] * 4 and np.isnan(p).all()
    # Columna 3: precio plano. Los empates dejan el primer tick como mínimo y máximo.
    assert points(price, 3) == ([70, 70, 70, 80], [110] * 4)
    assert out.volume[4096][:4].tolist() == [3.75, 1, 0, 2]
    assert out.last_ids[4096][:4].tolist() == [13, 14, -1, 16]
    assert out.ticks == len(ROWS)


def test_hand_computed_day_at_coarsest_level():
    out = reduce_day([ticks_batch(DAY, ROWS)], DAY)
    # A w = 128 la columna 0 (675 s) tiene los siete ticks.
    assert points(out.price[128], 0) == ([1, 3, 70, 80], [100, 95, 110, 110])
    assert out.volume[128][0] == 6.75
    assert out.last_ids[128][0] == 16
    t, p = points(out.price[128], 1)
    assert t == [675.0] * 4 and np.isnan(p).all()
    assert out.volume[128][1] == 0
    assert out.last_ids[128][1] == -1


def test_time_is_finite_and_non_decreasing_in_every_level():
    out = reduce_day([ticks_batch(DAY, ROWS)], DAY)
    for w in LEVELS:
        times = out.price[w][: 4 * w]
        assert np.isfinite(times).all()
        assert (np.diff(times) >= 0).all()


def test_batching_does_not_change_the_result():
    whole = reduce_day([ticks_batch(DAY, ROWS)], DAY)
    # Un tick por lote: la columna 0 se completa a través de cuatro lotes.
    split = reduce_day([ticks_batch(DAY, [r]) for r in ROWS], DAY)
    for w in LEVELS:
        np.testing.assert_array_equal(whole.price[w], split.price[w])
        np.testing.assert_array_equal(whole.volume[w], split.volume[w])
        np.testing.assert_array_equal(whole.last_ids[w], split.last_ids[w])


def test_sliced_batch_is_read_with_its_offset():
    batch = ticks_batch(DAY, ROWS)
    whole = reduce_day([batch.slice(2, 3)], DAY)
    expected = reduce_day([ticks_batch(DAY, ROWS[2:5])], DAY)
    np.testing.assert_array_equal(whole.price[4096], expected.price[4096])


def test_ticks_outside_the_day_are_ignored():
    before = (1, -5, 999, 7)
    after = (99, 86_400, 999, 7)
    out = reduce_day([ticks_batch(DAY, [before, *ROWS, after])], DAY)
    expected = reduce_day([ticks_batch(DAY, ROWS)], DAY)
    assert out.ticks == len(ROWS)
    np.testing.assert_array_equal(out.price[4096], expected.price[4096])
    np.testing.assert_array_equal(out.volume[4096], expected.volume[4096])


def test_day_without_ticks_is_all_nan():
    out = reduce_day([], DAY)
    assert out.ticks == 0
    assert np.isnan(out.price[128][4 * 128 :]).all()
    assert not out.volume[128].any()
    assert (out.last_ids[128] == -1).all()


def test_price_is_float32_of_the_exact_decimal():
    out = reduce_day([ticks_batch(DAY, [(1, 1, "4285.08", 1)])], DAY)
    assert out.price[4096][4 * 4096] == np.float32(4285.08)


def test_volume_sums_exact_integers_before_converting():
    # 0,1 + 0,2 en float64 da 0,30000000000000004; en enteros da 0,3 exacto.
    rows = [(1, 1, 100, "0.1"), (2, 2, 100, "0.2")]
    out = reduce_day([ticks_batch(DAY, rows)], DAY)
    assert out.volume[4096][0] == np.float32(0.3)


def test_unsorted_ticks_are_rejected():
    with pytest.raises(ValueError, match="ordenados"):
        reduce_day([ticks_batch(DAY, [(2, 5, 100, 1), (1, 3, 100, 1)])], DAY)
    with pytest.raises(ValueError, match="ordenados"):
        reduce_day(
            [ticks_batch(DAY, [(2, 5, 100, 1)]), ticks_batch(DAY, [(1, 3, 100, 1)])],
            DAY,
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
        reduce_day([batch], DAY)


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


def test_real_day_invariants():
    day = date(2017, 8, 18)
    t0 = day_start_us(day)
    in_day = [
        (int(r["agg_trade_id"]), Decimal(r["price"]), int(r["agg_trade_id"]) % 7 + 1)
        for r in read_ticks_csv()
        if 0 <= int(r["transact_time"]) - t0 < 86_400_000_000
    ]
    out = reduce_day(real_day_batches(), day)
    assert out.ticks == len(in_day) == 4735
    for w in LEVELS:
        n = 4 * w
        t, p = out.price[w][:n].reshape(w, 4), out.price[w][n:].reshape(w, 4)
        full = ~np.isnan(p).any(axis=1)
        assert full.any()
        first, lo, hi, last = (p[full, k] for k in range(4))
        # Mínimo y máximo son dos de los puntos y no se cruzan: min <= first, last <= max.
        assert (np.minimum(lo, hi) <= first).all() and (
            first <= np.maximum(lo, hi)
        ).all()
        assert (np.minimum(lo, hi) <= last).all() and (last <= np.maximum(lo, hi)).all()
        assert (np.diff(t, axis=1) >= 0).all() and (np.diff(t.ravel()) >= 0).all()
        # El volumen sumado de todas las columnas es el total del día (centésimas de BTC).
        total = sum(q for *_, q in in_day) / 100
        assert out.volume[w].sum(dtype=np.float64) == pytest.approx(total, rel=1e-6)
        # Una columna tiene precio exactamente cuando tiene volumen.
        assert ((out.volume[w] > 0) == full).all()
        assert ((out.last_ids[w] >= 0) == full).all()
    prices = [float(p) for _, p, _ in in_day]
    top = out.price[128][4 * 128 :]
    assert np.nanmin(top) == np.float32(min(prices))
    assert np.nanmax(top) == np.float32(max(prices))
