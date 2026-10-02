"""El fan-out de dc_pyo3 desde Python: sin lógica propia, mismo resultado que dc_core."""

import random
from array import array

import dc_pyo3
import pytest

SCALE = dc_pyo3.SCALE
THETA_10_PCT = 10_000_000


def price_bytes(prices):
    """`decimal128` de Arrow: 16 bytes little-endian por precio entero."""
    return memoryview(b"".join(p.to_bytes(16, "little", signed=True) for p in prices))


def columns(ticks):
    """Ticks `(price, time, id)` a las tres entradas de `feed_batch`."""
    prices, times, ids = zip(*ticks, strict=True) if ticks else ((), (), ())
    return (
        price_bytes(prices),
        memoryview(array("q", times)),
        memoryview(array("q", ids)),
    )


def series(n, seed=7):
    """Camino aleatorio con empates de `transact_time`, en escala SCALE."""
    rng = random.Random(seed)
    price, time = 50_000 * SCALE, 1_000
    ticks = []
    for agg_trade_id in range(1, n + 1):
        price += price // 30_000 * rng.randint(-10, 10)
        if rng.random() < 0.75:
            time += rng.randint(1, 3)
        ticks.append((price, time, agg_trade_id))
    return ticks


def feed_all(fanout, ticks, size):
    """Alimenta `ticks` en lotes de `size`; devuelve los eventos por θ."""
    out = [[] for _ in fanout.thetas]
    for start in range(0, len(ticks), size):
        closed = fanout.feed_batch(*columns(ticks[start : start + size]))
        for acc, events in zip(out, closed, strict=True):
            acc.extend(events)
    return out


def test_constants():
    assert SCALE == 10**8
    assert dc_pyo3.PRICE_LIMIT == 10**18
    assert dc_pyo3.STATE_VERSION == "1.0.0"


def test_first_event_by_hand():
    # θ = 10 %: el umbral de subida desde 100 es 110, y 111 lo cruza en t=2.
    # El tick de t=3 cierra el grupo de empate, y 107 (t=4, ≤ 108 = 0.9 × 120)
    # confirma la baja, que cierra el evento de subida con extremo 120.
    ticks = [
        (100 * SCALE, 1, 1),
        (111 * SCALE, 2, 2),
        (120 * SCALE, 3, 3),
        (107 * SCALE, 4, 4),
        (100 * SCALE, 5, 5),
    ]
    fanout = dc_pyo3.FanOut([THETA_10_PCT])
    ((event,),) = fanout.feed_batch(*columns(ticks))
    assert event.direction == 1
    assert event.reference == (100 * SCALE, 1, 1)
    assert event.confirm == (111 * SCALE, 2, 2)
    assert event.extreme == (120 * SCALE, 3, 3)


def test_results_do_not_depend_on_how_the_series_is_split():
    ticks = series(5_000)
    thetas = [20_000, 100_000, 500_000]
    whole = feed_all(dc_pyo3.FanOut(thetas), ticks, len(ticks))
    assert all(len(events) > 1 for events in whole)
    for size in (1, 7, 999):
        assert feed_all(dc_pyo3.FanOut(thetas), ticks, size) == whole


def test_more_ticks_than_one_decode_chunk_and_any_thread_count():
    # 70 000 ticks superan el tramo de 65 536: el resultado es el mismo.
    ticks = series(70_000, seed=3)
    thetas = [100_000, 2_000_000]
    one_thread = feed_all(dc_pyo3.FanOut(thetas, threads=1), ticks, len(ticks))
    two_threads = feed_all(dc_pyo3.FanOut(thetas, threads=2), ticks, 1_000)
    assert one_thread == two_threads
    assert all(len(events) > 1 for events in one_thread)


def test_finish_closes_the_open_tie_group():
    # El grupo confirma a 111 y sigue en t=2; el evento pendiente se abre al cerrarlo.
    ticks = [(100 * SCALE, 1, 1), (111 * SCALE, 2, 2), (112 * SCALE, 2, 3)]
    fanout = dc_pyo3.FanOut([THETA_10_PCT])
    assert fanout.feed_batch(*columns(ticks)) == [[]]
    with pytest.raises(ValueError, match="grupo de empate abierto"):
        fanout.carry_overs()
    assert fanout.finish() == [None]  # no había evento previo que cerrar
    (carry,) = fanout.carry_overs()
    assert carry.pending == ((100 * SCALE, 1, 1), (111 * SCALE, 2, 3))
    assert fanout.discarded() == [0]


def test_carry_over_continues_where_the_month_ended():
    ticks = series(4_000, seed=11)
    # Cortar donde cambia transact_time: ningún grupo cruza el borde de mes.
    cut = next(i for i in range(2_000, len(ticks)) if ticks[i][1] != ticks[i - 1][1])
    thetas = [50_000, 1_000_000]
    whole = feed_all(dc_pyo3.FanOut(thetas), ticks, len(ticks))

    first = dc_pyo3.FanOut(thetas)
    head = feed_all(first, ticks[:cut], 500)
    closed_at_end = first.finish()
    carry = first.carry_overs()
    second = dc_pyo3.FanOut.from_carry_over(thetas, carry)
    tail = feed_all(second, ticks[cut:], 500)

    for i in range(len(thetas)):
        events = head[i] + ([closed_at_end[i]] if closed_at_end[i] else []) + tail[i]
        assert events == whole[i]


def test_carry_over_fields_and_bytes_round_trip():
    ticks = series(3_000, seed=5)
    fanout = dc_pyo3.FanOut([100_000])
    feed_all(fanout, ticks, 3_000)
    fanout.finish()
    (carry,) = fanout.carry_overs()
    assert carry.theta == 100_000
    assert carry.state_version == dc_pyo3.STATE_VERSION
    assert carry.direction in (-1, 1)
    assert carry.ext_high[0] >= carry.ext_low[0]
    assert carry.pending is not None
    restored = dc_pyo3.CarryOver.from_bytes(carry.to_bytes())
    assert restored == carry
    rebuilt = dc_pyo3.CarryOver(
        carry.theta,
        carry.state_version,
        carry.direction,
        carry.ext_high,
        carry.ext_low,
        carry.pending,
    )
    assert rebuilt == carry
    assert "CarryOver(theta=100000" in repr(carry)


def test_from_carry_over_rejects_a_wrong_state():
    ticks = series(500)
    fanout = dc_pyo3.FanOut([100_000])
    feed_all(fanout, ticks, 500)
    fanout.finish()
    (carry,) = fanout.carry_overs()
    with pytest.raises(ValueError, match="otro theta"):
        dc_pyo3.FanOut.from_carry_over([200_000], [carry])
    with pytest.raises(ValueError, match="1 carry-over para 2 theta"):
        dc_pyo3.FanOut.from_carry_over([100_000, 200_000], [carry])
    old = dc_pyo3.CarryOver(
        100_000, "0.9.0", carry.direction, carry.ext_high, carry.ext_low
    )
    with pytest.raises(ValueError, match="state_version"):
        dc_pyo3.FanOut.from_carry_over([100_000], [old])


@pytest.mark.parametrize("theta", [0, -1, SCALE])
def test_invalid_theta(theta):
    with pytest.raises(ValueError, match="theta"):
        dc_pyo3.FanOut([theta])


@pytest.mark.parametrize("price", [0, -5, dc_pyo3.PRICE_LIMIT, 2**70])
def test_invalid_price_is_rejected_before_feeding_anything(price):
    fanout = dc_pyo3.FanOut([THETA_10_PCT])
    ticks = [(100 * SCALE, 1, 1), (price, 2, 2)]
    with pytest.raises(ValueError, match=r"prices\[1\]"):
        fanout.feed_batch(*columns(ticks))
    # El lote no se alimentó: sigue sin haber visto ningún tick.
    with pytest.raises(ValueError, match="ningún tick"):
        fanout.carry_overs()


def test_columns_of_different_length():
    prices, times, ids = columns([(100 * SCALE, 1, 1), (101 * SCALE, 2, 2)])
    fanout = dc_pyo3.FanOut([THETA_10_PCT])
    with pytest.raises(ValueError, match="distinto largo"):
        fanout.feed_batch(prices, times[:1], ids)
    with pytest.raises(ValueError, match="distinto largo"):
        fanout.feed_batch(prices, times, ids[:1])
    with pytest.raises(ValueError, match="distinto largo"):
        fanout.feed_batch(prices[:16], times, ids)


def test_len_and_thetas():
    fanout = dc_pyo3.FanOut([THETA_10_PCT, 20_000_000])
    assert len(fanout) == 2
    assert fanout.thetas == [THETA_10_PCT, 20_000_000]


def test_buffers_must_be_contiguous_and_int64():
    prices, times, ids = columns([(100 * SCALE, 1, 1), (101 * SCALE, 2, 2)])
    fanout = dc_pyo3.FanOut([THETA_10_PCT])
    strided = memoryview(array("q", [1, 0, 2, 0]))[::2]
    with pytest.raises(ValueError, match="contiguo"):
        fanout.feed_batch(prices, strided, ids)
    with pytest.raises(BufferError, match="not compatible"):
        fanout.feed_batch(prices, memoryview(array("d", [1.0, 2.0])), ids)
    with pytest.raises(TypeError):
        fanout.feed_batch([1, 2], times, ids)


def decode_columns(block):
    """Un `EventColumns` a `(direction, reference, confirm, extreme)` por evento."""
    raw = block.buffers()
    assert len(raw) == 10
    n = len(block)

    def ints(buffer, width):
        assert len(buffer) == n * width
        return [
            int.from_bytes(buffer[i * width : (i + 1) * width], "little", signed=True)
            for i in range(n)
        ]

    points = [
        list(zip(ints(raw[k], 16), ints(raw[k + 1], 8), ints(raw[k + 2], 8)))
        for k in (0, 3, 6)
    ]
    directions = ints(raw[9], 1)
    return list(zip(directions, *points, strict=True))


def as_tuples(events):
    return [(e.direction, e.reference, e.confirm, e.extreme) for e in events]


def test_columns_carry_the_same_events_as_the_objects():
    ticks = series(6_000, seed=13)
    thetas = [20_000, 100_000, 5_000_000]
    by_objects, by_columns = dc_pyo3.FanOut(thetas), dc_pyo3.FanOut(thetas)
    for start in range(0, len(ticks), 1_500):
        batch = columns(ticks[start : start + 1_500])
        objects = by_objects.feed_batch(*batch)
        blocks = by_columns.feed_batch_columns(*batch)
        assert [len(b) for b in blocks] == [len(e) for e in objects]
        for block, events in zip(blocks, objects, strict=True):
            assert decode_columns(block) == as_tuples(events)
    # Con el θ de 20 % (el último) no hay eventos: bloque vacío, 10 buffers vacíos.
    assert len(blocks[-1]) == 0
    assert all(len(b) == 0 for b in blocks[-1].buffers())
    assert sum(len(b) for b in blocks) > 0


def test_finish_columns_matches_finish():
    # θ = 10 %: el grupo de t=4 confirma la baja y sigue abierto (107 y 106 en
    # t=4); `finish` lo cierra y con él el evento de subida, con extremo 120.
    ticks = [
        (100 * SCALE, 1, 1),
        (111 * SCALE, 2, 2),
        (120 * SCALE, 3, 3),
        (107 * SCALE, 4, 4),
        (106 * SCALE, 4, 5),
    ]
    thetas = [THETA_10_PCT, 50_000_000]
    a, b = dc_pyo3.FanOut(thetas), dc_pyo3.FanOut(thetas)
    feed_all(a, ticks, 5)
    feed_all(b, ticks, 5)
    last = a.finish()
    blocks = b.finish_columns()
    assert last[0] is not None and last[1] is None
    for block, event in zip(blocks, last, strict=True):
        assert decode_columns(block) == ([] if event is None else as_tuples([event]))
    assert a.carry_overs() == b.carry_overs()


def test_feed_batch_columns_validates_like_feed_batch():
    fanout = dc_pyo3.FanOut([THETA_10_PCT])
    prices, times, ids = columns([(100 * SCALE, 1, 1), (0, 2, 2)])
    with pytest.raises(ValueError, match="prices"):
        fanout.feed_batch_columns(prices, times, ids)
    with pytest.raises(ValueError, match="distinto largo"):
        fanout.feed_batch_columns(prices, times, memoryview(array("q", [1])))
    # No se tocó: sigue aceptando una serie válida desde cero.
    ok = [(100 * SCALE, 1, 1), (111 * SCALE, 2, 2)]
    assert [len(b) for b in fanout.feed_batch_columns(*columns(ok))] == [0]


def _wide_series(n):
    """`n` ticks en zigzag: muchos eventos y cómputo suficiente para medir el GIL."""
    prices = array("q")
    for i in range(n):
        prices.extend((50_000 * SCALE + (i * 7_919 % 1_001) * 20 * SCALE // 1_000, 0))
    return (
        memoryview(prices).cast("B"),
        memoryview(array("q", range(1_000, 1_000 + n))),
        memoryview(array("q", range(1, n + 1))),
    )


def test_feed_batch_columns_releases_the_gil():
    """Mientras un hilo detecta, otro de Python sigue corriendo (ITSC-290)."""
    import threading
    import time

    n = 3_000_000
    prices, times, ids = _wide_series(n)
    fanout = dc_pyo3.FanOut([THETA_10_PCT // 100 * k for k in range(1, 9)], 1)
    ticking, stop, gaps = threading.Event(), threading.Event(), []

    def watch():
        """Anota cuánto tarda en volver a correr cada vez que cede el GIL."""
        last = time.perf_counter()
        while not stop.is_set():
            time.sleep(0)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
            if len(gaps) > 100:
                ticking.set()

    watcher = threading.Thread(target=watch)
    watcher.start()
    ticking.wait()
    gaps.clear()
    started = time.perf_counter()
    fanout.feed_batch_columns(prices, times, ids)
    seconds = time.perf_counter() - started
    stop.set()
    watcher.join()
    # Reteniendo el GIL el vigía quedaría parado casi toda la llamada.
    assert seconds > 0.2, "la serie es muy corta para medir el GIL"
    assert max(gaps) < seconds / 4
    assert len(gaps) > 100


def test_writing_while_detecting_does_not_change_the_events():
    """Un hilo hashea los eventos del lote anterior mientras otro detecta el siguiente."""
    import hashlib
    import threading

    chunks = [_wide_series(400_000) for _ in range(3)]
    thetas = [THETA_10_PCT // 100 * k for k in range(1, 9)]

    def digest(closed):
        h = hashlib.sha256()
        for columns in closed:
            for buffer in columns.buffers():
                h.update(buffer)
        return h.hexdigest()

    serial = dc_pyo3.FanOut(thetas, 2)
    expected = [digest(serial.feed_batch_columns(*c)) for c in chunks]

    fanout = dc_pyo3.FanOut(thetas, 2)
    got, previous = [], None
    for chunk in chunks:
        closed = []

        def detect(chunk=chunk, closed=closed):
            closed.append(fanout.feed_batch_columns(*chunk))

        detector = threading.Thread(target=detect)
        detector.start()
        if previous is not None:
            got.append(digest(previous))  # escribe lo anterior en este hilo
        detector.join()
        previous = closed[0]
    got.append(digest(previous))
    assert got == expected
