"""Eventos exactos de los θ de un día (TRD-viz §7.4).

Cada archivo se compara con un oráculo que no pasa por `events.py`: sale de `ticks.csv`
y `events_v0.csv` del fixture, con enteros y listas de Python.
"""

import io
import itertools
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from lake_fixture import DAY, build_lake, read_events
from test_viz_cli import day_directory, run
from viz_helpers import encode_to, month_events, ticks_batch
from viz_tiles.chain import PendingEvent
from viz_tiles.contract import (
    DAY_MS,
    EVENT_BYTES,
    EVENTS_FILE,
    FLAG_CONFIRM_CLIPPED,
    FLAG_EXTREME_CLIPPED,
    FLAG_PROVISIONAL,
    FLAG_REF_CLIPPED,
    FLAG_UP,
    TICK_OUTSIDE,
    TICKS_CHUNK,
    price_scale,
)
from viz_tiles.events import EventRows, EventsBuffer, event_rows
from viz_tiles.ticks import (
    DayTicks,
    TicksReader,
    day_start_us,
    decode_ticks,
    encode_day,
)
from viz_tiles.write import ThetaEvents, write_day

SCALE = price_scale("BTCUSDT")
DAY_DATE = date(2017, 8, 18)
T0 = day_start_us(DAY_DATE)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("events-lake")
    roots, ticks, pending = build_lake(base)
    assert run(roots, "--day", DAY) == 0
    directory = day_directory(roots)
    index = json.loads((directory / "index.json").read_text())
    return directory, index, ticks, pending


def read_sections(directory: Path, index: dict) -> dict[str, np.ndarray]:
    raw = (directory / index["events"]).read_bytes()
    total = sum(t["events"] for t in index["thetas"])
    assert len(raw) == EVENT_BYTES * total
    return {
        "ref": np.frombuffer(raw, "<i4", total, 0),
        "confirm": np.frombuffer(raw, "<i4", total, 4 * total),
        "extreme": np.frombuffer(raw, "<i4", total, 8 * total),
        "ref_tick": np.frombuffer(raw, "<u4", total, 12 * total),
        "confirm_tick": np.frombuffer(raw, "<u4", total, 16 * total),
        "extreme_tick": np.frombuffer(raw, "<u4", total, 20 * total),
        "flags": np.frombuffer(raw, "u1", total, 24 * total),
    }


def oracle_rows(theta: int, pending: dict) -> list[tuple[int, int, int, int]]:
    """`(ref_ms, confirm_ms, extreme_ms, flags)` de los eventos del θ, de las filas del CSV."""
    csv_rows = read_events()[theta]
    *closed, last = csv_rows
    rows = []
    for r in closed:
        up = int(r["direction"]) == 1
        rows.append(
            (
                (int(r["reference_time"]) - T0) // 1000,
                (int(r["confirm_time"]) - T0) // 1000,
                (int(r["extreme_time"]) - T0) // 1000,
                FLAG_UP if up else 0,
            )
        )
    up = pending["direction"] == 1
    rows.append(
        (
            (int(last["reference_time"]) - T0) // 1000,
            (int(last["confirm_time"]) - T0) // 1000,
            (pending["extreme"]["time"] - T0) // 1000,
            (FLAG_UP if up else 0) | FLAG_PROVISIONAL,
        )
    )
    return rows


def oracle_points(theta: int, pending: dict) -> list[tuple[tuple[int, str, int], ...]]:
    """Por evento, `((agg_trade_id, precio, tiempo µs), …)` de referencia, confirmación y extremo en L2.

    Sale de `events_v0.csv`; el extremo de la cola pendiente es el candidato del carry-over.
    """
    *closed, last = read_events()[theta]
    out = [
        tuple(
            (int(r[f"{p}_agg_trade_id"]), r[f"{p}_price"], int(r[f"{p}_time"]))
            for p in ("reference", "confirm", "extreme")
        )
        for r in closed
    ]
    top = pending["extreme"]
    out.append(
        (
            (
                int(last["reference_agg_trade_id"]),
                last["reference_price"],
                int(last["reference_time"]),
            ),
            (
                int(last["confirm_agg_trade_id"]),
                last["confirm_price"],
                int(last["confirm_time"]),
            ),
            (top["id"], str(top["price"]), int(top["time"])),
        )
    )
    return out


def test_index_lists_the_events_file_and_the_events_offsets(built):
    directory, index, _, _ = built
    assert index["events"] == EVENTS_FILE
    assert (directory / EVENTS_FILE).is_file()
    offsets = [t["events_offset"] for t in index["thetas"]]
    counts = [t["events"] for t in index["thetas"]]
    assert offsets == [sum(counts[:k]) for k in range(len(counts))]


def test_events_file_has_the_exact_rows_of_every_theta(built):
    directory, index, _, pending = built
    sections = read_sections(directory, index)
    for doc in index["thetas"]:
        theta = int(doc["theta"].split(".")[1])
        expected = oracle_rows(theta, pending[theta])
        assert doc["events"] == len(expected)
        lo, hi = doc["events_offset"], doc["events_offset"] + doc["events"]
        got = list(
            zip(
                sections["ref"][lo:hi].tolist(),
                sections["confirm"][lo:hi].tolist(),
                sections["extreme"][lo:hi].tolist(),
                sections["flags"][lo:hi].tolist(),
                strict=True,
            )
        )
        assert got == expected, doc["theta"]
        # Se encadenan: el extremo de uno es la referencia del siguiente.
        assert all(a[2] == b[0] for a, b in itertools.pairwise(got))


def test_every_event_point_round_trips_to_its_tick(built):
    """Ida y vuelta (TRD-viz §7.5): la posición apunta a un tick con el instante y el precio de L2.

    La referencia y el extremo son el tick de su `agg_trade_id`. La confirmación es el primer
    tick del grupo de empate con el `confirm_price` de L2, así que su id es menor o igual que
    `confirm_agg_trade_id` (el último del grupo, ADR-L2-03): el triángulo queda en el instante y
    el precio de la confirmación.
    """
    directory, index, ticks, pending = built
    sections = read_sections(directory, index)
    day_ticks = decode_ticks(
        (directory / index["ticks_file"]).read_bytes(), index["ticks"]
    )
    position_of = {t["id"]: p for p, t in enumerate(ticks)}
    checked = moved = 0
    for doc in index["thetas"]:
        theta = int(doc["theta"].split(".")[1])
        lo, hi = doc["events_offset"], doc["events_offset"] + doc["events"]
        points = oracle_points(theta, pending[theta])
        assert len(points) == hi - lo
        for k, row in enumerate(points):
            for name, ms, (agg_id, price, time) in zip(
                ("ref", "confirm", "extreme"),
                (sections["ref"], sections["confirm"], sections["extreme"]),
                row,
                strict=True,
            ):
                position = int(sections[f"{name}_tick"][lo + k])
                assert position != TICK_OUTSIDE
                tick = ticks[position]
                where = (doc["theta"], k, name)
                # El tick apuntado comparte instante y precio con el evento de L2.
                assert tick["time"] == time, where
                assert tick["price"] == Decimal(price), where
                # Y es también el de `ticks.bin`: mismo precio y mismo ms.
                assert day_ticks.price[position] == int(tick["price"] * SCALE)
                assert day_ticks.time_ms[position] == ms[lo + k]
                if name != "confirm":
                    assert tick["id"] == agg_id, where
                    checked += 1
                    continue
                assert tick["id"] <= agg_id, where
                # Nunca antes de la referencia ni después del último tick del grupo.
                reference = int(sections["ref_tick"][lo + k])
                assert reference < position <= position_of[agg_id], where
                # El primero del grupo con ese precio, no uno posterior.
                first = next(
                    t["id"]
                    for t in ticks
                    if t["time"] == time and t["price"] == Decimal(price)
                )
                assert tick["id"] == first, where
                moved += tick["id"] != agg_id
                checked += 1
    assert checked == 3 * sum(t["events"] for t in index["thetas"])
    assert (
        moved > 0
    )  # el fixture sí trae confirmaciones cuyo último tick del grupo tiene otro precio


def test_the_extreme_tick_of_an_event_is_the_reference_tick_of_the_next(built):
    directory, index, _, _ = built
    s = read_sections(directory, index)
    for doc in index["thetas"]:
        sl = slice(doc["events_offset"], doc["events_offset"] + doc["events"])
        ref, ext = s["ref_tick"][sl], s["extreme_tick"][sl]
        assert ext[:-1].tolist() == ref[1:].tolist(), doc["theta"]
        # Una posición nunca retrocede: referencia ≤ confirmación ≤ extremo.
        assert (s["ref_tick"][sl] <= s["confirm_tick"][sl]).all()
        assert (s["confirm_tick"][sl] <= s["extreme_tick"][sl]).all()


def test_only_the_tail_is_provisional(built):
    directory, index, _, _ = built
    flags = read_sections(directory, index)["flags"]
    for doc in index["thetas"]:
        mine = flags[doc["events_offset"] : doc["events_offset"] + doc["events"]]
        assert (mine & FLAG_PROVISIONAL).tolist() == [0] * (len(mine) - 1) + [
            FLAG_PROVISIONAL
        ]
        assert not (mine & (FLAG_REF_CLIPPED | FLAG_CONFIRM_CLIPPED)).any()


def test_events_are_ordered_in_time(built):
    directory, index, _, _ = built
    s = read_sections(directory, index)
    for doc in index["thetas"]:
        sl = slice(doc["events_offset"], doc["events_offset"] + doc["events"])
        for name in ("ref", "confirm", "extreme"):
            assert (np.diff(s[name][sl]) >= 0).all(), (doc["theta"], name)
        assert (s["ref"][sl] <= s["confirm"][sl]).all()
        assert (s["confirm"][sl] <= s["extreme"][sl]).all()


# -- unidades ----------------------------------------------------------------


def day_ticks_of(ids: list[int], day: date = DAY_DATE) -> DayTicks:
    """El `DayTicks` de un día cuyos ticks, de a uno por segundo, tienen esos `agg_trade_id`."""
    rows = [(agg_id, k + 1, 100 + k, 1) for k, agg_id in enumerate(ids)]
    return encode_day([ticks_batch(day, rows)], day, SCALE, io.BytesIO())


def test_event_rows_clip_to_the_day_and_flag_it():
    table = month_events(
        DAY_DATE,
        (1, 2, 3, 1, -5.0, 10.0, 20.0),  # empieza antes del día
        (3, 4, 5, -1, 20.0, 90_000.0, 95_000.0),  # confirma y termina después
    )
    # Los ticks del día son los de id 2 y 3: la referencia 1 es anterior y los ids 4 y 5, posteriores.
    rows = event_rows(table, None, False, T0, day_ticks_of([2, 3]))
    assert rows.reference.tolist() == [0, 20_000]
    assert rows.confirm.tolist() == [10_000, DAY_MS]
    assert rows.extreme.tolist() == [20_000, DAY_MS]
    assert rows.flags.tolist() == [
        FLAG_UP | FLAG_REF_CLIPPED,
        FLAG_CONFIRM_CLIPPED | FLAG_EXTREME_CLIPPED,
    ]
    for column in (rows.reference, rows.confirm, rows.extreme):
        assert column.dtype == np.dtype("<i4")
    # Cada punto recortado lleva el centinela y solo ellos; el resto, la posición de su tick.
    outside = TICK_OUTSIDE
    assert rows.reference_tick.tolist() == [outside, 1]
    assert rows.confirm_tick.tolist() == [0, outside]
    assert rows.extreme_tick.tolist() == [1, outside]
    for column in (rows.reference_tick, rows.confirm_tick, rows.extreme_tick):
        assert column.dtype == np.dtype("<u4")
    for tick, flag in (
        (rows.reference_tick, FLAG_REF_CLIPPED),
        (rows.confirm_tick, FLAG_CONFIRM_CLIPPED),
        (rows.extreme_tick, FLAG_EXTREME_CLIPPED),
    ):
        assert ((tick == outside) == ((rows.flags & flag) != 0)).all()


def test_event_rows_truncate_microseconds_to_milliseconds():
    table = month_events(DAY_DATE, (1, 2, 3, 1, 0.001999, 0.9999, 1.0))
    rows = event_rows(table, None, False, T0, day_ticks_of([1, 2, 3]))
    assert (rows.reference[0], rows.confirm[0], rows.extreme[0]) == (1, 999, 1000)


def test_event_rows_append_the_pending_tail_with_its_candidate():
    table = month_events(DAY_DATE, (1, 2, 3, 1, 10.0, 20.0, 30.0))
    tail = PendingEvent(3, 5, 9, -1, T0 + 30_000_000, T0 + 40_000_000, T0 + 70_000_000)
    ticks = day_ticks_of(list(range(1, 10)))
    rows = event_rows(table, tail, True, T0, ticks)
    assert len(rows) == 2
    assert rows.extreme[1] == 70_000  # el candidato vigente
    assert rows.flags[1] == FLAG_PROVISIONAL  # baja y provisional
    # La cola apunta a los ticks de su referencia (id 3), su confirmación (5) y su candidato (9).
    assert (
        rows.reference_tick[1],
        rows.confirm_tick[1],
        rows.extreme_tick[1],
    ) == (2, 4, 8)
    final = event_rows(table, tail, False, T0, ticks)
    assert final.flags[1] == 0


def test_a_candidate_outside_the_day_gets_the_sentinel():
    """Un candidato posterior al último tick del día no tiene tick en el día."""
    table = month_events(DAY_DATE)
    tail = PendingEvent(2, 3, 50, 1, T0 + 2_000_000, T0 + 3_000_000, T0 + 80_000_000)
    rows = event_rows(table, tail, False, T0, day_ticks_of([1, 2, 3, 4]))
    assert rows.extreme_tick.tolist() == [TICK_OUTSIDE]
    assert (rows.reference_tick[0], rows.confirm_tick[0]) == (1, 2)


def test_event_rows_follow_the_gaps_of_the_provider():
    """Con huecos en `agg_trade_id` la posición es la del tick, no `id − primer id`."""
    table = month_events(DAY_DATE, (10, 21, 30, 1, 1.0, 2.0, 3.0))
    ticks = day_ticks_of([10, 11, 12, 20, 21, 22, 30])
    rows = event_rows(table, None, False, T0, ticks)
    assert (rows.reference_tick[0], rows.confirm_tick[0], rows.extreme_tick[0]) == (
        0,
        4,
        6,
    )
    # Un id dentro del día que ningún tick tiene es una entrada rota: no se adivina.
    bad = month_events(DAY_DATE, (10, 15, 30, 1, 1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="no es un tick"):
        event_rows(bad, None, False, T0, ticks)


def test_event_rows_need_the_times_of_the_tail():
    tail = PendingEvent(3, 5, 9, -1)
    with pytest.raises(ValueError, match="tiempos"):
        event_rows(month_events(DAY_DATE), tail, True, T0, day_ticks_of([1, 2]))


def test_pack_events_has_seven_aligned_sections():
    def rows(ref, conf, ext, ref_tick, conf_tick, ext_tick, flags):
        return EventRows(
            np.array(ref, "<i4"),
            np.array(conf, "<i4"),
            np.array(ext, "<i4"),
            np.array(ref_tick, "<u4"),
            np.array(conf_tick, "<u4"),
            np.array(ext_tick, "<u4"),
            np.array(flags, "u1"),
        )

    a = rows([1, 2], [3, 4], [5, 6], [10, 11], [12, 13], [14, TICK_OUTSIDE], [1, 0])
    b = rows([7], [8], [9], [15], [16], [17], [3])
    buffer = EventsBuffer()
    for part in (a, EventRows.empty(), b):
        buffer.add(part)
    raw = bytes(buffer.packed())
    assert len(buffer) == 3 and len(raw) == 3 * EVENT_BYTES
    assert np.frombuffer(raw, "<i4", 9).tolist() == [1, 2, 7, 3, 4, 8, 5, 6, 9]
    assert np.frombuffer(raw, "<u4", 9, 36).tolist() == [
        10,
        11,
        15,
        12,
        13,
        16,
        14,
        TICK_OUTSIDE,
        17,
    ]
    assert list(raw[72:]) == [1, 0, 3]
    assert bytes(buffer.packed()) == raw  # idempotente
    assert bytes(EventsBuffer().packed()) == b""
    with pytest.raises(ValueError, match="empaquet"):
        buffer.add(a)


def test_events_buffer_grows_and_packs_in_place():
    """Más eventos que la capacidad inicial: crece y las secciones quedan en orden."""
    rng = np.random.default_rng(7)
    parts = []
    buffer = EventsBuffer()
    for n in (3, 2_000, 0, 5_000, 1):
        rows = EventRows(
            *(rng.integers(0, 86_400_000, n, dtype="<i4") for _ in range(3)),
            *(rng.integers(0, 2**32 - 1, n, dtype="<u4") for _ in range(3)),
            rng.integers(0, 32, n, dtype="u1"),
        )
        parts.append(rows)
        buffer.add(rows)
    total = sum(len(r) for r in parts)
    raw = np.frombuffer(buffer.packed(), "u1")
    assert len(raw) == total * EVENT_BYTES
    names = (
        "reference",
        "confirm",
        "extreme",
        "reference_tick",
        "confirm_tick",
        "extreme_tick",
    )
    for i, name in enumerate(names):
        got = raw[i * 4 * total : (i + 1) * 4 * total].view("<i4" if i < 3 else "<u4")
        assert (
            got.tolist() == np.concatenate([getattr(r, name) for r in parts]).tolist()
        ), name
    assert (
        raw[24 * total :].tolist() == np.concatenate([r.flags for r in parts]).tolist()
    )


def test_write_day_rejects_rows_that_disagree_with_the_event_count(tmp_path):
    ticks = encode_to(
        tmp_path, DAY_DATE, [ticks_batch(DAY_DATE, [(1, 1, 100, 1)])], SCALE
    )
    bad = ThetaEvents("0.00010000", 3, None)
    with pytest.raises(ValueError, match="filas"):
        write_day(
            tmp_path,
            provider="binance",
            market="spot",
            asset="BTCUSDT",
            day=DAY_DATE,
            ticks=ticks,
            thetas=[bad],
            input_hash="x",
            image_version="v",
        )


# -- el caso de aceptación: cuatro eventos en menos de un segundo --------------

# 12:40:26 UTC en segundos desde el inicio del día.
FLASH = 12 * 3600 + 40 * 60 + 26
# El instante (s) en que caen 4 090 ticks en un solo milisegundo, como el de 2026-09-30.
BURST = FLASH + 0.350
BURST_TICKS = 4_090

# Los ticks que cada evento de `write_flash_day` marca con un triángulo, además de los de fondo y
# los de cada 0,1 s: `(segundos desde el inicio del día, precio)`. Tres ticks en cada uno de los
# milisegundos .051, .943 y .980 hacen un instante compartido (un segmento del mínimo al máximo), con
# el tick del evento en medio: el caso de aceptación de ITSC-319 (▼ en .051 a 85 034,99, ▽ en .943 a
# 84 601,13 y ▲ en .980 a 84 000,01).
SHARED = {
    FLASH + 0.051: (85_020.00, 85_034.99, 85_050.00),
    FLASH + 0.943: (84_580.00, 84_601.13, 84_620.00),
    FLASH + 0.980: (84_030.00, 84_000.01, 84_010.00),
}
EVENT_TICKS = {
    44_000.0: 84_100,
    44_500.0: 84_200,
    45_700.0: 84_300,
    46_500.0: 84_400,
    FLASH - 5.0: 84_700,
    FLASH + 0.982: 84_050,
    FLASH + 2.0: 84_900,
}


def flash_ticks() -> list[tuple]:
    """Los ticks del día de `write_flash_day`: `(id, segundos, precio, cantidad)`.

    Un tick cada 7 s a lo largo del día, diez ticks cada 0,1 s desde `FLASH`, 4 090 ticks
    dentro del milisegundo de `BURST`, con precios entre 84 900 y 85 100, los tres ticks
    de cada instante de `SHARED` y un tick en cada punto de `EVENT_TICKS`. El id es la
    posición del tick en el día.
    """
    rows = [(0, i * 7.0, 84_000 + (i % 7), 1) for i in range(12_000)]
    rows += [(0, FLASH + i / 10, 85_000 - 50 * i, 1) for i in range(10)]
    rows += [
        (0, BURST + i * 1e-7, 85_000 + (i * 37) % 201 - 100, "0.01")
        for i in range(BURST_TICKS)
    ]
    for at, prices in SHARED.items():
        rows += [(0, at + i * 1e-6, price, 1) for i, price in enumerate(prices)]
    rows += [(0, at, price, 1) for at, price in EVENT_TICKS.items()]
    rows.sort(key=lambda r: r[1])
    return [(i, t, p, q) for i, (_, t, p, q) in enumerate(rows)]


def tick_at(seconds: float, nth: int = 0) -> tuple[int, float]:
    """`(id, segundos)` del tick `nth` (desde 0) del milisegundo de ese instante en `flash_ticks()`.

    Un evento de `write_flash_day` apunta siempre a un tick: así su tiempo y su `agg_trade_id`
    son los del tick, como en L2.
    """
    ms = round(seconds * 1_000_000) // 1000
    found = [r for r in flash_ticks() if round(r[1] * 1_000_000) // 1000 == ms]
    return found[nth][0], found[nth][1]


def flash_event(up: bool, ref: tuple, confirm: tuple, extreme: tuple) -> tuple:
    """Un evento de `month_events` desde tres `(id, segundos)`."""
    return (
        ref[0],
        confirm[0],
        extreme[0],
        1 if up else -1,
        ref[1],
        confirm[1],
        extreme[1],
    )


def write_flash_day(
    root: Path, day: date = date(2026, 9, 30), chunk: int = TICKS_CHUNK
) -> Path:
    """Un día con el flash crash de 2026-09-30: cuatro eventos DC en menos de un segundo.

    Un evento largo que termina en el pico, cuatro eventos entre `FLASH` y `FLASH + 0,982`,
    otro largo que arranca donde termina el cuarto y, en `BURST` (en medio del tercer evento), 4 090 ticks en un
    milisegundo. Un segundo θ confirma en el mismo instante que el segundo evento del primero. Un
    tercer θ trae siete eventos que alternan alza y baja, con los tres puntos del caso de aceptación
    (▼ en .051, ▽ en .943 y ▲ en .980) en instantes de tres ticks. Cada punto de cada evento es un
    tick de `flash_ticks()`.
    """
    ticks = encode_to(root, day, [ticks_batch(day, flash_ticks())], SCALE, chunk)
    at = tick_at
    first = month_events(
        day,
        flash_event(False, at(44_000.0), at(44_500.0), at(FLASH)),
        flash_event(True, at(FLASH), at(FLASH + 0.051, 1), at(FLASH + 0.300)),
        flash_event(False, at(FLASH + 0.300), at(FLASH + 0.400), at(FLASH + 0.600)),
        flash_event(True, at(FLASH + 0.600), at(FLASH + 0.700), at(FLASH + 0.800)),
        flash_event(False, at(FLASH + 0.800), at(FLASH + 0.900), at(FLASH + 0.982)),
        flash_event(True, at(FLASH + 0.982), at(45_700.0), at(46_500.0)),
    )
    second = month_events(
        day,
        flash_event(True, at(FLASH - 5.0), at(FLASH + 0.051, 2), at(FLASH + 2.0)),
    )
    # Siete eventos: el extremo de cada uno es la referencia del siguiente.
    x = [at(FLASH - 5.0), at(FLASH + 0.051, 1), at(FLASH + 0.980, 1)]
    x += [at(FLASH + 7.0 * k) for k in (1, 3, 5, 7, 9)]
    c = [at(FLASH), at(FLASH + 0.943, 1), at(FLASH + 2.0)]
    c += [at(FLASH + 7.0 * k) for k in (2, 4, 6, 8)]
    third = month_events(
        day, *(flash_event(k % 2 == 0, x[k], c[k], x[k + 1]) for k in range(7))
    )
    buffer = EventsBuffer()
    thetas = []
    for name, table in (
        ("0.00509931", first),
        ("0.01000000", second),
        ("0.05000000", third),
    ):
        rows = event_rows(table, None, False, day_start_us(day), ticks)
        buffer.add(rows)
        thetas.append(ThetaEvents(name, len(rows), None))
    write_day(
        root,
        provider="binance",
        market="spot",
        asset="BTCUSDT",
        day=day,
        ticks=ticks,
        events=buffer,
        thetas=thetas,
        input_hash="ab" * 32,
        image_version="0.1.0+test",
    )
    return (
        Path(root) / f"provider=binance/market=spot/asset=BTCUSDT/day={day.isoformat()}"
    )


def test_the_flash_crash_day_has_four_events_inside_one_second(tmp_path):
    directory = write_flash_day(tmp_path)
    index = json.loads((directory / "index.json").read_text())
    s = read_sections(directory, index)
    first = index["thetas"][0]
    rows = slice(first["events_offset"], first["events_offset"] + first["events"])
    inside = (s["ref"][rows] >= FLASH * 1000) & (
        s["extreme"][rows] <= (FLASH + 1) * 1000
    )
    assert inside.sum() == 4


def test_the_flash_crash_day_keeps_every_tick_of_the_burst_in_one_millisecond(tmp_path):
    """Ningún tick se pierde ni se funde: 4 090 ticks comparten un instante y se leen todos."""
    directory = write_flash_day(tmp_path)
    index = json.loads((directory / "index.json").read_text())
    got = decode_ticks((directory / index["ticks_file"]).read_bytes(), index["ticks"])
    at = int(BURST * 1000)
    burst = got.price[got.time_ms == at]
    assert len(burst) == BURST_TICKS
    assert burst.min() == 84_900 * SCALE and burst.max() == 85_100 * SCALE
    assert index["ticks"] == len(flash_ticks())


def test_the_acceptance_points_are_the_exact_ticks_in_their_shared_milliseconds(
    tmp_path,
):
    """▼ .051 a 85 034,99, ▽ .943 a 84 601,13 y ▲ .980 a 84 000,01: tres ticks en cada instante."""
    directory = write_flash_day(tmp_path)
    index = json.loads((directory / "index.json").read_text())
    s = read_sections(directory, index)
    got = decode_ticks((directory / index["ticks_file"]).read_bytes(), index["ticks"])
    third = index["thetas"][2]
    first = third["events_offset"]
    assert third["events"] == 7
    for event, column, ms, price in (
        (1, "ref_tick", 51, 85_034.99),  # ▼: el extremo que inicia el evento 2 (baja)
        (1, "confirm_tick", 943, 84_601.13),  # ▽: su confirmación
        (2, "ref_tick", 980, 84_000.01),  # ▲: el extremo que inicia el evento 3 (alza)
    ):
        position = int(s[column][first + event])
        assert got.time_ms[position] == FLASH * 1000 + ms
        assert got.price[position] == round(price * SCALE)
        # Tres ticks comparten ese milisegundo y el del evento no es el mínimo ni el máximo (salvo ▲).
        shared = got.price[got.time_ms == FLASH * 1000 + ms]
        assert len(shared) == 3
        if column == "ref_tick" and event == 2:  # ▲ es el mínimo del milisegundo
            assert got.price[position] == shared.min()
        else:  # ▼ y ▽ ni el mínimo ni el máximo
            assert shared.min() < got.price[position] < shared.max()
    # Los siete eventos se encadenan: el extremo de uno es la referencia del siguiente.
    sl = slice(first, first + 7)
    assert s["extreme_tick"][sl][:-1].tolist() == s["ref_tick"][sl][1:].tolist()


def test_event_rows_point_the_confirmation_at_the_tick_with_the_confirm_price():
    """Con `reader`, la confirmación va al tick del instante que tiene el `confirm_price` de L2."""
    # Ids 1 a 6: el segundo 3 trae tres ticks, ids 3 (100), 4 (110) y 5 (100); L2 da id 5.
    rows = [
        (1, 1, 90, 1),
        (2, 2, 95, 1),
        (3, 3, 100, 1),
        (4, 3, 110, 1),
        (5, 3, 100, 1),
        (6, 4, 99, 1),
    ]
    out = io.BytesIO()
    day_ticks = encode_day([ticks_batch(DAY_DATE, rows)], DAY_DATE, SCALE, out)
    data = out.getvalue()
    reader = TicksReader(day_ticks, lambda offset, size: data[offset : offset + size])
    table = month_events(DAY_DATE, (1, 5, 6, 1, 1.0, 3.0, 4.0))
    scaled = 10**8
    for price, position in ((110, 3), (100, 2)):
        events = replace(table, confirm_price=np.array([price * scaled], np.int64))
        assert event_rows(events, None, False, T0, day_ticks).confirm_tick.tolist() == [
            4
        ]
        moved = event_rows(events, None, False, T0, day_ticks, reader)
        assert moved.confirm_tick.tolist() == [position]
        # La referencia y el extremo siguen en el tick de su `agg_trade_id`.
        assert (moved.reference_tick[0], moved.extreme_tick[0]) == (0, 5)
    # El pendiente también: sin su precio de confirmación no se puede apuntar.
    tail = PendingEvent(2, 5, 6, -1, T0 + 2_000_000, T0 + 3_000_000, T0 + 4_000_000)
    with pytest.raises(ValueError, match="precio de su confirmación"):
        event_rows(month_events(DAY_DATE), tail, False, T0, day_ticks, reader)
    tail = replace(tail, confirm_price=110 * scaled)
    pending = event_rows(month_events(DAY_DATE), tail, False, T0, day_ticks, reader)
    assert pending.confirm_tick.tolist() == [3]
