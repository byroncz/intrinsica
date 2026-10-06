"""Eventos exactos de los θ de un día (TRD-viz §7.4).

Cada archivo se compara con un oráculo que no pasa por `events.py`: sale de `ticks.csv`
y `events_v0.csv` del fixture, con enteros y listas de Python.
"""

import itertools
import json
from datetime import date
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
    TICKS_CHUNK,
    price_scale,
)
from viz_tiles.events import EventRows, EventsBuffer, event_rows
from viz_tiles.ticks import day_start_us, decode_ticks
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
        "flags": np.frombuffer(raw, "u1", total, 12 * total),
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


def test_event_rows_clip_to_the_day_and_flag_it():
    table = month_events(
        DAY_DATE,
        (1, 2, 3, 1, -5.0, 10.0, 20.0),  # empieza antes del día
        (3, 4, 5, -1, 20.0, 90_000.0, 95_000.0),  # confirma y termina después
    )
    rows = event_rows(table, None, False, T0)
    assert rows.reference.tolist() == [0, 20_000]
    assert rows.confirm.tolist() == [10_000, DAY_MS]
    assert rows.extreme.tolist() == [20_000, DAY_MS]
    assert rows.flags.tolist() == [
        FLAG_UP | FLAG_REF_CLIPPED,
        FLAG_CONFIRM_CLIPPED | FLAG_EXTREME_CLIPPED,
    ]
    for column in (rows.reference, rows.confirm, rows.extreme):
        assert column.dtype == np.dtype("<i4")


def test_event_rows_truncate_microseconds_to_milliseconds():
    table = month_events(DAY_DATE, (1, 2, 3, 1, 0.001999, 0.9999, 1.0))
    rows = event_rows(table, None, False, T0)
    assert (rows.reference[0], rows.confirm[0], rows.extreme[0]) == (1, 999, 1000)


def test_event_rows_append_the_pending_tail_with_its_candidate():
    table = month_events(DAY_DATE, (1, 2, 3, 1, 10.0, 20.0, 30.0))
    tail = PendingEvent(3, 5, 9, -1, T0 + 30_000_000, T0 + 40_000_000, T0 + 70_000_000)
    rows = event_rows(table, tail, True, T0)
    assert len(rows) == 2
    assert rows.extreme[1] == 70_000  # el candidato vigente
    assert rows.flags[1] == FLAG_PROVISIONAL  # baja y provisional
    final = event_rows(table, tail, False, T0)
    assert final.flags[1] == 0


def test_event_rows_need_the_times_of_the_tail():
    tail = PendingEvent(3, 5, 9, -1)
    with pytest.raises(ValueError, match="tiempos"):
        event_rows(month_events(DAY_DATE), tail, True, T0)


def test_pack_events_has_four_aligned_sections():
    a = EventRows(
        np.array([1, 2], "<i4"),
        np.array([3, 4], "<i4"),
        np.array([5, 6], "<i4"),
        np.array([1, 0], "u1"),
    )
    b = EventRows(
        np.array([7], "<i4"),
        np.array([8], "<i4"),
        np.array([9], "<i4"),
        np.array([3], "u1"),
    )
    buffer = EventsBuffer()
    for rows in (a, EventRows.empty(), b):
        buffer.add(rows)
    raw = bytes(buffer.packed())
    assert len(buffer) == 3 and len(raw) == 3 * EVENT_BYTES
    assert np.frombuffer(raw, "<i4", 9).tolist() == [1, 2, 7, 3, 4, 8, 5, 6, 9]
    assert list(raw[36:]) == [1, 0, 3]
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
            rng.integers(0, 86_400_000, n, dtype="<i4"),
            rng.integers(0, 86_400_000, n, dtype="<i4"),
            rng.integers(0, 86_400_000, n, dtype="<i4"),
            rng.integers(0, 32, n, dtype="u1"),
        )
        parts.append(rows)
        buffer.add(rows)
    total = sum(len(r) for r in parts)
    raw = np.frombuffer(buffer.packed(), "u1")
    assert len(raw) == total * EVENT_BYTES
    for i, name in enumerate(("reference", "confirm", "extreme")):
        got = raw[i * 4 * total : (i + 1) * 4 * total].view("<i4")
        assert (
            got.tolist() == np.concatenate([getattr(r, name) for r in parts]).tolist()
        )
    assert (
        raw[12 * total :].tolist() == np.concatenate([r.flags for r in parts]).tolist()
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


def flash_ticks() -> list[tuple]:
    """Los ticks del día de `write_flash_day`: `(id, segundos, precio, cantidad)`.

    Un tick cada 7 s a lo largo del día, diez ticks cada 0,1 s desde `FLASH` y
    4 090 ticks dentro del milisegundo de `BURST`, con precios entre 84 900 y 85 100.
    """
    rows = [(0, i * 7.0, 84_000 + (i % 7), 1) for i in range(12_000)]
    rows += [(0, FLASH + i / 10, 85_000 - 50 * i, 1) for i in range(10)]
    rows += [
        (0, BURST + i * 1e-7, 85_000 + (i * 37) % 201 - 100, "0.01")
        for i in range(BURST_TICKS)
    ]
    rows.sort(key=lambda r: r[1])
    return [(i, t, p, q) for i, (_, t, p, q) in enumerate(rows)]


def write_flash_day(
    root: Path, day: date = date(2026, 9, 30), chunk: int = TICKS_CHUNK
) -> Path:
    """Un día con el flash crash de 2026-09-30: cuatro eventos DC en menos de un segundo.

    Un evento largo que termina en el pico, cuatro eventos entre `FLASH` y `FLASH + 0,982`,
    otro largo que arranca donde termina el cuarto y, en `BURST` (en medio del tercer evento), 4 090 ticks en un
    milisegundo. Un segundo θ confirma en el mismo instante que el segundo evento del primero.
    """
    ticks = encode_to(root, day, [ticks_batch(day, flash_ticks())], SCALE, chunk)
    first = month_events(
        day,
        (1, 2, 3, -1, 44_000.0, 44_500.0, FLASH),
        (3, 4, 5, 1, FLASH, FLASH + 0.051, FLASH + 0.300),
        (5, 6, 7, -1, FLASH + 0.300, FLASH + 0.400, FLASH + 0.600),
        (7, 8, 9, 1, FLASH + 0.600, FLASH + 0.700, FLASH + 0.800),
        (9, 10, 11, -1, FLASH + 0.800, FLASH + 0.900, FLASH + 0.982),
        (11, 12, 13, 1, FLASH + 0.982, 45_700.0, 46_500.0),
    )
    second = month_events(day, (1, 2, 3, 1, FLASH - 5.0, FLASH + 0.051, FLASH + 2.0))
    buffer = EventsBuffer()
    thetas = []
    for name, table in (("0.00509931", first), ("0.01000000", second)):
        rows = event_rows(table, None, False, day_start_us(day))
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
