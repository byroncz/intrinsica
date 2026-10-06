"""Eventos exactos, conteo de ticks y confirmaciones multiescala (TRD-viz §7.4 y §7.5).

Cada tile se compara con un oráculo que no pasa por `events.py` ni `reduce.py`: sale
de `ticks.csv` y `events_v0.csv` del fixture, con enteros y listas de Python.
"""

import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from lake_fixture import DAY, build_lake, read_events
from test_viz_cli import day_directory, run
from viz_helpers import UP, events, ticks_batch, timed_events
from viz_tiles.contract import (
    DAY_MS,
    DAY_US,
    EVENT_BYTES,
    EVENTS_FILE,
    FLAG_CONFIRM_CLIPPED,
    FLAG_EXTREME_CLIPPED,
    FLAG_PROVISIONAL,
    FLAG_REF_CLIPPED,
    FLAG_UP,
    LEVELS,
    price_scale,
)
from viz_tiles.direction import PendingEvent, direction_tiles
from viz_tiles.events import (
    ConfirmAccumulator,
    EventRows,
    event_rows,
    pack_events,
)
from viz_tiles.reduce import day_start_us, reduce_day
from viz_tiles.write import ThetaTiles, write_day

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


def test_index_lists_the_new_files_and_the_events_offsets(built):
    directory, index, _, _ = built
    assert index["events"] == EVENTS_FILE
    for kind, template in (
        ("count", "count-{w}.u32"),
        ("confirms", "confirms-{w}.u8"),
        ("simul", "simul-{w}.u8"),
    ):
        assert index[kind] == {str(w): template.format(w=w) for w in LEVELS}
        for name in index[kind].values():
            assert (directory / name).is_file()
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


def test_count_tiles_are_the_ticks_of_every_column(built):
    directory, index, ticks, _ = built
    for w in LEVELS:
        expected = np.zeros(w, dtype="<u4")
        for t in ticks:
            expected[(t["time"] - T0) * w // DAY_US] += 1
        got = np.fromfile(directory / index["count"][str(w)], dtype="<u4")
        np.testing.assert_array_equal(got, expected, err_msg=f"w={w}")
        assert got.sum() == index["ticks"]


def confirm_times(pending: dict) -> dict[int, list[int]]:
    """Las confirmaciones (µs desde el inicio del día) de cada θ, de las filas del CSV."""
    out = {}
    for theta, rows in read_events().items():
        out[theta] = [int(r["confirm_time"]) - T0 for r in rows]
    return out


def test_confirms_and_simul_tiles_match_a_brute_force_oracle(built):
    directory, index, _, pending = built
    by_theta = confirm_times(pending)
    for w in LEVELS:
        confirms = np.zeros(w, dtype=int)
        columns: dict[int, dict[int, set[int]]] = {}  # columna -> instante -> θ
        for theta, times in by_theta.items():
            seen = set()
            for t in times:
                if not 0 <= t < DAY_US:
                    continue
                col = t * w // DAY_US
                columns.setdefault(col, {}).setdefault(t, set()).add(theta)
                seen.add(col)
            for col in seen:
                confirms[col] += 1
        simul = np.zeros(w, dtype=int)
        for col, instants in columns.items():
            simul[col] = max(len(thetas) for thetas in instants.values())
        got_c = np.fromfile(directory / index["confirms"][str(w)], dtype="u1")
        got_s = np.fromfile(directory / index["simul"][str(w)], dtype="u1")
        np.testing.assert_array_equal(got_c, confirms, err_msg=f"confirms w={w}")
        np.testing.assert_array_equal(got_s, simul, err_msg=f"simul w={w}")
        assert (got_s <= got_c).all()
    # La fixture tiene confirmaciones y alguna columna con más de un θ.
    assert np.fromfile(directory / index["confirms"]["128"], dtype="u1").max() >= 2


# -- unidades ----------------------------------------------------------------


def test_event_rows_clip_to_the_day_and_flag_it():
    table = timed_events(
        DAY_DATE,
        (1, 2, 3, 1, -5.0, 10.0, 20.0),  # empieza antes del día
        (3, 4, 5, -1, 20.0, 90_000.0, 95_000.0),  # confirma y termina después
    )
    rows, confirm_us = event_rows(table, None, False, T0)
    assert rows.reference.tolist() == [0, 20_000]
    assert rows.confirm.tolist() == [10_000, DAY_MS]
    assert rows.extreme.tolist() == [20_000, DAY_MS]
    assert rows.flags.tolist() == [
        FLAG_UP | FLAG_REF_CLIPPED,
        FLAG_CONFIRM_CLIPPED | FLAG_EXTREME_CLIPPED,
    ]
    # Solo la primera confirmación cae dentro del día.
    assert confirm_us.tolist() == [10_000_000]
    for column in (rows.reference, rows.confirm, rows.extreme):
        assert column.dtype == np.dtype("<i4")


def test_event_rows_truncate_microseconds_to_milliseconds():
    table = timed_events(DAY_DATE, (1, 2, 3, 1, 0.001999, 0.9999, 1.0))
    rows, _ = event_rows(table, None, False, T0)
    assert (rows.reference[0], rows.confirm[0], rows.extreme[0]) == (1, 999, 1000)


def test_event_rows_append_the_pending_tail_with_its_candidate():
    table = timed_events(DAY_DATE, (1, 2, 3, 1, 10.0, 20.0, 30.0))
    tail = PendingEvent(3, 5, 9, -1, T0 + 30_000_000, T0 + 40_000_000, T0 + 70_000_000)
    rows, confirm_us = event_rows(table, tail, True, T0)
    assert len(rows) == 2
    assert rows.extreme[1] == 70_000  # el candidato vigente
    assert rows.flags[1] == FLAG_PROVISIONAL  # baja y provisional
    assert confirm_us.tolist() == [20_000_000, 40_000_000]
    final, _ = event_rows(table, tail, False, T0)
    assert final.flags[1] == 0


def test_event_rows_need_the_times_of_the_tail():
    tail = PendingEvent(3, 5, 9, -1)
    with pytest.raises(ValueError, match="tiempos"):
        event_rows(timed_events(DAY_DATE), tail, True, T0)


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
    raw = bytes(pack_events([a, EventRows.empty(), b]))
    assert len(raw) == 3 * EVENT_BYTES
    assert np.frombuffer(raw, "<i4", 9).tolist() == [1, 2, 7, 3, 4, 8, 5, 6, 9]
    assert list(raw[36:]) == [1, 0, 3]
    assert bytes(pack_events([])) == b""


def test_a_theta_counts_once_per_column_and_instant():
    acc = ConfirmAccumulator()
    # Dos confirmaciones del mismo θ en la misma columna del nivel 128 (675 s) cuentan una vez.
    acc.add(np.array([1_000_000, 2_000_000, 1_000_000]))
    acc.add(np.array([1_000_000, 900_000_000]))
    result = acc.finish()
    assert result.confirms[128][0] == 2 and result.confirms[128][1] == 1
    # Los dos θ comparten el instante 1 s, y es el mayor grupo de la columna.
    assert result.simul[128][0] == 2 and result.simul[128][1] == 1
    assert result.confirms[128].dtype == np.uint8 and len(result.simul[4096]) == 4096


def test_the_same_instant_is_not_simultaneous_across_levels_by_accident():
    acc = ConfirmAccumulator()
    acc.add(np.array([0]))
    acc.add(np.array([DAY_US // 128 - 1]))  # última µs de la columna 0 del nivel 128
    result = acc.finish()
    # En 128 confirman dos θ en la columna, pero en instantes distintos: simul vale 1.
    assert (result.confirms[128][0], result.simul[128][0]) == (2, 1)
    assert result.confirms[4096].sum() == 2  # en 4 096 cada uno tiene su columna


def test_accumulator_refuses_more_theta_than_a_byte_holds():
    acc = ConfirmAccumulator()
    for _ in range(255):
        acc.add(np.array([1]))
    with pytest.raises(ValueError, match="uint8"):
        acc.add(np.array([1]))


def test_empty_accumulator_is_all_zero():
    result = ConfirmAccumulator().finish()
    assert all(not a.any() for a in (*result.confirms.values(), *result.simul.values()))


def test_write_day_rejects_rows_that_disagree_with_the_event_count(tmp_path):
    reduction = reduce_day([], DAY_DATE, price_scale("BTCUSDT"))
    bad = ThetaTiles(
        "0.00010000",
        3,
        None,
        direction_tiles(reduction.last_ids, events(UP)),
        EventRows.empty(),
    )
    from viz_tiles.events import Confirmations

    with pytest.raises(ValueError, match="filas"):
        write_day(
            tmp_path,
            provider="binance",
            market="spot",
            asset="BTCUSDT",
            day=DAY_DATE,
            reduction=reduction,
            confirmations=Confirmations.empty(),
            thetas=[bad],
            input_hash="x",
            image_version="v",
        )


# -- el caso de aceptación: cuatro eventos en menos de un segundo --------------

# 12:40:26 UTC en segundos desde el inicio del día.
FLASH = 12 * 3600 + 40 * 60 + 26


def write_flash_day(root: Path, day: date = date(2026, 9, 30)) -> Path:
    """Un día con el flash crash de 2026-09-30: cuatro eventos DC en menos de un segundo.

    Un evento largo que termina en el pico, cuatro eventos entre `FLASH` y `FLASH + 0,982` y
    otro largo que arranca donde termina el cuarto.
    """
    rows = [(100 + i, 40_000 + i * 1000, 84_000 + (i % 7), 1) for i in range(12_000)]
    rows += [(20_000 + i, FLASH + i / 10, 85_000 - 50 * i, 1) for i in range(10)]
    rows.sort(key=lambda r: r[1])
    rows = [(i, t, p, q) for i, (_, t, p, q) in enumerate(rows)]
    reduction = reduce_day([ticks_batch(day, rows)], day, price_scale("BTCUSDT"))
    table = timed_events(
        day,
        (1, 2, 3, -1, 44_000.0, 44_500.0, FLASH),
        (3, 4, 5, 1, FLASH, FLASH + 0.051, FLASH + 0.300),
        (5, 6, 7, -1, FLASH + 0.300, FLASH + 0.400, FLASH + 0.600),
        (7, 8, 9, 1, FLASH + 0.600, FLASH + 0.700, FLASH + 0.800),
        (9, 10, 11, -1, FLASH + 0.800, FLASH + 0.900, FLASH + 0.982),
        (11, 12, 13, 1, FLASH + 0.982, 45_700.0, 46_500.0),
    )
    acc = ConfirmAccumulator()
    event, confirm_us = event_rows(table, None, False, day_start_us(day))
    acc.add(confirm_us)
    theta = ThetaTiles(
        "0.00509931",
        len(event),
        None,
        direction_tiles(reduction.last_ids, events((1, 2, 3, -1), (3, 4, 5, 1))),
        event,
    )
    # Un segundo θ confirma el mismo instante que el segundo evento (FLASH + 0,051).
    other = timed_events(day, (1, 2, 3, 1, FLASH - 5.0, FLASH + 0.051, FLASH + 2.0))
    other_rows, other_confirm = event_rows(other, None, False, day_start_us(day))
    acc.add(other_confirm)
    second = ThetaTiles(
        "0.01000000",
        len(other_rows),
        None,
        direction_tiles(reduction.last_ids, events((1, 2, 3, 1))),
        other_rows,
    )
    write_day(
        root,
        provider="binance",
        market="spot",
        asset="BTCUSDT",
        day=day,
        reduction=reduction,
        confirmations=acc.finish(),
        thetas=[theta, second],
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
    # El segundo θ confirma en el mismo instante que el segundo evento del primero.
    w = 4096
    col = (FLASH * 1000 + 51) * w // DAY_MS
    assert np.fromfile(directory / index["confirms"][str(w)], "u1")[col] == 2
    simul = np.fromfile(directory / index["simul"][str(w)], "u1")
    assert simul[col] == 2
