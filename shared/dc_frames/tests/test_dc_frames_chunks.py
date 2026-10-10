"""Modo por chunks, `max_event_ticks` y `ticks_before_month` contra un oráculo en Python puro.

El oráculo cuenta con listas de ids y comparaciones, sin `searchsorted` ni Arrow.
"""

import bisect
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import EventFrames, Frame, frames_of, read_frames
from frames_lake import (
    EVENT_NAMES,
    MONTH,
    SECOND_MONTH,
    boundary_ids,
    build_lake,
    build_two_month_lake,
)

PICKED = (100000, 250000)
THETAS = [Decimal(t).scaleb(-8) for t in PICKED]
ROW_GROUP = 50


def ids_of(frame: Frame) -> list[int]:
    """Los ids de una trama, que el fixture guarda en `quantity`."""
    return [int(q.as_py()) for q in frame.quantity]


def drain(chunks) -> list[Frame]:
    return list(chunks)


def merged_ids(chunks: list[Frame]) -> list[int]:
    return [i for chunk in chunks for i in ids_of(chunk)]


def key_of(event: EventFrames) -> tuple[int, int]:
    return (
        int(event.theta.scaleb(8)),
        event.event.column("confirm_agg_trade_id")[0].as_py(),
    )


def rows_by_key(events: dict[int, list[dict]]) -> dict[tuple[int, int], dict]:
    return {(t, boundary_ids(r)[1]): r for t, rows in events.items() for r in rows}


def phases(ids: list[int], row: dict) -> tuple[list[int], list[int]]:
    r, c, e = boundary_ids(row)
    return [i for i in ids if r < i <= c], [i for i in ids if c < i <= e]


def test_chunks_are_the_event_mode_frames_split_by_row_group(tmp_path):
    l1_root, l2_root, ticks, events = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    ids = [t["id"] for t in ticks]
    rows = rows_by_key(events)
    seen = crossing = 0
    for event in read_frames(THETAS, l1_root, l2_root, MONTH, MONTH, chunks=True):
        row = rows[key_of(event)]
        confirmation, overshoot = phases(ids, row)
        got_c, got_o = drain(event.confirmation), drain(event.overshoot)
        assert merged_ids(got_c) == confirmation
        assert merged_ids(got_o) == overshoot
        for phase, wanted in ((got_c, confirmation), (got_o, overshoot)):
            # Un trozo por row group con ticks de la fase, ninguno vacío ni más grande.
            assert all(0 < len(chunk) <= ROW_GROUP for chunk in phase)
            assert len(phase) == len(
                {bisect.bisect_left(ids, i) // ROW_GROUP for i in wanted}
            )
            assert all(isinstance(chunk, Frame) for chunk in phase)
        # Un evento sin meses previos ni presupuesto no trae más que sus tramas.
        assert event.ticks_before_month == 0
        assert not event.over_budget
        crossing += len(got_c) + len(got_o) > 2
        seen += 1
    assert seen == sum(len(events[t]) for t in PICKED)
    assert crossing > 50


def test_chunks_carry_the_same_columns_as_the_event_mode(tmp_path):
    l1_root, l2_root, _, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    whole = list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))
    chunked = list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH, chunks=True))
    assert len(whole) == len(chunked) > 100
    for a, b in zip(whole, chunked, strict=True):
        assert a.event.equals(b.event)
        for frame, chunks in (
            (a.confirmation, b.confirmation),
            (a.overshoot, b.overshoot),
        ):
            parts = drain(chunks)
            for name in ("transact_time", "price", "quantity", "is_buyer_maker"):
                joined = pa.chunked_array(
                    [getattr(p, name) for p in parts], getattr(frame, name).type
                )
                assert joined.equals(pa.chunked_array([getattr(frame, name)]))


def test_chunks_can_be_consumed_after_the_pass_moves_on(tmp_path):
    l1_root, l2_root, ticks, events = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    ids = [t["id"] for t in ticks]
    rows = rows_by_key(events)
    held = list(read_frames(THETAS, l1_root, l2_root, MONTH, MONTH, chunks=True))
    for event in held[::7]:
        confirmation, overshoot = phases(ids, rows[key_of(event)])
        assert merged_ids(drain(event.overshoot)) == overshoot
        assert merged_ids(drain(event.confirmation)) == confirmation
        # El iterador se recorre una vez.
        assert drain(event.confirmation) == []


def test_chunks_reread_only_the_row_groups_of_a_crossing_event(tmp_path, monkeypatch):
    l1_root, l2_root, ticks, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    reads = []
    original = pq.ParquetFile.read_row_group

    def read_row_group(parquet, i, columns=None, **kwargs):
        if columns is not None:
            reads.append(i)
        return original(parquet, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_row_group)
    groups = pq.ParquetFile(next(l1_root.rglob("consolidated.parquet"))).num_row_groups
    event = None
    for event in read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH, chunks=True):
        pass
    # Sin consumir los iterador no se relee nada: el pase decodifica cada row group una vez.
    assert len(reads) <= groups
    ids = [t["id"] for t in ticks]
    long = max(
        read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH, chunks=True),
        key=lambda e: (
            e.event.column("extreme_agg_trade_id")[0].as_py()
            - e.event.column("reference_agg_trade_id")[0].as_py()
        ),
    )
    reads.clear()
    spanned = drain(long.confirmation) + drain(long.overshoot)
    r, _, x = (long.event.column(f"{n}_agg_trade_id")[0].as_py() for n in EVENT_NAMES)
    first, last = (
        bisect.bisect_right(ids, r) // ROW_GROUP,
        bisect.bisect_left(ids, x) // ROW_GROUP,
    )
    assert last > first
    # Relee los row groups anteriores al del cierre; el del cierre ya viene en el evento.
    assert len(reads) == last - first
    assert sum(len(c) for c in spanned) == bisect.bisect_right(
        ids, x
    ) - bisect.bisect_right(ids, r)


def test_over_budget_event_comes_without_frames(tmp_path):
    l1_root, l2_root, ticks, events = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    ids = [t["id"] for t in ticks]
    rows = rows_by_key(events)
    sizes = sorted(
        boundary_ids(r)[2] - boundary_ids(r)[0] for rs in events.values() for r in rs
    )
    budget = sizes[len(sizes) // 2]
    over = fits = 0
    for event in read_frames(
        THETAS, l1_root, l2_root, MONTH, MONTH, max_event_ticks=budget
    ):
        r, _, e = boundary_ids(rows[key_of(event)])
        if e - r > budget:
            over += 1
            assert event.over_budget
            for frame in (event.confirmation, event.overshoot):
                assert len(frame) == 0
                assert frame.transact_time.type == pa.int64()
                assert frame.price.type == pa.decimal128(18, 8)
                assert frame.is_buyer_maker.type == pa.bool_()
                # Sin tramas no se ancla ningún row group.
                assert (
                    frame.price.buffers()[1] is None
                    or frame.price.buffers()[1].size == 0
                )
        else:
            fits += 1
            assert not event.over_budget
            assert ids_of(event.confirmation) == phases(ids, rows[key_of(event)])[0]
    assert over and fits


def test_an_event_exactly_at_the_budget_is_computed(tmp_path):
    l1_root, l2_root, _, events = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    largest = max(boundary_ids(r)[2] - boundary_ids(r)[0] for r in events[100000])
    at = list(
        read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH, max_event_ticks=largest)
    )
    assert not any(e.over_budget for e in at)
    under = list(
        read_frames(
            THETAS[0], l1_root, l2_root, MONTH, MONTH, max_event_ticks=largest - 1
        )
    )
    assert [e.over_budget for e in under].count(True) >= 1
    # Los demás eventos salen iguales con o sin presupuesto, también al cruzar row groups.
    free = list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))
    for a, b in zip(free, under, strict=True):
        if not b.over_budget:
            assert ids_of(a.confirmation) == ids_of(b.confirmation)
            assert ids_of(a.overshoot) == ids_of(b.overshoot)


def test_budget_arguments_are_validated(tmp_path):
    with pytest.raises(ValueError, match="chunks no tiene presupuesto"):
        read_frames(
            "0.001", tmp_path, tmp_path, MONTH, MONTH, chunks=True, max_event_ticks=10
        )
    with pytest.raises(ValueError, match="negativo"):
        read_frames("0.001", tmp_path, tmp_path, MONTH, MONTH, max_event_ticks=-1)


def expected_before(row: dict, ids: list[int], cut: int, month_is_second: bool) -> int:
    """Ticks de (R, E] en el primer mes, para un evento de la partición indicada."""
    if not month_is_second:
        return 0
    r, _, e = boundary_ids(row)
    return len([i for i in ids if r < i <= e and i <= cut])


@pytest.mark.parametrize("chunks", [False, True])
def test_ticks_before_month_matches_an_independent_count(tmp_path, chunks):
    l1_root, l2_root, ticks, events, cut, late = build_two_month_lake(tmp_path)
    ids = [t["id"] for t in ticks]
    rows = rows_by_key(events)
    kinds = {"first": 0, "spans": 0, "late": 0, "inside": 0}
    for month, in_second in ((MONTH, False), (SECOND_MONTH, True)):
        for event in read_frames(THETAS, l1_root, l2_root, month, month, chunks=chunks):
            key = key_of(event)
            row = rows[key]
            r, _, e = boundary_ids(row)
            want = expected_before(row, ids, cut, in_second)
            assert event.ticks_before_month == want, key
            if not in_second:
                kinds["first"] += 1
            elif key in late:
                # Extremo anterior al primer tick del mes: todos sus ticks son anteriores.
                assert e <= cut
                assert want == len([i for i in ids if r < i <= e]) > 0
                kinds["late"] += 1
            elif r <= cut:
                assert 0 < want < len([i for i in ids if r < i <= e])
                kinds["spans"] += 1
            else:
                assert want == 0
                kinds["inside"] += 1
    assert all(kinds.values()), kinds


def test_chunks_cross_months_and_agree_with_the_oracle(tmp_path):
    l1_root, l2_root, ticks, events, _, _ = build_two_month_lake(tmp_path)
    ids = [t["id"] for t in ticks]
    rows = rows_by_key(events)
    crossed = 0
    for event in read_frames(
        THETAS, l1_root, l2_root, SECOND_MONTH, SECOND_MONTH, chunks=True
    ):
        confirmation, overshoot = phases(ids, rows[key_of(event)])
        assert merged_ids(drain(event.confirmation)) == confirmation
        assert merged_ids(drain(event.overshoot)) == overshoot
        crossed += event.ticks_before_month > 0
    assert crossed


def test_ticks_before_month_with_a_gap_in_the_ids(tmp_path):
    """El conteo es de ticks de L1, no de ids: un hueco no suma."""
    l1_root, l2_root, ticks, events, cut, _ = build_two_month_lake(tmp_path)
    ids = [t["id"] for t in ticks]
    rows = rows_by_key(events)
    # Hay eventos que abarcan el corte, así que el rango de ids supera al de ticks.
    spanning = [
        e
        for e in read_frames(THETAS, l1_root, l2_root, SECOND_MONTH, SECOND_MONTH)
        if e.ticks_before_month
    ]
    assert spanning
    for event in spanning:
        r, _, x = boundary_ids(rows[key_of(event)])
        assert event.ticks_before_month == len([i for i in ids if r < i <= min(x, cut)])


def test_frames_of_counts_ticks_before_the_given_month(tmp_path):
    l1_root, l2_root, ticks, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    ids = [t["id"] for t in ticks]
    event = next(iter(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH)))
    again = frames_of(event.event, l1_root)
    assert again.ticks_before_month == 0 and not again.over_budget
    total = len(event.confirmation) + len(event.overshoot)
    later = frames_of(event.event, l1_root, month=SECOND_MONTH)
    assert later.ticks_before_month == total > 0
    with pytest.raises(ValueError, match="entre 1 y 12"):
        frames_of(event.event, l1_root, month=(2017, 13))
    assert ids  # el oráculo de ids lo usan las demás pruebas
