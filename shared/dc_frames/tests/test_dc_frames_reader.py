"""El lector de tramas contra un oráculo en Python puro sobre `ticks.csv` y `events_v0.csv`.

El oráculo no usa `searchsorted` ni Arrow: filtra la lista de ids por comparación.
"""

import bisect
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import (
    FrameBoundaryError,
    FramesInputError,
    frames_of,
    read_frames,
)
from frames_lake import (
    EVENT_NAMES,
    MONTH,
    build_lake,
    is_buyer_maker,
    read_events,
    read_ticks,
    write_l1,
    write_l2,
)

THETAS = [Decimal(t).scaleb(-8) for t in sorted(read_events())]


def ids_of(frame) -> list[int]:
    """Los ids de una trama, que el fixture guarda en `quantity`."""
    return [int(q.as_py()) for q in frame.quantity]


def boundaries(row: dict) -> tuple[int, int, int]:
    return tuple(int(row[f"{n}_agg_trade_id"]) for n in EVENT_NAMES)


def expected(ids: list[int], row: dict) -> tuple[list[int], list[int]]:
    r, c, e = boundaries(row)
    return (
        [i for i in ids if r < i <= c],
        [i for i in ids if c < i <= e],
    )


def check_event(
    event, theta_rows: dict[int, dict], ids: list[int], by_id: dict
) -> None:
    row = theta_rows[event.event.column("confirm_agg_trade_id")[0].as_py()]
    confirmation, overshoot = expected(ids, row)
    assert ids_of(event.confirmation) == confirmation
    assert ids_of(event.overshoot) == overshoot
    for frame, wanted in (
        (event.confirmation, confirmation),
        (event.overshoot, overshoot),
    ):
        assert [t.as_py() for t in frame.transact_time] == [
            by_id[i]["time"] for i in wanted
        ]
        assert [p.as_py() for p in frame.price] == [by_id[i]["price"] for i in wanted]
        assert [b.as_py() for b in frame.is_buyer_maker] == [
            is_buyer_maker(i) for i in wanted
        ]


def run_all(l1_root, l2_root, ticks, events, **kwargs):
    """Lee todos los θ y compara cada evento con el oráculo. Devuelve los eventos por θ."""
    ids = [t["id"] for t in ticks]
    by_id = {t["id"]: t for t in ticks}
    rows = {
        theta: {boundaries(r)[1]: r for r in events[theta]} for theta in sorted(events)
    }
    delivered: dict[Decimal, list] = {}
    for event in read_frames(THETAS, l1_root, l2_root, MONTH, MONTH, **kwargs):
        theta = int(event.theta.scaleb(8))
        check_event(event, rows[theta], ids, by_id)
        delivered.setdefault(event.theta, []).append(event)
    return delivered


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    return build_lake(tmp_path_factory.mktemp("frames-lake"))


def test_each_event_matches_the_oracle(lake):
    l1_root, l2_root, ticks, events = lake
    delivered = run_all(l1_root, l2_root, ticks, events)
    for theta, rows in events.items():
        got = delivered[Decimal(theta).scaleb(-8)]
        assert [e.event.column("confirm_agg_trade_id")[0].as_py() for e in got] == [
            boundaries(r)[1] for r in rows
        ]


def test_phase_edges(lake):
    l1_root, l2_root, ticks, _ = lake
    ids = [t["id"] for t in ticks]
    for event in read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH):
        e = event.event
        r, c, x = (e.column(f"{n}_agg_trade_id")[0].as_py() for n in EVENT_NAMES)
        confirmation = ids_of(event.confirmation)
        assert confirmation[0] == ids[bisect.bisect_right(ids, r)]
        assert confirmation[-1] == c
        assert (
            event.confirmation.transact_time[-1].as_py()
            == e.column("confirm_time")[0].as_py()
        )
        if x == c:
            assert len(event.overshoot) == 0
        else:
            assert ids_of(event.overshoot)[-1] == x


def test_ticks_are_accounted_for(lake):
    l1_root, l2_root, ticks, events = lake
    total = len(ticks)
    ids = [t["id"] for t in ticks]
    for theta, rows in events.items():
        delivered = list(
            read_frames(Decimal(theta).scaleb(-8), l1_root, l2_root, MONTH, MONTH)
        )
        inside = sum(len(e.confirmation) + len(e.overshoot) for e in delivered)
        before = bisect.bisect_right(ids, boundaries(rows[0])[0])
        after = total - bisect.bisect_right(ids, boundaries(rows[-1])[2])
        assert before + inside + after == total, theta


def test_empty_overshoot_is_an_empty_frame(lake):
    l1_root, l2_root, _, _ = lake
    empty = [
        e
        for e in read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH)
        if e.event.column("extreme_agg_trade_id")[0].as_py()
        == e.event.column("confirm_agg_trade_id")[0].as_py()
    ]
    assert empty
    frame = empty[0].overshoot
    assert len(frame) == 0
    assert frame.transact_time.type == pa.int64()
    assert frame.price.type == pa.decimal128(18, 8)
    assert frame.quantity.type == pa.decimal128(18, 8)
    assert frame.is_buyer_maker.type == pa.bool_()


def test_frame_has_only_the_four_columns(lake):
    l1_root, l2_root, _, _ = lake
    event = next(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))
    assert list(event.confirmation.__slots__) == [
        "transact_time",
        "price",
        "quantity",
        "is_buyer_maker",
    ]
    assert event.event.num_columns == 11


@pytest.mark.parametrize("row_group", [2, 7, 64, 100_000])
def test_events_crossing_small_row_groups_come_out_whole(tmp_path, row_group):
    l1_root, l2_root, ticks, events = build_lake(tmp_path, l1_row_group=row_group)
    delivered = run_all(l1_root, l2_root, ticks, events)
    # Con un row group más chico que el evento, el evento cruza al menos dos.
    assert any(
        len(e.confirmation) + len(e.overshoot) > row_group
        for es in delivered.values()
        for e in es
    ) == (row_group <= 64)


def test_gaps_in_ids_do_not_shift_any_boundary(tmp_path):
    ticks = read_ticks()
    events = read_events()
    protected = {i for rows in events.values() for r in rows for i in boundaries(r)}
    # Un hueco en cada tercer tick que no es frontera de ningún θ.
    thinned = [t for n, t in enumerate(ticks) if t["id"] in protected or n % 3]
    assert len(thinned) < len(ticks) - 1000
    l1_root, l2_root, thinned, events = build_lake(
        tmp_path, ticks=thinned, l1_row_group=64
    )
    delivered = run_all(l1_root, l2_root, thinned, events)
    # Hay eventos cuyo rango de ids tiene más ids que ticks: el hueco no se llenó.
    assert any(
        len(e.confirmation) < boundaries_span(e)
        for es in delivered.values()
        for e in es
    )


def boundaries_span(event) -> int:
    row = event.event
    return (
        row.column("confirm_agg_trade_id")[0].as_py()
        - row.column("reference_agg_trade_id")[0].as_py()
    )


@pytest.mark.parametrize("which", [0, 1, 2])
def test_a_boundary_in_a_gap_is_an_error(tmp_path, which):
    ticks = read_ticks()
    rows = read_events()[100000]
    target = boundaries(rows[40])[which]
    kept = [t for t in ticks if t["id"] != target]
    l1_root, l2_root, kept, _ = build_lake(tmp_path, ticks=kept, l1_row_group=64)
    with pytest.raises(FrameBoundaryError, match=str(target)):
        list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))


@pytest.mark.parametrize("row_group", [2, 64, 500])
def test_a_confirmation_tick_with_another_time_is_an_error(tmp_path, row_group):
    ticks = read_ticks()
    rows = read_events()[100000]
    c = boundaries(rows[40])[1]
    # L1 reprocesado: el tick C conserva su id pero cambió de tiempo.
    moved = [{**t, "time": t["time"] + 1} if t["id"] == c else t for t in ticks]
    l1_root, l2_root, _, _ = build_lake(tmp_path, ticks=moved, l1_row_group=row_group)
    with pytest.raises(FrameBoundaryError, match=rf"confirmación {c} .*transact_time"):
        list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))


def test_a_boundary_in_a_gap_between_row_groups_is_an_error(tmp_path):
    ticks = read_ticks()
    rows = read_events()[100000]
    c = boundaries(rows[40])[1]
    # Se quita un tramo entero que incluye la confirmación y llega al borde de un row group.
    kept = [t for t in ticks if not (c - 5 <= t["id"] <= c + 5)]
    l1_root, l2_root, kept, _ = build_lake(tmp_path, ticks=kept, l1_row_group=5)
    with pytest.raises(FrameBoundaryError):
        list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))


def test_l1_that_ends_before_the_last_extreme_is_an_error(tmp_path):
    ticks = read_ticks()
    rows = read_events()[100000]
    cut = boundaries(rows[10])[2]
    kept = [t for t in ticks if t["id"] <= cut + 2]
    l1_root, l2_root, _, _ = build_lake(tmp_path, ticks=kept)
    with pytest.raises(FrameBoundaryError, match="L1 terminó"):
        list(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH))


def test_events_crossing_months_come_out_whole(tmp_path):
    ticks = read_ticks()
    events = read_events()
    cut = ticks[len(ticks) // 2]["id"]
    first, second = (2017, 8), (2017, 9)
    l1_root, l2_root = tmp_path / "l1", tmp_path / "l2"
    write_l1(l1_root, [t for t in ticks if t["id"] <= cut], 50, first)
    write_l1(l1_root, [t for t in ticks if t["id"] > cut], 50, second)
    by_month: dict[tuple[int, int], dict[int, list[dict]]] = {first: {}, second: {}}
    for theta, rows in events.items():
        for month, part in (
            (first, [r for r in rows if boundaries(r)[2] <= cut]),
            (second, [r for r in rows if boundaries(r)[2] > cut]),
        ):
            by_month[month][theta] = part
            write_l2(l2_root, theta, part, 3, month)
    ids = [t["id"] for t in ticks]
    by_id = {t["id"]: t for t in ticks}
    crossed = 0
    for month, start in ((second, second), (first, first)):
        got = list(read_frames(THETAS, l1_root, l2_root, start, second))
        want = sum(
            len(rows)
            for m in (second, first)[: 1 if start == second else 2]
            for rows in by_month[m].values()
        )
        assert len(got) == want
        for event in got:
            theta = int(event.theta.scaleb(8))
            rows = {boundaries(r)[1]: r for r in events[theta]}
            check_event(event, rows, ids, by_id)
            r, _, x = boundaries(
                rows[event.event.column("confirm_agg_trade_id")[0].as_py()]
            )
            crossed += r <= cut < x
    assert crossed


def test_a_missing_l1_month_is_an_error(tmp_path):
    ticks = read_ticks()
    events = read_events()
    cut = ticks[len(ticks) // 2]["id"]
    l1_root, l2_root = tmp_path / "l1", tmp_path / "l2"
    write_l1(l1_root, [t for t in ticks if t["id"] <= cut], 50, (2017, 8))
    for theta, rows in events.items():
        write_l2(l2_root, theta, rows, 3, (2017, 8))
        write_l2(l2_root, theta, [], 3, (2017, 9))
    with pytest.raises(FramesInputError, match="L1 del mes 2017-09"):
        list(read_frames(THETAS[0], l1_root, l2_root, (2017, 8), (2017, 9)))


def test_a_theta_without_events_needs_no_l1(tmp_path):
    l2_root = tmp_path / "l2"
    write_l2(l2_root, 100000, [], 3)
    assert list(read_frames(THETAS[0], tmp_path / "l1", l2_root, MONTH, MONTH)) == []


def test_unordered_events_are_rejected(tmp_path):
    ticks = read_ticks()
    rows = read_events()[100000][:6]
    l1_root = tmp_path / "l1"
    write_l1(l1_root, ticks, 500)
    write_l2(tmp_path / "l2", 100000, rows[::-1], 100)
    with pytest.raises(FramesInputError, match="no cumplen"):
        list(read_frames(THETAS[0], l1_root, tmp_path / "l2", MONTH, MONTH))


def test_arguments_are_validated(tmp_path):
    with pytest.raises(ValueError, match="posterior"):
        read_frames("0.001", tmp_path, tmp_path, (2017, 9), (2017, 8))
    with pytest.raises(ValueError, match="repetidos"):
        read_frames(["0.001", "0.00100000"], tmp_path, tmp_path, MONTH, MONTH)
    with pytest.raises(TypeError):
        read_frames(0.001, tmp_path, tmp_path, MONTH, MONTH)
    with pytest.raises(ValueError, match="8 decimales"):
        read_frames("0.000000001", tmp_path, tmp_path, MONTH, MONTH)
    for bad in [(2017, 0), (2017, 13)]:
        with pytest.raises(ValueError, match="entre 1 y 12"):
            read_frames("0.001", tmp_path, tmp_path, bad, (2018, 1))
        with pytest.raises(ValueError, match="entre 1 y 12"):
            read_frames("0.001", tmp_path, tmp_path, (2016, 1), bad)


class Spy:
    """Cuenta los row groups de L1 que se decodifican (los que piden columnas)."""

    def __init__(self, monkeypatch):
        self.reads = 0
        original = pq.ParquetFile.read_row_group

        def read_row_group(parquet, i, columns=None, **kwargs):
            if columns is not None:
                self.reads += 1
            return original(parquet, i, columns=columns, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_row_group)


def test_fan_out_decodes_each_row_group_once(tmp_path, monkeypatch):
    l1_root, l2_root, _, _ = build_lake(tmp_path, l1_row_group=500)
    groups = pq.ParquetFile(next(l1_root.rglob("consolidated.parquet"))).num_row_groups
    spy = Spy(monkeypatch)
    assert sum(1 for _ in read_frames(THETAS, l1_root, l2_root, MONTH, MONTH)) > 1000
    assert spy.reads == groups


def test_frames_of_matches_read_frames_and_decodes_only_its_row_groups(
    tmp_path, monkeypatch
):
    l1_root, l2_root, ticks, events = build_lake(tmp_path, l1_row_group=50)
    ids = [t["id"] for t in ticks]
    by_id = {t["id"]: t for t in ticks}
    rows = {boundaries(r)[1]: r for r in events[250000]}
    spy = Spy(monkeypatch)
    for event in list(read_frames(THETAS[1], l1_root, l2_root, MONTH, MONTH))[::25]:
        spy.reads = 0
        again = frames_of(event.event, l1_root)
        check_event(again, rows, ids, by_id)
        assert again.event.equals(event.event)
        r, _, x = (
            event.event.column(f"{n}_agg_trade_id")[0].as_py() for n in EVENT_NAMES
        )
        first = bisect.bisect_left(ids, r) // 50
        last = bisect.bisect_left(ids, x) // 50
        assert spy.reads == last - first + 1
    mapped = {k: v[0] for k, v in event.event.to_pydict().items()}
    assert ids_of(frames_of(mapped, l1_root).confirmation) == ids_of(event.confirmation)
