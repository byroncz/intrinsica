"""Entrega por row group (TRD-L3 §7.4, ADR-L3-14) contra un oráculo en Python puro.

El oráculo ubica cada frontera con `bisect` sobre las listas de ids de cada row group,
sin `searchsorted` ni Arrow, y compara además los slices por posiciones con lo que
entrega `read_frames`.
"""

import bisect
import contextvars
import threading
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import (
    FrameBoundaryError,
    FramesInputError,
    RowGroupFrames,
    read_frames,
    read_row_groups,
)
from dc_frames import lake as lake_module
from frames_lake import (
    MONTH,
    SECOND_MONTH,
    boundary_ids,
    build_lake,
    build_two_month_lake,
    read_events,
    read_ticks,
    write_l1,
    write_l2,
)

ALL = (100000, 250000, 500000, 1000000, 2000000)
THETAS = [Decimal(t).scaleb(-8) for t in ALL]
ROW_GROUP = 50
COLUMNS = ("transact_time", "price", "quantity", "is_buyer_maker")


def key_of(theta: Decimal, batch: pa.RecordBatch, i: int) -> tuple[int, int]:
    return int(theta.scaleb(8)), batch.column("confirm_agg_trade_id")[i].as_py()


def appearances(deliveries: list[RowGroupFrames]) -> dict[tuple[int, int], list[dict]]:
    """Cada evento con lo que se entregó de él, un elemento por row group en que sale."""
    out: dict[tuple[int, int], list[dict]] = {}
    for n, delivery in enumerate(deliveries):
        for te in delivery.thetas:
            assert (
                len(te) == len(te.events) == len(te.positions) == len(te.starts_before)
            )
            assert te.positions.shape == (len(te), 3)
            for i in range(len(te)):
                out.setdefault(key_of(te.theta, te.events, i), []).append(
                    {
                        "rg": n,
                        "delivery": delivery,
                        "pos": tuple(int(x) for x in te.positions[i]),
                        "starts_before": bool(te.starts_before[i]),
                        "ends_after": bool(te.ends_after[i]),
                        "before": int(te.ticks_before_month[i]),
                    }
                )
    return out


def expected(
    ids_by_rg: list[list[int]], row: dict
) -> list[tuple[int, tuple, bool, bool]]:
    """Dónde sale un evento: `(row group, posiciones, empieza antes, termina después)`."""
    r, c, e = boundary_ids(row)
    first = next(n for n, ids in enumerate(ids_by_rg) if r in ids)
    last = next(n for n, ids in enumerate(ids_by_rg) if e in ids)
    out = []
    for n in range(first, last + 1):
        ids = ids_by_rg[n]
        pos = tuple(bisect.bisect_right(ids, x) for x in (r, c, e))
        out.append((n, pos, r < ids[0], e > ids[-1]))
    return out


def split(ticks: list[dict], size: int) -> list[list[int]]:
    ids = [t["id"] for t in ticks]
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def rows_by_key(events: dict[int, list[dict]]) -> dict[tuple[int, int], dict]:
    return {(t, boundary_ids(r)[1]): r for t, rows in events.items() for r in rows}


def check_against_oracle(deliveries, ids_by_rg, events, skip=lambda row: False):
    got = appearances(deliveries)
    rows = rows_by_key(events)
    for key, row in rows.items():
        if skip(row):
            continue
        want = expected(ids_by_rg, row)
        assert [
            (a["rg"], a["pos"], a["starts_before"], a["ends_after"]) for a in got[key]
        ] == want, key
    return got


def test_positions_match_the_oracle_on_every_event(tmp_path):
    l1_root, l2_root, ticks, events = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    deliveries = list(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH))
    got = check_against_oracle(deliveries, split(ticks, ROW_GROUP), events)
    assert len(got) == sum(len(rows) for rows in events.values())
    # Cada evento cierra una sola vez, y solo ahí trae `ticks_before_month` (0 en un mes).
    assert all(
        [not a["ends_after"] for a in ap].count(True) == 1 for ap in got.values()
    )
    assert all(ap[-1]["before"] == 0 for ap in got.values())
    spans = [len(ap) for ap in got.values()]
    assert max(spans) > 2 and min(spans) == 1


def test_a_gap_in_the_ids_does_not_shift_membership(tmp_path):
    events = read_events()
    protected = {i for rows in events.values() for r in rows for i in boundary_ids(r)}
    thinned = [t for n, t in enumerate(read_ticks()) if t["id"] in protected or n % 3]
    l1_root, l2_root, ticks, events = build_lake(
        tmp_path, l1_row_group=ROW_GROUP, ticks=thinned
    )
    deliveries = list(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH))
    ids_by_rg = split(ticks, ROW_GROUP)
    check_against_oracle(deliveries, ids_by_rg, events)
    # Hay huecos: restar posiciones daría otra cosa.
    assert any(b - a > len(ids) for ids in ids_by_rg for a, b in [(ids[0], ids[-1])])


def test_slices_by_position_are_the_phases_of_read_frames(tmp_path):
    l1_root, l2_root, _, events = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    deliveries = list(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH))
    got = appearances(deliveries)
    seen = 0
    for event in read_frames(THETAS, l1_root, l2_root, MONTH, MONTH):
        key = key_of(event.theta, event.event, 0)
        parts: dict[str, list[list[pa.Array]]] = {"conf": [], "over": []}
        for a in got[key]:
            r, c, e = a["pos"]
            frame = a["delivery"].frame
            parts["conf"].append([getattr(frame, n).slice(r, c - r) for n in COLUMNS])
            parts["over"].append([getattr(frame, n).slice(c, e - c) for n in COLUMNS])
        for name, whole in (
            ("conf", event.confirmation),
            ("over", event.overshoot),
        ):
            for j, column in enumerate(COLUMNS):
                joined = pa.chunked_array(
                    [p[j] for p in parts[name]], getattr(whole, column).type
                )
                assert joined.equals(pa.chunked_array([getattr(whole, column)])), key
        assert got[key][-1]["before"] == event.ticks_before_month
        seen += 1
    assert seen == len(got) == sum(len(rows) for rows in events.values())


def test_the_arrays_are_zero_copy_views_with_l1_types(tmp_path):
    l1_root, l2_root, _, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    first = next(iter(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH)))
    frame = first.frame
    assert len(frame) == ROW_GROUP
    assert frame.transact_time.type == pa.int64()
    assert frame.price.type == frame.quantity.type == pa.decimal128(18, 8)
    assert frame.is_buyer_maker.type == pa.bool_()
    # Un solo buffer por columna, del largo del row group: no se copió nada.
    assert frame.transact_time.buffers()[1].size >= ROW_GROUP * 8
    assert not hasattr(frame, "agg_trade_id")


def test_event_rows_carry_the_seven_columns_or_the_requested_ones(tmp_path):
    l1_root, l2_root, _, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    seven = (
        "reference_agg_trade_id",
        "confirm_agg_trade_id",
        "confirm_time",
        "extreme_agg_trade_id",
        "direction",
        "confirm_price",
        "extreme_price",
    )
    default = next(iter(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH)))
    assert all(set(te.events.schema.names) == set(seven) for te in default.thetas)
    own = next(
        iter(
            read_row_groups(
                THETAS,
                l1_root,
                l2_root,
                MONTH,
                MONTH,
                columns=["theta", "confirm_time"],
            )
        )
    )
    assert set(own.thetas[0].events.schema.names) == {
        "reference_agg_trade_id",
        "confirm_agg_trade_id",
        "confirm_time",
        "extreme_agg_trade_id",
        "theta",
    }
    # Las filas son las de L2 sin transformar.
    event = next(iter(read_frames(THETAS[0], l1_root, l2_root, MONTH, MONTH)))
    first = next(iter(read_row_groups(THETAS[0], l1_root, l2_root, MONTH, MONTH)))
    assert (
        first.thetas[0]
        .events.slice(0, 1)
        .equals(event.event.select(first.thetas[0].events.schema.names))
    )


def test_a_theta_with_no_event_touching_the_row_group_is_left_out(tmp_path):
    l1_root, l2_root, _, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    sizes = {
        len(d.thetas) for d in read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH)
    }
    assert min(sizes) < len(THETAS) and max(sizes) >= 1
    for d in read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH):
        assert all(len(te) for te in d.thetas)
        assert [te.theta for te in d.thetas] == [
            t for t in THETAS if t in {x.theta for x in d.thetas}
        ]


def count_decodes(monkeypatch) -> list[tuple[str, int]]:
    """Cada row group de L1 que se decodifica, con gancho en `lake.open_parquet`."""
    reads: list[tuple[str, int]] = []
    real = lake_module.open_parquet

    def open_parquet(
        path: str, what: str, metadata=None, pre_buffer=False
    ) -> pq.ParquetFile:
        parquet = real(path, what, metadata, pre_buffer)
        if path.endswith("consolidated.parquet"):
            original = parquet.read_row_group

            def read_row_group(i, columns=None, **kwargs):
                reads.append((path, i))
                return original(i, columns=columns, **kwargs)

            parquet.read_row_group = read_row_group
        return parquet

    monkeypatch.setattr(lake_module, "open_parquet", open_parquet)
    return reads


def test_five_thetas_decode_each_row_group_once(tmp_path, monkeypatch):
    l1_root, l2_root, _, _ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    path = next(l1_root.rglob("consolidated.parquet"))
    groups = pq.ParquetFile(path).num_row_groups
    reads = count_decodes(monkeypatch)
    delivered = sum(1 for _ in read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH))
    # El pase se salta los row groups que ningún evento toca; los demás, una vez cada uno.
    assert delivered == len(reads) == len(set(reads)) >= groups - 2


def test_two_months_decode_each_row_group_once_and_mark_the_previous_month(
    tmp_path, monkeypatch
):
    l1_root, l2_root, *_ = build_two_month_lake(tmp_path)
    reads = count_decodes(monkeypatch)
    deliveries = list(
        read_row_groups(THETAS, l1_root, l2_root, SECOND_MONTH, SECOND_MONTH)
    )
    assert len(reads) == len(set(reads)) == len(deliveries)
    months = [d.month for d in deliveries]
    assert (
        months == sorted(months) and months[0] == MONTH and months[-1] == SECOND_MONTH
    )
    assert [d.before_start for d in deliveries] == [m < SECOND_MONTH for m in months]
    # Los eventos de la partición del segundo mes: ticks_before_month como read_frames.
    got = appearances(deliveries)
    for event in read_frames(THETAS, l1_root, l2_root, SECOND_MONTH, SECOND_MONTH):
        key = key_of(event.theta, event.event, 0)
        closing = got[key][-1]
        assert not closing["ends_after"]
        assert closing["before"] == event.ticks_before_month
        assert all(a["before"] == 0 for a in got[key][:-1])
    assert any(a["before"] for ap in got.values() for a in ap)
    # Y en el primer mes ninguno cuenta ticks anteriores.
    first = list(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH))
    assert not any(a["before"] for ap in appearances(first).values() for a in ap)
    assert not any(d.before_start for d in first)


# Un lago a mano: 8 row groups de 100 ticks con id = tiempo, para poner las fronteras
# justo en los bordes. `confirm_time` de una fila se puede fijar aparte.
SIZE = 100
GROUPS = 8


def event(reference: int, confirm: int, extreme: int, confirm_time: int | None = None):
    row = {"direction": 1}
    for name, tick in (
        ("reference", reference),
        ("confirm", confirm),
        ("extreme", extreme),
    ):
        row[f"{name}_price"] = Decimal(1)
        row[f"{name}_time"] = tick
        row[f"{name}_agg_trade_id"] = tick
    if confirm_time is not None:
        row["confirm_time"] = confirm_time
    return row


def hand_lake(tmp_path, events: dict[int, list[dict]], ticks=None):
    ticks = ticks or [
        {"id": i, "price": Decimal(1), "time": i} for i in range(GROUPS * SIZE)
    ]
    write_l1(tmp_path / "l1", ticks, SIZE)
    for theta, rows in events.items():
        write_l2(tmp_path / "l2", theta, rows, 100)
    return tmp_path / "l1", tmp_path / "l2", ticks


def run(l1_root, l2_root, events) -> dict[tuple[int, int], list[dict]]:
    thetas = [Decimal(t).scaleb(-8) for t in events]
    return appearances(list(read_row_groups(thetas, l1_root, l2_root, MONTH, MONTH)))


def test_boundaries_on_the_first_and_last_tick_of_a_row_group(tmp_path):
    events = {
        100000: [
            event(0, 99, 99),  # R en el primer tick, C = E en el último
            event(99, 100, 199),  # R último de un row group, C primero del siguiente
            event(199, 350, 590),  # abarca los row groups 1 a 5
            event(590, 591, 700),  # E en el primer tick de un row group
            event(700, 750, 799),  # E en el último tick de L1
        ]
    }
    got = run(*hand_lake(tmp_path, events)[:2], events)
    by_confirm = {c: ap for (_, c), ap in got.items()}

    def summary(c):
        return [
            (a["rg"], a["pos"], a["starts_before"], a["ends_after"])
            for a in by_confirm[c]
        ]

    # Overshoot vacío: pos_E = pos_C; R en el primer tick deja pos_R = 1.
    assert summary(99) == [(0, (1, 100, 100), False, False)]
    # R es el último tick del row group 0: está en él con posición = largo, y la
    # confirmación [pos_R, pos_C) = [0, 1) del row group 1 es el tick 100.
    assert summary(100) == [
        (0, (100, 100, 100), False, True),
        (1, (0, 1, 100), True, False),
    ]
    long = summary(350)
    assert [n for n, *_ in long] == [1, 2, 3, 4, 5]
    assert long[0] == (1, (100, 100, 100), False, True)  # R último del row group 1
    assert long[1] == (2, (0, 100, 100), True, True)  # lo cruza entero
    assert long[2] == (3, (0, 51, 100), True, True)  # C es el tick 350
    assert long[4] == (5, (0, 0, 91), True, False)  # cierra en 590
    # Las fases consecutivas comparten corte: E del largo es R del siguiente.
    assert summary(591)[0][1][0] == long[4][1][2] == 91
    assert summary(591)[-1] == (7, (0, 0, 1), True, False)  # E es el primer tick
    assert summary(750)[-1][3] is False and summary(750)[-1][1] == (1, 51, 100)


def test_confirmation_instant_split_between_two_row_groups(tmp_path):
    ticks = [
        {"id": i, "price": Decimal(1), "time": 399 if i == 400 else i}
        for i in range(GROUPS * SIZE)
    ]
    # La confirmación es el tick 400, primero del row group 4; su instante (tiempo 399)
    # es también el del último tick del row group 3.
    events = {100000: [event(390, 400, 450, confirm_time=399)]}
    l1_root, l2_root, _ = hand_lake(tmp_path, events, ticks)
    deliveries = list(read_row_groups(Decimal("0.001"), l1_root, l2_root, MONTH, MONTH))
    assert [d.month for d in deliveries] == [MONTH] * 2
    third, fourth = deliveries
    assert [te.positions.tolist() for te in third.thetas] == [[[91, 100, 100]]]
    assert [te.positions.tolist() for te in fourth.thetas] == [[[0, 1, 51]]]
    last = third.frame.transact_time[99].as_py()
    pos_c = int(fourth.thetas[0].positions[0, 1])
    assert last == fourth.frame.transact_time[pos_c - 1].as_py() == 399
    assert third.thetas[0].ends_after.tolist() == [True]


def test_gaps_in_the_ids_do_not_shift_membership_in_a_hand_lake(tmp_path):
    ticks = [
        {"id": i, "price": Decimal(1), "time": i}
        for i in range(GROUPS * SIZE)
        if i % 7 or i in (150, 250, 299, 300, 399, 400)
    ]
    ticks = [t for t in ticks if t["id"] not in (100, 200)]
    events = {100000: [event(150, 250, 400), event(400, 401, 500)]}
    l1_root, l2_root, _ = hand_lake(tmp_path, events, ticks)
    ids = [t["id"] for t in ticks]
    got = run(l1_root, l2_root, events)
    for (_, confirm), appearances_ in got.items():
        row = next(r for r in events[100000] if boundary_ids(r)[1] == confirm)
        r, c, e = boundary_ids(row)
        for a in appearances_:
            # En este lago el tiempo de un tick es su id.
            rg_ids = a["delivery"].frame.transact_time.to_pylist()
            assert set(rg_ids) <= set(ids)
            assert a["pos"] == tuple(bisect.bisect_right(rg_ids, x) for x in (r, c, e))


@pytest.mark.parametrize(
    ("row", "field", "got"),
    [
        # Un hueco dentro de un row group: el id de la frontera no es un tick.
        (event(251, 260, 270), "reference_agg_trade_id", "251"),
        (event(250, 261, 270), "confirm_agg_trade_id", "261"),
        (event(250, 260, 271), "extreme_agg_trade_id", "271"),
        # Un hueco entre dos row groups (99 | 200): la referencia no está en ninguno.
        (event(150, 210, 220), "reference_agg_trade_id", "150"),
        # L1 termina antes del extremo.
        (event(250, 260, 5000), "extreme_agg_trade_id", "ausente"),
        # L1 y L2 no cuadran en el instante de confirmación.
        (event(250, 260, 270, confirm_time=259), "confirm_time", "260"),
    ],
)
def test_a_boundary_that_is_not_a_tick_fails_closed(tmp_path, row, field, got):
    holes = {251, 261, 271}
    ticks = [
        {"id": i, "price": Decimal(1), "time": i}
        for i in [*range(100), *range(200, 300)]
        if i not in holes
    ]
    l1_root, l2_root, _ = hand_lake(tmp_path, {100000: [row]}, ticks)
    with pytest.raises(FrameBoundaryError) as error:
        for _ in read_row_groups("0.001", l1_root, l2_root, MONTH, MONTH):
            pass
    assert error.value.field == field
    assert error.value.got == got
    assert error.value.theta == "0.00100000"


def test_a_theta_without_events_delivers_nothing(tmp_path):
    l1_root, l2_root, _ = hand_lake(tmp_path, {100000: []})
    assert list(read_row_groups("0.001", l1_root, l2_root, MONTH, MONTH)) == []


def test_arguments_are_validated_before_iterating(tmp_path):
    with pytest.raises(ValueError, match="al menos un θ"):
        read_row_groups([], tmp_path, tmp_path, MONTH, MONTH)
    with pytest.raises(ValueError, match="sin repetidos"):
        read_row_groups(["0.001", "0.001000"], tmp_path, tmp_path, MONTH, MONTH)
    with pytest.raises(ValueError, match="posterior a end"):
        read_row_groups("0.001", tmp_path, tmp_path, SECOND_MONTH, MONTH)
    with pytest.raises(ValueError, match="entre 1 y 12"):
        read_row_groups("0.001", tmp_path, tmp_path, (2017, 0), MONTH)
    with pytest.raises(ValueError, match="8 decimales"):
        read_row_groups("0.0000000001", tmp_path, tmp_path, MONTH, MONTH)


# `readers` > 1: R row groups se leen a la vez y salen en orden, con lo mismo que el lector
# secuencial. La comparación es row group a row group.


def assert_same_delivery(got: RowGroupFrames, want: RowGroupFrames) -> None:
    assert (got.month, got.before_start) == (want.month, want.before_start)
    for name in COLUMNS:
        assert getattr(got.frame, name).equals(getattr(want.frame, name)), name
    assert [te.theta for te in got.thetas] == [te.theta for te in want.thetas]
    for a, b in zip(got.thetas, want.thetas, strict=True):
        assert a.events.equals(b.events)
        assert a.positions.tolist() == b.positions.tolist()
        assert a.starts_before.tolist() == b.starts_before.tolist()
        assert a.ends_after.tolist() == b.ends_after.tolist()
        assert a.ticks_before_month.tolist() == b.ticks_before_month.tolist()


@pytest.mark.parametrize("readers", [2, 3, 8])
def test_readers_deliver_the_same_row_groups_as_the_sequential_reader(
    tmp_path, readers
):
    l1_root, l2_root, *_ = build_two_month_lake(tmp_path)
    for month in (MONTH, SECOND_MONTH):
        want = list(read_row_groups(THETAS, l1_root, l2_root, month, month))
        got = list(
            read_row_groups(THETAS, l1_root, l2_root, month, month, readers=readers)
        )
        assert len(got) == len(want) > 2
        for g, w in zip(got, want, strict=True):
            assert_same_delivery(g, w)


def test_readers_skip_the_same_row_groups_when_a_gap_leaves_some_unneeded(tmp_path):
    # Nada toca los row groups 3 a 5: se saltan con las estadísticas, con o sin hilos.
    events = {100000: [event(10, 50, 60), event(660, 670, 690)]}
    l1_root, l2_root, _ = hand_lake(tmp_path, events)
    thetas = Decimal("0.001")
    want = list(read_row_groups(thetas, l1_root, l2_root, MONTH, MONTH))
    got = list(read_row_groups(thetas, l1_root, l2_root, MONTH, MONTH, readers=3))
    assert len(want) < GROUPS - 1
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert_same_delivery(g, w)


def test_readers_decode_each_row_group_at_most_once(tmp_path, monkeypatch):
    l1_root, l2_root, *_ = build_two_month_lake(tmp_path)
    sequential = count_decodes(monkeypatch)
    delivered = list(
        read_row_groups(THETAS, l1_root, l2_root, SECOND_MONTH, SECOND_MONTH)
    )
    sequential_reads = list(sequential)
    sequential.clear()
    together = list(
        read_row_groups(THETAS, l1_root, l2_root, SECOND_MONTH, SECOND_MONTH, readers=3)
    )
    assert len(together) == len(delivered)
    assert len(sequential) == len(set(sequential))
    # Se decodifican los mismos y, a lo sumo, los `readers` que el cierre del último evento dejó sin pedir.
    assert set(sequential_reads) <= set(sequential)
    assert len(sequential) - len(sequential_reads) <= 3


def test_readers_must_be_at_least_one(tmp_path):
    l1_root, l2_root, *_ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    with pytest.raises(ValueError, match="readers 0"):
        read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH, readers=0)


@pytest.mark.parametrize("readers", [1, 3])
def test_ids_that_do_not_grow_fail_the_same_way_with_readers(tmp_path, readers):
    ticks = [{"id": i, "price": Decimal(1), "time": i} for i in range(SIZE)]
    ticks += [
        {"id": i, "price": Decimal(1), "time": i} for i in range(SIZE - 1, 3 * SIZE)
    ]
    events = {100000: [event(10, 150, 250)]}
    l1_root, l2_root, _ = hand_lake(tmp_path, events, ticks)
    with pytest.raises(FramesInputError, match="no supera el id"):
        list(
            read_row_groups(
                Decimal("0.001"), l1_root, l2_root, MONTH, MONTH, readers=readers
            )
        )


def test_closing_the_reader_early_ends_its_threads(tmp_path):
    l1_root, l2_root, *_ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    before = threading.active_count()
    rows = read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH, readers=3)
    next(rows)
    rows.close()
    assert threading.active_count() == before


def test_the_reader_threads_run_in_the_context_of_the_consumer(tmp_path, monkeypatch):
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="")
    seen: list[str] = []
    real = lake_module.open_parquet

    def open_parquet(path, what, metadata=None, pre_buffer=False):
        if metadata is not None:  # la apertura de un hilo del lector
            seen.append(marker.get())
        return real(path, what, metadata, pre_buffer)

    monkeypatch.setattr(lake_module, "open_parquet", open_parquet)
    l1_root, l2_root, *_ = build_lake(tmp_path, l1_row_group=ROW_GROUP)
    marker.set("unidad-7")
    assert list(read_row_groups(THETAS, l1_root, l2_root, MONTH, MONTH, readers=2))
    assert seen and set(seen) == {"unidad-7"}
