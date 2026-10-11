"""Sin estado entre meses: el corte de meses no cambia las filas y cada mes concilia (TRD-L3 §13, 4 y 5)."""

from bisect import bisect_right
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from frames_lake import write_carry_over
from l3_dc_descriptors import SUMMARIES_SCHEMA, Reconciliation, summarize_month
from l3_lake import M1, Tick, build_lake, fixture_ticks, month_cuts
from pyutils import content_hash

TICKS = fixture_ticks()
BIG = 2_000_000  # el θ con menos eventos: un evento abarca varios meses


def run(lake):
    return {
        month: summarize_month(
            lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
        )
        for month in lake.months
    }


def logical_hash(results, theta) -> str:
    """El hash de las filas del θ de todos los meses, en orden, como una sola tabla."""
    tables = [
        pq.read_table(per_theta[theta].path, schema=SUMMARIES_SCHEMA)
        for per_theta in results.values()
    ]
    return content_hash(pa.concat_tables(tables))


def build(tmp_path, *positions):
    lake = build_lake(tmp_path, TICKS, month_cuts(TICKS, *positions))
    return lake, run(lake)


@pytest.fixture(scope="module")
def one_pass(tmp_path_factory):
    """Una sola unidad con todos los ticks: la referencia de la invariancia."""
    lake = build_lake(tmp_path_factory.mktemp("one"), TICKS)
    results = run(lake)
    return {
        theta: logical_hash(results, theta) for theta in next(iter(results.values()))
    }


def naive_reconciliation(lake, month, theta) -> dict:
    """§9.1 con listas de Python: sin pie de Parquet, sin `searchsorted`."""
    key = int(theta.scaleb(8))
    index = lake.months.index(month)
    ticks = lake.ticks[month]
    ids = [t.id for t in ticks]
    first = ids[0]
    rows = lake.events[key, month]
    carry = lake.carry[key, month]
    previous = lake.carry[key, lake.months[index - 1]].direction if index else 0
    if previous != 0:
        tau = -1
    elif carry.direction != 0:
        tau = rows[0]["reference_agg_trade_id"] if rows else carry.pending[0][2]
    else:
        tau = min(carry.ext_high[2], carry.ext_low[2])
    last = max([r["extreme_agg_trade_id"] for r in rows], default=-1)
    every = all_ids(lake)  # crece: es el orden de la serie
    closed = before = 0
    for row in rows:
        r, e = row["reference_agg_trade_id"], row["extreme_agg_trade_id"]
        inside = every[bisect_right(every, r) : bisect_right(every, e)]
        closed += len(inside)
        before += sum(1 for i in inside if i < first)
    return {
        "n": len(ticks),
        "closed": closed,
        "before_month": before,
        "tail": sum(1 for i in ids if i > max(tau, last)),
        "unassigned": sum(1 for i in ids if i <= tau),
    }


def all_ids(lake) -> list[int]:
    return [t.id for t in lake.all_ticks()]


def check_reconciliation(lake, results):
    for month, per_theta in results.items():
        for theta, result in per_theta.items():
            fields = result.reconciliation
            naive = naive_reconciliation(lake, month, theta)
            assert {
                "n": fields.n,
                "closed": fields.closed,
                "before_month": fields.before_month,
                "tail": fields.tail,
                "unassigned": fields.unassigned,
            } == naive
            assert fields.balanced


@pytest.mark.parametrize(
    "positions",
    [
        (1500, 3200),  # eventos que cruzan el borde y eventos tardíos
        (1000, 1100),  # un evento de θ grande abarca tres meses
        (60, 120, 600),  # direction = 0 en los dos primeros meses
        (100, 180),  # R₁ en el penúltimo mes y la confirmación en el último
        (400, 900, 1700, 2600, 3500),  # cinco cortes
    ],
)
def test_cutting_months_does_not_change_the_rows(tmp_path, one_pass, positions):
    lake, results = build(tmp_path, *positions)
    for theta in next(iter(results.values())):
        assert logical_hash(results, theta) == one_pass[theta]
    check_reconciliation(lake, results)


def test_events_cross_the_border_and_close_before_the_month(tmp_path):
    lake, results = build(tmp_path, 1500, 3200)
    crossing = late = 0
    for month, per_theta in results.items():
        first = lake.ticks[month][0].id
        for theta, result in per_theta.items():
            crossing += result.reconciliation.before_month > 0
            rows = lake.events[int(theta.scaleb(8)), month]
            late += any(r["extreme_agg_trade_id"] < first for r in rows)
    assert crossing and late


def test_an_event_spans_two_earlier_months(tmp_path):
    lake, results = build(tmp_path, 1000, 1100)
    first = [lake.ticks[m][0].id for m in lake.months]
    spans = [
        row
        for k, month in enumerate(lake.months)
        for row in lake.events[BIG, month]
        if k >= 2 and row["reference_agg_trade_id"] < first[k - 1]
    ]
    assert spans
    # Sus ticks de dos meses anteriores se cuentan en P_M del mes que lo cierra.
    closing = next(
        r for t, r in results[lake.months[2]].items() if t == Decimal("0.02")
    )
    assert closing.reconciliation.before_month > len(lake.ticks[lake.months[1]])
    check_reconciliation(lake, results)


@pytest.mark.parametrize(
    ("positions", "reference_in_last"),
    [((60, 120), True), ((100, 180), False)],
)
def test_direction_zero_for_several_months(tmp_path, positions, reference_in_last):
    lake, results = build(tmp_path, *positions)
    theta = BIG
    first, second, last = lake.months
    # Dos meses seguidos sin ninguna confirmación: todo lo que queda atrás es no asignado.
    for month in (first, second):
        assert lake.carry[theta, month].direction == 0
        assert lake.events[theta, month] == []
    for month in (first, second):
        outcome = next(
            r for t, r in results[month].items() if int(t.scaleb(8)) == theta
        )
        assert outcome.rows == 0
        assert outcome.reconciliation.tail > 0
    first_outcome = next(
        r for t, r in results[first].items() if int(t.scaleb(8)) == theta
    )
    assert first_outcome.reconciliation.unassigned > 0
    assert lake.carry[theta, last].direction != 0
    in_last = next(
        r for t, r in results[last].items() if int(t.scaleb(8)) == theta
    ).reconciliation
    # R₁ dentro del mes de la confirmación: hay no asignados; si quedó en un mes
    # anterior, el mes de la confirmación ya no los tiene.
    assert (in_last.unassigned > 0) == reference_in_last
    check_reconciliation(lake, results)


def test_unassigned_ticks_in_several_consecutive_months(tmp_path):
    """`U_M` ≠ 0 en tres meses seguidos con `direction = 0`: τ es el menor extremo de cada mes."""
    ticks = [
        Tick(i, 1_000 + i, Decimal(100), Decimal(1), i % 2 == 0) for i in range(1, 301)
    ]
    lake = build_lake(tmp_path, ticks, [100, 200], thetas=(2_000_000,))
    lake_events = tmp_path / "l2"
    for number, (high, low) in zip(
        (8, 9, 10), ((50, 40), (180, 130), (250, 210)), strict=True
    ):
        write_carry_over(
            lake_events, 2_000_000, (2017, number), 0, ext_high=high, ext_low=low
        )
    wanted = [(40, 60), (30, 70), (10, 90)]
    for month, (unassigned, tail) in zip(lake.months, wanted, strict=True):
        (outcome,) = summarize_month(
            lake.l1, lake.l2, lake.l3, month, ["0.02000000"], series_start=M1
        ).values()
        assert outcome.rows == 0
        assert outcome.reconciliation == Reconciliation(100, 0, 0, tail, unassigned)
