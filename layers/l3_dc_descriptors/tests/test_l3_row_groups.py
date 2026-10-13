"""El resumidor por row group contra el camino por evento y contra la agregación ingenua
(TRD-L3 §13, criterios 3 y 7; ADR-L3-14).

Tres implementaciones de la agregación deben dar la misma tabla: la de producción (por row
group), el camino por evento (`by_event`, Arrow compute sobre el lector en modo por chunks)
y la ingenua de `oracle`. Los row groups de L1 son de 50 ticks para que los eventos crucen
row groups y las fronteras caigan en sus bordes.
"""

import bisect
from decimal import Decimal

import by_event
import oracle
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors import SUMMARIES_SCHEMA, summarize_month
from l3_lake import M1, build_lake, fixture_ticks, month_cuts
from pyutils import content_hash

ROW_GROUP = 50


def unique(lake):
    """Los eventos de todos los θ y meses como `(R, C, E, confirm_time)`, sin repetir."""
    seen = {}
    for rows in lake.events.values():
        for e in rows:
            R, C, E = (
                e[f"{n}_agg_trade_id"] for n in ("reference", "confirm", "extreme")
            )
            seen[R, C, E] = e["confirm_time"]
    return [(*key, time) for key, time in seen.items()]


def edges_of(lake) -> dict[str, int]:
    """Cuántos eventos tocan cada borde que el criterio 7 de §13 pide probar."""
    firsts, lasts, highs = set(), set(), []
    for month in lake.months:
        ids = [t.id for t in lake.ticks[month]]
        for i in range(0, len(ids), ROW_GROUP):
            firsts.add(ids[i])
            lasts.add(ids[min(i + ROW_GROUP, len(ids)) - 1])
            highs.append(ids[min(i + ROW_GROUP, len(ids)) - 1])
    ticks = lake.all_ticks()
    position = {t.id: i for i, t in enumerate(ticks)}
    group = lambda tick_id: bisect.bisect_left(highs, tick_id)
    found = dict.fromkeys(
        ("first", "last", "crosses", "over_two", "empty_overshoot", "split_instant"), 0
    )
    for R, C, E, time in unique(lake):
        found["first"] += sum(v in firsts for v in (R, C, E))
        found["last"] += sum(v in lasts for v in (R, C, E))
        found["crosses"] += group(R) != group(E)
        found["over_two"] += group(E) - group(R) >= 2
        found["empty_overshoot"] += E == C
        i = j = position[C]
        while j and ticks[j - 1].time == time and ticks[j - 1].id > R:
            j -= 1
        found["split_instant"] += group(ticks[j].id) != group(C)
    return found


@pytest.fixture(scope="module")
def edge_lake(tmp_path_factory):
    ticks = fixture_ticks(seed=11, drop=0.12)
    lake = build_lake(
        tmp_path_factory.mktemp("rowgroups"),
        ticks,
        month_cuts(ticks, 2300),
        l1_row_group=ROW_GROUP,
    )
    results = {
        month: summarize_month(
            lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
        )
        for month in lake.months
    }
    return lake, results


def test_the_lake_covers_the_edges_of_criterion_7(edge_lake):
    lake, _ = edge_lake
    found = edges_of(lake)
    assert all(found.values()), found


def test_same_table_and_content_hash_as_the_event_path(edge_lake, tmp_path):
    lake, results = edge_lake
    compared = 0
    for month, per_theta in results.items():
        oracle_results = by_event.summarize_month_by_event(
            lake.l1, lake.l2, tmp_path, month, lake.theta_texts
        )
        for theta, result in per_theta.items():
            wanted = oracle_results[theta]
            assert result.rows == wanted["rows"]
            assert result.content_hash == wanted["content_hash"]
            table = pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
            assert table.equals(pq.read_table(wanted["path"], schema=SUMMARIES_SCHEMA))
            assert content_hash(table) == result.content_hash
            # Lo que alimenta la conciliación de §9.1 sale igual por los dos caminos.
            assert result.reconciliation.closed == wanted["closed_ticks"]
            assert result.reconciliation.before_month == wanted["before_month"]
            compared += result.rows
    assert compared == sum(len(rows) for rows in lake.events.values())


def test_matches_the_naive_aggregation(edge_lake):
    lake, results = edge_lake
    ticks = lake.all_ticks()
    for month, per_theta in results.items():
        for theta, result in per_theta.items():
            rows = lake.events[int(theta.scaleb(8)), month]
            table = pq.read_table(result.path, schema=SUMMARIES_SCHEMA).to_pylist()
            assert len(table) == len(rows)
            for row, event in zip(table, rows, strict=True):
                assert row == oracle.from_ticks(event | {"theta": theta}, ticks)


def test_rerun_gives_the_same_file(edge_lake):
    lake, results = edge_lake
    month = lake.months[-1]
    again = summarize_month(
        lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
    )
    for theta, result in again.items():
        assert result.content_hash == results[month][theta].content_hash


def test_overflowing_notional_in_one_tick_and_across_row_groups(tmp_path):
    """Criterio 3 de §13: el producto de un solo tick ya desborda `int64` (10²⁴ contra
    9,2·10¹⁸), y una fase con esa suma también cruza row groups.
    """
    ticks = fixture_ticks(seed=5, price_factor=25, quantity=5_000)
    limit = 2**63
    assert max(t.price * t.quantity for t in ticks) * 10**16 > limit
    lake = build_lake(
        tmp_path, ticks, thetas=(250_000, 1_000_000), l1_row_group=ROW_GROUP
    )
    results = summarize_month(
        lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1
    )
    by_id = {t.id: i for i, t in enumerate(ticks)}
    big_and_crossing = 0
    for theta, result in results.items():
        table = pq.read_table(result.path, schema=SUMMARIES_SCHEMA).to_pylist()
        events = lake.events[int(theta.scaleb(8)), M1]
        assert len(table) == len(events) > 0
        for row, event in zip(table, events, strict=True):
            assert row == oracle.from_ticks(event | {"theta": theta}, ticks)
            first = by_id[event["reference_agg_trade_id"]] + 1
            last = by_id[event["confirm_agg_trade_id"]]
            notional = sum(t.price * t.quantity for t in ticks[first : last + 1])
            crosses = first // ROW_GROUP != last // ROW_GROUP
            big_and_crossing += crosses and notional * Decimal(10) ** 16 > limit
    assert big_and_crossing
