"""`summaries.parquet` de un mes contra la agregación ingenua y contra su contrato (TRD-L3 §13, 1 a 3 y 12)."""

from decimal import Decimal

import oracle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import read_frames
from l3_dc_descriptors import (
    STATE_VERSION,
    SUMMARIES_SCHEMA,
    summarize_month,
)
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts
from pyutils import content_hash

DEC = pa.decimal128(18, 8)
CONTRACT = [
    ("theta", pa.decimal128(9, 8), False),
    ("confirm_agg_trade_id", pa.int64(), False),
    ("c_ticks", pa.int64(), False),
    ("c_quantity_buy", DEC, False),
    ("c_quantity_sell", DEC, False),
    ("c_price_max", DEC, False),
    ("c_price_min", DEC, False),
    ("c_duration", pa.int64(), False),
    ("c_vwap", DEC, False),
    ("os_ticks", pa.int64(), False),
    ("os_quantity_buy", DEC, False),
    ("os_quantity_sell", DEC, False),
    ("os_price_max", DEC, True),
    ("os_price_min", DEC, True),
    ("os_duration", pa.int64(), True),
    ("os_vwap", DEC, True),
    ("confirm_instant_ticks", pa.int64(), False),
    ("confirm_instant_price_best", DEC, False),
    ("confirm_instant_price_worst", DEC, False),
]


@pytest.fixture(scope="module")
def gappy(tmp_path_factory):
    """Dos meses con huecos de `agg_trade_id` (un 12 % de ticks borrados)."""
    ticks = fixture_ticks(seed=11, drop=0.12)
    lake = build_lake(tmp_path_factory.mktemp("gappy"), ticks, month_cuts(ticks, 2300))
    results = {
        month: summarize_month(
            lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
        )
        for month in lake.months
    }
    return lake, results


def table_of(result) -> pa.Table:
    return pq.read_table(result.path, schema=SUMMARIES_SCHEMA)


def test_contract_columns_and_types(gappy):
    _, results = gappy
    for per_theta in results.values():
        for result in per_theta.values():
            schema = pq.read_schema(result.path)
            assert [(f.name, f.type, f.nullable) for f in schema] == CONTRACT
            assert len(schema) == 19
            assert not {"c_quantity", "os_quantity"} & set(schema.names)
            assert schema.metadata[b"state_version"] == STATE_VERSION.encode()


def test_skeleton_of_l2(gappy):
    lake, results = gappy
    multiple = 0
    for month, per_theta in results.items():
        for theta, result in per_theta.items():
            text = f"{theta:.8f}"
            events = pq.ParquetFile(
                lake.l2 / f"provider=binance/market=spot/asset=BTCUSDT/theta={text}"
                f"/year={month[0]}/month={month[1]:02d}/events.parquet"
            )
            summaries = pq.ParquetFile(result.path)
            sizes = [
                events.metadata.row_group(i).num_rows
                for i in range(events.num_row_groups)
            ]
            multiple += len(sizes) > 1
            assert [
                summaries.metadata.row_group(i).num_rows
                for i in range(summaries.num_row_groups)
            ] == sizes
            keys = ["confirm_agg_trade_id"]
            assert (
                summaries.read(columns=keys).column(0).to_pylist()
                == events.read(columns=keys).column(0).to_pylist()
            )
            theta_column = summaries.read(columns=["theta"]).column(0).to_pylist()
            assert set(theta_column) == {theta}
            ids = summaries.read(columns=keys).column(0).to_pylist()
            assert ids == sorted(set(ids))
            assert result.rows == sum(sizes)
    assert multiple  # hay θ con varios row groups de eventos


def test_physical_properties(gappy):
    _, results = gappy
    result = next(iter(results[M2].values()))
    meta = pq.ParquetFile(result.path).metadata
    group = meta.row_group(0)
    for i in range(meta.num_columns):
        column = group.column(i)
        assert column.compression == "ZSTD"
        assert column.statistics is not None and column.statistics.has_min_max
        assert not {"PLAIN_DICTIONARY", "RLE_DICTIONARY"} & set(column.encodings)
        assert not column.has_dictionary_page
    sorting = group.sorting_columns
    assert [meta.schema.column(s.column_index).name for s in sorting] == [
        "confirm_agg_trade_id"
    ]


def test_matches_the_naive_aggregation_over_the_reader_frames(gappy):
    lake, results = gappy
    seen = 0
    for month, per_theta in results.items():
        expected = {}
        events = read_frames(
            [Decimal(t).scaleb(-8) for t in lake.thetas], lake.l1, lake.l2, month, month
        )
        for ev in events:
            row = oracle.from_reader(ev)
            expected[row["theta"], row["confirm_agg_trade_id"]] = row
        for theta, result in per_theta.items():
            actual = table_of(result).to_pylist()
            assert [(r["theta"], r["confirm_agg_trade_id"]) for r in actual] == sorted(
                key for key in expected if key[0] == theta
            )
            for row in actual:
                assert row == expected[row["theta"], row["confirm_agg_trade_id"]]
                seen += 1
    assert seen == sum(len(r) for r in lake.events.values())


def test_matches_a_filter_of_the_tick_list_by_id(gappy):
    """Sin lector ni Arrow: la pertenencia es `R < id ≤ C` y `C < id ≤ E` sobre la lista."""
    lake, results = gappy
    ticks = lake.all_ticks()
    for month, per_theta in results.items():
        for theta, result in per_theta.items():
            rows = lake.events[int(theta.scaleb(8)), month]
            actual = table_of(result).to_pylist()
            assert len(actual) == len(rows)
            for row, event in zip(actual, rows, strict=True):
                wanted = oracle.from_ticks(event | {"theta": theta}, ticks)
                assert row == wanted


def test_the_lake_covers_the_edge_cases(gappy):
    """Las pruebas de arriba cubren huecos en R, C y E y overshoot vacío de verdad."""
    lake, _results = gappy
    ids = {t.id for t in lake.all_ticks()}
    after = {"reference": False, "confirm": False, "extreme": False}
    empty = inside = 0
    for rows in lake.events.values():
        for event in rows:
            for name in after:
                if event[f"{name}_agg_trade_id"] + 1 not in ids:
                    after[name] = True
            empty += event["extreme_agg_trade_id"] == event["confirm_agg_trade_id"]
            span = range(event["reference_agg_trade_id"], event["extreme_agg_trade_id"])
            inside += any(i not in ids for i in span)
    assert all(after.values())
    assert empty and inside


def test_empty_overshoot_has_zero_counts_and_null_prices(gappy):
    _, results = gappy
    empties = 0
    for per_theta in results.values():
        for result in per_theta.values():
            for row in table_of(result).to_pylist():
                if row["os_ticks"]:
                    assert row["os_price_max"] is not None
                    continue
                empties += 1
                assert row["os_quantity_buy"] == row["os_quantity_sell"] == 0
                for name in ("os_price_max", "os_price_min", "os_duration", "os_vwap"):
                    assert row[name] is None
    assert empties


def test_row_invariants_hold(gappy):
    _, results = gappy
    for per_theta in results.values():
        for result in per_theta.values():
            for row in table_of(result).to_pylist():
                assert row["c_ticks"] >= row["confirm_instant_ticks"] >= 1
                assert row["c_price_min"] <= row["c_vwap"] <= row["c_price_max"]
                assert (row["c_duration"] == 0) == (
                    row["c_ticks"] == row["confirm_instant_ticks"]
                )
                if row["os_ticks"]:
                    assert row["os_price_min"] <= row["os_vwap"] <= row["os_price_max"]


def test_rerun_gives_the_same_content_hash(tmp_path, gappy):
    lake, results = gappy
    again = summarize_month(
        lake.l1, lake.l2, lake.l3, M2, lake.theta_texts, series_start=M1
    )
    for theta, result in again.items():
        assert result.content_hash == results[M2][theta].content_hash
        assert content_hash(table_of(result)) == result.content_hash
