from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l1_ingest.schema import OUTPUT_SCHEMA
from l1_ingest.write import (
    content_hash,
    day_filename,
    partition_path,
    write_partition,
)


def _table(n: int = 5) -> pa.Table:
    return pa.table(
        {
            "agg_trade_id": list(range(n)),
            "price": [Decimal("100.12345678")] * n,
            "quantity": [Decimal("0.5")] * n,
            "first_trade_id": list(range(n)),
            "last_trade_id": list(range(n)),
            "transact_time": [1_700_000_000_000_000 + i for i in range(n)],
            "is_buyer_maker": [i % 2 == 0 for i in range(n)],
            "is_best_match": [True] * n,
        },
        schema=OUTPUT_SCHEMA,
    )


def test_partition_path(tmp_path):
    path = partition_path(
        tmp_path, "binance", "spot", "BTCUSDT", 2024, 3, "consolidated.parquet"
    )
    assert path == (
        f"{tmp_path}/provider=binance/market=spot/asset=BTCUSDT"
        "/year=2024/month=03/consolidated.parquet"
    )


def test_partition_path_gcs_root():
    path = partition_path("gs://b/l1/", "binance", "spot", "BTCUSDT", 2024, 12, "x")
    assert path == (
        "gs://b/l1/provider=binance/market=spot/asset=BTCUSDT/year=2024/month=12/x"
    )


def test_day_filename():
    assert day_filename(3) == "provisional-day=03.parquet"
    assert day_filename(21) == "provisional-day=21.parquet"


def test_write_partition_physical_properties(tmp_path):
    path = partition_path(tmp_path, "binance", "spot", "BTCUSDT", 2024, 3, "c.parquet")
    assert write_partition(_table(), path) == path

    pf = pq.ParquetFile(path)
    assert pf.schema_arrow.equals(OUTPUT_SCHEMA)
    meta = pf.metadata
    assert meta.num_rows == 5
    for rg in range(meta.num_row_groups):
        row_group = meta.row_group(rg)
        assert [(s.column_index, s.descending) for s in row_group.sorting_columns] == [
            (5, False),
            (0, False),
        ]
        for c in range(row_group.num_columns):
            col = row_group.column(c)
            assert col.compression == "ZSTD"
            assert col.is_stats_set
            assert col.statistics.has_min_max


def test_write_partition_rejects_other_schema(tmp_path):
    with pytest.raises(ValueError):
        write_partition(pa.table({"a": [1]}), str(tmp_path / "x.parquet"))


def test_overwrite_leaves_single_file(tmp_path):
    path = partition_path(tmp_path, "binance", "spot", "BTCUSDT", 2024, 3, "c.parquet")
    write_partition(_table(3), path)
    write_partition(_table(5), path)
    assert pq.read_table(path).num_rows == 5
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["c.parquet"]


def test_content_hash_ignores_chunking():
    table = _table(6)
    chunked = pa.concat_tables([table.slice(0, 2), table.slice(2)])
    assert chunked.column(0).num_chunks == 2
    assert content_hash(chunked) == content_hash(table)


def test_content_hash_changes_with_a_row():
    table = _table()
    ids = table.column("agg_trade_id").to_pylist()
    ids[2] += 100
    other = table.set_column(0, OUTPUT_SCHEMA.field(0), pa.array(ids, pa.int64()))
    assert content_hash(other) != content_hash(table)


def test_content_hash_survives_roundtrip(tmp_path):
    table = _table()
    path = str(tmp_path / "c.parquet")
    write_partition(table, path)
    assert content_hash(pq.read_table(path, schema=OUTPUT_SCHEMA)) == content_hash(
        table
    )
