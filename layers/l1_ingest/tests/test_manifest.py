from dataclasses import replace

import pyarrow.parquet as pq
import pytest
from l1_ingest.manifest import MANIFEST_SCHEMA, ManifestEntry, write_manifest

ENTRY = ManifestEntry(
    provider="binance",
    market="spot",
    asset="BTCUSDT",
    year=2024,
    month=1,
    granularity="monthly",
    source_url="https://data.binance.vision/x.zip",
    sha256="a" * 64,
    downloaded_at=1_700_000_000_000_000,
    file_bytes=900,
    image_version="0.1.0+abc1234",
)


def test_schema_columns_types_and_no_nulls():
    assert [(f.name, str(f.type)) for f in MANIFEST_SCHEMA] == [
        ("provider", "string"),
        ("market", "string"),
        ("asset", "string"),
        ("year", "int32"),
        ("month", "int32"),
        ("granularity", "string"),
        ("source_url", "string"),
        ("sha256", "string"),
        ("downloaded_at", "int64"),
        ("file_bytes", "int64"),
        ("image_version", "string"),
    ]
    assert not any(f.nullable for f in MANIFEST_SCHEMA)


def test_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="granularity"):
        replace(ENTRY, granularity="weekly")


def test_write_path_schema_compression_and_statistics(tmp_path):
    (path,) = write_manifest([ENTRY], tmp_path, "run-1")

    rel = path.removeprefix(str(tmp_path.resolve()) + "/")
    directory, name = rel.rsplit("/", 1)
    assert directory == (
        "provider=binance/market=spot/asset=BTCUSDT/year=2024/month=01"
    )
    assert name.startswith("run-1-") and name.endswith(".parquet")

    meta = pq.ParquetFile(path).metadata
    assert pq.read_schema(path).equals(MANIFEST_SCHEMA)
    assert meta.num_rows == 1
    column = meta.row_group(0).column(0)
    assert column.compression == "ZSTD"
    assert column.statistics.has_min_max
    assert pq.read_table(path).to_pylist() == [{**ENTRY.__dict__}]


def test_one_file_per_scope(tmp_path):
    other = replace(ENTRY, month=2)
    paths = write_manifest([ENTRY, ENTRY, other], tmp_path, "run-1")
    assert len(paths) == 2
    assert sorted(pq.ParquetFile(p).metadata.num_rows for p in paths) == [1, 2]


def test_append_only_and_empty(tmp_path):
    first = write_manifest([ENTRY], tmp_path, "run-1")
    second = write_manifest([ENTRY], tmp_path, "run-1")
    assert first != second
    assert len(list(tmp_path.rglob("*.parquet"))) == 2
    assert write_manifest([], tmp_path, "run-1") == []
    assert len(list(tmp_path.rglob("*.parquet"))) == 2
