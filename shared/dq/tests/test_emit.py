import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from dq import Finding, emit_findings
from dq.schema import FINDING_SCHEMA

DAY1 = int(datetime(2026, 9, 24, 12, tzinfo=UTC).timestamp() * 1_000_000)
DAY2 = int(datetime(2026, 9, 25, 12, tzinfo=UTC).timestamp() * 1_000_000)


def make(**overrides) -> Finding:
    values = {
        "detected_at": DAY1,
        "layer": "l1",
        "mode": "batch",
        "check_type": "gap",
        "severity": "warning",
        "stage": "provisional",
        "status": "fail",
        "provider": "binance",
        "market": "spot",
        "asset": "BTCUSDT",
        "year": 2026,
        "month": 9,
        "details": {"missing": 3},
        "run_id": "run-1",
        "image_version": "0.1.0",
    }
    return Finding(**(values | overrides))


def test_writes_partitioned_parquet(tmp_path):
    findings = [make(), make()]
    (path,) = emit_findings(findings, tmp_path)
    file = Path(path)
    assert file.parent == tmp_path / "detected_date=2026-09-24"
    assert file.name.startswith("run-1-")
    assert file.suffix == ".parquet"
    assert pq.read_table(file).num_rows == 2


def test_append_only(tmp_path):
    findings = [make()]
    first = emit_findings(findings, tmp_path)
    second = emit_findings(findings, tmp_path)
    assert first != second
    assert len(list((tmp_path / "detected_date=2026-09-24").iterdir())) == 2


def test_one_file_per_detection_date(tmp_path):
    paths = emit_findings([make(), make(detected_at=DAY2)], tmp_path)
    parents = {Path(p).parent.name for p in paths}
    assert parents == {"detected_date=2026-09-24", "detected_date=2026-09-25"}


def test_empty_list_writes_nothing(tmp_path):
    assert emit_findings([], tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_schema_compression_and_statistics(tmp_path):
    (path,) = emit_findings([make()], tmp_path)
    parquet = pq.ParquetFile(path)
    assert parquet.schema_arrow.equals(FINDING_SCHEMA)
    row_group = parquet.metadata.row_group(0)
    for i in range(row_group.num_columns):
        column = row_group.column(i)
        assert column.compression == "ZSTD"
        assert column.is_stats_set
    # pyarrow no expone el nivel de compresión en los metadatos; se comprueba
    # que el archivo se lee y es ZSTD. El nivel 3 lo fija la llamada a write_table.


@pytest.mark.parametrize(
    ("severity", "level"),
    [("info", logging.INFO), ("warning", logging.WARNING), ("error", logging.ERROR)],
)
def test_log_line_per_finding(tmp_path, caplog, severity, level):
    finding = make(severity=severity)
    with caplog.at_level(logging.DEBUG, logger="dq"):
        emit_findings([finding], tmp_path)
    (record,) = caplog.records
    assert record.name == "dq"
    assert record.levelno == level
    message = record.getMessage()
    for expected in (
        "check_type=gap",
        "status=fail",
        "stage=provisional",
        "provider=binance",
        "market=spot",
        "asset=BTCUSDT",
        "year=2026",
        "month=9",
        f"finding_id={finding.finding_id}",
    ):
        assert expected in message
