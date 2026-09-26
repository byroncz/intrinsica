import json
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dq.reader import current_findings
from l1_ingest.cli import main
from l1_ingest.schema import OUTPUT_SCHEMA
from l1_ingest.write import CONSOLIDATED, day_filename, partition_path, write_partition


def _put(root, year, month, first, last, name=CONSOLIDATED):
    n = last - first + 1
    table = pa.table(
        {
            "agg_trade_id": list(range(first, last + 1)),
            "price": [Decimal(1)] * n,
            "quantity": [Decimal(1)] * n,
            "first_trade_id": list(range(first, last + 1)),
            "last_trade_id": list(range(first, last + 1)),
            "transact_time": [year * 100 + month + i for i in range(n)],
            "is_buyer_maker": [True] * n,
            "is_best_match": [True] * n,
        },
        schema=OUTPUT_SCHEMA,
    )
    path = partition_path(root, "binance", "spot", "BTCUSDT", year, month, name)
    return write_partition(table, path)


def _env(tmp_path):
    return {
        "L1_LANDING_ROOT": str(tmp_path / "landing"),
        "L1_DQ_ROOT": str(tmp_path / "dq"),
        "L1_MANIFEST_ROOT": str(tmp_path / "manifest"),
        "IMAGE_VERSION": "0.3.0+test",
    }


def _run(tmp_path, to):
    argv = ["--mode", "seam-check", "--from", "2024-01", "--to", to]
    assert main(argv, _env(tmp_path)) == 0
    rows = current_findings(tmp_path / "dq").to_pylist()
    return sorted(rows, key=lambda r: r["month"])


def test_seam_pass_fail_and_missing(tmp_path, monkeypatch):
    landing = tmp_path / "landing"
    _put(landing, 2024, 1, 1, 10)
    _put(landing, 2024, 2, 11, 20)  # pass
    _put(landing, 2024, 3, 25, 30)  # hueco de 4
    _put(landing, 2024, 4, 28, 40)  # solape de 3 (max 30, min 28)
    # 2024-05 ausente; 2024-06 sí
    _put(landing, 2024, 6, 50, 60)

    def boom(*args, **kwargs):
        raise AssertionError("se leyeron datos")

    for name in ("read", "read_row_group", "read_row_groups", "iter_batches"):
        monkeypatch.setattr(pq.ParquetFile, name, boom)

    rows = _run(tmp_path, "2024-06")
    assert [r["month"] for r in rows] == [2, 3, 4, 5, 6]
    assert {r["mode"] for r in rows} == {"seam-check"}
    assert {r["stage"] for r in rows} == {"canonical"}
    assert {r["check_type"] for r in rows} == {"seam_discontinuity"}
    feb, mar, apr, may, jun = rows
    assert (feb["status"], feb["metric_value"]) == ("pass", 0)
    assert (mar["status"], mar["severity"], mar["metric_value"]) == (
        "fail",
        "warning",
        4,
    )
    d = json.loads(mar["details"])
    assert (d["prev_max"], d["next_min"]) == (20, 25)
    assert d["prev_path"].endswith(CONSOLIDATED) and d["next_path"].endswith(
        CONSOLIDATED
    )
    assert (apr["status"], apr["metric_value"]) == ("fail", -3)
    for missing in (may, jun):
        assert missing["status"] == "fail" and missing["metric_value"] is None
        assert "month=05" in json.loads(missing["details"])["missing"][0]


def test_seam_uses_provisionals_ordered_by_day(tmp_path):
    landing = tmp_path / "landing"
    _put(landing, 2024, 1, 1, 10, day_filename(2))
    _put(landing, 2024, 1, 11, 20, day_filename(10))  # último por día, no por nombre
    _put(landing, 2024, 2, 21, 30, day_filename(1))
    _put(landing, 2024, 2, 99, 100, day_filename(9))
    (feb,) = _run(tmp_path, "2024-02")
    assert (feb["status"], feb["metric_value"]) == ("pass", 0)


def test_seam_prefers_consolidated(tmp_path):
    landing = tmp_path / "landing"
    _put(landing, 2024, 1, 1, 10)
    _put(landing, 2024, 1, 500, 600, day_filename(1))
    _put(landing, 2024, 2, 11, 20)
    (feb,) = _run(tmp_path, "2024-02")
    assert feb["status"] == "pass"


@pytest.mark.parametrize("to", ["2024-01", "2023-12"])
def test_seam_empty_or_reversed_range(tmp_path, to):
    code = main(
        ["--mode", "seam-check", "--from", "2024-01", "--to", to], _env(tmp_path)
    )
    assert code == (0 if to == "2024-01" else 2)
