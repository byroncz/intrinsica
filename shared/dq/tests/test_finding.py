import json
import time
import uuid

import pytest
from dq import Finding, Severity, Stage, Status
from dq.finding import to_table
from dq.schema import FINDING_SCHEMA


def make(**overrides) -> Finding:
    values = {
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


def test_defaults():
    before = time.time_ns() // 1000
    finding = make()
    after = time.time_ns() // 1000
    assert str(uuid.UUID(finding.finding_id)) == finding.finding_id
    assert before <= finding.detected_at <= after
    assert finding.metric_value is None


def test_default_ids_are_unique():
    assert make().finding_id != make().finding_id


def test_enums_are_coerced():
    finding = make()
    assert finding.severity is Severity.WARNING
    assert finding.stage is Stage.PROVISIONAL
    assert finding.status is Status.FAIL


@pytest.mark.parametrize("field", ["severity", "stage", "status"])
def test_invalid_enum_raises(field):
    with pytest.raises(ValueError):
        make(**{field: "nope"})


def test_to_table_matches_schema():
    table = to_table([make(metric_value=1.5), make()])
    assert table.schema.equals(FINDING_SCHEMA)
    assert table.num_rows == 2
    assert table["metric_value"].to_pylist() == [1.5, None]
    assert json.loads(table["details"][0].as_py()) == {"missing": 3}
    assert table["severity"].to_pylist() == ["warning", "warning"]


def test_to_table_empty():
    assert to_table([]).schema.equals(FINDING_SCHEMA)
