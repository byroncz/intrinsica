import pyarrow as pa
from dq.schema import FINDING_SCHEMA

EXPECTED = [
    ("finding_id", pa.string()),
    ("detected_at", pa.int64()),
    ("layer", pa.string()),
    ("mode", pa.string()),
    ("check_type", pa.string()),
    ("severity", pa.string()),
    ("stage", pa.string()),
    ("status", pa.string()),
    ("provider", pa.string()),
    ("market", pa.string()),
    ("asset", pa.string()),
    ("year", pa.int32()),
    ("month", pa.int32()),
    ("metric_value", pa.float64()),
    ("details", pa.string()),
    ("run_id", pa.string()),
    ("image_version", pa.string()),
]


def test_names_order_and_types():
    assert [(f.name, f.type) for f in FINDING_SCHEMA] == EXPECTED


def test_only_metric_value_is_nullable():
    nullable = [f.name for f in FINDING_SCHEMA if f.nullable]
    assert nullable == ["metric_value"]
