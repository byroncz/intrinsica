"""Esquema Arrow del hallazgo de DQ (§7.3 del TRD-L1)."""

import pyarrow as pa

FINDING_SCHEMA = pa.schema(
    [
        pa.field("finding_id", pa.string(), nullable=False),
        # Microsegundos desde la época, UTC.
        pa.field("detected_at", pa.int64(), nullable=False),
        pa.field("layer", pa.string(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
        pa.field("check_type", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset", pa.string(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("month", pa.int32(), nullable=False),
        pa.field("metric_value", pa.float64(), nullable=True),
        # Texto JSON.
        pa.field("details", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("image_version", pa.string(), nullable=False),
    ]
)
