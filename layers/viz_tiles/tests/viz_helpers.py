import csv
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
from viz_tiles.reduce import day_start_us

FIXTURES = Path(__file__).resolve().parents[3] / "shared/dc_core/tests/fixtures"

TICKS_SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", pa.decimal128(18, 8), nullable=False),
        pa.field("quantity", pa.decimal128(18, 8), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
    ]
)


def ticks_batch(day, rows) -> pa.RecordBatch:
    """Un lote de ticks a partir de `(id, segundos desde el día, precio, cantidad)`."""
    t0 = day_start_us(day)
    return pa.record_batch(
        [
            pa.array([r[0] for r in rows], pa.int64()),
            pa.array([Decimal(str(r[2])) for r in rows], pa.decimal128(18, 8)),
            pa.array([Decimal(str(r[3])) for r in rows], pa.decimal128(18, 8)),
            pa.array([t0 + round(r[1] * 1_000_000) for r in rows], pa.int64()),
        ],
        schema=TICKS_SCHEMA,
    )


EVENT_SCHEMA = pa.schema(
    [
        pa.field("reference_agg_trade_id", pa.int64()),
        pa.field("confirm_agg_trade_id", pa.int64()),
        pa.field("extreme_agg_trade_id", pa.int64()),
        pa.field("direction", pa.int8()),
    ]
)


def events(*rows: tuple[int, int, int, int]) -> pa.Table:
    """Eventos `(referencia, confirmación, extremo, dirección)` con ids de tick."""
    return (
        pa.Table.from_arrays(
            [
                pa.array(col, f.type)
                for col, f in zip(zip(*rows), EVENT_SCHEMA, strict=True)
            ],
            schema=EVENT_SCHEMA,
        )
        if rows
        else EVENT_SCHEMA.empty_table()
    )


# Un alza (10, 20, 30] y una baja (30, 40, 55]: la referencia de una es el
# extremo de la otra, como en L2.
UP = (10, 20, 30, 1)
DOWN = (30, 40, 55, -1)


def read_ticks_csv() -> list[dict[str, str]]:
    """Las filas de `ticks.csv` (agg_trade_id, price, transact_time) como texto."""
    with open(FIXTURES / "ticks.csv") as f:
        return list(csv.DictReader(f))
