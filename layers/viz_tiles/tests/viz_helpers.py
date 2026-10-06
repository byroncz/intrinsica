import csv
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
from viz_tiles.lake import MonthEvents
from viz_tiles.ticks import day_start_us

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


def read_ticks_csv() -> list[dict[str, str]]:
    """Las filas de `ticks.csv` (agg_trade_id, price, transact_time) como texto."""
    with open(FIXTURES / "ticks.csv") as f:
        return list(csv.DictReader(f))


def month_events(day, *rows: tuple) -> MonthEvents:
    """Eventos `(ref_id, confirm_id, extremo_id, dirección, ref_s, confirm_s, extremo_s)`.

    Los tres últimos son segundos desde el inicio del día (pueden salirse de él).
    `confirm_id` no viaja: el día ya no lo necesita.
    """
    if not rows:
        return MonthEvents.empty()
    t0 = day_start_us(day)
    us = [[t0 + round(r[i] * 1_000_000) for r in rows] for i in (4, 5, 6)]
    return MonthEvents(
        np.array([r[0] for r in rows], np.int64),
        np.array([r[2] for r in rows], np.int64),
        *(np.array(col, np.int64) for col in us),
        np.array([r[3] for r in rows], np.int8),
    )


# El alza (10, 20, 30] de 0,5 s a 50 s y la baja (30, 40, 55] de 50 s a 85 s: la
# referencia de una es el extremo de la otra, como en L2.
UP_T = (10, 20, 30, 1, 0.5, 20, 50)
DOWN_T = (30, 40, 55, -1, 50, 60, 85)
