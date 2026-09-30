from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l2_dc_events.landing import CONSOLIDATED, consolidated_path

FIXTURES = Path(__file__).resolve().parents[3] / "shared/dc_core/tests/fixtures"

# Las columnas de la landing de L1 que L2 lee, más `quantity` para comprobar
# que el lector no la trae.
SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", pa.decimal128(18, 8), nullable=False),
        pa.field("quantity", pa.decimal128(18, 8), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
    ]
)


def price_int(text: str) -> int:
    """`"4285.08000000"` → `428508000000`, el entero sin escalar del DECIMAL."""
    return int(Decimal(text).scaleb(8))


def table_of(ticks: list[tuple[int, int, int]]) -> pa.Table:
    """Una tabla con el esquema de la landing a partir de `(price, time, id)`."""
    prices = [Decimal(p).scaleb(-8) for p, _, _ in ticks]
    return pa.table(
        {
            "agg_trade_id": pa.array([i for _, _, i in ticks], pa.int64()),
            "price": pa.array(prices, pa.decimal128(18, 8)),
            "quantity": pa.array([Decimal(1)] * len(ticks), pa.decimal128(18, 8)),
            "transact_time": pa.array([t for _, t, _ in ticks], pa.int64()),
        },
        schema=SCHEMA,
    )


@pytest.fixture
def fixture_ticks() -> list[tuple[int, int, int]]:
    """Los 4 735 ticks reales de 2017-08-18 como `(price, time, agg_trade_id)`."""
    lines = (FIXTURES / "ticks.csv").read_text().splitlines()[1:]
    rows = (line.split(",") for line in lines)
    return [(price_int(p), int(t), int(i)) for i, p, t in rows]


@pytest.fixture
def write_month(tmp_path):
    """Escribe el consolidado de un mes con `row_group_size` filas por row group."""

    def write(
        ticks, *, row_group_size, year=2017, month=8, asset="BTCUSDT", name=None
    ) -> Path:
        landing = tmp_path / "landing"
        path = Path(
            consolidated_path(str(landing), "binance", "spot", asset, year, month)
        )
        if name is not None:
            path = path.with_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table_of(ticks), path, row_group_size=row_group_size)
        return path

    write.landing = tmp_path / "landing"
    write.consolidated = CONSOLIDATED
    return write
