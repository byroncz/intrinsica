"""Esquemas Arrow de la salida de L2: `events.parquet` y `carry_over.parquet`.

Contrato en `docs/data-contracts.md` ("Salida Parquet de L2") y TRD-L2 §7.2 y
§7.4. Una prueba rompe el CI si esa tabla se desvía de estos esquemas.
"""

import pyarrow as pa

PRICE_TYPE = pa.decimal128(18, 8)
THETA_TYPE = pa.decimal128(9, 8)


def _point(name: str, *, nullable: bool = False) -> list[pa.Field]:
    """Las tres columnas de un punto: precio, tiempo (µs UTC) y `agg_trade_id`."""
    return [
        pa.field(f"{name}_price", PRICE_TYPE, nullable=nullable),
        pa.field(f"{name}_time", pa.int64(), nullable=nullable),
        pa.field(f"{name}_agg_trade_id", pa.int64(), nullable=nullable),
    ]


EVENTS_SCHEMA = pa.schema(
    [
        *_point("reference"),
        *_point("confirm"),
        *_point("extreme"),
        pa.field("direction", pa.int8(), nullable=False),
        pa.field("theta", THETA_TYPE, nullable=False),
    ]
)

# Orden físico de `events.parquet`: el de confirmación dentro de la partición.
EVENTS_SORT_ORDER = [("confirm_time", "ascending")]

CARRY_OVER_SCHEMA = pa.schema(
    [
        pa.field("provider", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset", pa.string(), nullable=False),
        pa.field("theta", THETA_TYPE, nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("month", pa.int32(), nullable=False),
        pa.field("state_version", pa.string(), nullable=False),
        pa.field("direction", pa.int8(), nullable=False),
        *_point("ext_high"),
        *_point("ext_low"),
        pa.field("has_pending_event", pa.bool_(), nullable=False),
        *_point("pending_reference", nullable=True),
        *_point("pending_confirm", nullable=True),
    ]
)
