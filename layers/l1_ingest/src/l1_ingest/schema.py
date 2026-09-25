"""Esquemas de L1: entrada (CSV de aggTrades, §7.1) y salida (§7.2)."""

import pyarrow as pa

# price y quantity quedan como texto: tiparlos (decimal) es de otro paso.
RAW_SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", pa.string(), nullable=False),
        pa.field("quantity", pa.string(), nullable=False),
        pa.field("first_trade_id", pa.int64(), nullable=False),
        pa.field("last_trade_id", pa.int64(), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
        pa.field("is_best_match", pa.bool_(), nullable=False),
    ]
)

# price y quantity exactos (ADR-L1-03); transact_time siempre en µs UTC (ADR-L1-02).
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", pa.decimal128(18, 8), nullable=False),
        pa.field("quantity", pa.decimal128(18, 8), nullable=False),
        pa.field("first_trade_id", pa.int64(), nullable=False),
        pa.field("last_trade_id", pa.int64(), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
        pa.field("is_best_match", pa.bool_(), nullable=False),
    ]
)
