"""Esquema de entrada de L1: las 8 columnas del CSV de aggTrades (§7.1)."""

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
