#!/usr/bin/env bash
# Prueba de humo de la imagen: procesa un día real (2017-08-18, 4 735 ticks de
# shared/dc_core/tests/fixtures/ticks.csv) dentro del contenedor y verifica las
# salidas. Uso: smoke.sh <imagen>
#
# La entrada no viene de L1: el CSV se convierte a un consolidated.parquet con
# el esquema de L1 usando el pyarrow de la propia imagen, así el humo no exige
# red ni una imagen de L1 y prueba el mismo pyarrow que corre en producción.
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ticks="$root/shared/dc_core/tests/fixtures/ticks.csv"
data="$(mktemp -d)"
trap 'docker run --rm -v "$data:/data" --entrypoint rm "$image" -rf /data/landing /data/events /data/dq; rm -rf "$data"' EXIT

partition=provider=binance/market=spot/asset=BTCUSDT/year=2017/month=08

# El contenedor corre como root: crear la ruta desde ahí evita que el trap
# tenga que borrar archivos ajenos.
docker run --rm -i -v "$data:/data" -v "$ticks:/ticks.csv:ro" \
  --entrypoint python "$image" - /ticks.csv "/data/landing/$partition/consolidated.parquet" <<'PY'
import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

src, dst = sys.argv[1:3]
ticks = pc.read_csv(
    src, convert_options=pc.ConvertOptions(column_types={"price": pa.string()})
)
n = ticks.num_rows
dec = pa.decimal128(18, 8)
# Esquema de OUTPUT_SCHEMA de L1 (layers/l1_ingest/src/l1_ingest/schema.py).
schema = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", dec, nullable=False),
        pa.field("quantity", dec, nullable=False),
        pa.field("first_trade_id", pa.int64(), nullable=False),
        pa.field("last_trade_id", pa.int64(), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
        pa.field("is_best_match", pa.bool_(), nullable=False),
    ]
)
ids = ticks["agg_trade_id"]
table = pa.table(
    {
        "agg_trade_id": ids,
        "price": pa.array([Decimal(p) for p in ticks["price"].to_pylist()], dec),
        "quantity": pa.array([Decimal(1)] * n, dec),
        "first_trade_id": ids,
        "last_trade_id": ids,
        "transact_time": ticks["transact_time"],
        "is_buyer_maker": pa.array([False] * n),
        "is_best_match": pa.array([True] * n),
    },
    schema=schema,
)
Path(dst).parent.mkdir(parents=True)
pq.write_table(table, dst)
print(f"consolidated.parquet: {n} ticks")
PY

docker run --rm -v "$data:/data" \
  -e L2_LANDING_ROOT=/data/landing \
  -e L2_EVENTS_ROOT=/data/events \
  -e L2_DQ_ROOT=/data/dq \
  "$image" --mode backfill --from 2017-08 --series-start 2017-08 2>&1 | tee "$data/run.log"

for name in events.parquet carry_over.parquet; do
  found="$(find "$data/events" -name "$name" -size +0 | wc -l)"
  [ "$found" -eq 50 ] \
    || { echo "::error::se esperaban 50 $name y hay $found"; exit 1; }
done
find "$data/dq" -name '*.parquet' -size +0 2>/dev/null | grep -q . \
  || { echo "::error::no hay Parquet en dq"; exit 1; }
grep -q content_hash= "$data/run.log" \
  || { echo "::error::la salida no trae content_hash"; exit 1; }
echo "humo OK: $(grep -o -m1 'events_content_hash=[0-9a-f]*' "$data/run.log")"
