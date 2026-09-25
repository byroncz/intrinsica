#!/usr/bin/env bash
# Prueba de humo de la imagen: procesa un día real (el primero de la serie)
# dentro del contenedor y verifica las tres salidas. Uso: smoke.sh <imagen>
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"
data="$(mktemp -d)"
trap 'docker run --rm -v "$data:/data" --entrypoint rm "$image" -rf /data/landing /data/dq /data/manifest; rm -rf "$data"' EXIT

docker run --rm -v "$data:/data" \
  -e L1_LANDING_ROOT=/data/landing \
  -e L1_DQ_ROOT=/data/dq \
  -e L1_MANIFEST_ROOT=/data/manifest \
  "$image" --mode daily --from 2017-08-17 2>&1 | tee "$data/run.log"

partition=provider=binance/market=spot/asset=BTCUSDT/year=2017/month=08
test -s "$data/landing/$partition/provisional-day=17.parquet" \
  || { echo "::error::falta landing/$partition/provisional-day=17.parquet"; exit 1; }
for root in dq manifest; do
  find "$data/$root" -name '*.parquet' -size +0 2>/dev/null | grep -q . \
    || { echo "::error::no hay Parquet en $root"; exit 1; }
done
grep -q content_hash= "$data/run.log" \
  || { echo "::error::la salida no trae content_hash"; exit 1; }
echo "humo OK: $(grep -o 'content_hash=[0-9a-f]*' "$data/run.log")"
