#!/usr/bin/env bash
# Prueba de humo de la imagen: arma un lago chico (L1, L2 y un catálogo temporal) con el
# generador de las pruebas (tests/l3_lake.py: ticks reales y eventos del detector real de
# L2), corre backfill dentro del contenedor y verifica una fila por evento; vuelve a
# correr y comprueba que no escribe nada (la frontera); y corre --force --from
# comprobando el mismo content_hash. Uso: smoke.sh <imagen>
#
# El lago se genera en el host con `uv run` (necesita dc_pyo3, que es Rust y no viaja en
# la imagen); todo lo demás corre en el contenedor, con el Python y el pyarrow de la
# imagen. El catálogo de θ es de este lago y no se versiona: L3 no lleva ninguno propio.
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
data="$(mktemp -d)"
trap 'docker run --rm -v "$data:/data" --entrypoint rm "$image" -rf /data/l1 /data/l2 /data/l3 /data/dq /data/thetas.yaml; rm -rf "$data"' EXIT

(cd "$root" && uv run --locked --no-dev --all-packages python \
  "$here/tests/smoke_lake.py" "$data")

cli() {
  docker run --rm -v "$data:/data" \
    -e L3_LANDING_ROOT=/data/l1 \
    -e L3_EVENTS_ROOT=/data/l2 \
    -e L3_SUMMARIES_ROOT=/data/l3 \
    -e L3_DQ_ROOT=/data/dq \
    -e L3_THETAS_URI=/data/thetas.yaml \
    "$image" --series-start 2017-08 "$@" 2>&1
}
check() {
  docker run --rm -v "$data:/data" -v "$here/smoke_check.py:/smoke_check.py:ro" \
    --entrypoint python "$image" /smoke_check.py "$@"
}
changed() { find "$data/l3" "$data/dq" -type f -newer "$data/marker" | wc -l; }

# 1. backfill: tres meses, cinco θ, una fila por evento y un hallazgo por unidad.
cli --mode backfill | tee "$data/run1.log"
thetas="$(grep -c '^  - ' "$data/thetas.yaml")"
check rows "$((thetas * 3))"
units="$(grep -c '"check_type": "unit_timing"' "$data/run1.log")"
[ "$units" -eq 3 ] || { echo "::error::se esperaban 3 unit_timing y hay $units"; exit 1; }
grep -q '"check_type": "summaries_summary"' "$data/run1.log" \
  || { echo "::error::la salida no trae summaries_summary"; exit 1; }
grep -q '"content_hash"' "$data/run1.log" \
  || { echo "::error::la salida no trae content_hash"; exit 1; }

# 2. segunda corrida: la frontera lo deja todo al día y no se escribe nada.
sleep 1 && touch "$data/marker"
cli --mode backfill | tee "$data/run2.log"
grep -q 'están al día' "$data/run2.log" \
  || { echo "::error::la segunda corrida no dijo que estaba al día"; exit 1; }
[ "$(changed)" -eq 0 ] \
  || { echo "::error::la segunda corrida escribió $(changed) archivos"; exit 1; }

# 3. --force --from: reprocesa dos meses y deja el mismo content_hash.
cli --mode backfill --force --from 2017-09 | tee "$data/run3.log"
check rows "$((thetas * 3))"
check hashes 2
echo "humo OK: $(grep -o -m1 '"content_hash": "[0-9a-f]*"' "$data/run1.log")"
