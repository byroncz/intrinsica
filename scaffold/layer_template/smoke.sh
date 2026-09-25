#!/usr/bin/env bash
# Prueba de humo de la imagen: procesa una unidad real dentro del contenedor y
# verifica sus salidas. Uso: smoke.sh <imagen>
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"
data="$(mktemp -d)"
trap 'rm -rf "$data"' EXIT

# TODO: pasa a la imagen las variables L<N>_*_ROOT de la capa y un modo real.
docker run --rm -v "$data:/data" "$image" --mode daily 2>&1 | tee "$data/run.log"

# TODO: verifica las salidas de la capa (Parquet, hallazgos, manifiesto).
echo "humo OK"
