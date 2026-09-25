#!/usr/bin/env bash
# Prueba de humo de la imagen: procesa una unidad real dentro del contenedor y
# verifica sus salidas. Uso: smoke.sh <imagen>
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"
data="$(mktemp -d)"
# El contenedor corre como root: borra sus salidas desde dentro antes de que el
# runner intente borrar $data. TODO: lista las rutas de salida de la capa.
trap 'docker run --rm -v "$data:/data" --entrypoint rm "$image" -rf /data/<salidas>; rm -rf "$data"' EXIT

# TODO: pasa a la imagen las variables L<N>_*_ROOT de la capa y un modo real.
docker run --rm -v "$data:/data" "$image" --mode daily 2>&1 | tee "$data/run.log"

# TODO: verifica las salidas de la capa (Parquet, hallazgos, manifiesto) y
# reemplaza las dos líneas siguientes por `echo "humo OK"`. Falla a propósito
# hasta completarlo: sin verificaciones, un "humo OK" sería evidencia falsa.
echo "::error::smoke.sh de l<N>_<nombre> sin completar (TODO)"
exit 1
