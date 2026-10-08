#!/usr/bin/env bash
# Prueba de humo de la imagen: trae las librerías que usan los scripts de
# operación y el entrypoint rechaza una ejecución sin --script con código 2
# (la de los args por defecto del módulo `layer`). Sin red ni GCS.
# Uso: smoke.sh <imagen>
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"

docker run --rm --entrypoint python "$image" -c '
import dc_frames, duckdb, dq, google.cloud.storage, pyarrow, pyutils
import pyarrow.fs as pafs
assert hasattr(pafs, "GcsFileSystem"), "pyarrow sin soporte de GCS"
print("librerías OK, pyarrow", pyarrow.__version__)
'

set +e
docker run --rm "$image" --mode script > /dev/null 2>&1
code=$?
set -e
[ "$code" -eq 2 ] || { echo "::error::sin --script se esperaba código 2 y salió $code"; exit 1; }
echo "humo OK"
