#!/usr/bin/env bash
# Prueba check-layer-versions.sh en un repo temporal (ITSC-355).
# Uso: .github/scripts/test-check-layer-versions.sh
# Cubre lo que entra y lo que no entra en las imágenes según el .dockerignore:
#   - shared/*/tests y shared/*/README.md no exigen subir ningún VERSION;
#   - shared/*/src, shared/*/pyproject.toml, uv.lock y el pyproject.toml raíz exigen
#     subir las cinco capas.
set -euo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check-layer-versions.sh"
layers='["l1_ingest","l2_dc_events","l3_dc_descriptors","ops_tools","viz_tiles"]'
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cd "$tmp"
git init -q -b main .
git config user.email test@example.com
git config user.name test
for l in $(jq -r '.[]' <<< "$layers"); do
  mkdir -p "layers/$l/tests"
  echo 0.1.0 > "layers/$l/VERSION"
  echo x > "layers/$l/tests/test_x.py"
done
mkdir -p shared/dc_frames/src/dc_frames shared/dc_frames/tests
echo x > shared/dc_frames/src/dc_frames/reader.py
echo x > shared/dc_frames/tests/test_dc_frames_memory.py
echo x > shared/dc_frames/README.md
echo x > shared/dc_frames/pyproject.toml
echo x > uv.lock
echo x > pyproject.toml
git add -A && git commit -q -m base

fail=0

# caso <nombre> <archivo a tocar> <salida esperada> <capas esperadas en la salida>
caso() {
  local name="$1" file="$2" want="$3" want_layers="$4" out got=0
  git checkout -q -B "t-$name" main
  echo cambio >> "$file"
  git commit -q -am "$name"
  out="$("$script" main "$layers" 2>&1)" || got=$?
  if [ "$got" != "$want" ]; then
    echo "FALLA $name: salió con $got, se esperaba $want"; echo "$out"; fail=1; return
  fi
  for l in $(jq -r '.[]' <<< "$layers"); do
    if grep -q "Cambió código de ${l} " <<< "$out"; then
      [ "$want_layers" = "todas" ] || { echo "FALLA $name: nombró ${l} y no debía"; fail=1; }
    elif [ "$want_layers" = "todas" ]; then
      echo "FALLA $name: no nombró ${l}"; fail=1
    fi
  done
  echo "ok $name"
}

caso shared-tests   shared/dc_frames/tests/test_dc_frames_memory.py 0 ninguna
caso shared-readme  shared/dc_frames/README.md                      0 ninguna
caso layer-tests    layers/l1_ingest/tests/test_x.py                0 ninguna
caso shared-src     shared/dc_frames/src/dc_frames/reader.py        1 todas
caso shared-pyproj  shared/dc_frames/pyproject.toml                 1 todas
caso uv-lock        uv.lock                                         1 todas
caso root-pyproj    pyproject.toml                                  1 todas

exit "$fail"
