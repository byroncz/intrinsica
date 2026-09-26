#!/usr/bin/env bash
# Falla si un cambio de código de una capa no sube su VERSION.
# Uso: check-layer-versions.sh <ref-base> '<json con las capas>'
# Código de una capa = todo bajo layers/<capa>/ salvo README.md, tests/ y VERSION.
set -euo pipefail

base="$1"
layers="$2"
fail=0

for layer in $(jq -r '.[]' <<< "$layers"); do
  dir="layers/${layer}"
  code="$(git diff --name-only "${base}...HEAD" -- "$dir" \
    ":(exclude)${dir}/README.md" ":(exclude)${dir}/tests" ":(exclude)${dir}/VERSION")"
  [ -z "$code" ] && continue
  if git diff --quiet "${base}...HEAD" -- "${dir}/VERSION"; then
    echo "::error file=${dir}/VERSION::Cambió código de ${layer} sin subir la versión: sube ${dir}/VERSION (parche para fixes, menor para features)."
    echo "$code" | sed 's/^/  - /'
    fail=1
  fi
done

exit "$fail"
