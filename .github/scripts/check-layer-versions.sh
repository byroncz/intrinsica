#!/usr/bin/env bash
# Falla si un cambio de código de una capa no sube su VERSION.
# Uso: check-layer-versions.sh <ref-base> '<json con las capas>'
# Código de una capa = todo bajo layers/<capa>/ salvo README.md, tests/ y VERSION,
# más shared/, uv.lock y pyproject.toml, que el Dockerfile copia a cada imagen.
# La VERSION nueva debe ser estrictamente mayor que la de la base.
set -euo pipefail

base="$1"
layers="$2"
fail=0

for layer in $(jq -r '.[]' <<< "$layers"); do
  dir="layers/${layer}"
  code="$(git diff --name-only "${base}...HEAD" -- "$dir" shared uv.lock pyproject.toml \
    ":(exclude)${dir}/README.md" ":(exclude)${dir}/tests" ":(exclude)${dir}/VERSION")"
  [ -z "$code" ] && continue
  new="$(tr -d '[:space:]' < "${dir}/VERSION")"
  # Si VERSION no existía en la base (capa nueva), cualquier versión es válida.
  if ! old="$(git show "${base}:${dir}/VERSION" 2>/dev/null)"; then
    continue
  fi
  old="$(tr -d '[:space:]' <<< "$old")"
  if [ "$new" = "$old" ] || [ "$(printf '%s\n%s\n' "$old" "$new" | sort -V | tail -n1)" != "$new" ]; then
    echo "::error file=${dir}/VERSION::Cambió código de ${layer} sin subir la versión (base ${old}, PR ${new}): sube ${dir}/VERSION (parche para fixes, menor para features)."
    echo "$code" | sed 's/^/  - /'
    fail=1
  fi
done

exit "$fail"
