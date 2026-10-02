#!/usr/bin/env bash
# Imprime el valor de --args que run-job.yml pasa a `gcloud run jobs execute`:
# las opciones separadas por comas, cada una como una sola pieza --opción=valor.
# gcloud rechaza un --args con un valor repetido en la lista (ITSC-232:
# to == from; ITSC-280: series_start == from), y así dos meses iguales no chocan.
#
# Uso: run-job-args.sh <job> [from] [to] [force] [series_start]
#   job:   <capa>-<modo>; el modo (--mode) es lo que sigue al primer guion
#   force: "true" agrega --force; cualquier otro valor, nada
#
# --to se omite si es igual a from, salvo en l2-backfill (ITSC-292). En L1 la
# CLI asume to = from cuando falta, y la de l2-monthly rechaza --to. En
# l2-backfill, desde 0.6.0 (ITSC-285), to ausente significa "hasta el último mes
# con consolidated.parquet en L1": omitirlo convertiría un mes en un rango abierto.
set -euo pipefail

[[ $# -ge 1 && $# -le 5 ]] || {
  echo "uso: run-job-args.sh <job> [from] [to] [force] [series_start]" >&2
  exit 2
}
job=$1
from=${2:-}
to=${3:-}
force=${4:-}
series_start=${5:-}

args="--mode,${job#*-}"
if [[ -n "$from" ]]; then
  args="${args},--from=${from}"
fi
if [[ -n "$to" && ( "$to" != "$from" || "$job" == l2-backfill ) ]]; then
  args="${args},--to=${to}"
fi
if [[ "$force" == "true" ]]; then
  args="${args},--force"
fi
if [[ -n "$series_start" ]]; then
  args="${args},--series-start=${series_start}"
fi
echo "$args"
