#!/usr/bin/env bash
# Imprime el valor de --args que run-job.yml pasa a `gcloud run jobs execute`:
# las opciones separadas por comas, cada una como una sola pieza --opción=valor.
# gcloud rechaza un --args con un valor repetido en la lista (ITSC-232:
# to == from; ITSC-280: series_start == from), y así dos meses iguales no chocan.
#
# Uso: run-job-args.sh <job> [from] [to] [force] [series_start] [script] [args]
#   job:   <capa>-<modo>; el modo (--mode) es lo que sigue al primer guion
#   force: "true" agrega --force; cualquier otro valor, nada
#   script, args: solo ops-script (ITSC-298); los demás jobs los ignoran
#
# --to se omite si es igual a from, salvo en l2-backfill (ITSC-292). En L1 la
# CLI asume to = from cuando falta, y la de l2-monthly rechaza --to. En
# l2-backfill, desde 0.6.0 (ITSC-285), to ausente significa "hasta el último mes
# con consolidated.parquet en L1": omitirlo convertiría un mes en un rango abierto.
set -euo pipefail

[[ $# -ge 1 && $# -le 7 ]] || {
  echo "uso: run-job-args.sh <job> [from] [to] [force] [series_start] [script] [args]" >&2
  exit 2
}
job=$1
from=${2:-}
to=${3:-}
force=${4:-}
series_start=${5:-}
script=${6:-}
script_args=${7:-}

# ops-script (ITSC-298): el texto libre de args puede llevar comas y valores
# repetidos (--a 5 --b 5), que el formato de lista de gcloud no tolera. Va como
# una sola pieza, con ;; de separador en vez de la coma (gcloud topic escaping),
# y el entrypoint la parte como un shell (--script-args). Sin --mode: el job no
# tiene modos.
if [[ "$job" == ops-script ]]; then
  for value in "$script" "$script_args"; do
    [[ "$value" != *';;'* ]] || {
      echo "run-job-args: ';;' no se admite en script ni en args" >&2
      exit 2
    }
  done
  out="^;;^--script=${script}"
  if [[ -n "$script_args" ]]; then
    out="${out};;--script-args=${script_args}"
  fi
  echo "$out"
  exit 0
fi

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
