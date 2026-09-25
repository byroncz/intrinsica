#!/usr/bin/env bash
# Imprime una línea por unidad de trabajo del rango; run-job.yml cuenta las
# líneas para fijar --tasks. Cada tarea del job toma from + CLOUD_RUN_TASK_INDEX.
#
# Uso: run-job-tasks.sh <mode> <from> [to]
#   backfill, monthly-close: meses YYYY-MM del rango inclusivo
#   daily:                   días YYYY-MM-DD del rango inclusivo
#   seam-check:              una sola unidad ("1")
# to por defecto es from. Entrada inválida: mensaje en stderr y salida 2.
set -euo pipefail

fail() {
  echo "run-job-tasks: $*" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || fail "uso: run-job-tasks.sh <mode> <from> [to]"
mode=$1
from=$2
to=${3:-$2}

# Un día válido sobrevive al viaje de ida y vuelta por date (rechaza 2025-02-30).
valid_day() {
  [[ $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "$(date -u -d "$1" +%F 2>/dev/null)" == "$1" ]]
}

valid_month() {
  [[ $1 =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]
}

case "$mode" in
  backfill | monthly-close)
    valid_month "$from" || fail "from inválido '$from': se espera YYYY-MM"
    valid_month "$to" || fail "to inválido '$to': se espera YYYY-MM"
    [[ "$to" < "$from" ]] && fail "to '$to' es menor que from '$from'"
    y=$((10#${from%-*}))
    m=$((10#${from#*-}))
    while :; do
      unit=$(printf '%04d-%02d' "$y" "$m")
      echo "$unit"
      [[ "$unit" == "$to" ]] && break
      m=$((m + 1))
      if ((m > 12)); then
        m=1
        y=$((y + 1))
      fi
    done
    ;;
  daily)
    valid_day "$from" || fail "from inválido '$from': se espera YYYY-MM-DD"
    valid_day "$to" || fail "to inválido '$to': se espera YYYY-MM-DD"
    [[ "$to" < "$from" ]] && fail "to '$to' es menor que from '$from'"
    day=$from
    while :; do
      echo "$day"
      [[ "$day" == "$to" ]] && break
      day=$(date -u -d "$day + 1 day" +%F)
    done
    ;;
  seam-check)
    # Formato de from/to pendiente de ITSC-203: se acepta mes o día, sin comas.
    for v in "$from" "$to"; do
      [[ $v =~ ^[0-9]{4}-[0-9]{2}(-[0-9]{2})?$ ]] || fail "'$v' inválido: se espera YYYY-MM o YYYY-MM-DD"
    done
    [[ "$to" < "$from" ]] && fail "to '$to' es menor que from '$from'"
    echo 1
    ;;
  *)
    fail "mode inválido '$mode': backfill, daily, monthly-close o seam-check"
    ;;
esac
