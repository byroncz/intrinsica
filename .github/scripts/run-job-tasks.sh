#!/usr/bin/env bash
# Imprime una línea por unidad de trabajo del rango; run-job.yml cuenta las
# líneas para fijar --tasks. En L1 cada tarea del job toma from +
# CLOUD_RUN_TASK_INDEX. L2 no usa task array: sus meses son secuenciales (cada
# uno lee el carry-over del anterior), así que su CLI exige una sola tarea.
#
# Uso: run-job-tasks.sh <job> <from> [to]   (job = <capa>-<modo>)
#   l1-backfill, l1-monthly-close: meses YYYY-MM del rango inclusivo
#   l1-daily:                      días YYYY-MM-DD del rango inclusivo
#   l1-seam-check:                 una sola unidad ("1")
#   l2-backfill:                   una sola unidad ("1") para todo el rango
#   l2-monthly:                    una sola unidad ("1"); to vacío o igual a from
# to por defecto es from. Entrada inválida: mensaje en stderr y salida 2.
set -euo pipefail

fail() {
  echo "run-job-tasks: $*" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || fail "uso: run-job-tasks.sh <job> <from> [to]"
job=$1
from=$2
to=${3:-$2}

# Un día válido sobrevive al viaje de ida y vuelta por date (rechaza 2025-02-30).
valid_day() {
  [[ $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "$(date -u -d "$1" +%F 2>/dev/null)" == "$1" ]]
}

valid_month() {
  [[ $1 =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]
}

# from y to son meses válidos y to no es menor que from.
check_month_range() {
  valid_month "$from" || fail "from inválido '$from': se espera YYYY-MM"
  valid_month "$to" || fail "to inválido '$to': se espera YYYY-MM"
  [[ "$to" < "$from" ]] && fail "to '$to' es menor que from '$from'"
  return 0
}

case "$job" in
  l1-backfill | l1-monthly-close)
    check_month_range
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
  l1-daily)
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
  l1-seam-check | l2-backfill)
    check_month_range
    echo 1
    ;;
  l2-monthly)
    valid_month "$from" || fail "from inválido '$from': se espera YYYY-MM"
    [[ "$to" == "$from" ]] || fail "l2-monthly procesa un solo mes: to '$to' debe estar vacío o ser igual a from '$from'"
    echo 1
    ;;
  *)
    fail "job inválido '$job': l1-backfill, l1-daily, l1-monthly-close, l1-seam-check, l2-backfill o l2-monthly"
    ;;
esac
