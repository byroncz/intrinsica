#!/usr/bin/env bash
# Tabla Markdown con los hallazgos de DQ de una ejecución (ITSC-295). Lee de la
# entrada estándar las líneas de log de la ejecución (las de `gcloud logging
# read`, con o sin prefijo de fecha) y se queda con las que traen un hallazgo
# en JSON, el que deja `dq.emit_findings`. Sin hallazgos imprime "sin hallazgos".
#
# Uso: gcloud logging read ... --format='value(textPayload,jsonPayload.message)' |
#        findings-summary.sh >> "$GITHUB_STEP_SUMMARY"
# Orden: error, warning, info; dentro de cada severidad, el del log. Muestra a
# lo más MAX_ROWS filas (por defecto 500) para no pasar el límite de 1 MiB del
# resumen de Actions.
set -euo pipefail

max_rows=${MAX_ROWS:-500}

jq -R -r -s --argjson max "$max_rows" '
  def cell: tostring | gsub("[\r\n]+"; " ") | gsub("\\|"; "\\|");
  def short: if length > 200 then .[0:200] + "…" else . end;
  def unit: "\(.layer // "?") " + (.details.unit // "\(.asset) \(.year)-\(.month | tostring | if length < 2 then "0" + . else . end)");
  def rank: {"error": 0, "warning": 1, "info": 2}[.severity] // 3;

  [ split("\n")[]
    | select(contains("{"))
    | sub("^[^{]*"; "")
    | (try fromjson catch null)
    | select(type == "object" and has("check_type") and has("severity")) ]
  | unique_by(.finding_id // .)  # los reintentos de Cloud Run no duplican filas
  | sort_by(rank) as $rows
  | "#### Hallazgos de calidad de datos",
    "",
    if ($rows | length) == 0 then "sin hallazgos" else
      "\($rows | length) hallazgos: \([$rows[] | select(.severity == "error")] | length) error, \([$rows[] | select(.severity == "warning")] | length) warning, \([$rows[] | select(.severity == "info")] | length) info.",
      "",
      "| Check | Severidad | Unidad | Valor | Detalle |",
      "| --- | --- | --- | --- | --- |",
      ($rows[:$max][] |
        "| \(.check_type | cell) | \(.severity | cell) | \(unit | cell) | \(.metric_value // "-" | cell) | \((.details.reason // (.details // {} | tojson)) | short | cell) |"),
      (if ($rows | length) > $max then "", "Se muestran \($max) de \($rows | length) filas." else empty end)
    end
'
