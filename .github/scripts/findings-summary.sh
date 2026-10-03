#!/usr/bin/env bash
# Tabla Markdown con los hallazgos de DQ de una ejecución (ITSC-295). Lee de la
# entrada estándar las líneas de log de la ejecución (las de `gcloud logging
# read`, con o sin prefijo de fecha) y se queda con las que traen un hallazgo
# en JSON, el que deja `dq.emit_findings`. Sin hallazgos imprime "sin hallazgos".
#
# Uso: gcloud logging read ... --format='value(textPayload,jsonPayload.message)' |
#        findings-summary.sh >> "$GITHUB_STEP_SUMMARY"
# Orden: error, warning, info; dentro de cada severidad, el del log. Un
# reintento de Cloud Run (`max_retries = 1`) repite la tarea y emite otra fila
# con otro `finding_id` (uuid4 por emisión): se deduplica por contenido, con el
# hallazgo completo menos los campos de emisión (`finding_id`, `run_id`,
# `detected_at`), y se conserva la primera aparición. Dos filas solo se juntan
# si son indistinguibles también en el lago. Muestra a lo más MAX_ROWS filas
# (por defecto 500) para no pasar el límite de 1 MiB del resumen de Actions.
set -euo pipefail

max_rows=${MAX_ROWS:-500}

jq -R -r -s --argjson max "$max_rows" '
  def cell: tostring | gsub("[\r\n]+"; " ") | gsub("\\|"; "\\|");
  def short: if length > 200 then .[0:200] + "…" else . end;
  def unit: "\(.layer // "?") " + (.details.unit // "\(.asset) \(.year)-\(.month | tostring | if length < 2 then "0" + . else . end)");
  def rank: {"error": 0, "warning": 1, "info": 2}[.severity] // 3;
  def content_key: del(.finding_id, .run_id, .detected_at) | tojson;

  [ split("\n")[]
    | select(contains("{"))
    | sub("^[^{]*"; "")
    | (try fromjson catch null)
    | select(type == "object" and has("check_type") and has("severity")) ]
  | reduce .[] as $f ({seen: {}, rows: []};
      ($f | content_key) as $k
      | if .seen[$k] then . else .seen[$k] = true | .rows += [$f] end)
  | .rows
  | sort_by(rank) as $rows  # estable: dentro de cada severidad queda el orden del log
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
