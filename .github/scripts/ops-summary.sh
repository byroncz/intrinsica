#!/usr/bin/env bash
# Resumen Markdown de una ejecución de ops-script (ITSC-298). Lee de la entrada
# estándar las líneas de log de la ejecución (las de `gcloud logging read`) y
# arma, con los marcadores que imprime layers/ops_tools/src/ops_tools/cli.py:
# la identidad del script (URI, generation, SHA-256), su contenido y las
# últimas líneas de su salida con su código de salida.
#
# Uso: gcloud logging read ... --format='value(textPayload,jsonPayload.message)' |
#        ops-summary.sh >> "$GITHUB_STEP_SUMMARY"
# Si el log no trae el encabezado (el job murió antes de ejecutar el script, por
# ejemplo un 403 o un 404 al descargarlo), muestra las últimas líneas del log
# completo: ahí está el motivo. Muestra a lo más MAX_OUTPUT_LINES líneas de
# salida (por defecto 40) y MAX_CONTENT_LINES de contenido (por defecto 300),
# para no pasar el límite de 1 MiB del resumen de Actions.
set -euo pipefail

max_output=${MAX_OUTPUT_LINES:-40}
max_content=${MAX_CONTENT_LINES:-300}

jq -R -r -s \
  --argjson max_output "$max_output" --argjson max_content "$max_content" '
  # Con value(textPayload,jsonPayload.message), una línea de texto sale con un
  # tabulador al final.
  def clean: sub("\t$"; "");
  # Un bloque de código con una valla más larga que cualquier racha de comillas
  # invertidas del texto: el contenido de un script puede traer las suyas.
  def fenced($lang):
    ([match("`+"; "g").length] | max // 0) as $run
    | ([3, $run + 1] | max) as $n
    | (reduce range($n) as $_ (""; . + "`")) as $f
    | "\($f)\($lang)\n\(.)\n\($f)";
  def first_at(pred; $from):
    [to_entries[] | select(.key >= $from and (.value | pred)) | .key] | first;

  [split("\n")[] | clean] as $all
  | (if ($all | length) > 0 and $all[-1] == "" then $all[:-1] else $all end) as $lines
  | ($lines | first_at(startswith("OPS_SCRIPT_META "); 0)) as $meta_at
  | ($lines | first_at(. == "----- ops-script: contenido -----"; $meta_at // 0)) as $begin
  | ($lines | first_at(. == "----- ops-script: fin del contenido -----"; $begin // 0)) as $end
  | ($lines | first_at(. == "----- ops-script: salida -----"; $end // 0)) as $out_at
  | ($lines | to_entries | map(select(.value | startswith("OPS_SCRIPT_EXIT "))) | last) as $exit
  | "#### Script",
    "",
    if $meta_at == null or $end == null then
      "No se encontró el encabezado del script en los logs: el job terminó antes de ejecutarlo (por ejemplo, el objeto no existe o la cuenta no puede leerlo). Últimas líneas del log:",
      "",
      ($lines[-$max_output:] | join("\n") | fenced(""))
    else
      ($lines[$meta_at] | sub("^OPS_SCRIPT_META "; "") | fromjson) as $m
      | ($lines[$begin + 1:$end]) as $content
      | (if $out_at == null then [] else $lines[$out_at + 1:($exit.key // ($lines | length))] end) as $output
      | "| Campo | Valor |",
        "| --- | --- |",
        "| URI | `\($m.uri)` |",
        "| Generation | `\($m.generation)` |",
        "| SHA-256 | `\($m.sha256)` |",
        "| Resultados | `\(if $m.results_uri == "" then "-" else $m.results_uri end)` |",
        "| Código de salida | \(if $exit == null then "sin código (el job murió: timeout u OOM)" else ($exit.value | sub("^OPS_SCRIPT_EXIT code="; "")) end) |",
        "",
        "<details><summary>Contenido del script (\($content | length) líneas)</summary>",
        "",
        ($content[:$max_content] | join("\n") | fenced("python")),
        (if ($content | length) > $max_content then "", "Se muestran \($max_content) de \($content | length) líneas." else empty end),
        "",
        "</details>",
        "",
        "#### Salida del script",
        "",
        if ($output | length) == 0 then "sin salida" else
          (if ($output | length) > $max_output then "Últimas \($max_output) de \($output | length) líneas:" else "\($output | length) líneas:" end),
          "",
          ($output[-$max_output:] | join("\n") | fenced(""))
        end
    end
'
