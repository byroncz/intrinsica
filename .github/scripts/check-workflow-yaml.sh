#!/usr/bin/env bash
# Valida que el source_contents del workflow del módulo orchestration sea YAML
# válido una vez que Terraform lo renderiza (ITSC-230). Cloud Workflows corta
# una expresión ${...} sin comillas si contiene ": " y el apply falla tarde.
# Uso: check-workflow-yaml.sh [main.tf]
set -euo pipefail

tf="${1:-infra/modules/orchestration/main.tf}"

# Extrae el heredoc, quita la indentación y renderiza lo que Terraform resuelve:
# $${ -> ${ (escape) y las interpolaciones ${...} restantes -> valor ficticio.
rendered="$(awk '/source_contents *= *<<-?EOT/{f=1;next} f&&/^ *EOT$/{exit} f' "$tf" \
  | sed -E 's/^ {4}//; s/\$\$\{/@@DOLLAR@@/g; s/\$\{[^}]*\}/x/g; s/@@DOLLAR@@/${/g')"

[ -n "$rendered" ] || { echo "No se encontró source_contents en $tf" >&2; exit 1; }

printf '%s\n' "$rendered" | uvx --quiet --from pyyaml python -c '
import sys, yaml
try:
    yaml.safe_load(sys.stdin)
except yaml.YAMLError as e:
    sys.exit(f"YAML inválido en el workflow renderizado: {e}")
print("Workflow renderizado: YAML válido")
'
