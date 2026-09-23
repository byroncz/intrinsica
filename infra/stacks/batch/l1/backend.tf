# Configuración parcial: el bucket no admite variables, se pasa al init con
# -backend-config="bucket=<project_id>-tfstate". Ver infra/README.md.
terraform {
  backend "gcs" {
    prefix = "stacks/batch/l1"
  }
}
