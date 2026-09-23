variable "project_id" {
  description = "ID del GCP project único. Globalmente único; prefija el nombre de todos los buckets."
  type        = string
}

variable "billing_account" {
  description = "ID de la cuenta de facturación (formato XXXXXX-XXXXXX-XXXXXX) que se asocia al project."
  type        = string
}

variable "region" {
  description = "Región de GCP para los buckets y el provider."
  type        = string
  default     = "us-east1"
}

variable "keep_tagged_versions" {
  description = "Cantidad de versiones más recientes que conserva la política de limpieza de Artifact Registry."
  type        = number
  default     = 10
}

variable "github_repo" {
  description = "Repositorio de GitHub (owner/nombre) al que Workload Identity Federation permite actuar como la service account ci."
  type        = string
  default     = "byroncz/intrinsica"
}
