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
