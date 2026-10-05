variable "tfstate_bucket" {
  description = "Bucket con el estado remoto (<project_id>-tfstate). Se usa para leer el estado del stack data."
  type        = string
}

variable "viz_viewer" {
  description = "Correo de la cuenta de Google que lee el bucket viz. Llega por TF_VAR_viz_viewer desde el secret de repositorio VIZ_VIEWER; nunca se escribe en el repo ni en un .tfvars."
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.viz_viewer))
    error_message = "viz_viewer debe ser un correo (el secret VIZ_VIEWER no tiene un correo válido)."
  }
}
