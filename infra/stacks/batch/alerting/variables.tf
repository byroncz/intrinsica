variable "tfstate_bucket" {
  description = "Bucket con el estado remoto (<project_id>-tfstate). Se usa para leer el estado del stack data."
  type        = string
}

variable "alert_email" {
  description = "Correo que recibe la alerta. Llega por TF_VAR_alert_email desde el secret de repositorio ALERT_EMAIL; nunca se escribe en el repo."
  type        = string
  sensitive   = true
}
