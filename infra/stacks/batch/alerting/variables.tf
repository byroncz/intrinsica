variable "tfstate_bucket" {
  description = "Bucket con el estado remoto (<project_id>-tfstate). Se usa para leer el estado del stack data."
  type        = string
}

variable "alert_email" {
  description = "Correo que recibe la alerta. Llega por TF_VAR_alert_email desde la variable ALERT_EMAIL del environment gcp; nunca se escribe en el repo."
  type        = string
  sensitive   = true
}
