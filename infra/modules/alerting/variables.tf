variable "project_id" {
  description = "ID del GCP project donde se crean el canal y la política de alerta."
  type        = string
}

variable "alert_email" {
  description = "Correo que recibe la alerta. Es dato personal: no se versiona, llega por TF_VAR_alert_email desde el environment gcp de Actions."
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email))
    error_message = "alert_email debe ser una dirección de correo."
  }
}

variable "rate_limit_seconds" {
  description = "Mínimo de segundos entre dos notificaciones de la política. Cloud Monitoring exige al menos 300 en alertas de log."
  type        = number
  default     = 300

  validation {
    condition     = var.rate_limit_seconds >= 300
    error_message = "rate_limit_seconds debe ser al menos 300."
  }
}

variable "auto_close_seconds" {
  description = "Segundos sin nuevos hallazgos tras los cuales Monitoring cierra el incidente solo (7 días por defecto; el máximo es 14)."
  type        = number
  default     = 604800

  validation {
    condition     = var.auto_close_seconds >= 1800 && var.auto_close_seconds <= 1209600
    error_message = "auto_close_seconds debe estar entre 1800 y 1209600."
  }
}
