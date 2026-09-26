variable "layer" {
  description = "Nombre corto de la capa (por ejemplo l1). Prefija la service account y el workflow como <layer>-<recurso>."
  type        = string
}

variable "project_id" {
  description = "ID del GCP project donde se crean el workflow y los schedulers."
  type        = string
}

variable "region" {
  description = "Región del workflow, de los schedulers y de los Cloud Run Jobs."
  type        = string
}

variable "job_names" {
  description = "Cloud Run Jobs que el workflow puede ejecutar: mapa modo → nombre del job (el output job_names del módulo layer). Solo se orquestan los modos de schedules."
  type        = map(string)
}

variable "scheduler_invoker_email" {
  description = "Service account con la que los schedulers inician ejecuciones del workflow (output scheduler_invoker_service_account_email del stack data)."
  type        = string
}

variable "schedules" {
  description = "Un scheduler por modo orquestado: mapa modo → {schedule, time_zone, paused, description}. Cada modo debe existir en job_names. paused = false enciende el scheduler."
  type = map(object({
    schedule    = string
    time_zone   = optional(string, "Etc/UTC")
    paused      = optional(bool, true)
    description = string
  }))

  validation {
    condition     = length(var.schedules) > 0
    error_message = "schedules no puede estar vacío."
  }
}
