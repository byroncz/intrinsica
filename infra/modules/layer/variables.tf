variable "layer" {
  description = "Nombre corto de la capa (por ejemplo l1). Prefija los jobs y las service accounts como <layer>-<modo>."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9]{1,25}$", var.layer))
    error_message = "layer debe tener 2 a 26 caracteres, empezar con letra minúscula y usar solo [a-z0-9] (sin guiones: run-job.yml deriva el modo del sufijo tras el primer guion)."
  }
}

variable "project_id" {
  description = "ID del GCP project donde se crean la service account y el job."
  type        = string
}

variable "region" {
  description = "Región del Cloud Run Job."
  type        = string
}

variable "image" {
  description = "Imagen de contenedor que corre el job (ruta completa de Artifact Registry o pública)."
  type        = string
}

variable "cpu" {
  description = "CPU del contenedor, como texto (formato de Cloud Run)."
  type        = string
  default     = "4"
}

variable "memory" {
  description = "Memoria del contenedor (formato de Cloud Run). Por defecto 16Gi, ADR-L1-09."
  type        = string
  default     = "16Gi"
}

variable "env" {
  description = "Variables de entorno del contenedor del job: mapa nombre → valor."
  type        = map(string)
  default     = {}
}

variable "modes" {
  description = "Modos de la capa: mapa modo → acceso. Cada modo crea el job <layer>-<modo> y la service account <layer>-<modo>. access es un mapa nombre de bucket → {role, prefixes}: el rol de storage y los prefijos de objeto donde lo tiene."
  type = map(object({
    access = map(object({
      role     = string
      prefixes = list(string)
    }))
  }))

  validation {
    condition     = length(var.modes) > 0 && alltrue([for m in keys(var.modes) : can(regex("^[a-z][a-z0-9-]{1,25}$", m))])
    error_message = "modes no puede estar vacío y cada modo debe usar [a-z0-9-] y empezar con letra minúscula."
  }

  validation {
    condition = alltrue([
      for cfg in values(var.modes) : alltrue([
        for grant in values(cfg.access) : alltrue([for p in grant.prefixes : length(p) > 1 && endswith(p, "/")])
      ])
    ])
    error_message = "Cada prefijo debe ser no vacío y terminar en \"/\" (por ejemplo \"l1/\")."
  }
}
