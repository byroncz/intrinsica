variable "layer" {
  description = "Nombre corto de la capa (por ejemplo l1). Nombra la service account <layer>-job y el job."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,25}$", var.layer))
    error_message = "layer debe tener 2 a 26 caracteres, empezar con letra minúscula y usar solo [a-z0-9-]."
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

variable "bucket_prefixes" {
  description = "Acceso de la capa a los datos: mapa nombre de bucket → lista de prefijos de objeto donde puede leer y escribir."
  type        = map(list(string))

  validation {
    condition     = alltrue([for prefixes in values(var.bucket_prefixes) : alltrue([for p in prefixes : length(p) > 1 && endswith(p, "/")])])
    error_message = "Cada prefijo debe ser no vacío y terminar en \"/\" (por ejemplo \"l1/\")."
  }
}
