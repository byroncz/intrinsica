variable "tfstate_bucket" {
  description = "Bucket con el estado remoto (<project_id>-tfstate). Se usa para leer el estado del stack data."
  type        = string
}
