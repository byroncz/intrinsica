output "project_id" {
  description = "ID del GCP project único."
  value       = google_project.this.project_id
}

output "project_number" {
  description = "Número del GCP project (lo usa Workload Identity Federation)."
  value       = google_project.this.number
}

output "region" {
  description = "Región de los buckets y del provider."
  value       = var.region
}

output "buckets" {
  description = "Mapa sufijo → nombre real de cada bucket."
  value = {
    "tfstate"     = google_storage_bucket.tfstate.name
    "landing"     = google_storage_bucket.landing.name
    "dc-events"   = google_storage_bucket.dc_events.name
    "frames"      = google_storage_bucket.frames.name
    "indicators"  = google_storage_bucket.indicators.name
    "dq-findings" = google_storage_bucket.dq_findings.name
    "manifest"    = google_storage_bucket.manifest.name
  }
}
