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

output "artifact_registry" {
  description = "Ruta del repositorio Docker: us-east1-docker.pkg.dev/<project_id>/<repo>."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "wif_provider_name" {
  description = "Nombre completo del provider WIF (projects/<número>/locations/global/workloadIdentityPools/<pool>/providers/<provider>)."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "ci_service_account_email" {
  description = "Email de la service account que asume GitHub Actions."
  value       = google_service_account.ci.email
}

output "deploy_service_account_email" {
  description = "Email de la service account con la que GitHub Actions aplica stacks de capa y ejecuta jobs."
  value       = google_service_account.deploy.email
}
