output "service_account_email" {
  description = "Email de la service account con la que corre el job."
  value       = google_service_account.job.email
}

output "job_name" {
  description = "Nombre del Cloud Run Job de la capa."
  value       = google_cloud_run_v2_job.this.name
}
