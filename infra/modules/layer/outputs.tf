output "service_account_emails" {
  description = "Email de la service account de cada modo: mapa modo → email."
  value       = { for mode, sa in google_service_account.job : mode => sa.email }
}

output "job_names" {
  description = "Nombre del Cloud Run Job de cada modo: mapa modo → nombre."
  value       = { for mode, job in google_cloud_run_v2_job.this : mode => job.name }
}
