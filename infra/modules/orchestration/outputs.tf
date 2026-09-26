output "workflow_name" {
  description = "Nombre del workflow que ejecuta los jobs."
  value       = google_workflows_workflow.run_job.name
}

output "workflow_service_account_email" {
  description = "Email de la service account del workflow."
  value       = google_service_account.workflow.email
}

output "scheduler_names" {
  description = "Nombre del scheduler de cada modo: mapa modo → nombre."
  value       = { for mode, s in google_cloud_scheduler_job.this : mode => s.name }
}
