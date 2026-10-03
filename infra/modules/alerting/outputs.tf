output "notification_channel_id" {
  description = "ID del canal de correo."
  value       = google_monitoring_notification_channel.email.id
}

output "alert_policy_id" {
  description = "ID de la política de alerta de hallazgos ERROR."
  value       = google_monitoring_alert_policy.finding_error.id
}
