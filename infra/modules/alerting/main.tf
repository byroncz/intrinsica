# Alerta por correo cuando cualquier Cloud Run Job de cualquier capa deja un
# hallazgo de calidad con severidad ERROR. No declara provider.

# Cloud Run guarda cada línea JSON de stdout como jsonPayload y promueve su
# campo `severity` a la severidad de la entrada (y lo quita del jsonPayload),
# por eso el filtro usa `severity=ERROR` y no `jsonPayload.severity`.
# `dq.configure_logging` agrega `finding_id` solo a las líneas de hallazgos:
# así un traceback u otro log ERROR no dispara esta alerta.
locals {
  filter = <<-EOT
    resource.type="cloud_run_job"
    severity=ERROR
    jsonPayload.finding_id:*
  EOT
}

resource "google_monitoring_notification_channel" "email" {
  project      = var.project_id
  display_name = "Hallazgos DQ: correo"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

resource "google_monitoring_alert_policy" "finding_error" {
  project      = var.project_id
  display_name = "Hallazgo de calidad con severidad ERROR"
  combiner     = "OR"

  conditions {
    display_name = "Línea de hallazgo con severity=ERROR en un Cloud Run Job"

    condition_matched_log {
      filter = local.filter

      # Salen de los campos JSON del hallazgo (dq.emit_findings).
      label_extractors = {
        check_type = "EXTRACT(jsonPayload.check_type)"
        layer      = "EXTRACT(jsonPayload.layer)"
        mode       = "EXTRACT(jsonPayload.mode)"
        asset      = "EXTRACT(jsonPayload.asset)"
        year       = "EXTRACT(jsonPayload.year)"
        month      = "EXTRACT(jsonPayload.month)"
        reason     = "EXTRACT(jsonPayload.details.reason)"
        run_id     = "EXTRACT(jsonPayload.run_id)"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    # Una notificación por ventana: los reintentos de Cloud Run (max_retries = 1)
    # repiten el hallazgo y no deben mandar dos correos.
    notification_rate_limit {
      period = "${var.rate_limit_seconds}s"
    }
    auto_close = "${var.auto_close_seconds}s"
  }

  documentation {
    mime_type = "text/markdown"
    # $${...} es el escape de Terraform: Monitoring sustituye las variables.
    content = <<-EOT
      Un job dejó un hallazgo de calidad de datos con severidad **error**.

      - Job: `$${resource.label.job_name}`
      - Capa y modo: `$${log.extracted_label.layer}` / `$${log.extracted_label.mode}`
      - Check: `$${log.extracted_label.check_type}`
      - Unidad: `$${log.extracted_label.asset}` `$${log.extracted_label.year}-$${log.extracted_label.month}`
      - Detalle: $${log.extracted_label.reason}
      - Run: `$${log.extracted_label.run_id}`

      El enlace "View logs" de este correo abre la línea completa (con la
      ejecución y `details`) en Logs Explorer. Qué hacer según el check:
      docs/runbooks/operacion-l1.md, sección "Alerta por correo".
    EOT
  }
}
