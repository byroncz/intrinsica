# Orquestación de una capa: un workflow que ejecuta el Cloud Run Job del modo
# pedido y un Cloud Scheduler por modo que lo dispara. No declara provider.

locals {
  modes = keys(var.schedules)
  # Misma fuente que el IAM: el workflow no reconstruye el nombre del job.
  job_names = { for m in local.modes : m => var.job_names[m] }
}

# Identidad propia del workflow. Solo puede ejecutar los jobs de schedules
# (roles/run.invoker por job, abajo); no tiene roles de project.
resource "google_service_account" "workflow" {
  project      = var.project_id
  account_id   = "${var.layer}-workflow"
  display_name = "Workflow de orquestación de la capa ${var.layer}"
}

resource "google_cloud_run_v2_job_iam_member" "workflow_invoker" {
  for_each = var.schedules

  project  = var.project_id
  location = var.region
  name     = var.job_names[each.key]
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

# Recibe {"mode": "..."} y llama jobs.run por HTTP con OAuth2 en vez del
# conector googleapis.run.v2: el conector espera la operación y pide
# run.operations.get, que es un permiso de project. Así la service account
# queda sin roles de project. Consecuencia: el workflow termina al aceptar la
# ejecución; el resultado se ve en Cloud Run, no en Workflows.
# Sin overrides: el día y el mes por defecto los pone el propio modo.
resource "google_workflows_workflow" "run_job" {
  project         = var.project_id
  region          = var.region
  name            = "${var.layer}-run-job"
  description     = "Ejecuta el Cloud Run Job ${var.layer}-<modo> (${join(" | ", local.modes)}) sin overrides."
  service_account = google_service_account.workflow.id

  source_contents = <<-EOT
    main:
      params: [args]
      steps:
        - mapa_jobs:
            assign:
              - jobs: ${jsonencode(local.job_names)}
        - validar_modo:
            switch:
              - condition: $${default(map.get(args, "mode"), "") in keys(jobs)}
                next: ejecutar_job
            next: modo_invalido
        - modo_invalido:
            raise: $${"mode invalido: " + default(map.get(args, "mode"), "(ausente)")}
        - ejecutar_job:
            call: http.post
            args:
              url: $${"https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/" + jobs[args.mode] + ":run"}
              auth:
                type: OAuth2
            result: ejecucion
        - fin:
            return: $${ejecucion.body.name}
  EOT

  depends_on = [google_cloud_run_v2_job_iam_member.workflow_invoker]
}

# Un scheduler por modo. Nacen en pausa: encenderlos es decisión del humano
# (paused = false por PR y apply).
resource "google_cloud_scheduler_job" "this" {
  for_each = var.schedules

  project     = var.project_id
  region      = var.region
  name        = "${var.layer}-${each.key}"
  description = each.value.description
  schedule    = each.value.schedule
  time_zone   = each.value.time_zone
  paused      = each.value.paused

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/${google_workflows_workflow.run_job.id}/executions"
    headers     = { "Content-Type" = "application/json" }
    body        = base64encode(jsonencode({ argument = jsonencode({ mode = each.key }) }))

    oauth_token {
      service_account_email = var.scheduler_invoker_email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }
}
