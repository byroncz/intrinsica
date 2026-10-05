# Orquestación de una capa: un workflow que ejecuta el Cloud Run Job del modo
# pedido y un Cloud Scheduler por modo que lo dispara. No declara provider.

locals {
  modes = keys(var.schedules)
  # Misma fuente que el IAM: el workflow no reconstruye el nombre del job.
  job_names = { for m in local.modes : m => var.job_names[m] }
  # Modos encadenados: modo → lista ordenada de jobs que se ejecutan uno tras
  # otro, cada uno cuando el anterior termina bien.
  next_jobs = { for m, s in var.schedules : m => s.next_jobs if length(s.next_jobs) > 0 }
  # Eslabones que el workflow espera: todos menos el último de cada cadena.
  waited_links = toset(flatten([for l in values(local.next_jobs) : slice(l, 0, length(l) - 1)]))
}

# Identidad propia del workflow. Solo puede ejecutar los jobs de schedules y los
# next_jobs, y leer las ejecuciones de los jobs que espera (roles por job,
# abajo); no tiene roles de project.
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

# Para encadenar, el workflow lee la ejecución del modo (executions.get, que
# pide run.executions.get) hasta que termine. roles/run.viewer es el rol
# predefinido más chico que lo incluye y se da sobre el job de origen, no sobre
# el project.
resource "google_cloud_run_v2_job_iam_member" "workflow_viewer" {
  for_each = local.next_jobs

  project  = var.project_id
  location = var.region
  name     = var.job_names[each.key]
  role     = "roles/run.viewer"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

# Lo mismo para cada eslabón que el workflow espera (el último de la cadena no:
# nadie lee su ejecución). Un eslabón que ya es el job de un modo de esta capa
# tiene su viewer arriba; setsubtract evita el binding duplicado.
resource "google_cloud_run_v2_job_iam_member" "workflow_chain_viewer" {
  for_each = setsubtract(local.waited_links, values(local.job_names))

  project  = var.project_id
  location = var.region
  name     = each.value
  role     = "roles/run.viewer"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

# Ejecutar cada eslabón (puede ser de otra capa, por eso no sale de job_names).
# toset evita repetir el binding si dos modos encadenan al mismo job.
resource "google_cloud_run_v2_job_iam_member" "workflow_next_invoker" {
  for_each = toset(flatten(values(local.next_jobs)))

  project  = var.project_id
  location = var.region
  name     = each.value
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

# Recibe {"mode": "..."} y llama jobs.run por HTTP con OAuth2 en vez del
# conector googleapis.run.v2: el conector espera la operación y pide
# run.operations.get, que es un permiso de project. Así la service account
# queda sin roles de project. Consecuencia: en un modo sin next_jobs el workflow
# termina al aceptar la ejecución; el resultado se ve en Cloud Run.
# En un modo con next_jobs sondea executions.get cada 30 s hasta que la
# ejecución trae completionTime (el nombre viene en body.metadata.name;
# body.name es el de la operación). Solo si hubo éxito (algún task exitoso,
# ninguno fallido ni cancelado) ejecuta el siguiente de la lista y repite la
# espera con él; el último no se espera. Un eslabón fallido termina el workflow
# con error y los que siguen no corren. El GET se reintenta con la política por
# defecto de Workflows; el POST no, porque repetirlo lanzaría otra ejecución.
# Sin overrides: el día y el mes por defecto los pone el propio modo.
resource "google_workflows_workflow" "run_job" {
  project         = var.project_id
  region          = var.region
  name            = "${var.layer}-run-job"
  description     = "Ejecuta el Cloud Run Job ${var.layer}-<modo> (${join(" | ", local.modes)}) sin overrides y encadena los next_jobs de los modos que los tienen."
  service_account = google_service_account.workflow.id

  # Regla: toda expresión ${...} del YAML va entre comillas simples. Si contiene
  # ": " y va sin comillas, el parser YAML de Workflows la corta y el apply falla
  # (ITSC-230). Aquí se escribe $${...} por el escape de Terraform.
  # .github/scripts/check-workflow-yaml.sh lo verifica en CI.
  source_contents = <<-EOT
    main:
      params: [args]
      steps:
        - mapa_jobs:
            assign:
              - jobs: ${jsonencode(local.job_names)}
              - siguientes: ${jsonencode(local.next_jobs)}
        - validar_modo:
            switch:
              - condition: '$${default(map.get(args, "mode"), "") in keys(jobs)}'
                next: ejecutar_job
            next: modo_invalido
        - modo_invalido:
            raise: '$${"mode invalido: " + default(map.get(args, "mode"), "(ausente)")}'
        - ejecutar_job:
            call: http.post
            args:
              url: '$${"https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/" + jobs[args.mode] + ":run"}'
              auth:
                type: OAuth2
            result: ejecucion
        - iniciar_cadena:
            assign:
              - cadena: '$${default(map.get(siguientes, args.mode), [])}'
              - indice: 0
        - hay_siguiente:
            switch:
              - condition: '$${indice < len(cadena)}'
                next: nombre_ejecucion
            next: fin
        - nombre_ejecucion:
            assign:
              - ejecucion_nombre: '$${ejecucion.body.metadata.name}'
        - pausa:
            call: sys.sleep
            args:
              seconds: 30
        - consultar_ejecucion:
            try:
              call: http.get
              args:
                url: '$${"https://run.googleapis.com/v2/" + ejecucion_nombre}'
                auth:
                  type: OAuth2
              result: estado
            retry: '$${http.default_retry}'
        - evaluar_estado:
            switch:
              - condition: '$${default(map.get(estado.body, "completionTime"), "") == ""}'
                next: pausa
              - condition: '$${default(map.get(estado.body, "succeededCount"), 0) > 0 and default(map.get(estado.body, "failedCount"), 0) == 0 and default(map.get(estado.body, "cancelledCount"), 0) == 0}'
                next: ejecutar_siguiente
            next: ejecucion_fallida
        - ejecucion_fallida:
            raise: '$${"la ejecucion " + ejecucion_nombre + " no termino con exito; no se ejecuta " + cadena[indice]}'
        - ejecutar_siguiente:
            call: http.post
            args:
              url: '$${"https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/" + cadena[indice] + ":run"}'
              auth:
                type: OAuth2
            result: ejecucion
        - avanzar:
            assign:
              - indice: '$${indice + 1}'
            next: hay_siguiente
        - fin:
            return: '$${ejecucion.body.name}'
  EOT

  depends_on = [
    google_cloud_run_v2_job_iam_member.workflow_invoker,
    google_cloud_run_v2_job_iam_member.workflow_viewer,
    google_cloud_run_v2_job_iam_member.workflow_chain_viewer,
    google_cloud_run_v2_job_iam_member.workflow_next_invoker,
  ]
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
