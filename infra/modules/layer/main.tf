# Cómputo y acceso de una capa. No crea buckets ni declara provider:
# los datos son del stack data y el provider lo fija el stack que instancia.

locals {
  # Un binding por (modo, bucket, prefijo).
  grants = merge([
    for mode, cfg in var.modes : merge([
      for bucket, grant in cfg.access : {
        for prefix in grant.prefixes : "${mode}/${bucket}/${prefix}" => {
          mode   = mode
          bucket = bucket
          prefix = prefix
          role   = grant.role
        }
      }
    ]...)
  ]...)
}

# Una service account por modo (TRD-L1 §11): permisos mínimos por modo.
resource "google_service_account" "job" {
  for_each = var.modes

  project      = var.project_id
  account_id   = "${var.layer}-${each.key}"
  display_name = "Job ${each.key} de la capa ${var.layer}"

  # Terraform >= 1.6 no valida entre variables: el largo de <capa>-<modo> se
  # comprueba aquí y no en el apply.
  lifecycle {
    precondition {
      condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", "${var.layer}-${each.key}"))
      error_message = "${var.layer}-${each.key} no es un account_id válido: 6 a 30 caracteres, empieza con letra y termina en letra o dígito."
    }
  }
}

# Un job por modo: la service account se fija en la plantilla, no en la
# ejecución. Los args por defecto sirven a Scheduler y Workflows, que ejecutan
# el job sin overrides.
resource "google_cloud_run_v2_job" "this" {
  for_each = var.modes

  project  = var.project_id
  location = var.region
  name     = "${var.layer}-${each.key}"

  # El cómputo se puede destruir sin tocar datos (criterio 4 de la Épica).
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.job[each.key].email

      containers {
        image = var.image
        args  = ["--mode", each.key]

        dynamic "env" {
          for_each = var.env
          content {
            name  = env.key
            value = env.value
          }
        }

        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }
      }
    }
  }
}

# Acceso solo a los prefijos recibidos. Requiere UBLA activo en el bucket.
# La condición tiene dos ramas porque GCS evalúa storage.objects.list sobre el
# bucket, no sobre el objeto: leer y escribir casan por la ruta del objeto, y
# listar casa por el bucket con el prefijo de listado dentro del permitido.
resource "google_storage_bucket_iam_member" "prefix" {
  for_each = local.grants

  bucket = each.value.bucket
  role   = each.value.role
  member = "serviceAccount:${google_service_account.job[each.value.mode].email}"

  condition {
    title       = "${var.layer}-${each.value.mode}-${replace(trimsuffix("${each.value.bucket}/${each.value.prefix}", "/"), "/", "-")}"
    description = "Acceso de ${var.layer}-${each.value.mode} solo bajo ${each.value.prefix}"
    expression  = "resource.name.startsWith(\"projects/_/buckets/${each.value.bucket}/objects/${each.value.prefix}\") || (resource.name == \"projects/_/buckets/${each.value.bucket}\" && api.getAttribute(\"storage.googleapis.com/objectListPrefix\", \"\").startsWith(\"${each.value.prefix}\"))"
  }
}
