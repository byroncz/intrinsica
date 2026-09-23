# Cómputo y acceso de una capa. No crea buckets ni declara provider:
# los datos son del stack data y el provider lo fija el stack que instancia.

resource "google_service_account" "job" {
  project      = var.project_id
  account_id   = "${var.layer}-job"
  display_name = "Job de la capa ${var.layer}"
}

resource "google_cloud_run_v2_job" "this" {
  project  = var.project_id
  location = var.region
  name     = "${var.layer}-job"

  # El cómputo se puede destruir sin tocar datos (criterio 4 de la Épica).
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.job.email

      containers {
        image = var.image

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

locals {
  # Un binding por par (bucket, prefijo).
  grants = merge([
    for bucket, prefixes in var.bucket_prefixes : {
      for prefix in prefixes : "${bucket}/${prefix}" => {
        bucket = bucket
        prefix = prefix
      }
    }
  ]...)
}

# Acceso solo a los prefijos recibidos. Requiere UBLA activo en el bucket.
resource "google_storage_bucket_iam_member" "prefix" {
  for_each = local.grants

  bucket = each.value.bucket
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.job.email}"

  condition {
    title       = "${var.layer}-${replace(each.key, "/", "-")}"
    description = "Acceso de ${var.layer} solo bajo ${each.value.prefix}"
    expression  = "resource.name.startsWith(\"projects/_/buckets/${each.value.bucket}/objects/${each.value.prefix}\")"
  }
}
