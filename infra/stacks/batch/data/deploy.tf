# Identidad con la que GitHub Actions aplica los stacks de capa y ejecuta jobs.
# Va aparte de `ci-github` (que solo publica imágenes) para que cada una tenga
# permisos mínimos. Sin llaves: entra solo por Workload Identity Federation.

resource "google_service_account" "deploy" {
  project      = google_project.this.project_id
  account_id   = "deploy-github"
  display_name = "Despliegue de GitHub Actions"

  depends_on = [google_project_service.apis]
}

# Mismo principalSet que `ci-github`: solo tokens del repositorio github_repo.
resource "google_service_account_iam_member" "deploy_wif" {
  service_account_id = google_service_account.deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

locals {
  deploy_member = "serviceAccount:${google_service_account.deploy.email}"
}

# Crear, actualizar, borrar y ejecutar Cloud Run Jobs (incluye run.jobs.run,
# por eso no hace falta roles/run.invoker).
resource "google_project_iam_member" "deploy_run_developer" {
  project = google_project.this.project_id
  role    = "roles/run.developer"
  member  = local.deploy_member
}

# Crear y borrar las service accounts de capa (<capa>-job).
resource "google_project_iam_member" "deploy_sa_admin" {
  project = google_project.this.project_id
  role    = "roles/iam.serviceAccountAdmin"
  member  = local.deploy_member
}

# Actuar como (actAs) las service accounts de capa al crear su Cloud Run Job.
resource "google_project_iam_member" "deploy_sa_user" {
  project = google_project.this.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = local.deploy_member
}

# Leer los logs de la ejecución de un job.
resource "google_project_iam_member" "deploy_logging_viewer" {
  project = google_project.this.project_id
  role    = "roles/logging.viewer"
  member  = local.deploy_member
}

# Leer y escribir el estado de Terraform (objetos del bucket tfstate).
resource "google_storage_bucket_iam_member" "deploy_tfstate" {
  bucket = google_storage_bucket.tfstate.name
  role   = "roles/storage.objectUser"
  member = local.deploy_member
}

# Leer y fijar la política IAM de un bucket sin tocar sus objetos: lo que
# necesita el módulo layer para sus google_storage_bucket_iam_member.
resource "google_project_iam_custom_role" "bucket_iam_admin" {
  project     = google_project.this.project_id
  role_id     = "bucketIamAdmin"
  title       = "Administrador de IAM de buckets"
  description = "Lee y fija la política IAM de un bucket, sin acceso a sus objetos."
  permissions = [
    "storage.buckets.get",
    "storage.buckets.getIamPolicy",
    "storage.buckets.setIamPolicy",
  ]
}

# Solo sobre los buckets a los que las capas piden acceso.
resource "google_storage_bucket_iam_member" "deploy_bucket_iam" {
  for_each = {
    landing     = google_storage_bucket.landing.name
    dq-findings = google_storage_bucket.dq_findings.name
    manifest    = google_storage_bucket.manifest.name
  }

  bucket = each.value
  role   = google_project_iam_custom_role.bucket_iam_admin.id
  member = local.deploy_member
}
