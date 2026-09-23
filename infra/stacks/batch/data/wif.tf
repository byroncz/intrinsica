# Workload Identity Federation: GitHub Actions publica imágenes sin llaves.
# WIF es la única vía; este stack no crea llaves de service account (RNF-08).

resource "google_iam_workload_identity_pool" "github" {
  project                   = google_project.this.project_id
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = google_project.this.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  # Solo tokens emitidos para este repositorio.
  attribute_condition = "assertion.repository == \"${var.github_repo}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Identidad con la que actúa el CI (resource `ci`; account_id exige 6+ caracteres).
resource "google_service_account" "ci" {
  project      = google_project.this.project_id
  account_id   = "ci-github"
  display_name = "CI de GitHub Actions"

  depends_on = [google_project_service.apis]
}

# Escritura solo sobre el repositorio de imágenes, nunca a nivel de project.
resource "google_artifact_registry_repository_iam_member" "ci_writer" {
  project    = google_artifact_registry_repository.images.project
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.ci.email}"
}

# Deja que los tokens de GitHub del repositorio actúen como la service account.
resource "google_service_account_iam_member" "ci_wif" {
  service_account_id = google_service_account.ci.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}
