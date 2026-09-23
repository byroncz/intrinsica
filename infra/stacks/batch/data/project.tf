# Cuenta personal sin organización: el project no lleva org_id ni folder_id.
resource "google_project" "this" {
  name            = var.project_id
  project_id      = var.project_id
  billing_account = var.billing_account
  deletion_policy = "PREVENT"
}

locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "storage.googleapis.com",
    "serviceusage.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)

  project            = google_project.this.project_id
  service            = each.value
  disable_on_destroy = false
}
