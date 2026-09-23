# Repositorio Docker de las imágenes de las capas. El free tier de Artifact
# Registry es 0,5 GB: la política de limpieza mantiene el costo en 0
# (TRD maestro §8.3).
resource "google_artifact_registry_repository" "images" {
  project       = google_project.this.project_id
  location      = var.region
  repository_id = "intrinsica"
  format        = "DOCKER"
  description   = "Imágenes Docker de las capas de intrinsica."

  cleanup_policy_dry_run = false

  # Borra toda imagen sin tag.
  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state = "UNTAGGED"
    }
  }

  # Conserva las N versiones con tag más recientes.
  cleanup_policies {
    id     = "keep-recent-tagged"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.keep_tagged_versions
    }
  }

  depends_on = [google_project_service.apis]
}
