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

  # Una política KEEP solo protege frente a una DELETE que coincida: sin una
  # DELETE que abarque las versiones con tag, nada las borraría nunca. Por eso
  # la DELETE cubre todas las versiones (con y sin tag) y la KEEP, que tiene
  # precedencia, decide cuáles sobreviven.
  cleanup_policies {
    id     = "delete-all"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }

  # Conserva las N versiones más recientes (cuenta todas, con o sin tag).
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.keep_tagged_versions
    }
  }

  depends_on = [google_project_service.apis]
}
