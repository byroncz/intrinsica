# Los buckets son datos: este stack es su único dueño y nunca se destruyen.
# Un stack de capa no crea buckets; los consume por nombre desde outputs.tf.

# Estado remoto de Terraform de todos los stacks.
resource "google_storage_bucket" "tfstate" {
  name                        = "${var.project_id}-tfstate"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Landing de L1: ZIP originales de data.binance.vision (TRD maestro §9.3).
resource "google_storage_bucket" "landing" {
  name                        = "${var.project_id}-landing"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Salida de L2: eventos Directional Change.
resource "google_storage_bucket" "dc_events" {
  name                        = "${var.project_id}-dc-events"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Salida de L3: frames.
resource "google_storage_bucket" "frames" {
  name                        = "${var.project_id}-frames"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Salida de L4: indicadores.
resource "google_storage_bucket" "indicators" {
  name                        = "${var.project_id}-indicators"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Lago de hallazgos de Data Quality (RF-20).
resource "google_storage_bucket" "dq_findings" {
  name                        = "${var.project_id}-dq-findings"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Manifiesto de carga.
resource "google_storage_bucket" "manifest" {
  name                        = "${var.project_id}-manifest"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}
