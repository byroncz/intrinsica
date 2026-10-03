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

  # ITSC-291: el job plan de _terraform-stack.yml guarda aquí, bajo plans/, el
  # plan que el job apply aplica tras la aprobación (privado: el repo es
  # público y un artefacto de Actions lo descargaría cualquiera). El apply lo
  # borra al terminar; esto limpia los que quedaron por un apply rechazado,
  # cancelado o fallido. Solo plans/: el estado (stacks/) nunca vence. Sin
  # with_state, igual que results/ de ops: toma vigentes y no vigentes.
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      with_state     = "ANY"
      matches_prefix = ["plans/"]
      age            = 7
    }
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

  # ITSC-235: toda versión no vigente se borra a los 30 días, que son la
  # ventana de recuperación ante una reescritura o un borrado equivocado.
  # Sin num_newer_versions: un objeto borrado (p. ej. los provisional-day de
  # monthly-close) no tiene versiones más nuevas y esa condición nunca se
  # cumplía, así que quedaba huérfano para siempre.
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      with_state                 = "ARCHIVED"
      days_since_noncurrent_time = 30
    }
  }

  # ITSC-234: PartitionWriter escribe a un temporal `.tmp` y lo mueve sobre
  # el destino con fs.move (copia más borrado); con el bucket versionado,
  # cada `.tmp` borrado queda como versión no vigente del mismo tamaño que
  # la partición. Esta regla lo borra al día siguiente sin esperar
  # num_newer_versions (un .tmp borrado no tiene versiones más nuevas y
  # nunca alcanza esa condición), evitando que la duplicación se acumule.
  # Es la mitigación definitiva: se acepta un .tmp no vigente de hasta un
  # día por escritura (docs/data-contracts.md, "Sobrescritura atómica").
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      with_state                 = "ARCHIVED"
      matches_suffix             = [".tmp"]
      days_since_noncurrent_time = 1
    }
  }

  # Nearline y no Coldline: L2 relee el histórico completo, y Nearline cobra
  # menos por recuperación. Los 30 días respetan su mínimo de permanencia.
  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      with_state     = "LIVE"
      age            = 30
      matches_prefix = ["l1/"]
      matches_suffix = ["consolidated.parquet"]
    }
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

# Scripts y salidas de operación (ITSC-298): el job ops-script lee de scripts/
# lo que el humano sube desde Cloud Shell y deja salidas grandes en results/.
# No es un dato del lago: es el espacio de trabajo de las verificaciones ad hoc.
resource "google_storage_bucket" "ops" {
  name                        = "${var.project_id}-ops"
  project                     = google_project.this.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # El script se sube siempre a la misma ruta y se sobrescribe: el versionado
  # es la trazabilidad de lo que corrió (el job registra su generation y hash).
  versioning {
    enabled = true
  }

  # Una versión no vigente de un script se conserva 90 días y se borra. La
  # vigente no vence: es el script que se vuelve a lanzar.
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      with_state                 = "ARCHIVED"
      matches_prefix             = ["scripts/"]
      days_since_noncurrent_time = 90
    }
  }

  # Todo lo de results/ se borra a los 7 días. Sin with_state, la regla toma
  # vigentes y no vigentes: con el bucket versionado, borrar una vigente la deja
  # como no vigente, y como su age ya pasó de 7 días, la siguiente evaluación la
  # borra de verdad.
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      with_state     = "ANY"
      matches_prefix = ["results/"]
      age            = 7
    }
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}
