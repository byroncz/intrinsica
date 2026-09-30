# Este stack no posee datos: lee los outputs del stack data y solo pide acceso
# a prefijos de sus buckets.
data "terraform_remote_state" "data" {
  backend = "gcs"

  config = {
    bucket = var.tfstate_bucket
    prefix = "stacks/batch/data"
  }
}

locals {
  data    = data.terraform_remote_state.data.outputs
  buckets = local.data.buckets

  writer_access = {
    (local.buckets["landing"])     = { role = "roles/storage.objectUser", prefixes = ["l1/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["l1/"] }
    (local.buckets["manifest"])    = { role = "roles/storage.objectUser", prefixes = ["l1/"] }
  }
}

provider "google" {
  project = local.data.project_id
  region  = local.data.region
}

module "layer" {
  source = "../../../modules/layer"

  layer      = "l1"
  project_id = local.data.project_id
  region     = local.data.region
  image      = "${local.data.artifact_registry}/l1_ingest:${trimspace(file("${path.module}/../../../../layers/l1_ingest/VERSION"))}"

  # Sonda §14.1 (docs/runbooks/sonda-l1.md): mes 2023-03, run del 2026-09-26
  # (imagen 0.1.1), RSS pico 5622 MiB (34 % de 16 GiB; 46 % del umbral de
  # 12.288 MiB), pared 577 s, sin OOM.
  # Cumple la regla (<= 75 % y sin OOM): 4 vCPU y 16 GiB.
  cpu    = "4"
  memory = "16Gi"

  # Timeout por defecto: 3600 s. Sobre la pared de 577 s de la sonda (2023-03)
  # el margen es 6x; con los 600 s por defecto de Cloud Run quedaban 23 s y un
  # mes más pesado moría por timeout. daily y seam-check lo bajan a 900 s;
  # backfill y monthly-close heredan este valor.
  timeout = 3600

  # 1 y no 0: el reintento cubre fallos transitorios de infraestructura. Un
  # fallo determinista (OOM, timeout, checksum) no lo arregla repetir, y cada
  # reintento se paga completo: lo resuelve el humano relanzando el rango, que
  # se reanuda porque backfill salta lo que ya existe con el mismo .CHECKSUM.
  max_retries = 1

  env = {
    L1_LANDING_ROOT  = "gs://${local.buckets["landing"]}/l1"
    L1_DQ_ROOT       = "gs://${local.buckets["dq-findings"]}/l1"
    L1_MANIFEST_ROOT = "gs://${local.buckets["manifest"]}/l1"
  }

  # TRD-L1 §11: una service account por modo. monthly-close borra provisionales,
  # por eso escribe igual que backfill y daily; seam-check solo lee landing.
  modes = {
    backfill = { access = local.writer_access }
    daily    = { timeout = 900, access = local.writer_access }
    # El consolidado mensual es la unidad más pesada: hereda el tope de 3600 s.
    monthly-close = { access = local.writer_access }
    seam-check = {
      timeout = 900
      access = {
        (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
        (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["l1/"] }
      }
    }
  }
}

# Orquestación del estado estacionario (TRD-L1 §10.1 y §11). Ambos schedulers
# nacen en pausa; encenderlos es decisión del humano (README del módulo).
module "orchestration" {
  source = "../../../modules/orchestration"

  layer                   = "l1"
  project_id              = local.data.project_id
  region                  = local.data.region
  job_names               = module.layer.job_names
  scheduler_invoker_email = local.data.scheduler_invoker_service_account_email

  schedules = {
    daily = {
      # Binance publica el día D durante D+1 (UTC). 03:00 UTC deja margen tras
      # el cierre de D y corre cuando el archivo de D ya está disponible.
      schedule    = "0 3 * * *"
      paused      = true
      description = "Ingesta diaria de L1: el día D se publica en D+1."
    }
    monthly-close = {
      # El primer lunes de M+1 cae entre el 1 y el 7, así que el día 8 siempre
      # es posterior a él, a las 06:00 UTC.
      schedule    = "0 6 8 * *"
      paused      = true
      description = "Cierre mensual de L1, tras el primer lunes de M+1."
      # L2 procesa el mes que L1 acaba de cerrar: el workflow espera el fin de
      # monthly-close y, solo si fue exitoso, ejecuta l2-monthly sin overrides
      # (la CLI toma el mes por defecto). El nombre sigue la convención
      # <capa>-<modo> del módulo layer y no sale del output job_names del stack
      # l2: leerlo exigiría un remote_state más y un outputs.tf en l2 que este
      # stack no necesita, porque el nombre no cambia sin cambiar la convención.
      # El stack l2 debe estar aplicado antes: el IAM se da sobre ese job.
      next_job = "l2-monthly"
    }
  }
}
