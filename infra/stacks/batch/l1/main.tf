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

  env = {
    L1_LANDING_ROOT  = "gs://${local.buckets["landing"]}/l1"
    L1_DQ_ROOT       = "gs://${local.buckets["dq-findings"]}/l1"
    L1_MANIFEST_ROOT = "gs://${local.buckets["manifest"]}/l1"
  }

  # TRD-L1 §11: una service account por modo. monthly-close borra provisionales,
  # por eso escribe igual que backfill y daily; seam-check solo lee landing.
  modes = {
    backfill      = { access = local.writer_access }
    daily         = { access = local.writer_access }
    monthly-close = { access = local.writer_access }
    seam-check = {
      access = {
        (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
        (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["l1/"] }
      }
    }
  }
}
