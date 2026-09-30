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

  # TRD-L2 §7: L2 lee la landing de L1 y escribe eventos y hallazgos de DQ. No
  # escribe manifiesto, por eso no pide nada sobre manifest.
  access = {
    (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
    (local.buckets["dc-events"])   = { role = "roles/storage.objectUser", prefixes = ["l2/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["l2/"] }
  }
}

provider "google" {
  project = local.data.project_id
  region  = local.data.region
}

module "layer" {
  source = "../../../modules/layer"

  layer      = "l2"
  project_id = local.data.project_id
  region     = local.data.region
  image      = "${local.data.artifact_registry}/l2_dc_events:${trimspace(file("${path.module}/../../../../layers/l2_dc_events/VERSION"))}"

  # Provisionales: 4 vCPU, 16 GiB, 3600 s y 1 reintento son los valores de L1,
  # no una medición de L2. La sonda de L2 (ITSC-281) mide RSS pico y pared con
  # 2, 4 y 8 vCPU sobre el mes más pesado, y la decisión ADR-04 (ITSC-282)
  # fija estos cuatro valores y si L2 corre en Cloud Run Jobs o en Cloud Batch.
  cpu         = "4"
  memory      = "16Gi"
  timeout     = 3600
  max_retries = 1

  env = {
    L2_LANDING_ROOT = "gs://${local.buckets["landing"]}/l1"
    L2_EVENTS_ROOT  = "gs://${local.buckets["dc-events"]}/l2"
    L2_DQ_ROOT      = "gs://${local.buckets["dq-findings"]}/l2"
    # Primer mes de la serie: el único que se procesa sin carry-over previo.
    L2_SERIES_START = "2017-08"
  }

  # Una service account por modo. Ambos modos escriben lo mismo; la separación
  # deja revocar o auditar cada uno por su cuenta.
  modes = {
    backfill = { access = local.access }
    monthly  = { access = local.access }
  }
}
