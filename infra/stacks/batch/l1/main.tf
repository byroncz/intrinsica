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

  env = {
    L1_LANDING_ROOT  = "gs://${local.buckets["landing"]}/l1"
    L1_DQ_ROOT       = "gs://${local.buckets["dq-findings"]}/l1"
    L1_MANIFEST_ROOT = "gs://${local.buckets["manifest"]}/l1"
  }

  bucket_prefixes = {
    (local.buckets["landing"])     = ["l1/"]
    (local.buckets["dq-findings"]) = ["l1/"]
    (local.buckets["manifest"])    = ["l1/"]
  }
}
