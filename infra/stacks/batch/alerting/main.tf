# Alertas del proyecto, no de una capa: una sola política cubre los Cloud Run
# Jobs de todas. Va en su propio stack para que destruir o aplicar l1 o l2 no
# toque las alertas. Lee los outputs de data.
data "terraform_remote_state" "data" {
  backend = "gcs"

  config = {
    bucket = var.tfstate_bucket
    prefix = "stacks/batch/data"
  }
}

locals {
  data = data.terraform_remote_state.data.outputs
}

provider "google" {
  project = local.data.project_id
  region  = local.data.region
}

module "alerting" {
  source = "../../../modules/alerting"

  project_id  = local.data.project_id
  alert_email = var.alert_email
}
