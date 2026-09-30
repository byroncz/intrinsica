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

  # ADR-04 (TRD-L2 §10.2 y §14, docs/runbooks/sonda-l2.md): L2 corre en Cloud
  # Run Jobs; Cloud Batch no hace falta. Sonda del 2026-09-30, mes 2023-03
  # (190.227.841 ticks, imagen 0.5.0, 4 GiB en las tres corridas, sin OOM):
  #
  #   vCPU | RSS pico (MiB) | pared (s) | vCPU-s | costo de lista (USD)
  #      2 |            446 |     292,2 |    584 | 0,0129
  #      4 |            446 |     261,5 |  1.046 | 0,0209
  #      8 |            445 |     301,1 |  2.409 | 0,0458
  #
  # La pared no baja con más vCPU (261 a 301 s entre 2 y 8), así que más CPU solo
  # encarece: 2 vCPU cuestan 3,5x menos que 8 por la misma pared.
  # Regla de L1: RSS <= 75 % de la memoria y sin OOM. 446 MiB son 11 % de 4 GiB.
  # Se fija 4 GiB y no menos porque es la memoria medida, no una extrapolada.
  cpu    = "2"
  memory = "4Gi"

  # Timeout por defecto, el de monthly: un solo mes. 1800 s son 6,2x la pared de
  # la sonda con 2 vCPU (292,2 s, el mes más pesado); monthly procesa un mes
  # recién cerrado, de ordinario mucho más liviano.
  timeout = 1800

  # 1 y no 0, por la misma razón que en L1: el reintento cubre fallos
  # transitorios de infraestructura y uno determinista (OOM, timeout, checksum)
  # lo resuelve el humano. En backfill un reintento no repite lo hecho: salta
  # los meses que ya tienen su carry-over (RF-L2-09).
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
    # Una sola tarea encadena los 109 meses por carry-over (sin task array).
    # Techo: 109 x 292,2 s = 31.850 s (8,85 h), con el mes más pesado repetido.
    # 54.000 s (15 h) son 1,7x ese techo (la regla pide >= 1,5x) y quedan bajo
    # el tope de 86.400 s que acepta el módulo.
    backfill = { timeout = 54000, access = local.access }
    monthly  = { access = local.access }
  }
}
