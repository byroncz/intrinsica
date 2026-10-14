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

  # TRD-L3 §11: L3 lee L1 (consolidated.parquet), L2 (eventos y carry-over) y el
  # catálogo de θ de L2, y escribe solo bajo l3/. El catálogo lo edita el humano
  # y lo siembra el stack l2, así que esta cuenta solo lo lee. Los descriptores
  # van a su propio bucket, dc-descriptors (ITSC-350): el módulo layer da un solo
  # rol por bucket y en dc-events L3 no podría leer l2/ y escribir l3/.
  access = {
    (local.buckets["landing"])        = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
    (local.buckets["dc-events"])      = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.buckets["manifest"])       = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.buckets["dc-descriptors"]) = { role = "roles/storage.objectUser", prefixes = ["l3/"] }
    (local.buckets["dq-findings"])    = { role = "roles/storage.objectUser", prefixes = ["l3/"] }
  }
}

provider "google" {
  project = local.data.project_id
  region  = local.data.region
}

module "layer" {
  source = "../../../modules/layer"

  layer      = "l3"
  project_id = local.data.project_id
  region     = local.data.region
  image      = "${local.data.artifact_registry}/l3_dc_descriptors:${trimspace(file("${path.module}/../../../../layers/l3_dc_descriptors/VERSION"))}"

  # PROVISIONAL hasta el backfill histórico (ITSC-335). 4 vCPU y 4 GiB, el tamaño
  # de L2, medidos el 2026-10-14 sobre 2023-03 (TRD-L3 §10.3): pared 873 s, RSS
  # pico 598 MiB (14,6 % de 4 GiB), sin OOM. Una tarea y un hilo: la suma estimada
  # de los 109 meses (≈ 29 000 s) cabe en 86 400 s con margen 1,5, así que no hace
  # falta paralelizar. El tamaño definitivo se fija tras el histórico, con el
  # máximo de wall_s y de rss_peak_mib de los 109 unit_timing.
  cpu    = "4"
  memory = "4Gi"

  # PROVISIONAL, el de monthly: 7 200 s son 8,2 veces la pared medida de 2023-03
  # (873 s, 2026-10-14), sobre el mínimo de 6 veces de TRD-L2 §10.2 (5 238 s). Se
  # confirma con el mes más pesado del histórico.
  timeout = 7200

  # 1 y no 0, como L1 y L2: el reintento cubre fallos transitorios de
  # infraestructura y uno determinista (OOM, timeout, conciliación) lo resuelve
  # el humano. En backfill un reintento no repite lo hecho: la frontera de cada θ
  # salta los meses cuyo summaries.parquet ya es válido.
  max_retries = 1

  env = {
    L3_LANDING_ROOT   = "gs://${local.buckets["landing"]}/l1"
    L3_EVENTS_ROOT    = "gs://${local.buckets["dc-events"]}/l2"
    L3_SUMMARIES_ROOT = "gs://${local.buckets["dc-descriptors"]}/l3"
    L3_DQ_ROOT        = "gs://${local.buckets["dq-findings"]}/l3"
    # Primer mes de la serie de L2: el único sin mes anterior. Se declara, no se
    # infiere.
    L3_SERIES_START = "2017-08"
    # El mismo objeto que usa L2. L3 no siembra ni copia el catálogo: agregar un
    # θ es editarlo en GCS (TRD-L2 §7.3) y lanzar l3-backfill sin from.
    L3_THETAS_URI = "gs://${local.buckets["manifest"]}/l2/thetas.yaml"
  }

  # Una service account por modo. Ambos modos leen y escriben lo mismo; la
  # separación deja revocar o auditar cada uno por su cuenta.
  modes = {
    # PROVISIONAL: 86 400 s es el tope del módulo y cubre con margen 1,5 la suma
    # estimada de los meses (≈ 29 000 s; 109 × 873 s = 95 157 s es solo una cota
    # superior). Una sola tarea recorre el rango. Cloud Run no la corta, pero el
    # run de Actions espera 6 h como máximo: el histórico va por rangos (ver
    # infra/README.md, stack l3). Se fija con la suma medida tras el histórico.
    backfill = { timeout = 86400, access = local.access }
    monthly  = { access = local.access }
  }
}
