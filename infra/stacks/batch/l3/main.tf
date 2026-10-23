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

  # Definitivo (ITSC-335, histórico de 110 meses, 2026-10-15; TRD-L3 §10.2). 4 vCPU
  # y 4 GiB, el tamaño de L2. RSS pico máximo 2 942 MiB (2018-01, 72 % de 4 GiB, con la
  # imagen 0.17.0; ITSC-375 mide su composición), sin OOM. Una tarea y un hilo: la suma medida de los 110 meses (41 256 s) con
  # margen 1,5 son 61 884 s, bajo 86 400 s, así que no hace falta paralelizar. El
  # job usa ≈ 15 % de los vCPU; usarlos (procesos por θ) o bajarlos es una
  # mejora aparte, no medida.
  cpu    = "4"
  memory = "4Gi"

  # El de monthly: 10 800 s son 7,6 veces la pared del mes más pesado (2021-01,
  # 1 419 s), sobre el mínimo de 6 veces de TRD-L2 §10.2 (8 514 s).
  timeout = 10800

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
    # 86 400 s, el tope del módulo, son 2,1 veces la suma medida de los 110 meses
    # (41 256 s; el mínimo de la regla de TRD-L2 §10.2 es 1,5 veces, 61 884 s). Una
    # sola tarea recorre el rango. Cloud Run no la corta, pero el run de Actions
    # espera 6 h como máximo (41 meses tomaron 4 h 57 min): el histórico va por
    # rangos (ver infra/README.md, stack l3).
    backfill = { timeout = 86400, access = local.access }
    monthly  = { access = local.access }
  }
}
