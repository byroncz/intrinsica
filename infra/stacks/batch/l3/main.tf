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
  # van al bucket frames: lo creó data para las tramas que la decisión del
  # 2026-10-08 descartó, estaba vacío y ninguna otra capa lo usa. El nombre
  # queda desfasado; se reutiliza porque un bucket no se renombra y porque el
  # módulo layer da un solo rol por bucket: en dc-events L3 no podría leer l2/ y
  # escribir l3/.
  access = {
    (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
    (local.buckets["dc-events"])   = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.buckets["manifest"])    = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.buckets["frames"])      = { role = "roles/storage.objectUser", prefixes = ["l3/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["l3/"] }
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

  # PROVISIONAL. Arranque de TRD-L3 §10.2, hipótesis sin medir: el tamaño de L2
  # (4 vCPU y 4 GiB), a cuyo mes más pesado se espera que L3 se parezca. Una sola
  # tarea, sin task array: sin carry-over los meses de L3 son independientes y
  # una tarea por mes sería posible, pero la partición la decide la sonda. Lo
  # cierra ITSC-335, que mide el job real (l3-backfill sobre 2023-03) y fija
  # tamaño, timeouts y partición.
  cpu    = "4"
  memory = "4Gi"

  # PROVISIONAL, el de monthly: un solo mes. 7200 s es un valor de partida
  # generoso, sin pared medida detrás. ITSC-335 lo fija con la regla de TRD-L2
  # §10.2: al menos 6 veces la pared del mes más pesado medida.
  timeout = 7200

  # 1 y no 0, como L1 y L2: el reintento cubre fallos transitorios de
  # infraestructura y uno determinista (OOM, timeout, conciliación) lo resuelve
  # el humano. En backfill un reintento no repite lo hecho: la frontera de cada θ
  # salta los meses cuyo summaries.parquet ya es válido.
  max_retries = 1

  env = {
    L3_LANDING_ROOT   = "gs://${local.buckets["landing"]}/l1"
    L3_EVENTS_ROOT    = "gs://${local.buckets["dc-events"]}/l2"
    L3_SUMMARIES_ROOT = "gs://${local.buckets["frames"]}/l3"
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
    # PROVISIONAL: 86 400 s es el tope del módulo. Una sola tarea recorre los
    # 109 meses; ITSC-335 fija el timeout con la suma medida de los meses (regla
    # de TRD-L2 §10.2: al menos 1,5 veces, sin pasar de 86 400 s) y decide si
    # el backfill se parte en una tarea por mes.
    backfill = { timeout = 86400, access = local.access }
    monthly  = { access = local.access }
  }
}
