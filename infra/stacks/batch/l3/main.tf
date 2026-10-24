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

  # Definitivo (ITSC-352, T5: 2021-01 a 1, 2, 4 y 8 vCPU con la imagen 0.19.0,
  # 2026-10-24; TRD-L3 §10.2). 2 vCPU y 4 GiB. Pared 282, 112, 96 y 75 s;
  # vCPU-s facturados (pared × vCPU) 283, 223, 383 y 599: 2 vCPU es el mínimo, y
  # 4 y 8 pagan CPU que el motor no usa (engine_busy 0,63 y 0,14). Cuesta +16 %
  # de pared en el mes denso frente a 4 vCPU. Los 50 content_hash son los de
  # ITSC-335 en los cuatro tamaños. Memoria: dos medidas distintas, como en
  # infra/README.md. Sobre los 110 meses (ITSC-335, núcleo original) el RSS pico
  # máximo fue 1 267 MiB; la cifra vigente es la de la corrida de seis meses
  # 2017-08 a 2018-01 con la imagen 0.19.0, 2 347 MiB (2018-01, 57 % de 4 GiB), y
  # 2021-01 pica a 2 141 MiB con 2 vCPU y a 2 654 con 8. Sin OOM. Una tarea.
  cpu    = "2"
  memory = "4Gi"

  # El de monthly: 1 200 s son 7,3 veces la pared estimada del mes más pesado con
  # 2 vCPU (≈ 164 s: 2022-11, 141,4 s a 4 vCPU en ITSC-364, por el 1,16 que 2 vCPU
  # le suma a 2021-01), sobre el mínimo de 6 veces de TRD-L2 §10.2 (984 s). 900 s,
  # el valor de ITSC-364 para 4 vCPU, ya no lo cumplen. Es una estimación: la
  # pared de 2022-11 a 2 vCPU sale del backfill de 110 meses.
  timeout = 1200

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
    # 10 800 s son 1,9 veces la suma estimada de los 110 meses con 2 vCPU (≈ 5 600 s:
    # los 4 828 s de ITSC-364 a 4 vCPU por 1,16; el mínimo de la regla de TRD-L2
    # §10.2 es 1,5 veces, 8 400 s), bajo el tope de 86 400 s del módulo. Una sola
    # tarea recorre el rango, y 5 600 s caben en las 6 h que espera el run de
    # Actions. La suma real sale del backfill completo con este tamaño.
    backfill = { timeout = 10800, access = local.access }
    monthly  = { access = local.access }
  }
}
