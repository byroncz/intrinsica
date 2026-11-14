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

  # El env común de los modos. En el módulo layer el env de un modo reemplaza a var.env, no
  # se mezcla: un modo que agrega una variable pasa merge(local.env, {...}).
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
  # 2017-08 a 2018-01 con la imagen 0.19.0, 2 347 MiB (2018-01, 57 % de 4 GiB;
  # corrida (a) de ITSC-376, run 40059849133, run_id
  # 62ed6bcc-d8cd-47cd-80d3-5c9928622e52, 300 de 300 en el run 40061351593), y
  # 2021-01 pica a 2 141 MiB con 2 vCPU y a 2 654 con 8. Sin OOM. Una tarea.
  cpu    = "2"
  memory = "4Gi"

  # El de monthly: 1 200 s son 7,3 veces la pared estimada del mes más pesado con
  # 2 vCPU (≈ 164 s: 2022-11, 141,4 s a 4 vCPU en ITSC-364, por el 1,16 que 2 vCPU
  # le suma a 2021-01), sobre el mínimo de 6 veces de TRD-L2 §10.2 (984 s). 900 s,
  # el valor de ITSC-364 para 4 vCPU, ya no lo cumplen. Es una estimación: la
  # pared de 2022-11 a 2 vCPU sale del backfill de 110 meses.
  #
  # Es solo el valor por defecto del módulo: desde ITSC-420 cada modo fija el suyo
  # explícito (abajo), para que ninguno herede estos 1 200 s sin decidirlo. Ya no
  # alcanzan en un modo mensual: la verificación de la frontera (TRD-L3 §8.2) midió
  # 1 945,8 s a 2 vCPU en la nube (ITSC-420).
  timeout = 1200

  # 1 y no 0, como L1 y L2: el reintento cubre fallos transitorios de
  # infraestructura y uno determinista (OOM, timeout, conciliación) lo resuelve
  # el humano. En backfill un reintento no repite lo hecho: la frontera de cada θ
  # salta los meses cuyo summaries.parquet ya es válido.
  max_retries = 1

  env = local.env

  # Una service account por modo. Ambos modos leen y escriben lo mismo; la
  # separación deja revocar o auditar cada uno por su cuenta.
  modes = {
    # 10 800 s son 1,9 veces la suma estimada de los 110 meses con 2 vCPU (≈ 5 600 s:
    # los 4 828 s de ITSC-364 a 4 vCPU por 1,16; el mínimo de la regla de TRD-L2
    # §10.2 es 1,5 veces, 8 400 s), bajo el tope de 86 400 s del módulo. Una sola
    # tarea recorre el rango, y 5 600 s caben en las 6 h que espera el run de
    # Actions. La suma real sale del backfill completo con este tamaño.
    #
    # Actualización (ITSC-402): 43 200 s (12 h), para que un backfill completo de la
    # familia A quepa en una ejecución. A mide 27 556 s en total (ITSC-401, TRD-L3
    # §10.2), 2,5 veces los 10 800 s de arriba, que se dimensionaron para S. La regla
    # de TRD-L2 §10.2 pide al menos 1,5 veces esa suma (41 334 s): 43 200 s la
    # cumplen con 1,57 veces y quedan bajo el tope de 86 400 s del módulo. Decisión
    # del arquitecto el 2026-11-06 (TRD-L3 v2.17.2). La espera de run-job.yml muere
    # a las 6 h (21 600 s) mientras la ejecución sigue en Cloud Run: el runbook
    # explica cómo seguirla.
    backfill = { timeout = 43200, access = local.access }

    # ITSC-420: 3 600 s. La frontera verifica cada mes de la serie de cada θ y en la nube
    # tardó 1 945,8 s (11 100 archivos, 22 200 pies, 50 θ, 2 vCPU, imagen 0.25.1, ejecución
    # l3-backfill-m7thd, familia path, 8 lectores). La regla de TRD-L3 §10.2 es que el
    # timeout de un job mensual cubra la verificación más 6 veces el mes más lento (183,7 s,
    # 2022-11, ITSC-378): 1 946 + 6 × 183,7 = 3 048 s. 3 600 s son 1,18 veces eso y 1,85
    # veces la verificación sola; S no verifica un techo aparte, así que lee menos pies
    # que esa cifra y el margen es mayor. Con los 1 200 s de antes, la cadena del
    # 2026-12-01 habría muerto por timeout. Se mide de nuevo con la verificación más rápida
    # de esta misma card (meta 300 s).
    monthly = { timeout = 3600, access = local.access }

    # El backfill de la familia F (ITSC-413): el modo backfill de la CLI con
    # --family spectrum, en un job de 32 GiB y 8 vCPU. F guarda en RAM los retornos de
    # cada evento abierto (4 B cada uno) y calcula una FFT exacta al cerrarlo, cuyo
    # transitorio pasa de lo que cabe en los 4 GiB de backfill (TRD-L3 ADR-L3-19).
    # ITSC-418 (decisión del humano con el arquitecto, 2026-11-12, Épica ITSC-410) sube
    # el presupuesto de F a 130 M de ticks por evento y el job de 8 GiB a 32 GiB, el
    # máximo de Cloud Run: la cota de pico es 24 576 MiB, el 75 % de la memoria, y el
    # cierre del peor evento (n = 130 M) pide 20 351 MiB según 16 B × (2n + 4m), con
    # m = 2^28 (TRD-L3 ADR-L3-19), más hasta 4 096 MiB de retornos en vuelo: 24 447 MiB
    # antes de la base del proceso, con ≈ 130 MiB de margen. Cloud Run pide 8 vCPU para 32 GiB (la cifra de la
    # card; el apply la valida: no se pudo consultar su documentación desde el
    # contenedor); se declaran aquí para que bajar el cpu de la capa no deje este modo
    # inválido sin que el apply lo diga. Mismo timeout que backfill (43 200 s); la
    # pared de F se mide en el mes previo y en el backfill completo (runbook de L3).
    # L3_INFLIGHT_BYTES sube de 768 MiB a 4 GiB: con el presupuesto nuevo los eventos
    # gigantes guardan sus retornos y el tope por defecto frenaría la unidad a un
    # tramo a la vez durante meses enteros. El nombre del modo no es el de la CLI, de
    # ahí los args propios; run-job-args.sh arma los mismos para l3-backfill-spectrum.
    backfill-spectrum = {
      args    = ["--mode", "backfill", "--family", "spectrum"]
      cpu     = "8"
      memory  = "32Gi"
      timeout = 43200
      env     = merge(local.env, { L3_INFLIGHT_BYTES = "4294967296" })
      access  = local.access
    }

    # La pasada de la familia T (ITSC-396): el mismo modo monthly de la CLI con
    # --family path, en su propio job para que el workflow la corra después de la
    # de S y un fallo de T no toque lo que S ya publicó. Mismos recursos que
    # monthly (los heredados del módulo) y misma imagen. El nombre del modo no es
    # el de la CLI, de ahí los args propios.
    #
    # ITSC-420: 5 400 s. Una familia distinta de S verifica la serie de la familia y el techo
    # de S, y esa verificación midió 1 945,8 s en la nube (ver monthly). Con la misma regla
    # de TRD-L3 §10.2: 1 946 + 6 × 183,7 s (el mes de S; T tarda un 11 % menos, ITSC-395) =
    # 3 048 s, y 5 400 s son 1,77 veces eso. La holgura es mayor que en monthly a propósito,
    # hasta que se mida la verificación nueva; la hija ITSC-415 ajusta los timeouts
    # mensuales con esa cifra.
    monthly-path = {
      args    = ["--mode", "monthly", "--family", "path"]
      timeout = 5400
      access  = local.access
    }

    # La pasada de la familia A (ITSC-402): el mismo patrón que monthly-path, con
    # --family fit, en su propio job para que un fallo de A no toque lo que S y T ya
    # publicaron. Misma imagen y mismos recursos que monthly, salvo el timeout: A
    # tarda más que S. El mes más lento de A es 2023-03, con 1 240 s (ITSC-401), que
    # ya supera los 1 200 s de monthly. 7 500 s son 6,05 veces esa pared, la regla de
    # TRD-L2 §10.2 que sigue el resto de los timeouts, bajo el tope de 86 400 s.
    monthly-fit = {
      args    = ["--mode", "monthly", "--family", "fit"]
      timeout = 7500
      access  = local.access
    }
  }
}
