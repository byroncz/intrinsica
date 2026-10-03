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

  # TRD-L2 §7: L2 lee la landing de L1 y el catálogo de θ (manifest/l2/thetas.yaml,
  # TRD-L2 §7.3), y escribe eventos y hallazgos de DQ. El catálogo lo edita el
  # humano, así que la cuenta solo lo lee: objectViewer, nunca escritura.
  access = {
    (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
    (local.buckets["manifest"])    = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.buckets["dc-events"])   = { role = "roles/storage.objectUser", prefixes = ["l2/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["l2/"] }
  }
}

# Semilla del catálogo de θ (ITSC-291): crea manifest/l2/thetas.yaml solo si no
# existe. No es un google_storage_bucket_object: el objeto ya existe (subido a
# mano el 2026-10-02), el proveedor no puede importarlo (el plan falla con
# "resource google_storage_bucket_object doesn't support import") y declararlo
# lo sobrescribiría con el contenido del repo. `gcloud storage cp --no-clobber`
# no pisa un objeto existente, y el provisioner corre solo al crear el recurso:
# sin triggers, una vez en el estado no se repite. El contenido vivo es del
# humano, que lo edita en GCS (TRD-L2 §7.3), así que un cambio en el archivo del
# repo ni se aplica ni aparece en el plan. Destruir el stack no borra el objeto.
# gcloud llega ya autenticado al job apply de _terraform-stack.yml.
resource "terraform_data" "thetas_seed" {
  provisioner "local-exec" {
    command = "gcloud storage cp --no-clobber \"$SEED_FILE\" \"$SEED_URI\""

    environment = {
      SEED_FILE = abspath("${path.module}/../../../../layers/l2_dc_events/src/l2_dc_events/config/thetas.yaml")
      SEED_URI  = "gs://${local.buckets["manifest"]}/l2/thetas.yaml"
    }
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
  # Run Jobs; Cloud Batch no hace falta. Dimensionado con la sonda de ITSC-286,
  # mes 2023-03 (190.227.841 ticks, 4 GiB en las tres, sin OOM), imagen 0.5.2
  # en 2 vCPU y 0.6.0 (mismo rendimiento) en 4 y 8:
  #
  #   vCPU | RSS pico (MiB) | pared (s) | freno cgroup (s) | vCPU-s | costo de lista (USD)
  #      2 |            487 |     311,7 |             99,4 |    623 | 0,0137
  #      4 |            489 |     184,1 |              4,5 |    736 | 0,0147
  #      8 |            491 |     133,4 |              0,0 |  1.067 | 0,0203
  #
  # Se fija 4 vCPU. Por costo de lista gana 2 vCPU, pero por 0,001 USD al mes
  # (menos de 10 %: empate, y el backfill cabe en el cupo gratis con cualquiera);
  # en el empate gana la pared (-41 %), y 2 vCPU está limitado por la CPU (32 %
  # de la pared frenada por el cgroup). 8 vCPU cuesta 38 % más por 27 % menos de
  # pared y sin freno: lo que frena ahí es un tramo serial (wait_s 38 %), no la
  # CPU, y va a otra card.
  # Regla de L1: RSS <= 75 % de la memoria y sin OOM. 491 MiB son 12 % de 4 GiB.
  # Se fija 4 GiB y no menos porque es la memoria medida, no una extrapolada.
  cpu    = "4"
  memory = "4Gi"

  # Timeout por defecto, el de monthly: un solo mes. 1200 s son 6,5x la pared de
  # la sonda con 4 vCPU (184,1 s, el mes más pesado); monthly procesa un mes
  # recién cerrado, de ordinario mucho más liviano.
  timeout = 1200

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
    # Catálogo de θ (TRD-L2 §7.3): agregar un θ es editar este objeto y lanzar
    # l2-backfill sin from; no requiere PR ni apply. Lo siembra
    # terraform_data.thetas_seed si falta.
    L2_THETAS_URI = "gs://${local.buckets["manifest"]}/l2/thetas.yaml"
  }

  # Una service account por modo. Ambos modos escriben lo mismo; la separación
  # deja revocar o auditar cada uno por su cuenta.
  modes = {
    # Una sola tarea encadena los 109 meses por carry-over (sin task array).
    # Techo: 109 x 184,1 s = 20.067 s (5,57 h), con el mes más pesado repetido.
    # 36.000 s (10 h) son 1,79x ese techo (la regla pide >= 1,5x) y quedan bajo
    # el tope de 86.400 s que acepta el módulo.
    backfill = { timeout = 36000, access = local.access }
    monthly  = { access = local.access }
  }
}
