# Job de operación (ITSC-298): corre un script Python ad hoc tomado de GCS. No es
# una capa de datos: no escribe en el lago. Este stack no posee buckets (el
# bucket ops lo crea data, como todos); solo pide acceso a sus prefijos.
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
  ops     = local.buckets["ops"]

  # Solo lectura sobre el lago (landing, eventos, hallazgos, manifiesto y tiles
  # de viz) y sobre los scripts. Nada de escritura en el lago: lo que un script
  # quiera dejar va a results/, abajo.
  access = {
    (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
    (local.buckets["dc-events"])   = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectViewer", prefixes = ["l1/", "l2/"] }
    (local.buckets["manifest"])    = { role = "roles/storage.objectViewer", prefixes = ["l1/", "l2/"] }
    (local.buckets["viz"])         = { role = "roles/storage.objectViewer", prefixes = ["tiles/"] }
    (local.ops)                    = { role = "roles/storage.objectViewer", prefixes = ["scripts/"] }
  }
}

provider "google" {
  project = local.data.project_id
  region  = local.data.region
}

module "layer" {
  source = "../../../modules/layer"

  layer      = "ops"
  project_id = local.data.project_id
  region     = local.data.region
  image      = "${local.data.artifact_registry}/ops_tools:${trimspace(file("${path.module}/../../../../layers/ops_tools/VERSION"))}"

  # Límites fijos: acotan lo que un script puede gastar (riesgo aceptado de la
  # card). 4 vCPU y 8 GiB son holgura para escanear pies Parquet y comparar
  # bordes; lo que pida más se gradúa a una capa con su propia sonda.
  cpu    = "4"
  memory = "8Gi"

  # Una hora y no más: un script que tarda más no es ad hoc.
  timeout = 3600

  # 0 y no 1: un script ad hoc no se reintenta solo. Si falla, el humano lee
  # el resumen y decide; repetirlo a ciegas paga dos veces y tapa el error.
  max_retries = 0

  env = {
    # El entrypoint arma OPS_RESULTS_URI = <esto>/<ejecución>/ para el script.
    OPS_RESULTS_ROOT = "gs://${local.ops}/results"
  }

  # Un solo modo, una sola tarea (run-job-tasks.sh devuelve una unidad). Job
  # ops-script y service account ops-script.
  modes = {
    script = { access = local.access }
  }
}

# Escritura solo en results/. objectCreator crea objetos nuevos y no pisa ni
# borra: un script no puede tocar lo que otro dejó, y escribir en otra parte
# (scripts/, o cualquier bucket del lago) falla con 403. Sin lectura ni listado
# aquí: eso lo da el viewer de scripts/, así que la condición solo necesita la
# rama del objeto.
resource "google_storage_bucket_iam_member" "results_creator" {
  bucket = local.ops
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${module.layer.service_account_emails["script"]}"

  condition {
    title       = "ops-script-${local.ops}-results"
    description = "Escritura de ops-script solo bajo results/"
    expression  = "resource.name.startsWith(\"projects/_/buckets/${local.ops}/objects/results/\")"
  }
}
