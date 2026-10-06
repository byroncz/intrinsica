# Visualización de los tiles (ITSC-307, ITSC-310). Corre dos jobs: viz-tiles
# reduce L1 y L2 a tiles por día y los deja, con su página HTML, en el bucket
# viz; viz-render vuelve a armar esas páginas desde los tiles ya escritos cuando
# cambia la plantilla. No es una capa Medallion: lee del lago y nunca escribe en
# él (landing, dc-events ni manifest). Este stack no posee buckets (el bucket viz
# lo crea data, como todos); solo pide acceso a sus prefijos y le da lectura al
# humano.
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
  viz     = local.buckets["viz"]

  # viz-tiles lee L1 y L2 (solo lectura) y escribe en dos lugares: los tiles, el
  # index.html de cada día y latest.html bajo viz/tiles/, y sus hallazgos bajo
  # dq-findings/viz/. objectUser y no objectCreator: regenerar un día pisa sus
  # archivos y la idempotencia por input_hash lee el index.json previo.
  access = {
    (local.buckets["landing"])     = { role = "roles/storage.objectViewer", prefixes = ["l1/"] }
    (local.buckets["dc-events"])   = { role = "roles/storage.objectViewer", prefixes = ["l2/"] }
    (local.viz)                    = { role = "roles/storage.objectUser", prefixes = ["tiles/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["viz/"] }
  }

  # viz-render no lee L1 ni L2: parte de los tiles y escribe solo páginas
  # (index.html del día y latest.html) y sus hallazgos. Sin acceso a landing ni a
  # dc-events, una plantilla con un error no puede tocar el lago. objectUser por
  # la misma razón que viz-tiles: regenerar una página pisa el index.html previo,
  # y su idempotencia lee la huella guardada en él.
  render_access = {
    (local.viz)                    = { role = "roles/storage.objectUser", prefixes = ["tiles/"] }
    (local.buckets["dq-findings"]) = { role = "roles/storage.objectUser", prefixes = ["viz/"] }
  }
}

provider "google" {
  project = local.data.project_id
  region  = local.data.region
}

module "layer" {
  source = "../../../modules/layer"

  layer      = "viz"
  project_id = local.data.project_id
  region     = local.data.region
  image      = "${local.data.artifact_registry}/viz_tiles:${trimspace(file("${path.module}/../../../../layers/viz_tiles/VERSION"))}"

  # Punto de partida, no una medida: el TRD-viz (§10.2) no fija el tamaño, se mide
  # como en L1 y L2. 4 vCPU y 4 GiB son lo que L2 mostró suficiente para leer un
  # mes de ticks con memoria acotada por lote (RSS ~0,5 GiB); un día de viz lee
  # lo mismo y acumula ~0,4 MB. La hija 8 de la Épica E6 (ITSC-303) lo
  # redimensiona con lo medido.
  cpu    = "4"
  memory = "4Gi"

  # 36 000 s (10 h): el mismo modo sirve al backfill por rangos de meses, que es
  # una sola tarea que recorre los días en orden; el tope del módulo es 86 400 s.
  # Es el mismo techo de l2-backfill y se ajusta con la pared medida.
  timeout = 36000

  # 1 y no 0, como L1 y L2: el reintento cubre fallos transitorios de
  # infraestructura, y uno determinista lo resuelve el humano. Reintentar no
  # repite lo hecho: un día cuyo input_hash no cambió se salta.
  max_retries = 1

  env = {
    VIZ_LANDING_ROOT = "gs://${local.buckets["landing"]}/l1"
    VIZ_EVENTS_ROOT  = "gs://${local.buckets["dc-events"]}/l2"
    VIZ_TILES_ROOT   = "gs://${local.viz}/tiles"
    VIZ_DQ_ROOT      = "gs://${local.buckets["dq-findings"]}/viz"
  }

  # Un job por modo, cada uno con una sola tarea (run-job-tasks.sh devuelve una
  # unidad): job y service account viz-tiles, y viz-render.
  modes = {
    tiles = { access = local.access }

    render = {
      access = local.render_access

      # 2 vCPU y 2 GiB: render lee los 18 arreglos de un día de uno en uno, los
      # codifica y los suelta (pico O(un arreglo), no O(día)); no hay ticks ni
      # Parquet, que es lo que pide los 4 GiB a viz-tiles. Punto de partida,
      # como el de viz-tiles: la hija 8 de la Épica E6 (ITSC-303) lo mide.
      cpu    = "2"
      memory = "2Gi"

      # 3 600 s (1 h): un re-render completo del histórico son ~3 300 días de
      # leer y escribir ~1 MB cada uno, minutos de cómputo; la hora deja margen
      # a la latencia de GCS. viz-tiles necesita 10 h porque reduce ticks; este
      # no. Un rango que no quepa se lanza por partes.
      timeout = 3600

      # Solo las dos raíces que render usa (ROOT_VARS de la CLI): sin
      # VIZ_LANDING_ROOT ni VIZ_EVENTS_ROOT, que el job no puede leer. Este env
      # reemplaza al del módulo. max_retries sigue siendo el del módulo, 1: el
      # reintento cubre fallos transitorios y repetir no rehace lo hecho, porque
      # un día con la misma huella de plantilla se salta.
      env = {
        VIZ_TILES_ROOT = "gs://${local.viz}/tiles"
        VIZ_DQ_ROOT    = "gs://${local.buckets["dq-findings"]}/viz"
      }
    }
  }
}

# El humano abre los archivos del bucket con su cuenta de Google. Lectura de todo
# el bucket y sin condición: lo único que hay es tiles/ y se regenera. El correo
# llega del secret VIZ_VIEWER; el plan lo muestra enmascarado por ser sensible.
# deploy-github fija este binding con bucketIamAdmin, sin acceso a los objetos.
resource "google_storage_bucket_iam_member" "viewer" {
  bucket = local.viz
  role   = "roles/storage.objectViewer"
  member = "user:${var.viz_viewer}"
}
