# infra/

Infraestructura como código. Definido en el
[TRD maestro §8.2](../docs/TRD/plataforma_directional_change.md#82-infraestructura-como-código).

## Qué contiene

- `modules/`: módulos Terraform reutilizables.
- `stacks/`: una carpeta por arquitectura, cada una con su propio estado.

## Qué no contiene

- Código de las capas ni de `shared/`.
- Entornos de despliegue (dev/qa/uat/prod): el término *environment* está
  reservado y hoy no se instancia. Por eso es `stacks/`, no `environments/`.

## Terraform en el contenedor

La imagen devkit no trae Terraform. Se instala en `~/.local/bin` (ya está en
el `PATH`), sin `sudo`, con la versión fijada en
`bin/install-terraform.sh` y verificando el SHA256 publicado por HashiCorp:

```bash
infra/bin/install-terraform.sh
terraform version
```

- Exporta `CHECKPOINT_DISABLE=1` para que Terraform no consulte el servicio
  de chequeo de versiones (dominio fuera de la lista blanca del proxy).
- Los dominios `releases.hashicorp.com` y `registry.terraform.io` están en
  `domains` de `.devkit/devkit.toml`; aplican tras `devkit recreate`.
- `~/.local/bin` no sobrevive a `devkit rebuild`: se repite el script.

## Bootstrap del stack `batch/data`

`stacks/batch/data` posee lo que nunca se destruye: el GCP project, las APIs,
el repositorio de imágenes (Artifact Registry), Workload Identity Federation
y los buckets (incluido el de estado de Terraform). Como el bucket de estado
lo crea el propio stack, el primer `apply` usa estado local y después el
estado se migra al bucket. Solo lo ejecuta el humano; cuesta 0 USD (buckets
vacíos, un repositorio sin imágenes y un project sin cómputo).

### 1. Cuenta y facturación

1. Crea la cuenta de Google Cloud y activa el free trial en
   <https://console.cloud.google.com/freetrial>. Es una cuenta personal, sin
   organización: el project no lleva `org_id` ni `folder_id`.
2. Obtén el `billing_account`:

   ```bash
   gcloud auth login
   gcloud billing accounts list   # columna ACCOUNT_ID: XXXXXX-XXXXXX-XXXXXX
   ```

3. Elige un `project_id` único globalmente (6 a 30 caracteres, minúsculas,
   dígitos y guiones). Prefija todos los buckets: `<project_id>-<sufijo>`.
4. Autentica Terraform: `gcloud auth application-default login`.

### 2. Primer apply con backend local

```bash
cd infra/stacks/batch/data
export CHECKPOINT_DISABLE=1
cat > terraform.tfvars <<'TFVARS'
project_id      = "<project_id>"
billing_account = "<XXXXXX-XXXXXX-XXXXXX>"
TFVARS
mv backend.tf backend.tf.off          # sin backend: estado local
terraform init
terraform apply
```

`terraform.tfvars` está en `.gitignore` (`*.tfvars`) y Terraform lo lee solo:
así ningún `plan` ni `apply` del stack vuelve a pedir las variables.

Crea el project, habilita las APIs y crea los buckets.

Si el apply falla en `google_project_service` con un 403 de
`serviceusage.googleapis.com` ("requires a quota project"), es porque las
credenciales de usuario aún no tienen quota project: el project no existía al
autenticar. A esa altura ya existe, así que asígnalo y repite el apply:

```bash
gcloud auth application-default set-quota-project <project_id>
terraform apply
```

Si el provider
exigiera `org_id` para crear el project, créalo a mano e impórtalo:

```bash
gcloud projects create <project_id>
gcloud billing projects link <project_id> --billing-account=<billing_account>
terraform import google_project.this projects/<project_id>
```

### 3. Migrar el estado al bucket `tfstate`

```bash
mv backend.tf.off backend.tf
terraform init -migrate-state -backend-config="bucket=<project_id>-tfstate"
```

Responde `yes` a copiar el estado. Verifica con `terraform plan` (sin
cambios) y borra `terraform.tfstate*` locales (están en `.gitignore`).

### 4. Conectar GitHub Actions (Artifact Registry y WIF)

Tras el apply, el stack publica estos outputs. Las tres últimas filas
(`GCP_DEPLOY_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`, `GCP_REGION`) se cargan al
final, siguiendo "Habilitar el despliegue desde GitHub Actions", después de
crear el environment `gcp`. Cárgalos en GitHub, en
*Settings → Secrets and variables → Actions → Variables*, como variables de
repositorio (no son secretos):

| Variable de GitHub           | Output de Terraform            |
| ---------------------------- | ------------------------------ |
| `GCP_ARTIFACT_REGISTRY`      | `artifact_registry`            |
| `GCP_WIF_PROVIDER`           | `wif_provider_name`            |
| `GCP_CI_SERVICE_ACCOUNT`     | `ci_service_account_email`     |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | `deploy_service_account_email` |
| `GCP_PROJECT_ID`             | `project_id`                   |
| `GCP_REGION`                 | `region`                       |

```bash
cd infra/stacks/batch/data
terraform output
```

- No hay llaves de service account: el CI se autentica con Workload
  Identity Federation y solo el repositorio `github_repo` (por defecto
  `byroncz/intrinsica`) puede actuar como las service accounts `ci-github` y
  `deploy-github`.
- `ci` solo escribe en el repositorio de imágenes, no a nivel de project.
- La política de limpieza borra toda versión y conserva las
  `keep_tagged_versions` (10) más recientes, con o sin tag (la KEEP tiene
  precedencia sobre la DELETE); así el repositorio se mantiene dentro del
  free tier de 0,5 GB.

### Habilitar el despliegue desde GitHub Actions

Lo hace el humano, en este orden. Ningún agente ejecuta el apply.

1. Aplica el stack `data` desde Cloud Shell. Crea la service account
   `deploy-github`, ligada al mismo pool WIF que `ci-github`. Actualiza antes
   el checkout a `main` y comprueba que `terraform.tfvars` (está en
   `.gitignore`) sigue en la carpeta; si no, recréalo como en la sección 2:

   ```bash
   git pull
   cd infra/stacks/batch/data
   terraform init -backend-config="bucket=<project_id>-tfstate"
   terraform plan
   terraform apply
   ```

   Este paso se repite cada vez que cambian los permisos de `deploy-github`
   (por ejemplo, al darle `roles/artifactregistry.reader` sobre el
   repositorio de imágenes, que Cloud Run exige para desplegar un job con
   imagen real). Después, relanza desde Actions el workflow que había
   fallado (Terraform → capa → apply).

   Aplica `data` desde Cloud Shell también antes del apply de `l1` con la
   orquestación (Workflows y Scheduler): habilita sus APIs, da a
   `deploy-github` los roles para gestionarlos, crea la service account
   `scheduler-invoker` (output `scheduler_invoker_service_account_email`) y
   aplica las reglas de ciclo de vida del bucket landing. Sin este apply, el
   de `l1` falla con 403 o con la API deshabilitada.

   Aplica `data` también antes del primer apply de `l2`: `deploy-github`
   necesita `bucketIamAdmin` sobre el bucket `dc-events` para fijar el IAM por
   prefijo. Sin este apply, el de `l2` falla con 403 al crear los
   `google_storage_bucket_iam_member` de `dc-events`.

   Aplica `data` también antes del primer apply de `l3` (ITSC-334, ITSC-350):
   `deploy-github` necesita `bucketIamAdmin` sobre `dc-descriptors`. Sin este
   apply, el de `l3` falla con 403 al crear los `google_storage_bucket_iam_member`
   de `dc-descriptors`.

2. Crea el environment `gcp` del repositorio (*Settings → Environments →
   New environment*) con tu usuario como revisor requerido. Debe existir
   antes del paso 3: un run que referencia un environment inexistente lo crea
   sin protección.
3. Carga como variables de repositorio `GCP_DEPLOY_SERVICE_ACCOUNT`
   (output `deploy_service_account_email`), `GCP_PROJECT_ID` (output
   `project_id`) y `GCP_REGION` (output `region`), como en la sección 4.

### 5. Agregar un stack de capa

Cada stack de capa es una carpeta `infra/stacks/batch/<capa>/` con su propio
estado en el mismo bucket, con prefix distinto:

```hcl
terraform {
  backend "gcs" {
    prefix = "stacks/batch/<capa>"
  }
}

data "terraform_remote_state" "data" {
  backend = "gcs"
  config = {
    bucket = var.tfstate_bucket
    prefix = "stacks/batch/data"
  }
}
```

- El bucket de estado no admite variables en `backend`, pero sí en
  `terraform_remote_state`: cada stack de capa declara `variable "tfstate_bucket"`.
- Se inicializa con `terraform init -backend-config="bucket=<project_id>-tfstate"`.
- Plan y apply piden la variable:
  `terraform plan -var tfstate_bucket=<project_id>-tfstate` (igual con `apply`).
- Consume solo los outputs de `data` (`project_id`, `project_number`,
  `region`, `buckets`, `artifact_registry`, `wif_provider_name`,
  `ci_service_account_email`), por ejemplo `data.terraform_remote_state.data.outputs.buckets["landing"]`.
- Un stack de capa nunca crea buckets: son datos y `data` es su único dueño.
- Los módulos hijo no declaran `provider` (TRD maestro §8.2).

## Limpieza única de los `.tmp` acumulados en `landing` (ITSC-234)

`PartitionWriter` escribe a un temporal y lo mueve sobre el destino con
`fs.move`: en GCS eso es copia más borrado, y con el bucket versionado cada
`.tmp` borrado queda como versión no vigente del mismo tamaño que la
partición, duplicando el almacenamiento (detalle completo en
docs/data-contracts.md, "Sobrescritura atómica"). Antes de ITSC-234, la
primera regla de lifecycle (`num_newer_versions`/`days_since_noncurrent_time`)
nunca alcanzaba a estos `.tmp`: un `.tmp` borrado no vuelve a tener versiones
más nuevas, así que la condición de `num_newer_versions` nunca se cumplía y
se acumulaban para siempre. Desde ITSC-235 esa primera regla ya no usa
`num_newer_versions`: toda versión no vigente de `landing` se borra a los 30
días (`days_since_noncurrent_time = 30`), lo que también alcanza a los
`provisional-day=NN.parquet` que `monthly-close` borra. La regla nueva (`matches_suffix = [".tmp"]`,
`days_since_noncurrent_time = 1`) baja esa espera a un día para lo que se
escriba de ahora en más, pero los `.tmp` acumulados antes de aplicarla siguen
ahí hasta que la regla los alcanza. Para no esperar, bórralos a mano una sola
vez desde Cloud Shell. Solo lo ejecuta el humano; el agente no tiene
credenciales de GCP.

Primero verifica qué hay, filtrando solo las versiones no vigentes:
`gsutil ls -a` lista todas las versiones y les agrega `#<generación>` al
nombre; comparar contra `gsutil ls` (sin `-a`, que solo lista la vigente) es
lo que distingue una versión no vigente de la vigente:

```bash
gsutil ls -a 'gs://<project_id>-landing/l1/**/.*.tmp'
```

Confirmado que son restos (no hay `.tmp` vigente: ningún `PartitionWriter`
en curso los necesita), bórralos:

```bash
gsutil rm -a 'gs://<project_id>-landing/l1/**/.*.tmp'
```

Verifica el resultado comparando el tamaño vigente contra el total con
versiones; deberían quedar aproximadamente iguales (una diferencia pequeña es
esperable: son los `.tmp` de escrituras recientes que la regla de un día
todavía no alcanzó):

```bash
gsutil du -sh 'gs://<project_id>-landing/l1'
gsutil du -sha 'gs://<project_id>-landing/l1'
```

## Timeouts y reintentos de los jobs de l1

Cada job fija `timeout` y `max_retries` de forma explícita (módulo `layer`,
valores por defecto 3600 s y 1; el stack `l1` solo sobrescribe el timeout en daily y seam-check):

| Job | timeout | max_retries |
|---|---|---|
| `l1-backfill` | 3600 s | 1 |
| `l1-monthly-close` | 3600 s | 1 |
| `l1-daily` | 900 s | 1 |
| `l1-seam-check` | 900 s | 1 |

3600 s es 6× la pared de la sonda (577 s, mes 2023-03); el tope por defecto de
Cloud Run (600 s) dejaba 23 s de margen. Se reintenta una vez y no cero
porque el reintento cubre fallos transitorios de infraestructura; un fallo
determinista (OOM, timeout, checksum) no lo arregla repetir y lo resuelve el
humano. La reanudación no es por reintento: el modo backfill salta lo que ya
existe con el mismo `.CHECKSUM`; si falla, relanza el rango y solo se procesan
los meses que faltan.

## Desplegar un cambio de capa

Todo cambio de código de una capa (incluidos `shared/`, `uv.lock` y el
`pyproject.toml` raíz, que entran en cada imagen) sube su `layers/<capa>/VERSION`
(estrictamente mayor que la de `main`) y CI publica la imagen con ese tag al
mergear. El job de Cloud Run no toma la imagen solo: lo mueve un `apply` del
stack, y desde ITSC-291 ese `apply` te llega solo, como una pregunta.

### Mergear y aprobar

1. Mergeas el PR. Si tocó `infra/stacks/batch/<stack>/`, `infra/modules/`,
   `layers/<capa>/VERSION` o los workflows `terraform.yml` y
   `_terraform-stack.yml`, al terminar CI en verde sobre `main` arranca el run
   *Terraform* titulado *Deploy: \<título del commit\>*. Un merge que no toca nada
   de eso deja un run corto (`changes`, sin plan) y ninguno esperando.
2. Por cada stack de capa afectado, el job `plan` (sin aprobación) corre
   `terraform plan -out` y deja el plan completo en su log y en el resumen del
   run. Si no hay cambios, el stack termina ahí: no hay `apply` ni nada espera.
3. Si hay cambios, el job `apply` queda en *waiting*. Lee el plan y, en la página
   del run, pulsa *Review deployments* (environment `gcp`): apruebas o rechazas.
   Aprobado, el `apply` baja el plan guardado, lo muestra y lo aplica tal cual.
   Sin respuesta, GitHub cancela el deployment pendiente a los 30 días y no se
   toca GCP. El plan debe mostrar, por ejemplo,
   `image: ...:<versión anterior> -> ...:<versión nueva>`.

Por qué el deploy cuelga de CI (`workflow_run`) y no de un `needs` al job de
build: el build vive en `ci.yml`, que corre en cada push y construye solo las
capas que cambiaron, y `terraform.yml` también corre en PR y a mano. Encadenarlos
con `needs` obligaría a fundir ambos workflows o a reconstruir las imágenes en
`terraform.yml`. Con `workflow_run`, "CI en verde sobre `main`" garantiza que la
imagen ya existe, que es exactamente lo que faltaba (aplicar antes falla con
"tag no existe"). Costo: si otro job de CI falla (lint, Rust), no hay deploy
aunque la imagen se haya publicado; se arregla en `main` o se lanza el manual.

**El plan guardado vive en un bucket, no en un artefacto.** El repositorio es
público y cualquier cuenta de GitHub descarga los artefactos de un run; un plan
guardado lleva dentro los valores de sus variables, entre ellos los correos de
`alerting` y `viz`. Por eso el job `plan` lo sube a
`gs://<proyecto>-tfstate/plans/<run>-<intento>/<stack>.tfplan` (privado), el
`apply` lo borra al terminar y una regla de ciclo de vida de `data` borra a los 7
días los que quedaron (rechazados, cancelados o con apply fallido). La regla
llega con el próximo apply de `data`; sin él el flujo funciona igual, solo que
los planes huérfanos no se limpian solos.

Otros workflows del repo con `deploy-github` también pueden escribir en ese
bucket, así que el job `plan` publica el `sha256` del archivo que produjo y el
`apply` lo verifica tras bajarlo: si no coincide, falla antes de `terraform apply`
y no aplica nada.

**Dos merges seguidos dejan dos runs esperando.** Aprueba el más reciente y
rechaza el anterior: su plan ya no coincide con `main`. Si aprobaras el viejo
después de aplicar el nuevo, Terraform lo rechaza con *Saved plan is stale*
porque el estado cambió. Esa es la red de seguridad; la regla es no depender de
ella.

**Si el apply falla a mitad de camino**, el plan guardado ya no sirve (el estado
cambió). Corrige la causa y re-ejecuta *todo* el run (*Re-run all jobs*), que
planea de nuevo y pide otra aprobación; o lanza el manual.

Límites: el cálculo de afectados compara el commit de `main` contra su primer
padre, así que cubre un *squash* o un *merge commit*. Un *rebase merge* de varios
commits solo ve el último: despliega a mano con el manual. El stack `data` nunca
entra: lo aplica el humano desde Cloud Shell.

### Manual: reaplicar sin cambio de código o destroy

*Actions → Terraform → Run workflow* (stack, `apply` o `destroy`) sirve cuando
no hay cambio de código que dispare el flujo anterior: cerrar la deriva de un
`gcloud run jobs update` hecho a mano, reintentar tras corregir un permiso en
`data`, o un `destroy`. Se divide igual: `plan` sin aprobación (`plan` o
`plan -destroy`, visible en el log y el resumen) y `apply` con aprobación del
plan guardado. Si el plan no tiene cambios, no hay nada que aprobar. Un `destroy` de
`l2` no borra el catálogo de θ: la semilla es un `terraform_data` y el objeto no
es de Terraform.

Si el merge no disparó el CI de push, el rescate es lanzar *CI* a mano en
`main` (Actions → *CI* → *Run workflow*, rama `main`): construye y publica las
tres capas, aunque no hayan cambiado. Los tags que ya existían se sobrescriben
con el mismo contenido. Un run manual sobre otra rama solo construye y corre la
prueba de humo, sin publicar. Ese run manual no dispara el deploy (solo lo hace
un push): lánzalo después con el manual de Terraform.

### Secret `ALERT_EMAIL`

El job `plan` no usa el environment `gcp`, así que no ve sus secrets.
`ALERT_EMAIL` (el correo del stack `alerting`) debe ser **secret de
repositorio** (*Settings → Secrets and variables → Actions → Secrets → Repository
secrets*). Es secret y no variable porque GitHub no enmascara una variable: la
imprime en claro en el encabezado `env:` de cada paso, y los logs de este repo
público los lee cualquiera. Un secret sale como `***`, y el workflow solo lo
exporta cuando el stack es `alerting`. Sin él el job `plan` de `alerting` falla de
entrada con el motivo, en vez de planear con un correo de relleno que luego se
aplicaría.

## Stack l2 (DC Events)

Instancia del módulo `layer` con dos modos: `l2-backfill` y `l2-monthly`, cada
uno con su service account. Lee la landing de L1 (`roles/storage.objectViewer`
bajo `l1/`) y escribe en `dc-events` y `dq-findings` (`roles/storage.objectUser`
bajo `l2/`). También lee el catálogo de θ, `manifest/l2/thetas.yaml`
(`roles/storage.objectViewer` bajo `l2/`; la variable `L2_THETAS_URI` apunta
a él, TRD-L2 §7.3): solo lectura, porque lo edita el humano (ver "Cómo agregar
θ" en el [README de la capa](../layers/l2_dc_events/README.md)). No tiene
scheduler propio: la encadena el workflow de L1 (ITSC-283).

**Semilla del catálogo (ITSC-291).** El stack declara
`terraform_data.thetas_seed`, que al crearse corre `gcloud storage cp
--no-clobber` con el contenido de
`layers/l2_dc_events/src/l2_dc_events/config/thetas.yaml`: sube el catálogo solo
si el objeto no existe y nunca lo pisa. No es un `google_storage_bucket_object`
porque el objeto ya existía (subido a mano el 2026-10-02), el proveedor no admite
importarlo (el plan falla con `doesn't support import`) y declararlo lo
sobrescribiría. El primer apply de `l2` muestra `terraform_data.thetas_seed will
be created`: es esperado y no toca el catálogo vivo. Una vez en el estado, el
recurso no se repite y editar el catálogo en GCS no aparece como deriva en el
plan. Un `destroy` de `l2` tampoco borra el objeto. Para sembrarlo de nuevo en un
bucket de prueba, borra el objeto y el recurso del estado (`terraform state rm
terraform_data.thetas_seed`) y aplica.

Para eso `deploy-github` necesita leer y crear ese único objeto (acceso sobre
`l2/thetas.yaml`, en `data`: `objectViewer` y `objectCreator`, sin update ni
delete). Solo lo usa el job `apply`: el plan no toca el
objeto, así que `data` se aplica antes de aprobar el primer deploy, no antes del
merge. Sin el permiso, el apply de `l2` falla con 403.

Orden de apply:

1. El humano aplica `data` desde Cloud Shell (ver "Habilitar el despliegue desde
   GitHub Actions"), para que `deploy-github` tenga `bucketIamAdmin` sobre
   `dc-events` y acceso al objeto de la semilla.
2. Mergear el PR que sube `layers/l2_dc_events/VERSION` o toca el stack. Al
   terminar CI en `main`, que publica `l2_dc_events:<versión>`, el run
   *Terraform* deja el `apply` de `l2` esperando.
3. Revisar el plan y aprobar en *Review deployments* (ver "Desplegar un cambio de
   capa").

Los recursos salen de la sonda de L2 ([runbook](../docs/runbooks/sonda-l2.md),
ITSC-281 e ITSC-286) y de la decisión ADR-04 ([TRD-L2 §14](../docs/TRD/l2.md)):
4 vCPU y 4 GiB en ambos jobs, sobre Cloud Run Jobs.

**Aplicar este cambio cierra la deriva de la sonda.** Las corridas de la sonda
cambiaron el job `l2-backfill` a mano con `gcloud run jobs update` (CPU variable,
4 GiB, timeout de 24 h, sin reintentos); la última lo dejó en 8 vCPU. El
`apply` de `l2` lo devuelve al código. El plan debe mostrar cambios de `cpu`,
`timeout` y `max_retries` (0 a 1) en `l2-backfill`, y de `cpu` (2 a 4) y
`timeout` (2400 s a 1200 s) en `l2-monthly`, que ya tenía 1 reintento. La
memoria no cambia. En `l2-backfill` también puede quitar `client` y
`client_version`, que `gcloud run jobs update` deja puestos y el módulo no fija;
eso es parte de cerrar la deriva. Cualquier otro cambio es una sorpresa y se
revisa antes de aprobar el `apply`.

## Stack l3 (Descriptores por evento)

Instancia del módulo `layer` con dos modos: `l3-backfill` y `l3-monthly`, cada
uno con su service account y una sola tarea (sin task array). Imagen
`l3_dc_descriptors:<VERSION>`, de `layers/l3_dc_descriptors/VERSION`. Diseño y
decisiones en [TRD-L3 §11](../docs/TRD/l3.md#11-operaciones).

**Buckets y accesos.** Las dos cuentas tienen el mismo acceso y nada más:

| Bucket | Rol | Prefijo | Para qué |
|---|---|---|---|
| `landing` | `objectViewer` | `l1/` | `consolidated.parquet` de L1 |
| `dc-events` | `objectViewer` | `l2/` | eventos y carry-over de L2 |
| `manifest` | `objectViewer` | `l2/` | catálogo de θ (`l2/thetas.yaml`) |
| `dc-descriptors` | `objectUser` | `l3/` | descriptores (`summaries.parquet`) |
| `dq-findings` | `objectUser` | `l3/` | hallazgos de DQ |

Escribir fuera de `l3/` da 403. Lo comprueba `permisos.py` del job `ops-script`
(`13 de 13`), que ahora incluye la escritura en `dc-descriptors` y en `dq-findings/l3/`
(ver el [runbook de operación ad hoc](../docs/runbooks/operacion-ops.md)).

**El bucket de L3 es `dc-descriptors`, no `frames`.** ITSC-334 reutilizó el bucket
`frames` (creado para las tramas que la decisión del 2026-10-08 descartó) para
ahorrar un apply de `data`, pero ese apply hizo falta de todos modos, así que el
argumento desapareció. El nombre `frames` no dice qué contiene; `dc-descriptors`
sigue la convención de nombrar cada bucket por su contenido y es simétrico con la
capa `l3_dc_descriptors`. Un bucket no se renombra: ITSC-350 crea el nuevo en
`data`, apunta L3 a él y elimina `frames`, que está vacío; hacerlo después de la
primera corrida (ITSC-335) exigiría copiar gigabytes y reverificar `content_hash`.
Se usa un bucket propio y no `dc-events` porque el módulo `layer` admite un solo
rol por bucket: en `dc-events`, L3 no podría leer `l2/` y escribir `l3/`. Decisión
del humano, 2026-10-14 (revierte la del 2026-10-13); detalle en
[TRD-L3 §11](../docs/TRD/l3.md#11-operaciones), historial 1.3.4.

**El catálogo de θ es el de L2.** `L3_THETAS_URI` apunta a
`gs://<proyecto>-manifest/l2/thetas.yaml`, el mismo objeto que usa L2. El stack
`l3` no lo siembra ni lo copia (lo siembra `l2`); si falta, L3 sale con código 2.

**Cómputo, definitivo (ITSC-352, 2026-10-24).** 2 vCPU, 4 GiB, `max_retries = 1`
y una sola tarea. Sale de medir 2021-01 (46,0 M de eventos) con 1, 2, 4 y 8 vCPU
con la imagen 0.19.0 y los valores por defecto:

| vCPU | `wall_s` | vCPU-s (pared × vCPU) | `cpu_s` | `cpu_throttled_s` | `rss_peak_mib` |
|---|---|---|---|---|---|
| 1 | 282,5 | 283 | 242 | 745 | 1 797 |
| **2** | **111,5** | **223** | 194 | 280 | 2 141 |
| 4 | 95,9 | 383 | 266 | 47 | 2 183 |
| 8 | 74,9 | 599 | 194 | 2 | 2 654 |

Los 50 `content_hash` de 2021-01 son los de ITSC-335 en los cuatro tamaños. De 4 a
8 vCPU la pared baja solo a 0,78 (el plan pedía ≤ 0,6) y el costo sube 1,6 veces:
el motor está ocupado el 14 % del tiempo y lo que no escala es la E/S, con 12
escritores y 2 lectores fijos. 2 vCPU es el mínimo de vCPU-s y cuesta 16 % más de
pared que 4; con 1 vCPU la pared se triplica y el job se estrangula 745 s.
Memoria: dos medidas distintas. Sobre los 110 meses (ITSC-335) el RSS pico
máximo fue 1 267 MiB (núcleo original). Cifra vigente, corrida de seis meses
2017-08 a 2018-01 con la imagen 0.19.0: 2 347 MiB en 2018-01 (57 % de 4 GiB).
Sin OOM. Timeouts con la regla de TRD-L2 §10.2: `l3-monthly` 1 200 s (7,3 veces la
pared estimada del mes más pesado a 2 vCPU, ≈ 164 s; el mínimo son 984 s) y
`l3-backfill` 10 800 s (1,9 veces la suma estimada, ≈ 5 600 s; el mínimo son
8 400 s). Las dos son estimaciones desde la medición de 4 vCPU de ITSC-364
(mes más pesado 2022-11, 141,4 s; suma 4 828 s) por el 1,16 de 2021-01; la suma
real sale del backfill de 110 meses con este tamaño. Decisión y cifras en
[TRD-L3 §10.2](../docs/TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición).
El backfill completo cuesta ≈ 11 200 vCPU-s y 22 400 GiB-s estimados (≈ 0,25 USD de
lista), el 6 % del cupo gratuito mensual de Cloud Run **si L3 corriera aislado**:
el cupo es de la cuenta de facturación y lo comparten L1, L2, viz y L3
([TRD-L3 §10.5](../docs/TRD/l3.md#105-costo)).

**Aplicar este cambio.** El plan del `apply` de `l3` debe mostrar `cpu` (4 a 2)
en los dos jobs, `timeout` en `l3-monthly` (10 800 s a 1 200 s) y en
`l3-backfill` (86 400 s a 10 800 s). Memoria, reintentos y env no cambian.
Cualquier otro cambio es una sorpresa y se revisa antes de aprobar.

**Backfill histórico por rangos (núcleo original).** Con el núcleo original la
suma de los 110 meses eran 11,5 h, más que las 6 h que espera el run de Actions
(un rango de 41 meses tomó 4 h 57 min); con el núcleo nuevo son ≈ 1,3 h y caben
en una sola ejecución (ITSC-364). Cloud Run no corta la tarea, pero el run
fallaría. Por eso el histórico se lanzó
con *Run job* → `l3-backfill`, uno a la vez, cada uno con éxito antes del
siguiente:

| # | `from` | `to` | Resultado |
|---|---|---|---|
| 1 | `2017-08` | `2020-12` | 4 h 57 min (41 meses) |
| 2 | `2021-01` | `2021-12` | 2 h 09 min |
| 3 | `2022-01` | `2023-02` | 2 h 07 min |
| 4 | vacío | `2024-12` | 1 h 07 min |
| 5 | vacío | `2026-08` | Falló: construyó 2025-01 a 2026-01 (13 meses, suma de `wall_s` 2 269 s) y abortó en 2026-02. Su reintento de Cloud Run y un relanzamiento fallaron igual |
| 6 | vacío | vacío | 19 min, tras ITSC-351: construyó 2026-02 a 2026-08 y reconstruyó 2026-09 |
| 7 | vacío | vacío | Run 38771287050 (`l3-backfill-74t6c`), sin `--force`: salida 0, los 50 θ con frontera en 2026-09, las 110 unidades saltadas («ningún θ la necesita»), sin escrituras, sin `theta_behind_frontier` y con la misma `generation`. Log final: «backfill: los 50 θ están al día en el rango» |

La frontera de cada θ es contigua desde `series_start`: un rango sin `from`
arranca en el mes siguiente al último construido. Con 2023-03 y 2026-09 ya
hechos fuera de orden, los rangos 3 y 5 terminaron un mes antes de cada uno para
no reconstruirlos (el 4 arrancó en 2023-04, porque la serie ya era contigua hasta
2023-03). Si el run de un rango agota las 6 h de Actions, la tarea sigue
en Cloud Run: el humano la sigue en *Cloud Run → Jobs* y lanza el siguiente rango
cuando termine. Relanzar un rango interrumpido es seguro: la frontera retoma donde
quedó.

El rango 5 abortó en 2026-02 con `l2_state_inconsistent`: L1 había dejado 2 000
`agg_trade_id` repetidos en ese mes (2 000 ids repetidos, 3 000 filas sobrantes, según
midió ITSC-351) y la compuerta de TRD-L3 §9.2 los detectó.
ITSC-351 lo corrigió en L1 y reprocesó en cascada L1 → L2 → viz desde 2026-02
(ver el [runbook de operación de L1](../docs/runbooks/operacion-l1.md)). Con esa
cascada hecha, el rango 6 (run 38674127712) corrió sin `from` ni `to`: la serie
era contigua hasta 2026-01, así que arrancó en 2026-02 y llegó hasta 2026-09, que
reconstruyó con los mismos 50 `content_hash` de la primera corrida (run
38493995515). Los `unit_timing` de la suma de 41 256 s salen de 103 meses
(40 294 s): 101 de los rangos 1 a 5, 2023-03 del run 38495040816 y 2026-09 del
run 38493995515; y del rango 6 (2026-02 a 2026-08, 7 meses); la reconstrucción de 2026-09 no
se suma dos veces.

**Regla para un backfill futuro de la serie completa** (por ejemplo, al agregar una
familia): fija cada rango por su **pared estimada**, no por su número de meses. La
suma de los `wall_s` esperados del rango debe quedar en **≈ 5 h (18 000 s) o menos**,
para dejar holgura bajo las 6 h de Actions. Estima cada mes con el ajuste de
[TRD-L3 §10.2](../docs/TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)
o con los `unit_timing` de `dq-findings/l3/`. El número de meses no sirve de medida:
los 41 meses de 2017-08 a 2020-12 promediaron ≈ 7,2 min por mes (suma de
`wall_s` 17 792 s), pero 2021 promedió ≈ 10,75 min por mes (2 h 09 min en 12 meses)
y el más pesado, 2021-01, tomó 24 min; un rango de 41 meses que cubriera 2020-2022
rondaría 7 h. Lo más simple es repetir los rangos de la tabla (1 a 4 y el resto
hasta el último mes), que ya se sabe que caben.

**Cadena mensual.** El stack `l1` pasa a
`next_jobs = ["l2-monthly", "viz-tiles", "l3-monthly"]` en `monthly-close`: el
workflow corre cada job solo si el anterior terminó bien, y L3 queda al final.
Costo asumido: si `viz-tiles` falla un mes, `l3-monthly` no corre y el humano lo
lanza desde *Run job*. Ninguno recibe overrides: `l3-monthly` toma el mes anterior.

**Orden de apply de un despliegue desde cero, todo por el humano**:

1. `data` desde Cloud Shell: da a `deploy-github` el IAM de `dc-descriptors` en
   `deploy_bucket_iam` (`bucketIamAdmin`, ver "Habilitar el despliegue desde
   GitHub Actions"). Sin este apply, el de `l3` falla con 403 al crear los
   bindings de `dc-descriptors`.
2. `l3`, tras mergear y cuando CI publique `l3_dc_descriptors:<versión>`. Va
   **antes** que `l1`.
3. `l1`: el workflow de la orquestación recibe `run.invoker` y `run.viewer`
   sobre `l3-monthly`, que debe existir. Si `l1` se aplica primero, falla.

**Migración de `frames` a `dc-descriptors` (ITSC-350)**, sobre un proyecto que ya
tiene `l3` aplicado con `frames`. Todo por el humano, después del merge:

0. En Cloud Shell: `gcloud storage ls gs://<proyecto>-frames/` debe salir vacío.
   Si no lo está, se detiene todo y se avisa.
1. `data` desde Cloud Shell, con el checkout de `main`. El plan debe mostrar **1
   bucket a crear (`dc-descriptors`), 1 a destruir (`frames`, vacío) y el IAM de
   `deploy-github` movido de uno a otro**. Quitar el bloque de `frames` del código
   basta para destruirlo: `prevent_destroy` solo protege mientras el bloque
   existe. Si aun así Terraform lo rechaza, se hace en dos pasos: primero
   `prevent_destroy = false` sobre `frames` y apply; luego quitar el bloque y
   apply.
2. `l3` en el environment `gcp`. El plan debe mostrar **los 2 bindings de
   `frames` destruidos, 2 creados sobre `dc-descriptors` y los 2 jobs
   actualizados en `L3_SUMMARIES_ROOT`**. Sin apply de `l1`: la cadena no cambia.
3. `permisos.py` desde *Run job* (`ops-script`): la última línea debe ser
   `permisos: 13 de 13 como se esperaba`.

El plan de `l3` en el PR muestra las dos cuentas, sus bindings y los dos jobs; el
de `l1`, el workflow `l1-run-job` con la cadena nueva y dos bindings nuevos:
`run.invoker` sobre `l3-monthly` y `run.viewer` sobre `viz-tiles`, que ahora el
workflow espera.

**Lanzar los jobs.** *Actions → Run job* (nunca un agente):

- `l3-backfill`: `from` y `to` son meses `YYYY-MM`. Sin `from`, la frontera de cada
  θ decide desde dónde avanza; sin `to`, llega al último mes con
  `consolidated.parquet` en L1. `force` reprocesa desde `from` e ignora la
  frontera (exige `from`). `series_start` es opcional.
- `l3-monthly`: `from` es el mes (vacío = el mes anterior); `to` vacío o igual a
  `from`; no admite `force`.

Regla: no lanzar a mano el mismo mes y modo que Scheduler esté ejecutando.

## Stack ops (scripts de operación)

Instancia del módulo `layer` con un solo modo: el job `ops-script` (y la service
account `ops-script`), que corre un script Python tomado de GCS (ITSC-298). No
es una capa de datos: no escribe en el lago, salvo crear `content_hash.json` en
`dc-events/l2/` para el backfill del linaje (ITSC-368, abajo). Cómo se usa, qué ve el humano y los
scripts de partida están en el
[runbook de operación ad hoc](../docs/runbooks/operacion-ops.md).

- **El bucket `<proyecto>-ops` lo crea `data`**, no `ops`: crear buckets exige
  permisos de proyecto que `deploy-github` no tiene y que no conviene darle (un
  rol de proyecto con `storage.buckets.setIamPolicy` alcanzaría a todos los
  buckets del lago), y los buckets son datos, de `data` como todos. Tiene
  versionado (cada subida de un script es una generation) y dos reglas de ciclo
  de vida: las versiones no vigentes de `scripts/` se borran a los 90 días y
  todo lo de `results/` a los 7.
- **Acceso de la service account.** Lectura (`roles/storage.objectViewer`) de
  `landing/l1/`, `dc-events/l2/`, `dc-descriptors/l3/`, `dq-findings/{l1,l2,l3}/`,
  `manifest/{l1,l2}/`, `viz/tiles/` y `ops/scripts/`, por el módulo `layer`
  (`dc-descriptors/l3/` y `dq-findings/l3/` desde ITSC-336, para
  `l3_check_month.py`; el apply de `ops` agrega 2 bindings y no destruye
  ninguno); y escritura solo en `ops/results/`
  (`roles/storage.objectCreator`, un binding propio del stack porque el módulo
  da un solo rol por bucket). Sin roles de IAM ni `run.jobs.update`.
- **Backfill del linaje (ITSC-368).** Un binding más, propio del stack:
  `roles/storage.objectCreator` sobre `dc-events` con la condición de que el nombre
  empiece por `l2/` y termine en `/content_hash.json`. Solo crea (no pisa ni
  borra): el script `l2_backfill_lineage.py` no puede tocar un Parquet ni una
  huella ya publicada, y cualquier otro nombre sigue dando 403. El apply de `ops`
  agrega 1 binding y no destruye ninguno. Va **antes** del backfill y del apply de
  `l3` ([Cambios requeridos de L3](../docs/runbooks/operacion-l3.md#cambios-requeridos)).
- **Cómputo.** 4 vCPU, 8 GiB, timeout 3600 s, `max_retries = 0` (un script ad
  hoc no se reintenta solo) y una sola tarea. La imagen es `ops_tools`
  (`layers/ops_tools/`, versión en su `VERSION`).
- `deploy-github` recibe `bucketIamAdmin` sobre el bucket `ops`, como sobre
  los demás, para fijar el IAM por prefijo.

Orden de apply, todo por el humano:

1. `data` desde Cloud Shell (ver "Habilitar el despliegue desde GitHub
   Actions"): crea el bucket y da a `deploy-github` `bucketIamAdmin` sobre él.
   Sin este apply, el plan de `ops` falla: el output `buckets` de `data` aún
   no trae `ops`. Por eso el check `stack (ops)` del PR está en rojo hasta
   entonces, y es esperado. Aplica `data` con el checkout de la rama del PR,
   antes de aprobar (`main` aún no tiene el bucket `ops`); luego re-ejecuta
   `stack (ops)`. Su plan (la service account, ocho bindings y el job) es la
   evidencia de que `ops` crea solo eso.
2. Mergear. Al terminar CI en `main`, que publica `ops_tools:<versión>` (la de
   `layers/ops_tools/VERSION`), el run *Terraform* deja el `apply` de `ops`
   esperando; no se aprueba antes porque el apply falla si el tag aún no existe.
3. Aprobar en *Review deployments*. El plan debe mostrar solo la service
   account, sus bindings (siete de lectura y uno de escritura) y el job.
4. Correr la prueba de permisos del runbook (`permisos.py`).

Un cambio de código de `ops_tools` sube su `VERSION` y se despliega como
cualquier capa (ver "Desplegar un cambio de capa").

## Stack viz (tiles de visualización)

Instancia del módulo `layer` con dos modos. `viz-tiles` (y su service account)
reduce L1 y L2 a tiles por día y deja, junto a ellos, el `index.html`
autocontenido de cada día y `tiles/latest.html` (ITSC-307). `viz-render` (y su
service account, ITSC-310) vuelve a armar esas páginas desde los tiles del
bucket cuando cambia la plantilla, sin leer L1 ni L2.
Diseño en el [TRD-viz](../docs/TRD/viz.md); la imagen es `viz_tiles`
(`layers/viz_tiles/`, versión en su `VERSION`). Terraform no publica ninguna
página ni archivo: lo único que declara es el bucket (en `data`), el job y quién
puede leer.

**Qué crea**

- En `data`: el bucket `<proyecto>-viz`, con acceso uniforme, acceso público
  prevenido, sin versionado y sin regla de ciclo de vida (todo lo que hay es
  `tiles/` y se regenera desde L1 y L2), y `bucketIamAdmin` sobre él para
  `deploy-github`. Ese rol basta para el binding del visor; `deploy-github` no
  tiene ningún rol de objetos sobre este bucket.
- En `viz`: la service account y el job `viz-tiles` (4 vCPU, 4 GiB, timeout de
  36 000 s, 1 reintento), la service account y el job `viz-render` (2 vCPU,
  2 GiB, timeout de 3 600 s, 1 reintento) y el binding del visor. El tamaño es un
  punto de partida y no una medida: la hija 8 de la Épica E6 lo redimensiona con
  lo medido. `viz-tiles` tiene el timeout alto porque el mismo modo sirve al
  backfill por rangos; `viz-render` solo lee y escribe un archivo de ~1 MB por
  día, sin ticks, y por eso pide la mitad del cómputo. El módulo `layer` admite
  `cpu`, `memory` y `env` por modo para esto: el env de `viz-render` trae solo
  `VIZ_TILES_ROOT` y `VIZ_DQ_ROOT`.

**Accesos**

| Quién | Rol | Dónde |
|---|---|---|
| `viz-tiles` | `objectViewer` | `landing/l1/` y `dc-events/l2/` |
| `viz-tiles` | `objectUser` | `viz/tiles/` (tiles, `index.html` de cada día y `latest.html`) y `dq-findings/viz/` |
| `viz-render` | `objectUser` | `viz/tiles/` y `dq-findings/viz/`; nada en `landing` ni en `dc-events` |
| El humano (`VIZ_VIEWER`) | `objectViewer` | todo el bucket `viz` |

`viz-tiles` y `viz-render` nunca escriben en `landing`, `dc-events` ni
`manifest`; `viz-render` ni siquiera los lee. Los dos usan `objectUser` y no
`objectCreator` porque regenerar un día pisa archivos que ya existen.

**Secret `VIZ_VIEWER`.** Es el correo de la cuenta de Google que abre los
archivos del bucket (variable sensible `viz_viewer`, `TF_VAR_viz_viewer`). Igual
que `ALERT_EMAIL` (ver "Secret `ALERT_EMAIL`"), debe ser **secret de
repositorio** (*Settings → Secrets and variables → Actions → Secrets →
Repository secrets*) y no del environment `gcp`: el job `plan` no usa ese
environment y no ve sus secrets. Es secret y no variable porque GitHub no
enmascara una variable y los logs de este repo público los lee cualquiera; el
workflow solo lo exporta cuando el stack es `viz`. Sin él, el job `plan` de
merge y manual falla de entrada con el motivo. El correo nunca va al repo ni a un
`.tfvars`. El humano lo crea antes del primer deploy de `viz`.

El plan de un PR usa el relleno `plan@example.invalid` y muestra
`~ update in place` en `google_storage_bucket_iam_member.viewer` (el `member`
cambia por el relleno): es ruido esperado, como en `alerting`, porque el PR no ve
el secret. El plan de un merge o del manual usa el correo real.

**Orden de aplicación**, todo por el humano:

1. `data` desde Cloud Shell (ver "Habilitar el despliegue desde GitHub
   Actions"): crea el bucket `viz` y da a `deploy-github` `bucketIamAdmin` sobre
   él. Sin este apply, el plan de `viz` falla: el output `buckets` de `data` aún
   no trae `viz`, y el check `stack (viz)` del PR queda en rojo hasta entonces
   (es esperado). Aplica `data` con el checkout de la rama del PR, antes de
   aprobar.
2. Crear el secret de repositorio `VIZ_VIEWER`.
3. Mergear. Al terminar CI en `main`, que publica `viz_tiles:<versión>` (la de
   `layers/viz_tiles/VERSION`), el run *Terraform* deja el `apply` de `viz`
   esperando; no se aprueba antes porque el apply falla si el tag aún no existe.
4. Aprobar en *Review deployments*. En el primer deploy el plan muestra las dos
   service accounts, sus seis bindings, el binding del visor y los dos jobs. Con
   `viz-tiles` ya aplicado, el deploy de `viz-render` (ITSC-310) agrega solo su
   service account, sus dos bindings y su job, y no cambia nada de `viz-tiles`.

**Lanzar tiles.** *Actions → Run job* con `job` = `viz-tiles`: `from` y `to` son
meses `YYYY-MM` (vacíos, la CLI toma el mes anterior); `force` regenera aunque el
`input_hash` no haya cambiado y exige `from`. `series_start`, `script` y `args`
se rechazan. Es una sola tarea que recorre los meses del rango en orden.

**Re-renderizar páginas.** Tras cambiar la plantilla (sube `VERSION` y se
despliega la imagen nueva), *Actions → Run job* con `job` = `viz-render`: mismos
`from` y `to` (meses `YYYY-MM`, vacíos = el mes anterior) y mismo `force`
(regenera aunque la huella de la plantilla no haya cambiado; exige `from`);
`series_start`, `script` y `args` se rechazan. Lee solo los tiles del bucket: en
un rango, los días sin `index.json` se omiten; `input_missing` (job en rojo) sale
si un mes no tiene ningún día con tiles o si un día tiene arreglos rotos. Un
re-render completo del histórico son unas 3 300 lecturas de `index.json` más 18
arreglos por día y otras tantas escrituras: minutos de cómputo y centavos de
operaciones, frente a repetir el backfill desde L1 y L2. Un rango que no
quepa en la hora del job se lanza por partes. Los agentes no lanzan este
workflow: lo corre el humano.

**Compartir un día.** El `index.html` del día es el exportable; no hay zip. Dos
caminos, sobre el bucket `<proyecto>-viz`:

- Abrirlo: `https://storage.cloud.google.com/<bucket>/tiles/provider=binance/market=spot/asset=BTCUSDT/day=YYYY-MM-DD/index.html`,
  o `.../tiles/latest.html` para el último día, con la cuenta de `VIZ_VIEWER`.
- Descargarlo para enviarlo:
  `gcloud storage cp "gs://<bucket>/tiles/provider=binance/market=spot/asset=BTCUSDT/day=YYYY-MM-DD/index.html" .`
  El archivo es autocontenido y abre desde disco en un navegador, sin servidor.
  Más detalle (gzip del bucket, qué hacer si llega comprimido) en el
  [README de la capa](../layers/viz_tiles/README.md#compartir-un-día).

## Stack alerting (alerta de hallazgos ERROR)

Instancia del módulo `alerting`: un canal de correo y una política de log match
de Cloud Monitoring que avisa cuando un Cloud Run Job de cualquier capa deja un
hallazgo de calidad con `severity=ERROR` (ITSC-296). Va en su propio stack,
con estado propio, para que aplicar o destruir `l1` o `l2` no toque las
alertas. Qué dispara, a quién llega y cómo silenciarla:
[runbook de operación](../docs/runbooks/operacion-l1.md#alerta-por-correo-itsc-296).

- El correo es la variable sensible `alert_email`, que el workflow toma del
  secret `ALERT_EMAIL` (`TF_VAR_alert_email`); no está en el repo. Debe ser
  **secret de repositorio**, no del environment `gcp` (ver "Secret `ALERT_EMAIL`"):
  el humano lo crea ahí antes del próximo deploy de `alerting`. Mientras no
  exista, el job `plan` de merge y manual falla con el motivo. El plan de los PR
  usa siempre un correo de relleno y no lo imprime.
- Antes del primer apply, el humano aplica `data` desde Cloud Shell: habilita
  `monitoring.googleapis.com` y da a `deploy-github` los roles
  `monitoring.alertPolicyEditor`, `monitoring.notificationChannelEditor` y
  `logging.configWriter` (una política log match crea por debajo una
  notification rule en Cloud Logging). Sin eso el apply falla con 403 o con la
  API deshabilitada.
- El `apply` de `alerting` llega por el flujo de merge o por el manual (ver
  "Desplegar un cambio de capa"). El plan debe mostrar solo el canal y la
  política.
- El plan de un PR usa el correo de relleno `plan@example.invalid` y muestra
  `~ update in place` en `google_monitoring_notification_channel.email`, en
  `labels` (valor sensible): es ruido esperado, porque el PR no ve el secret. El
  plan del job `plan` tras un merge o el manual sí usa el correo real: ahí
  cualquier cambio en `alerting` es una sorpresa.
- Costo: 0 USD hoy. Google anunció 0,35 USD/mes por referencia de métrica "no
  antes del 1 de septiembre de 2027"; una alerta log match cuenta como una. El
  canal de correo no cobra y los logs caben en los 50 GiB/mes gratis
  ([precios](https://cloud.google.com/products/observability/pricing)).

## Timeouts y reintentos de los jobs de l2

Cada job fija `timeout` y `max_retries` de forma explícita (módulo `layer`; el
stack `l2` fija 1200 s y 1 reintento para todos y sobrescribe el timeout en
backfill):

| Job | timeout | max_retries |
|---|---|---|
| `l2-backfill` | 36000 s | 1 |
| `l2-monthly` | 1200 s | 1 |

- **`l2-monthly`: 1200 s** es 6,5× la pared de la sonda con 4 vCPU (184,1 s, mes
  2023-03, el más pesado, imagen 0.6.0). La regla de L1 pide al menos 6×.
- **`l2-backfill`: 36000 s (10 h)** es 1,79× la pared extrapolada de los 109
  meses (109 × 184,1 s = 20.067 s, 5,57 h). La regla pide al menos 1,5× y no
  más de 86.400 s. Es un techo: el mes más pesado se repite 109 veces y los
  demás tardan menos. L2 no usa task array, porque los meses se encadenan por
  carry-over, y el timeout de Cloud Run es por tarea, así que la tarea única
  debe caber entera.
- **Un reintento, no cero**: cubre fallos transitorios de infraestructura. Un
  fallo determinista (OOM, timeout, checksum) no lo arregla repetir y lo
  resuelve el humano. En backfill el reintento no repite trabajo: la
  reanudación (RF-L2-09) salta los meses que ya tienen su `carry_over.parquet`.
  Un mes a medias se reprocesa entero.
