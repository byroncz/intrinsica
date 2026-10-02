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

2. Crea el environment `gcp` del repositorio (*Settings → Environments →
   New environment*) con tu usuario como revisor requerido. Debe existir
   antes del paso 3: un `workflow_dispatch` que referencia un environment
   inexistente lo crea sin protección.
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
(estrictamente mayor que la de `main`) y CI
publica la imagen con ese tag al mergear. El job de Cloud Run no la toma solo:
el humano espera a que el run de CI en `main` termine de publicar
`<capa>:<versión>` y, recién entonces, lanza desde Actions
*Terraform → `<capa>` → apply* para que el job pase al tag nuevo. Un apply
anterior falla porque el tag aún no existe. El plan debe mostrar
`image: ...:<versión anterior> -> ...:<versión nueva>`.

## Stack l2 (DC Events)

Instancia del módulo `layer` con dos modos: `l2-backfill` y `l2-monthly`, cada
uno con su service account. Lee la landing de L1 (`roles/storage.objectViewer`
bajo `l1/`) y escribe en `dc-events` y `dq-findings` (`roles/storage.objectUser`
bajo `l2/`). También lee el catálogo de θ, `manifest/l2/thetas.yaml`
(`roles/storage.objectViewer` bajo `l2/`; la variable `L2_THETAS_URI` apunta
a él, TRD-L2 §7.3): solo lectura, porque lo edita el humano. Antes de la
primera corrida hay que subirlo una vez desde Cloud Shell (ver "Cómo agregar
θ" en el [README de la capa](../layers/l2_dc_events/README.md)). No tiene
scheduler propio: la encadena el workflow de L1 (ITSC-283).

Orden de apply, todo por el humano:

1. `data` desde Cloud Shell (ver "Habilitar el despliegue desde GitHub
   Actions"), para que `deploy-github` tenga `bucketIamAdmin` sobre `dc-events`.
2. Esperar a que el run de CI en `main` publique `l2_dc_events:<versión>`, con
   la versión de `layers/l2_dc_events/VERSION`. Un apply anterior falla
   porque el tag aún no existe.
3. Actions → *Terraform* → `l2` → `apply`.

Los recursos salen de la sonda de L2 ([runbook](../docs/runbooks/sonda-l2.md),
ITSC-281) y de la decisión ADR-04 (ITSC-282, [TRD-L2 §14](../docs/TRD/l2.md)):
2 vCPU y 4 GiB en ambos jobs, sobre Cloud Run Jobs.

**Aplicar este cambio cierra la deriva de la sonda.** Las corridas de la sonda
cambiaron el job `l2-backfill` a mano con `gcloud run jobs update` (CPU variable,
4 GiB, timeout de 24 h, sin reintentos). El `apply` de `l2` lo devuelve al
código. El plan debe mostrar cambios de `cpu`, `memory` y `timeout` en los dos
jobs, y de `max_retries` (0 a 1) solo en `l2-backfill`: `l2-monthly` ya tenía 1
reintento. En `l2-backfill` también puede quitar `client` y `client_version`,
que `gcloud run jobs update` deja puestos y el módulo no fija; eso es parte de
cerrar la deriva. Cualquier otro cambio es una sorpresa y se revisa antes de
aprobar el `apply`.

## Timeouts y reintentos de los jobs de l2

Cada job fija `timeout` y `max_retries` de forma explícita (módulo `layer`; el
stack `l2` fija 2400 s y 1 reintento para todos y sobrescribe el timeout en
backfill):

| Job | timeout | max_retries |
|---|---|---|
| `l2-backfill` | 54000 s | 1 |
| `l2-monthly` | 2400 s | 1 |

- **`l2-monthly`: 2400 s** es 8,2× la pared de la sonda con 2 vCPU (292,2 s, mes
  2023-03, el más pesado) y 8,0× la peor corrida (301,1 s, con 8 vCPU). La regla
  de L1 pide al menos 6×.
- **`l2-backfill`: 54000 s (15 h)** es 1,7× la pared extrapolada de los 109
  meses (109 × 292,2 s = 31.850 s, 8,85 h). La regla pide al menos 1,5× y no
  más de 86.400 s. Es un techo: el mes más pesado se repite 109 veces y los
  demás tardan menos. L2 no usa task array, porque los meses se encadenan por
  carry-over, y el timeout de Cloud Run es por tarea, así que la tarea única
  debe caber entera.
- **Un reintento, no cero**: cubre fallos transitorios de infraestructura. Un
  fallo determinista (OOM, timeout, checksum) no lo arregla repetir y lo
  resuelve el humano. En backfill el reintento no repite trabajo: la
  reanudación (RF-L2-09) salta los meses que ya tienen su `carry_over.parquet`.
  Un mes a medias se reprocesa entero.
