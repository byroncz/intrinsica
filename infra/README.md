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
