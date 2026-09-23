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

`stacks/batch/data` posee lo que nunca se destruye: el GCP project, las APIs
y los buckets (incluido el de estado de Terraform). Como el bucket de estado
lo crea el propio stack, el primer `apply` usa estado local y después el
estado se migra al bucket. Solo lo ejecuta el humano; cuesta 0 USD (buckets
vacíos y un project sin cómputo).

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

Tras el apply, el stack publica tres outputs. Cárgalos en GitHub, en
*Settings → Secrets and variables → Actions → Variables*, como variables de
repositorio (no son secretos):

| Variable de GitHub       | Output de Terraform        |
| ------------------------ | -------------------------- |
| `GCP_ARTIFACT_REGISTRY`  | `artifact_registry`        |
| `GCP_WIF_PROVIDER`       | `wif_provider_name`        |
| `GCP_CI_SERVICE_ACCOUNT` | `ci_service_account_email` |

```bash
cd infra/stacks/batch/data
terraform output
```

- No hay llaves de service account: el CI se autentica con Workload
  Identity Federation y solo el repositorio `github_repo` (por defecto
  `byroncz/intrinsica`) puede actuar como la service account `ci`.
- `ci` solo escribe en el repositorio de imágenes, no a nivel de project.
- La política de limpieza borra toda versión y conserva las
  `keep_tagged_versions` (10) más recientes, con o sin tag (la KEEP tiene
  precedencia sobre la DELETE); así el repositorio se mantiene dentro del
  free tier de 0,5 GB.

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
    bucket = "<project_id>-tfstate"
    prefix = "stacks/batch/data"
  }
}
```

- Se inicializa con `terraform init -backend-config="bucket=<project_id>-tfstate"`.
- Consume solo los outputs de `data` (`project_id`, `project_number`,
  `region`, `buckets`), por ejemplo `data.terraform_remote_state.data.outputs.buckets["landing"]`.
- Un stack de capa nunca crea buckets: son datos y `data` es su único dueño.
- Los módulos hijo no declaran `provider` (TRD maestro §8.2).
