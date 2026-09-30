# infra/modules/orchestration/

Orquesta el estado estacionario de una capa: un Cloud Workflow que ejecuta el
Cloud Run Job del modo pedido y un Cloud Scheduler por modo que lo dispara.
TRD-L1 §10.1 y §11.

## Qué crea

- `<layer>-run-job` (Workflow): recibe `{"mode": "<modo>"}`, llama
  `jobs.run` de `<layer>-<modo>` por HTTP con OAuth2. Un modo sin `next_job`
  termina al aceptar la ejecución (el resultado se ve en Cloud Run); uno con
  `next_job` espera su fin y encadena (ver "Encadenar un modo"). Un `mode` que
  no esté en `schedules` termina el workflow con error. Usa HTTP y no el
  conector `googleapis.run.v2` porque el conector espera la operación y pide
  `run.operations.get`, permiso de project.
- `<layer>-workflow` (service account): sin roles de project. Por job recibe
  `roles/run.invoker` sobre los jobs de `schedules` y sobre cada `next_job`, y
  `roles/run.viewer` sobre el job de cada modo encadenado.
- `<layer>-<modo>` (Scheduler, uno por entrada de `schedules`): inicia una
  ejecución del workflow con la identidad `scheduler-invoker` del stack data.

## Encadenar un modo

`next_job` en una entrada de `schedules` es el nombre de un Cloud Run Job (de
la capa o de otra) que corre cuando el job del modo termina bien. Ejemplo, el
cierre mensual de L1 dispara `l2-monthly`:

```hcl
monthly-close = {
  schedule = "0 6 8 * *"
  # ...
  next_job = "l2-monthly"
}
```

Qué hace el workflow con ese modo:

1. Ejecuta el job del modo (`jobs.run`) y toma el nombre de la ejecución de
   `body.metadata.name`.
2. Cada 30 s llama `executions.get` hasta que la ejecución trae
   `completionTime`.
3. Si hubo algún task exitoso y ninguno fallido ni cancelado, ejecuta
   `next_job` con `jobs.run` y sin overrides: el job aplica sus valores por
   defecto. Si no, el workflow termina con error y `next_job` no corre.

Los modos sin `next_job` no cambian. Como el workflow espera, la ejecución dura
lo que dure el job (hasta su timeout, más el reintento): se ve en Workflows como
"Active" todo ese tiempo.

Requisito: el job de `next_job` debe existir antes del apply, porque el IAM se
da sobre él (aplica primero su stack).

### Probarlo a mano

Con los schedulers en pausa, el humano lanza el workflow desde Cloud Shell
(`run-job.yml` ejecuta jobs, no el workflow):

```sh
gcloud workflows run l1-run-job --location=us-east1 --data '{"mode":"monthly-close"}'
```

El comando espera el resultado. Si terminó bien, devuelve el nombre de la
operación de `l2-monthly`, cuya ejecución aparece en Cloud Run Jobs y escribe
el mes cerrado en `dc-events`. Si `monthly-close` falló, el estado es `FAILED`
con el motivo y no hay ejecución de `l2-monthly`. Con `{"mode":"daily"}` el
workflow no encadena nada.

## Encender un scheduler

Los schedulers se crean con `paused = true`. Encenderlos es decisión del
humano y se hace por PR:

1. En `schedules` del módulo (en el `main.tf` del stack), pon `paused = false`
   en el modo que quieras.
2. Abre el PR y aplica el stack desde `terraform.yml`.

No uses `gcloud scheduler jobs resume`: cambia el estado fuera de Terraform y
el siguiente `apply` lo revierte a `paused = true` (drift). Hasta que se
enciendan, lanza los modos con `run-job.yml`.

## Requisitos

El stack data (ITSC-222) debe estar aplicado: APIs, `scheduler-invoker` y los
roles de `deploy-github`.
