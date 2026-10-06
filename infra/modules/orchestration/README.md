# infra/modules/orchestration/

Orquesta el estado estacionario de una capa: un Cloud Workflow que ejecuta el
Cloud Run Job del modo pedido y un Cloud Scheduler por modo que lo dispara.
TRD-L1 §10.1 y §11.

## Qué crea

- `<layer>-run-job` (Workflow): recibe `{"mode": "<modo>"}`, llama
  `jobs.run` de `<layer>-<modo>` por HTTP con OAuth2. Un modo sin `next_jobs`
  termina al aceptar la ejecución (el resultado se ve en Cloud Run); uno con
  `next_jobs` espera su fin y encadena (ver "Encadenar un modo"). Un `mode` que
  no esté en `schedules` termina el workflow con error. Usa HTTP y no el
  conector `googleapis.run.v2` porque el conector espera la operación y pide
  `run.operations.get`, permiso de project.
- `<layer>-workflow` (service account): sin roles de project. Por job recibe
  `roles/run.invoker` sobre los jobs de `schedules` y sobre cada eslabón de
  `next_jobs`, y `roles/run.viewer` sobre el job de cada modo encadenado y
  sobre cada eslabón que el workflow espera (todos menos el último de la
  cadena).
- `<layer>-<modo>` (Scheduler, uno por entrada de `schedules`): inicia una
  ejecución del workflow con la identidad `scheduler-invoker` del stack data.

## Encadenar un modo

`next_jobs` en una entrada de `schedules` es una lista ordenada de nombres de
Cloud Run Jobs (de la capa o de otras). Corren uno tras otro: cada uno solo
cuando el anterior terminó bien, y el primero cuando termina el job del modo.
Ejemplo, el cierre mensual de L1 encadena `l2-monthly` y luego `viz-tiles`:

```hcl
monthly-close = {
  schedule = "0 6 8 * *"
  # ...
  next_jobs = ["l2-monthly", "viz-tiles"]
}
```

Qué hace el workflow con ese modo:

1. Ejecuta el job del modo (`jobs.run`) y toma el nombre de la ejecución de
   `body.metadata.name`.
2. Cada 30 s llama `executions.get` hasta que la ejecución trae
   `completionTime`.
3. Si hubo algún task exitoso y ninguno fallido ni cancelado, ejecuta el
   siguiente de la lista con `jobs.run` y sin overrides: el job aplica sus
   valores por defecto.
4. Repite los pasos 2 y 3 con ese eslabón, y así hasta el final. El último de
   la lista no se espera: el workflow termina al aceptar su ejecución y
   devuelve el nombre de su operación.

Casos de borde:

- **Modo sin `next_jobs`**: no cambia nada. El workflow termina al aceptar la
  ejecución y el resultado se ve en Cloud Run.
- **Un eslabón falla** (o se cancela): el workflow termina con error, con un
  mensaje que nombra la ejecución fallida y el job que no se ejecuta, y los
  eslabones que siguen no corren. Si falla `l2-monthly`, `viz-tiles` no corre.

Como el workflow espera, su ejecución dura lo que duren los jobs esperados
(hasta su timeout, más el reintento): se ve en Workflows como "Active" todo ese
tiempo.

Requisito: los jobs de `next_jobs` deben existir antes del apply, porque el IAM
se da sobre ellos (aplica primero sus stacks).

### Cierre mensual de L1: `l2-monthly` y `viz-tiles`

`l1-run-job` con `monthly-close` ejecuta `l1-monthly-close`, luego
`l2-monthly` y luego `viz-tiles`, sin overrides. `viz-tiles` sin argumentos
construye el mes anterior al actual (UTC): el mismo que `l2-monthly` acaba de
cerrar, porque el scheduler corre el día 8. Los stacks `l2` y `viz` deben estar
aplicados antes del apply de `l1`.

### Probarlo a mano

Con los schedulers en pausa, el humano lanza el workflow desde Cloud Shell
(`run-job.yml` ejecuta jobs, no el workflow):

```sh
gcloud workflows run l1-run-job --location=us-east1 --data '{"mode":"monthly-close"}'
```

El comando espera el resultado. Qué se espera ver: en Cloud Run Jobs, las
ejecuciones de `l1-monthly-close`, después de `l2-monthly` y después de
`viz-tiles`, en ese orden, cada una iniciada cuando la anterior terminó bien.
El comando devuelve el nombre de la operación de `viz-tiles` apenas la acepta
(esa última ejecución no se espera); sus tiles del mes aparecen en el bucket
viz. Si `monthly-close` falló, el estado es `FAILED` con el motivo y no hay
ejecución de `l2-monthly` ni de `viz-tiles`; si falló `l2-monthly`, no hay
ejecución de `viz-tiles`. Con `{"mode":"daily"}` el workflow no encadena nada.

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
