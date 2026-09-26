# infra/modules/orchestration/

Orquesta el estado estacionario de una capa: un Cloud Workflow que ejecuta el
Cloud Run Job del modo pedido y un Cloud Scheduler por modo que lo dispara.
TRD-L1 §10.1 y §11.

## Qué crea

- `<layer>-run-job` (Workflow): recibe `{"mode": "<modo>"}`, llama
  `jobs.run` de `<layer>-<modo>` por HTTP con OAuth2 y termina sin esperar el
  fin del job (el resultado se ve en Cloud Run). Un `mode` que no esté en
  `schedules` termina el workflow con error. Usa HTTP y no el conector
  `googleapis.run.v2` porque el conector espera la operación y pide
  `run.operations.get`, permiso de project.
- `<layer>-workflow` (service account): solo `roles/run.invoker` sobre los
  jobs de `schedules`, dado por job.
- `<layer>-<modo>` (Scheduler, uno por entrada de `schedules`): inicia una
  ejecución del workflow con la identidad `scheduler-invoker` del stack data.

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
