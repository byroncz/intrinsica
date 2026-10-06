# infra/modules/

Módulos Terraform reutilizables. Definido en el
[TRD maestro §8.2](../../docs/TRD/plataforma_directional_change.md#82-infraestructura-como-código).

## Qué contiene

- `layer/`: módulo parametrizable (bucket/prefijo GCS, ruta de imagen, IAM
  mínimo, plantilla de job), se instancia una vez por capa.
- `orchestration/`: workflow y schedulers de una capa (ver su README).
- `alerting/`: canal de correo y política de alerta de hallazgos ERROR (ver
  el stack `batch/alerting`).
- `registry/` y `observability/`.

## Variable `modes` del módulo `layer`

`modes` es un mapa modo → `{ access }`, donde `access` asigna a cada bucket un
`role` de storage y los `prefixes` donde aplica. Por cada modo el módulo crea
el Cloud Run Job `<layer>-<modo>` (args por defecto `--mode <modo>`, misma
imagen, CPU, RAM y env) y la service account `<layer>-<modo>` con solo su
acceso. Hay un job por modo porque la service account se fija en la plantilla
del Job, no en la ejecución: la única forma de dar a cada modo su propia
identidad y permisos mínimos (TRD-L1 §11) es un job por modo. Un modo puede
sobrescribir `timeout`, `cpu` y `memory` (opcionales; sin ellos usa los del
módulo) y `env`, que reemplaza al `env` del módulo en vez de mezclarse: así un
modo que no lee un bucket tampoco recibe su ruta (ITSC-310). Los outputs
`job_names` y `service_account_emails` son mapas indexados por modo.

## Qué no contiene

- Declaración de `provider`: los módulos hijo no la declaran, la fija el stack.
- Estado ni recursos concretos: eso vive en `stacks/`.
