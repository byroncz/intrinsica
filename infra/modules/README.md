# infra/modules/

Módulos Terraform reutilizables. Definido en el
[TRD maestro §8.2](../../docs/TRD/plataforma_directional_change.md#82-infraestructura-como-código).

## Qué contiene

- `layer/`: módulo parametrizable (bucket/prefijo GCS, ruta de imagen, IAM
  mínimo, plantilla de job), se instancia una vez por capa.
- `registry/` y `observability/`.

## Qué no contiene

- Declaración de `provider`: los módulos hijo no la declaran, la fija el stack.
- Estado ni recursos concretos: eso vive en `stacks/`.
