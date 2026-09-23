# infra/stacks/batch/

Stack de la arquitectura batch (la actual). Definido en el
[TRD maestro §8.2](../../../docs/TRD/plataforma_directional_change.md#82-infraestructura-como-código).

## Qué contiene

La instanciación de los módulos de `infra/modules/` para el pipeline batch,
con backend `gcs` y estado propio.

## Qué no contiene

- Definiciones de módulos: van en `infra/modules/`.
- El stack `kappa/`: es futuro y otra arquitectura, con estado separado.
- Secretos ni credenciales.
