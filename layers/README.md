# layers/

Una carpeta por capa del pipeline. Definido en el
[TRD maestro §8.1](../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota).

## Qué contiene

Una carpeta `l<N>_<nombre>/` por capa (`l1_ingest`, `l2_dc_events`,
`l3_frames`, `l4_indicators`). Cada una produce **una imagen Docker**,
con el modo de ejecución como parámetro (ADR-07). Estructura esperada en
[`scaffold/layer_template/`](../scaffold/layer_template/README.md).

## Qué no contiene

- Código compartido entre capas: va en `shared/`.
- Infraestructura (Terraform): va en `infra/`.
- Capas que aún no se implementan: la carpeta se crea con la primera
  línea de código de la capa.
