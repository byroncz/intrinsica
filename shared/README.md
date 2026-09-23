# shared/

Código compartido entre capas. Definido en el
[TRD maestro §8.1](../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota).

## Qué contiene

Un paquete por subcarpeta, cada uno con su `pyproject.toml` (o `Cargo.toml`).

## Nombres reservados (aún no existen)

| Nombre | Qué será | Épica |
|---|---|---|
| `dc_core` | Crate Rust del núcleo de Directional Change | E4 |
| `dc_pyo3` | Bindings PyO3 sobre `dc_core` | E4 |
| `pyutils` | Utilidades Python comunes | E1 |
| `dq` | `emit_findings()` y contrato del hallazgo de DQ | E1 |

## Qué no contiene

- Lógica propia de una capa: va en `layers/`.
- Directorios sin `pyproject.toml`: rompen el workspace de uv. Cada
  directorio se crea junto con su manifiesto, no antes.
