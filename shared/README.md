# shared/

Código compartido entre capas. Definido en el
[TRD maestro §8.1](../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota).

## Qué contiene

Un paquete por subcarpeta, cada uno con su `pyproject.toml` (o `Cargo.toml`).

## Paquetes

| Paquete | Qué es |
|---|---|
| [`dq`](dq/) | Contrato del hallazgo de DQ (esquema Arrow) |
| [`pyutils`](pyutils/) | Escritor Parquet atómico, `content_hash` y `resolve_fs` compartidos por L1 y L2 (ITSC-245) |
| [`dc_core`](dc_core/) | Crate Rust del núcleo de Directional Change: detector DC de un θ (ITSC-239) |
| [`dc_pyo3`](dc_pyo3/) | Bindings PyO3 sobre `dc_core`: fan-out de N θ y carry-over para Python (ITSC-243) |

## Qué no contiene

- Lógica propia de una capa: va en `layers/`.
- Directorios sin `pyproject.toml`: rompen el workspace de uv. Cada
  directorio se crea junto con su manifiesto, no antes; un paquete solo
  Rust (como `dc_core`) se excluye en `[tool.uv.workspace] exclude` del
  `pyproject.toml` raíz. `dc_pyo3` sí tiene `pyproject.toml` (maturin), así
  que es miembro de los dos workspaces.
