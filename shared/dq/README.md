# dq

Contrato del hallazgo de calidad de datos (DQ) de intrinsica y utilidades
para escribirlo y leerlo.

## Qué contiene

- `dq.FINDING_SCHEMA` (en `dq.schema`): esquema Arrow de las 17 columnas.
- `dq.Finding`: dataclass validada, con los enumerados `Severity`, `Stage` y
  `Status`.
- `dq.emit_findings(findings, root)`: escribe los hallazgos como Parquet en
  una raíz local o `gs://`, y deja una línea de log por hallazgo.
- `dq.reader.current_findings(root)`: estado actual por `finding_id`
  (extra `reader`).

## Cómo lo instala una capa

En el `pyproject.toml` de la capa:

```toml
[tool.uv.sources]
dq = { workspace = true }
```

Luego, desde la raíz del repo:

```bash
uv add dq
```

## Extra `reader`

`emit_findings` solo necesita `pyarrow`. El lector usa DuckDB, que pesa más y
no todas las capas lo necesitan, por eso vive en un extra opcional:
`uv add "dq[reader]"`.

## Contrato

Esquema, disposición física, reglas y consulta de referencia en
[docs/data-contracts.md](../../docs/data-contracts.md#lago-de-hallazgos-de-calidad-de-datos).
