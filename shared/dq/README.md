# dq

Contrato del hallazgo de calidad de datos (DQ) de intrinsica y utilidades
para escribirlo y leerlo.

## Qué contiene

- `dq.schema.FINDING_SCHEMA`: esquema Arrow de las 17 columnas.
- `dq.Finding`: dataclass validada, con los enumerados `Severity`, `Stage` y
  `Status`.
- `dq.emit_findings(findings, root)`: escribe los hallazgos como Parquet en
  una raíz local o `gs://`, y deja una línea de log por hallazgo.
- `dq.reader.current_findings(root)`: estado actual por `finding_id`
  (extra `reader`).

## Cómo lo instala una capa

Desde la raíz del repo, indica la capa con `--package`:

```bash
uv add --package <capa> dq
```

Es equivalente a correr `uv add dq` dentro de `layers/<capa>/`. Sin
`--package`, `uv add` modifica el `pyproject.toml` de la raíz, no el de la
capa. Como `dq` es miembro del workspace, `uv add` agrega solo
`dq = { workspace = true }` en `[tool.uv.sources]` de la capa.

## Extra `reader`

`emit_findings` solo necesita `pyarrow`. El lector usa DuckDB, que pesa más y
no todas las capas lo necesitan, por eso vive en un extra opcional:
`uv add --package <capa> "dq[reader]"`.

## Contrato

Esquema, disposición física, reglas y consulta de referencia en
[docs/data-contracts.md](../../docs/data-contracts.md#lago-de-hallazgos-de-calidad-de-datos).
