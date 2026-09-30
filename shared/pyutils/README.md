# pyutils

Utilidades Python comunes de las capas: el escritor Parquet atómico y el hash
de contenido. Salen de `l1_ingest` (ITSC-245) para que L2 no mantenga una
copia; las capas no se importan entre sí (RNF-14 del maestro), así que lo
común vive aquí.

## Qué contiene

- `pyutils.PartitionWriter(path, schema, sort_order=None, row_group_size=None)`:
  escribe un Parquet por lotes (ZSTD-3, estadísticas, `sorting_columns` si hay
  `sort_order`) a un temporal y `commit` lo renombra sobre el destino. Sin
  `row_group_size`, cada lote o tabla es su propio row group. Detalle de la
  sobrescritura atómica y de GCS en
  [`docs/data-contracts.md`](../../docs/data-contracts.md).
- `pyutils.ContentHasher(schema)` y `pyutils.content_hash(table)`: SHA-256 del
  contenido lógico (esquema más filas en orden), sin importar el chunking ni
  los bytes del Parquet. Las columnas enteras, decimales o booleanas no
  nulables se hashean sobre sus bytes; las demás, valor a valor.
- `pyutils.resolve_fs(path)`: sistema de archivos Arrow y ruta, local o `gs://`.

## Cómo lo instala una capa

```bash
uv add --package <capa> pyutils
```

Lo propio de cada capa (esquema, orden, tamaño de row group, rutas) se pasa
por parámetro; ver `l1_ingest.write.output_writer`.
