"""Hash del contenido lógico de una tabla, independiente del chunking y del archivo."""

import hashlib

import pyarrow as pa
import pyarrow.compute as pc


def _raw_bytes(column: pa.Array) -> memoryview:
    """Bytes de valores de una columna de ancho fijo, respetando el offset del slice."""
    if pa.types.is_boolean(column.type):
        # El bitmap de un slice arrastra bits vecinos: se pasa a uint8.
        column = pc.cast(column, pa.uint8())
    if column.null_count:
        raise ValueError("una columna no nulable trae nulos: no se puede hashear")
    width = column.type.bit_width // 8
    start = column.offset * width
    return memoryview(column.buffers()[1])[start : start + len(column) * width]


def _is_raw(field: pa.Field) -> bool:
    """Si los bytes de valores de la columna bastan para hashearla."""
    kind = field.type
    fixed = (
        pa.types.is_integer(kind)
        or pa.types.is_decimal(kind)
        or pa.types.is_boolean(kind)
    )
    return fixed and not field.nullable


class ContentHasher:
    """SHA-256 incremental del contenido lógico: esquema más filas en orden.

    Un hash por columna; el digest final combina el esquema con los digests de
    columna. No depende del chunking ni de los bytes del Parquet: dos
    escrituras con las mismas filas dan el mismo hash aunque el archivo
    difiera, y cada lote se ve una sola vez.

    Las columnas de ancho fijo y no nulables (enteros, decimales y booleanos)
    se hashean sobre los bytes de sus valores. Las demás (texto o nulables)
    valor a valor, porque el buffer de un valor nulo no está definido. Se
    decide por el esquema y no por cada lote, para que el hash no dependa de
    cómo se parta la serie.
    """

    def __init__(self, schema: pa.Schema) -> None:
        # La metadata del esquema (clave-valor) no es contenido lógico.
        self._schema = schema.remove_metadata()
        self._columns = [hashlib.sha256() for _ in self._schema]
        self._by_value = [not _is_raw(field) for field in self._schema]

    def update(self, batch: pa.RecordBatch) -> None:
        for digest, by_value, column in zip(
            self._columns, self._by_value, batch.columns, strict=True
        ):
            if by_value:
                for value in column.to_pylist():
                    digest.update(repr(value).encode() + b"\0")
            else:
                digest.update(_raw_bytes(column))

    def hexdigest(self) -> str:
        digest = hashlib.sha256(self._schema.serialize().to_pybytes())
        for column in self._columns:
            digest.update(column.digest())
        return digest.hexdigest()


def content_hash(table: pa.Table) -> str:
    """SHA-256 del contenido lógico de `table` (ver `ContentHasher`)."""
    hasher = ContentHasher(table.schema)
    for batch in table.to_batches():
        hasher.update(batch)
    return hasher.hexdigest()
