"""Hash del contenido lógico de una tabla, independiente del chunking y del archivo."""

import hashlib
from decimal import Decimal

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


def _is_fixed(field: pa.Field) -> bool:
    kind = field.type
    return (
        pa.types.is_integer(kind)
        or pa.types.is_decimal(kind)
        or pa.types.is_boolean(kind)
    )


def _is_raw(field: pa.Field) -> bool:
    """Si los bytes de valores de la columna bastan para hashearla."""
    return _is_fixed(field) and not field.nullable


_LARGE = pa.large_string()
_END = 2**31 - 1
_NULL_CELL = pa.scalar("None\0", _LARGE)
# repr(Decimal) pasa a notación científica con exponente ajustado menor que −6 y
# el cero con escala lleva exponente ("0E-8"). El texto de Arrow coincide hoy, pero
# esa coincidencia no es un contrato de Arrow: las celdas con |valor| < 10⁻⁶ se
# rehacen con `repr` de Python. Son pocas, y las pruebas lo verifican.
_TINY = pa.scalar(Decimal("0.000001"))


def _wrap(text: pa.LargeStringArray, prefix: str, suffix: str) -> pa.LargeStringArray:
    """`prefix + texto + suffix` por celda; un nulo sigue nulo.

    `binary_replace_slice` cuesta ≈ 4 veces menos que `binary_join_element_wise`.
    Una posición mayor que el largo de la cadena se recorta a su final: insertar en
    `_END` es agregar al final.
    """
    if prefix:
        text = pc.binary_replace_slice(text, 0, 0, prefix)
    return pc.binary_replace_slice(text, _END, _END, suffix)


def _cells(column: pa.Array) -> pa.LargeStringArray:
    """`repr(valor) + "\\0"` de cada celda de una columna nulable.

    Es lo que el camino valor a valor hashea celda a celda, construido por
    columna completa con kernels de Arrow y sin objetos de Python por celda.
    """
    kind = column.type
    if pa.types.is_boolean(kind):
        cells = pc.if_else(
            column, pa.scalar("True\0", _LARGE), pa.scalar("False\0", _LARGE)
        )
    elif pa.types.is_decimal(kind):
        cells = _wrap(pc.cast(column, _LARGE), "Decimal('", "')\0")
        tiny = pc.less(pc.abs(column), _TINY)
        if pc.any(tiny).as_py():  # `any` ignora los nulos
            tiny = pc.fill_null(tiny, False)
            fixed = [repr(value) + "\0" for value in column.filter(tiny).to_pylist()]
            cells = pc.replace_with_mask(cells, tiny, pa.array(fixed, _LARGE))
    else:
        cells = _wrap(pc.cast(column, _LARGE), "", "\0")
    return pc.fill_null(cells, _NULL_CELL)


def _cells_bytes(cells: pa.LargeStringArray) -> memoryview:
    """Los bytes concatenados de `cells`, sin nulos, respetando su offset."""
    offsets = memoryview(cells.buffers()[1]).cast("q")
    return memoryview(cells.buffers()[2])[
        offsets[cells.offset] : offsets[cells.offset + len(cells)]
    ]


class ContentHasher:
    """SHA-256 incremental del contenido lógico: esquema más filas en orden.

    Un hash por columna; el digest final combina el esquema con los digests de
    columna. No depende del chunking ni de los bytes del Parquet: dos
    escrituras con las mismas filas dan el mismo hash aunque el archivo
    difiera, y cada lote se ve una sola vez.

    Hay tres caminos, elegidos por el esquema y no por cada lote, para que el
    hash no dependa de cómo se parta la serie:

    - Enteros, decimales y booleanos no nulables: los bytes de sus valores.
    - Enteros, decimales y booleanos nulables: la representación `repr(valor)`
      de cada celda más `\\0` (`None` para un nulo), porque el buffer de un
      valor nulo no está definido. Se construye por columna completa con Arrow
      (`_cells`) y se entrega a SHA-256 de una sola vez; el digest es el mismo
      que el de hashear celda a celda con `repr` de Python.
    - Texto y cualquier otro tipo: valor a valor, con `repr` de Python.
    """

    def __init__(self, schema: pa.Schema) -> None:
        # La metadata del esquema (clave-valor) no es contenido lógico.
        self._schema = schema.remove_metadata()
        self._columns = [hashlib.sha256() for _ in self._schema]
        self._raw = [_is_raw(field) for field in self._schema]
        self._fixed = [_is_fixed(field) for field in self._schema]

    def update(self, batch: pa.RecordBatch) -> None:
        for digest, raw, fixed, column in zip(
            self._columns, self._raw, self._fixed, batch.columns, strict=True
        ):
            if raw:
                digest.update(_raw_bytes(column))
            elif fixed:
                if len(column):
                    # La cadena de la columna vive solo durante este update.
                    digest.update(_cells_bytes(_cells(column)))
            else:
                for value in column.to_pylist():
                    digest.update(repr(value).encode() + b"\0")

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
