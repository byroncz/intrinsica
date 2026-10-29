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


def _is_float(field: pa.Field) -> bool:
    return pa.types.is_float64(field.type)


def _is_raw(field: pa.Field) -> bool:
    """Si los bytes de valores de la columna bastan para hashearla."""
    return (_is_fixed(field) or _is_float(field)) and not field.nullable


# Versión del algoritmo de `ContentHasher`. Se publica con cada huella (`content_hash.json`) y
# sube solo si el mismo contenido pasara a dar otro hash: entonces las huellas publicadas con
# la versión anterior ya no son comparables.
HASHER_VERSION = "1"

_LARGE = pa.large_string()
_END = 2**31 - 1
_NULL_CELL = pa.scalar("None\0", _LARGE)
_NULL_FLOAT_CELL = pa.scalar(bytes(9), pa.large_binary())
# repr(Decimal) pasa a notación científica con exponente ajustado menor que −6 y
# el cero con escala lleva exponente ("0E-8"). El texto de Arrow coincide hoy, pero
# esa coincidencia no es un contrato de Arrow: las celdas con |valor| < 10⁻⁶ se
# rehacen con `repr` de Python. Son pocas, y las pruebas lo verifican.
_TINY = Decimal("0.000001")


def _wrap(text: pa.LargeStringArray, prefix: str, suffix: str) -> pa.LargeStringArray:
    """`prefix + texto + suffix` por celda; un nulo sigue nulo.

    `binary_replace_slice` cuesta ≈ 4 veces menos que `binary_join_element_wise`.
    Una posición mayor que el largo de la cadena se recorta a su final: insertar en
    `_END` es agregar al final.
    """
    if prefix:
        text = pc.binary_replace_slice(text, 0, 0, prefix)
    return pc.binary_replace_slice(text, _END, _END, suffix)


def _tiny(column: pa.Array) -> pa.BooleanArray | None:
    """Celdas de un decimal cuyo texto Arrow podría diferir de `repr`; `None` si ninguna.

    Con escala entre 0 y 6 el exponente de `repr` nunca baja de −6: no hay celdas.
    Con escala mayor, el umbral se construye en el tipo de la columna: compararlo
    contra un decimal de otra escala lleva ambos lados a una precisión que puede
    pasar de 38. Si la precisión no pasa de `escala − 6`, 10⁻⁶ no cabe en el tipo,
    pero todo valor es menor que 10⁻⁶: van todas. Con escala negativa `repr`
    siempre lleva exponente: también van todas.
    """
    scale = column.type.scale
    if 0 <= scale <= 6:
        return None
    if scale < 0 or column.type.precision <= scale - 6:
        return pc.is_valid(column)
    return pc.less(pc.abs(column), pa.scalar(_TINY, column.type))


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
        tiny = _tiny(column)
        if tiny is not None and pc.any(tiny).as_py():  # `any` ignora los nulos
            tiny = pc.fill_null(tiny, False)
            fixed = [repr(value) + "\0" for value in column.filter(tiny).to_pylist()]
            cells = pc.replace_with_mask(cells, tiny, pa.array(fixed, _LARGE))
    else:
        cells = _wrap(pc.cast(column, _LARGE), "", "\0")
    return pc.fill_null(cells, _NULL_CELL)


def _float_cells(column: pa.Array) -> pa.LargeBinaryArray:
    """Celdas de 9 bytes de una columna float64 nulable: validez y valor.

    Cada celda es `\\x01` más los 8 bytes little-endian del valor, o nueve ceros
    si es nulo. El ancho fijo distingue un nulo de cualquier valor (incluido 0.0,
    cuyos 8 bytes son también ceros) y no depende de cómo se parta la columna.
    Se construye por columna completa con kernels de Arrow, como `_cells`: la vista
    `binary(8)` reinterpreta el buffer sin copiarlo y el prefijo se inserta con
    `binary_replace_slice`.
    """
    raw = pc.cast(column.view(pa.binary(8)), pa.large_binary())
    return pc.fill_null(pc.binary_replace_slice(raw, 0, 0, b"\x01"), _NULL_FLOAT_CELL)


def _cells_bytes(cells: pa.LargeStringArray | pa.LargeBinaryArray) -> memoryview:
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

    Hay cuatro caminos, elegidos por el esquema y no por cada lote, para que el
    hash no dependa de cómo se parta la serie:

    - Enteros, decimales, booleanos y float64 no nulables: los bytes de sus
      valores (en float64, 8 bytes little-endian por valor, sin copia).
    - Enteros, decimales y booleanos nulables: la representación `repr(valor)`
      de cada celda más `\\0` (`None` para un nulo), porque el buffer de un
      valor nulo no está definido. Se construye por columna completa con Arrow
      (`_cells`) y se entrega a SHA-256 de una sola vez; el digest es el mismo
      que el de hashear celda a celda con `repr` de Python.
    - float64 nulables: una celda de 9 bytes por fila, `\\x01` más los 8 bytes
      del valor, o nueve ceros si es nulo (`_float_cells`). Se construye por
      columna completa con Arrow y no pasa por el texto, que no es un contrato de
      Arrow para los flotantes.
    - Texto y cualquier otro tipo (float32 incluido): valor a valor, con `repr`
      de Python.

    float64 se hashea por sus bits, no por su valor: NaN e infinitos no fallan y
    dan los bytes crudos que traiga el buffer, sin canonizarlos (dos NaN con
    payload distinto dan digests distintos, y 0.0 y -0.0 también). Es
    determinista: el mismo buffer da el mismo digest. L3 nunca produce NaN ni inf.

    `HASHER_VERSION` no subió al agregar float64: ninguna huella publicada tiene
    columnas float64 y el digest de los demás tipos no cambia.
    """

    def __init__(self, schema: pa.Schema) -> None:
        # La metadata del esquema (clave-valor) no es contenido lógico.
        self._schema = schema.remove_metadata()
        self._columns = [hashlib.sha256() for _ in self._schema]
        self._raw = [_is_raw(field) for field in self._schema]
        self._fixed = [_is_fixed(field) for field in self._schema]
        self._float = [_is_float(field) for field in self._schema]

    def update(self, batch: pa.RecordBatch) -> None:
        for digest, raw, fixed, floating, column in zip(
            self._columns,
            self._raw,
            self._fixed,
            self._float,
            batch.columns,
            strict=True,
        ):
            if raw:
                digest.update(_raw_bytes(column))
            elif fixed:
                if len(column):
                    # La cadena de la columna vive solo durante este update.
                    digest.update(_cells_bytes(_cells(column)))
            elif floating:
                if len(column):
                    digest.update(_cells_bytes(_float_cells(column)))
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
