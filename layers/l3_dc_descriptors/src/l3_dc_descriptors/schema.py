"""Esquema de `summaries.parquet` (TRD-L3 §7.2) y su codificación en Arrow.

Las filas se arman en un buffer de NumPy por columna, con los decimales en su escala
(10⁻⁸): el valor de 64 bits es la palabra baja del decimal de 128 bits. Al llenar un row
group, Arrow envuelve esos buffers sin copiarlos y sin objetos de Python por valor.
"""

import numpy as np
import pyarrow as pa

STATE_VERSION = "1.0.0"
FAMILY = "summaries"

DECIMAL = pa.decimal128(18, 8)
THETA = pa.decimal128(9, 8)
SCALE = 8
LIMIT = 10**18  # los valores de DECIMAL(18,8) son menores que 10^18 sin escalar

# (nombre, tipo, nulable) de las 18 columnas que van en el buffer de cada columna, tras `theta`.
_COLUMNS = (
    ("confirm_agg_trade_id", pa.int64(), False),
    ("c_ticks", pa.int64(), False),
    ("c_quantity_buy", DECIMAL, False),
    ("c_quantity_sell", DECIMAL, False),
    ("c_price_max", DECIMAL, False),
    ("c_price_min", DECIMAL, False),
    ("c_duration", pa.int64(), False),
    ("c_vwap", DECIMAL, False),
    ("os_ticks", pa.int64(), False),
    ("os_quantity_buy", DECIMAL, False),
    ("os_quantity_sell", DECIMAL, False),
    ("os_price_max", DECIMAL, True),
    ("os_price_min", DECIMAL, True),
    ("os_duration", pa.int64(), True),
    ("os_vwap", DECIMAL, True),
    ("confirm_instant_ticks", pa.int64(), False),
    ("confirm_instant_price_best", DECIMAL, False),
    ("confirm_instant_price_worst", DECIMAL, False),
)
SUMMARIES_SCHEMA = pa.schema(
    [
        pa.field("theta", THETA, nullable=False),
        *(pa.field(name, kind, nullable=null) for name, kind, null in _COLUMNS),
    ]
)
WIDTH = len(_COLUMNS)
COLUMN = {name: i for i, (name, _, _) in enumerate(_COLUMNS)}
# Los `os_*` nulables son nulos juntos: cuando el overshoot es vacío (`os_ticks` = 0).
OVERSHOOT_TICKS = COLUMN["os_ticks"]
_WIDE = tuple(pa.types.is_decimal(kind) for _, kind, _ in _COLUMNS)


class RowGroupBuffer:
    """El row group de salida de un θ, en un buffer de NumPy por columna.

    Cada evento llena su fila directamente en el buffer de cada columna; los decimales
    ocupan 16 bytes (palabra baja y extensión de signo). `to_batch` envuelve esos
    buffers en Arrow sin copiarlos: no hay otra representación del mismo dato.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.fill = 0
        self._data = [np.empty((size, 2) if wide else size, "<i8") for wide in _WIDE]

    def append(self, values: list[int]) -> None:
        """La fila del siguiente evento: `WIDTH` enteros, los decimales sin escalar."""
        i = self.fill
        for column, wide, value in zip(self._data, _WIDE, values, strict=True):
            if wide:
                column[i, 0] = value
                column[i, 1] = value >> 63
            else:
                column[i] = value
        self.fill += 1

    def extend(self, block: np.ndarray) -> None:
        """Las filas de varios eventos: `block` es `(k, WIDTH)` de `int64`, con los
        decimales sin escalar. Es `append` por columna, sin bucle por evento.
        """
        i, k = self.fill, len(block)
        for j, (column, wide) in enumerate(zip(self._data, _WIDE, strict=True)):
            if wide:
                column[i : i + k, 0] = block[:, j]
                column[i : i + k, 1] = block[:, j] >> 63
            else:
                column[i : i + k] = block[:, j]
        self.fill += k

    def to_batch(self, theta: int) -> pa.RecordBatch:
        """El row group lleno. `theta` es el θ sin escalar (× 10⁸).

        Los valores de las columnas nulables se ignoran donde `os_ticks` es 0.
        """
        valid = self._data[OVERSHOOT_TICKS] > 0
        validity, nulls = None, 0
        if not valid.all():
            validity = pa.py_buffer(np.packbits(valid, bitorder="little"))
            nulls = self.size - int(valid.sum())
        constant = np.zeros((self.size, 2), "<i8")
        constant[:, 0] = theta
        arrays = [
            pa.Array.from_buffers(THETA, self.size, [None, pa.py_buffer(constant)])
        ]
        for column, (_, kind, nullable) in zip(self._data, _COLUMNS, strict=True):
            buffers = [validity if nullable else None, pa.py_buffer(column)]
            arrays.append(
                pa.Array.from_buffers(
                    kind, self.size, buffers, nulls if nullable else 0
                )
            )
        return pa.RecordBatch.from_arrays(arrays, schema=SUMMARIES_SCHEMA)
