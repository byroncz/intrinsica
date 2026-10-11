"""Esquema de `summaries.parquet` (TRD-L3 §7.2) y su codificación en Arrow.

Las filas se arman en una matriz de enteros: una fila por evento y una columna por
dato, con los decimales en su escala (10⁻⁸). Al llenar un row group se pasan a Arrow
sin objetos de Python por valor: el valor de 64 bits es la palabra baja del decimal de
128 bits.
"""

import numpy as np
import pyarrow as pa

STATE_VERSION = "1.0.0"
FAMILY = "summaries"

DECIMAL = pa.decimal128(18, 8)
THETA = pa.decimal128(9, 8)
SCALE = 8
LIMIT = 10**18  # los valores de DECIMAL(18,8) son menores que 10^18 sin escalar

# (nombre, tipo, nulable) de las 18 columnas que van en la matriz, tras `theta`.
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
# Los `os_*` nulables son nulos juntos: cuando el overshoot es vacío.
OVERSHOOT_NULLABLE = tuple(COLUMN[name] for name, _, null in _COLUMNS if null)
OVERSHOOT_TICKS = COLUMN["os_ticks"]


def _wide(values: np.ndarray) -> np.ndarray:
    """Los enteros de 64 bits como decimales de 128: palabra baja y extensión de signo."""
    out = np.empty((len(values), 2), "<i8")
    out[:, 0] = values
    out[:, 1] = values >> 63
    return out


def _array(kind: pa.DataType, values: np.ndarray, valid: np.ndarray | None) -> pa.Array:
    count = len(values)
    validity, nulls = None, 0
    if valid is not None and not valid.all():
        validity = pa.py_buffer(np.packbits(valid, bitorder="little"))
        nulls = count - int(valid.sum())
    data = _wide(values) if pa.types.is_decimal(kind) else values.astype("<i8")
    return pa.Array.from_buffers(kind, count, [validity, pa.py_buffer(data)], nulls)


def to_batch(theta: int, rows: np.ndarray) -> pa.RecordBatch:
    """Un row group de `summaries.parquet` a partir de la matriz de filas.

    `theta` es el θ sin escalar (× 10⁸). Los valores de las columnas nulables se ignoran
    donde `os_ticks` es 0.
    """
    count = len(rows)
    valid = rows[:, OVERSHOOT_TICKS] > 0
    arrays = [_array(THETA, np.full(count, theta, np.int64), None)]
    for i, (_, kind, nullable) in enumerate(_COLUMNS):
        column = np.ascontiguousarray(rows[:, i])
        arrays.append(_array(kind, column, valid if nullable else None))
    return pa.RecordBatch.from_arrays(arrays, schema=SUMMARIES_SCHEMA)
