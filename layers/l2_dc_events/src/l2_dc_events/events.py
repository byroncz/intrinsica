"""Los eventos de un θ como columnas de Arrow y su escritura por tramos.

El fan-out devuelve objetos `Event`; aquí se vuelcan a columnas y se escriben a
`events.parquet` a medida que llegan, sin acumular el mes: por θ solo hay un
tramo de a lo más `FLUSH_ROWS` filas en RAM (RNF-L2-01).
"""

import sys
from array import array
from collections.abc import Sequence
from operator import attrgetter
from typing import Self

import dc_pyo3
import pyarrow as pa

from l2_dc_events.schema import EVENTS_SCHEMA, EVENTS_SORT_ORDER, PRICE_TYPE, THETA_TYPE
from l2_dc_events.write import ContentHasher, PartitionWriter

# Filas por row group de `events.parquet`. Un tramo son 11 columnas: 4 DECIMAL
# de 16 B, 6 INT64 y 1 INT8, unos 113 B por fila (~3,5 MiB con 32 768 filas);
# con 50 θ, el peor caso es ~175 MiB solo si los 50 llegan al límite a la vez,
# y los θ grandes casi nunca emiten.
FLUSH_ROWS = 32_768

_LITTLE_ENDIAN = sys.byteorder == "little"

_REFERENCE = attrgetter("reference")
_CONFIRM = attrgetter("confirm")
_EXTREME = attrgetter("extreme")
_DIRECTION = attrgetter("direction")


def _int64(values: array) -> pa.Array:
    return pa.Array.from_buffers(pa.int64(), len(values), [None, pa.py_buffer(values)])


def _decimal(values: array, kind: pa.DataType) -> pa.Array:
    """`DECIMAL` desde su entero sin escalar: 16 B little-endian por valor.

    Precios y θ son positivos (`0 < price < PRICE_LIMIT`), así que la palabra
    alta es cero y basta escribir la baja; sin pasar por `Decimal`.
    """
    if not _LITTLE_ENDIAN:
        raise RuntimeError("el volcado de DECIMAL asume una plataforma little-endian")
    wide = array("q", bytes(16 * len(values)))
    wide[0::2] = values
    return pa.Array.from_buffers(kind, len(values), [None, pa.py_buffer(wide)])


class EventColumns:
    """Acumula eventos en columnas planas y las entrega como un `RecordBatch`."""

    def __init__(self, theta: int) -> None:
        self.theta = theta
        self._reset()

    def _reset(self) -> None:
        # Precio, tiempo e id de referencia, confirmación y extremo.
        self._points = [array("q") for _ in range(9)]
        self._direction = array("b")

    def __len__(self) -> int:
        return len(self._direction)

    def extend(self, events: Sequence[dc_pyo3.Event]) -> None:
        """Vuelca los eventos por columnas: un `extend` por columna, no un
        `append` por valor (con 12 M de eventos por mes eso era el 75 % del
        tiempo de la unidad)."""
        if not events:
            return
        columns = self._points
        for start, point in ((0, _REFERENCE), (3, _CONFIRM), (6, _EXTREME)):
            price, time, agg_trade_id = zip(*map(point, events))
            columns[start].extend(price)
            columns[start + 1].extend(time)
            columns[start + 2].extend(agg_trade_id)
        self._direction.extend(map(_DIRECTION, events))

    def take(self) -> pa.RecordBatch:
        """Las filas acumuladas como lote; deja el acumulador vacío.

        El lote posee los buffers: no se reutilizan, se crean nuevos.
        """
        n = len(self)
        p = self._points
        arrays = [
            _decimal(p[0], PRICE_TYPE),
            _int64(p[1]),
            _int64(p[2]),
            _decimal(p[3], PRICE_TYPE),
            _int64(p[4]),
            _int64(p[5]),
            _decimal(p[6], PRICE_TYPE),
            _int64(p[7]),
            _int64(p[8]),
            pa.Array.from_buffers(pa.int8(), n, [None, pa.py_buffer(self._direction)]),
            _decimal(array("q", [self.theta]) * n, THETA_TYPE),
        ]
        self._reset()
        return pa.RecordBatch.from_arrays(arrays, schema=EVENTS_SCHEMA)


class EventWriter:
    """`events.parquet` de un θ y mes, alimentado por tramos.

    Escribe a un temporal (`PartitionWriter`), calcula el `content_hash` a
    medida que escribe y no publica nada hasta `commit`.
    """

    def __init__(self, path: str, theta: int) -> None:
        self._columns = EventColumns(theta)
        self._writer = PartitionWriter(path, EVENTS_SCHEMA, EVENTS_SORT_ORDER)
        self._hasher = ContentHasher(EVENTS_SCHEMA)
        self.n_events = 0

    def __enter__(self) -> Self:
        self._writer.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._writer.__exit__(*exc_info)

    def add(self, events: Sequence[dc_pyo3.Event]) -> None:
        self._columns.extend(events)
        if len(self._columns) >= FLUSH_ROWS:
            self._flush()

    def _flush(self) -> None:
        batch = self._columns.take()
        self.n_events += batch.num_rows
        self._hasher.update(batch)
        self._writer.write_batch(batch)

    def commit(self) -> str:
        """Escribe lo que quede, publica el archivo y devuelve su `content_hash`."""
        if len(self._columns):
            self._flush()
        self._writer.commit()
        return self._hasher.hexdigest()
