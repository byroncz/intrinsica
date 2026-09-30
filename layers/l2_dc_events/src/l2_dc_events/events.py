"""Los eventos de un θ como columnas de Arrow y su escritura por tramos.

El fan-out entrega, por θ y tramo, un `dc_pyo3.EventColumns`: buffers ya en el
layout de Arrow. Aquí se envuelven sin copia y se escriben a `events.parquet`
a medida que llegan, sin acumular el mes: por θ solo hay un tramo de a lo más
`FLUSH_ROWS` filas (más lo que traiga el último tramo) en RAM (RNF-L2-01).
"""

import time
from typing import Self

import dc_pyo3
import pyarrow as pa
from pyutils import ContentHasher, PartitionWriter

from l2_dc_events.schema import EVENTS_SCHEMA, EVENTS_SORT_ORDER, THETA_TYPE

# Filas por row group de `events.parquet`. Un tramo son 11 columnas: 4 DECIMAL
# de 16 B, 6 INT64 y 1 INT8, unos 113 B por fila (~3,5 MiB con 32 768 filas);
# con 50 θ, el peor caso es ~175 MiB solo si los 50 llegan al límite a la vez,
# y los θ grandes casi nunca emiten.
FLUSH_ROWS = 32_768


def to_batch(columns: dc_pyo3.EventColumns, theta: int) -> pa.RecordBatch:
    """Los eventos de `dc_pyo3` como `RecordBatch`, sin copiar: los arrays de
    Arrow apuntan a los `bytes` que entregó el binding. Solo `theta`, constante,
    se materializa (16 B por fila).
    """
    n = len(columns)
    # Las 10 columnas del binding son las primeras del esquema; `theta` es la última.
    arrays = [
        pa.Array.from_buffers(field.type, n, [None, pa.py_buffer(buffer)])
        for field, buffer in zip(EVENTS_SCHEMA, columns.buffers(), strict=False)
    ]
    # DECIMAL(9, 8) positivo: la palabra alta es cero.
    arrays.append(
        pa.Array.from_buffers(
            THETA_TYPE, n, [None, pa.py_buffer(theta.to_bytes(16, "little") * n)]
        )
    )
    return pa.RecordBatch.from_arrays(arrays, schema=EVENTS_SCHEMA)


class EventWriter:
    """`events.parquet` de un θ y mes, alimentado por tramos.

    Escribe a un temporal (`PartitionWriter`), calcula el `content_hash` a
    medida que escribe y no publica nada hasta `commit`. Un escritor lo usa un
    hilo a la vez; distintos escritores pueden correr en hilos distintos
    (pyarrow suelta el GIL al codificar y `hashlib` al hashear).
    """

    def __init__(self, path: str, theta: int) -> None:
        self._theta = theta
        self._pending: list[pa.RecordBatch] = []
        self._rows = 0
        self._writer = PartitionWriter(path, EVENTS_SCHEMA, EVENTS_SORT_ORDER)
        self._hasher = ContentHasher(EVENTS_SCHEMA)
        self.n_events = 0
        # Segundos en `add` y `commit` (codificar, hashear, subir): los mide el
        # hilo que esté escribiendo, uno a la vez por escritor, así que no
        # necesita candado. `pipeline` suma los de los 50 θ.
        self.write_s = 0.0

    def __enter__(self) -> Self:
        self._writer.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._writer.__exit__(*exc_info)

    def add(self, columns: dc_pyo3.EventColumns) -> None:
        if not len(columns):
            return
        started = time.perf_counter()
        self._pending.append(to_batch(columns, self._theta))
        self._rows += len(columns)
        if self._rows >= FLUSH_ROWS:
            self._flush()
        self.write_s += time.perf_counter() - started

    def _flush(self) -> None:
        """Escribe los tramos acumulados como un row group y los suelta."""
        batches, self._pending, self._rows = self._pending, [], 0
        for batch in batches:
            self._hasher.update(batch)
        table = pa.Table.from_batches(batches, schema=EVENTS_SCHEMA)
        self.n_events += table.num_rows
        self._writer.write_table(table)

    def commit(self) -> str:
        """Escribe lo que quede, publica el archivo y devuelve su `content_hash`."""
        started = time.perf_counter()
        if self._rows:
            self._flush()
        self._writer.commit()
        self.write_s += time.perf_counter() - started
        return self._hasher.hexdigest()
