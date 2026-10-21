"""Las fronteras de L2 de los θ, leídas por delante del motor (ADR-L3-16).

Python mueve bytes: lee `events.parquet` de cada θ y se lo entrega al motor
(`native.BoundaryFeed`), que decide qué eventos toca cada row group. Aquí no se calcula
nada sobre las filas.

Cada θ abre su archivo una vez (`dc_frames.lake.EventFile`) y lo lee por tramos de varios
row groups (`L2_LOT_BYTES` comprimidos): pocas peticiones grandes por archivo, no una por
row group. Un tramo se decodifica en lotes de `L2_BATCH_ROWS` eventos, cada uno con sus
buffers, y el motor suelta un lote en cuanto sus eventos cierran. `BatchStream` trae esos
lotes en un pool de E/S, `L2_AHEAD` por delante de quien los consume, así que los 50 θ se
leen en paralelo y el hilo principal casi nunca espera. Lo retenido por θ es el caché
comprimido del tramo en curso más `L2_AHEAD + 1` lotes decodificados (≈ 2 MiB cada uno).
"""

import threading
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Executor
from types import MappingProxyType
from typing import Self

import pyarrow as pa
from dc_frames.lake import DECIMAL, EventFile, check_columns

from l3_dc_descriptors.timing import ReadMeter

# Bytes comprimidos por tramo: varios row groups de `events.parquet` (≈ 0,5 MB cada uno en
# 2021-01) por lectura. Lo que ITSC-362 midió: la latencia por viaje, no el ancho de banda,
# era el límite (1 335 lecturas de ≈ 126 ms para 610 MB).
L2_LOT_BYTES = 8 << 20
# Eventos por lote que se entrega al motor: ≈ 2 MiB decodificados (65 B por evento).
L2_BATCH_ROWS = 32_768
# Los tipos de las columnas de L2 que el motor lee como bytes de Arrow sin convertirlas
# (las cuatro de ids y tiempo, `int64`, las comprueba `EventFile`).
L2_TYPES = MappingProxyType(
    {
        "direction": pa.int8(),
        "confirm_price": DECIMAL,
        "extreme_price": DECIMAL,
    }
)
# Lotes que `BatchStream` trae por delante de quien los consume, por θ.
L2_AHEAD = 2


def check_types(file: EventFile) -> None:
    """`FramesInputError` si una columna de L2 no tiene el tipo que el motor espera.

    El motor recibe los buffers tal cual, sin convertirlos: un precio en otro ancho se
    leería mal en silencio.
    """
    check_columns(file.schema, dict(L2_TYPES), file.path)


class BatchStream:
    """Los lotes de eventos de un `EventFile`, en orden y por delante de quien los pide.

    Es un iterador. Con `pool`, una tarea a la vez (nunca dos sobre el mismo archivo) trae
    el siguiente lote mientras el consumidor trabaja con el anterior, hasta `ahead` listos;
    sin `pool`, cada `next` lee en el hilo que lo pide. Mide en `meter` los bytes de cada
    tramo y la pared de cada lectura. Un error de lectura sale en el `next` que lo habría
    traído. `close()` espera a la tarea en curso y suelta lo que traía; hay que llamarlo
    antes de cerrar el archivo.
    """

    def __init__(
        self,
        file: EventFile,
        pool: Executor | None,
        meter: ReadMeter,
        *,
        lot_bytes: int | None = None,
        batch_rows: int | None = None,
        ahead: int | None = None,
    ) -> None:
        # Los valores por defecto se leen aquí, no al definir la clase: las pruebas los
        # reducen para recorrer muchos tramos y lotes con un lago chico.
        self._file = file
        self._meter = meter
        self._lot_bytes = L2_LOT_BYTES if lot_bytes is None else lot_bytes
        self._batch_rows = L2_BATCH_ROWS if batch_rows is None else batch_rows
        self._source = self._read()
        self._pool = pool
        self._ahead = (L2_AHEAD if ahead is None else ahead) if pool is not None else 0
        self._cond = threading.Condition()
        self._ready: deque[pa.RecordBatch] = deque()
        self._busy = False  # una tarea está trayendo un lote
        self._done = False  # el archivo se agotó
        self._closed = False
        self._error: BaseException | None = None
        with self._cond:
            self._schedule()

    def _read(self) -> Iterator[pa.RecordBatch]:
        for lot in self._file.lots(self._lot_bytes):
            self._meter.add(self._file.compressed(lot))
            yield from self._file.batches(lot, self._batch_rows, self._lot_bytes)

    def _pull(self) -> pa.RecordBatch | None:
        with self._meter.reading():
            return next(self._source, None)

    def _schedule(self) -> None:
        """Lanza la tarea que trae lotes si hay lugar y no hay otra (con `_cond`)."""
        if (
            self._pool is None
            or self._busy
            or self._done
            or self._closed
            or self._error is not None
            or len(self._ready) >= self._ahead
        ):
            return
        self._busy = True
        self._pool.submit(self._fill)

    def _fill(self) -> None:
        while True:
            try:
                batch = self._pull()
            except BaseException as error:  # noqa: BLE001 - sale en `__next__`
                with self._cond:
                    self._error, self._busy = error, False
                    self._cond.notify_all()
                return
            with self._cond:
                if batch is None:
                    self._done = True
                else:
                    self._ready.append(batch)
                self._cond.notify_all()
                if self._done or self._closed or len(self._ready) >= self._ahead:
                    self._busy = False
                    return

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._pool is None:
            batch = self._pull()
            if batch is None:
                raise StopIteration
            return batch
        with self._cond:
            while not self._ready and not self._done and self._error is None:
                self._cond.wait()
            if self._ready:
                batch = self._ready.popleft()
                self._schedule()
                return batch
            if self._error is not None:
                raise self._error
            raise StopIteration

    def close(self) -> None:
        with self._cond:
            self._closed = True
            while self._busy:
                self._cond.wait()
            self._ready.clear()
        self._source.close()
