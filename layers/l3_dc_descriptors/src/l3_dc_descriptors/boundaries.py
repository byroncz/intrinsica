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

Los lotes traen las 11 columnas del archivo y se decodifican una sola vez. El hilo que los
lee los pasa por `L2Events.tap` (ADR-L3-16, ITSC-365): hashea las 11 (el `l2_events_content_hash`
del linaje), anota la huella de las claves de cada row group (el esqueleto con que el
escritor comprueba lo que el motor selló, sin releer el archivo) y entrega al motor solo las
siete que usa, así que las otras cuatro no quedan retenidas.
"""

import hashlib
import threading
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Executor
from decimal import Decimal
from types import MappingProxyType
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from dc_frames import FramesInputError
from dc_frames.lake import DECIMAL, EventFile, check_columns
from pyutils import ContentHasher

from l3_dc_descriptors.event import EVENT_COLUMNS
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


class L2Events:
    """Lo que L3 saca de `events.parquet` de un θ mientras lo lee, sin volver a él.

    Es el `tap` de su `BatchStream` y, a la vez, el esqueleto (`dc_frames.family.Skeleton`)
    de su `FamilyWriter`:

    - **Linaje.** `hexdigest()` es el `content_hash` lógico del archivo, el mismo que L2
      publica en `events_summary` y que `file_hash` calculaba releyéndolo: un
      `ContentHasher` sobre el esquema del archivo, alimentado con los lotes en orden y con
      todas las columnas.
    - **Esqueleto.** Por cada row group no vacío guarda la huella (SHA-256) de sus
      `confirm_agg_trade_id` y si su `theta` es el de la partición en todas las filas. El
      escritor compara la huella de lo que el motor selló, que copió esos ids de L2, con la
      del row group que L3 leyó, y su `theta` con el de la partición: mismas filas, mismo
      orden y mismas claves, en 33 B por row group y sin un viaje a GCS. Las filas por row
      group salen del pie del archivo.

    Un lote no cruza row groups (`EventFile.batches`); aun así `tap` reparte por filas, así
    que un lote que los cruzara tampoco movería ninguna huella. Un solo hilo a la vez llama
    a `tap` (el de `BatchStream`); el escritor consulta desde otros, de ahí el candado.
    """

    def __init__(self, file: EventFile, theta: Decimal) -> None:
        if "theta" not in file.schema.names:
            raise FramesInputError(f"{file.path}: falta la columna 'theta'")
        self.path = file.path
        self.row_groups = [rows for rows in file.rows if rows]
        self._order = file.schema.names
        self._hasher = ContentHasher(file.schema)
        self._theta = theta
        self._lock = threading.Lock()
        self._done: dict[int, tuple[bytes, bool]] = {}
        self._group = 0  # el row group que se está llenando
        self._filled = 0
        self._ids = hashlib.sha256()
        self._same_theta = True

    def tap(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Observa un lote con las 11 columnas y devuelve las siete que el motor usa."""
        self._hasher.update(batch.select(self._order))
        ids = _ids(batch.column("confirm_agg_trade_id"))
        theta = batch.column("theta")
        start = 0
        while start < batch.num_rows and self._group < len(self.row_groups):
            take = min(
                batch.num_rows - start, self.row_groups[self._group] - self._filled
            )
            self._ids.update(ids[start : start + take])
            other = _first_other(theta.slice(start, take), self._theta)
            self._same_theta &= other is None
            self._filled += take
            start += take
            if self._filled == self.row_groups[self._group]:
                with self._lock:
                    self._done[self._group] = (self._ids.digest(), self._same_theta)
                self._group += 1
                self._filled = 0
                self._ids = hashlib.sha256()
                self._same_theta = True
        return batch.select(EVENT_COLUMNS)

    def hexdigest(self) -> str:
        """El `content_hash` de lo que `tap` vio; completo cuando el archivo se leyó entero."""
        return self._hasher.hexdigest()

    def differing_key(
        self, n: int, first_row: int, batch: pa.RecordBatch
    ) -> str | None:
        with self._lock:
            seen = self._done.pop(n, None)
        if seen is None:
            return f"row group {n}: L3 aún no leyó ese row group del esqueleto"
        digest, same_theta = seen
        if not same_theta:
            return f"row group {n}, fila {first_row}: theta no es el del esqueleto"
        # Lo sellado se compara por sus dos claves: el `theta` que escribió el motor contra el
        # de la partición y sus ids contra la huella de los que L3 leyó de L2.
        other = _first_other(batch.column("theta"), self._theta)
        if other is not None:
            return (
                f"row group {n}, fila {first_row + other}: theta no es el del esqueleto"
            )
        sealed = batch.column("confirm_agg_trade_id")
        if sealed.null_count:
            row = pc.index(sealed.is_null(), True).as_py()
            return (
                f"row group {n}, fila {first_row + row}: "
                "confirm_agg_trade_id no es el del esqueleto"
            )
        if hashlib.sha256(_ids(sealed)).digest() != digest:
            return (
                f"row group {n}, desde la fila {first_row}: confirm_agg_trade_id "
                "no es el del esqueleto"
            )
        return None

    def close(self) -> None:
        """No tiene nada abierto: el archivo es de quien lo abrió."""


def _ids(column: pa.Array) -> np.ndarray:
    """Los ids como `int64` contiguo, sin copia cuando se puede; un nulo es un error."""
    if column.null_count:
        raise FramesInputError("confirm_agg_trade_id trae nulos")
    return np.ascontiguousarray(column.to_numpy(zero_copy_only=False))


def _first_other(column: pa.Array, theta: Decimal) -> int | None:
    """La primera fila que no es `theta` (un nulo cuenta como distinto), o `None`."""
    if not len(column):
        return None
    same = pc.equal(column, pa.scalar(theta, column.type)).fill_null(False)
    if pc.all(same).as_py():
        return None
    return pc.index(same, False).as_py()


class BatchStream:
    """Los lotes de eventos de un `EventFile`, en orden y por delante de quien los pide.

    Es un iterador. Con `pool`, una tarea a la vez (nunca dos sobre el mismo archivo) trae
    el siguiente lote mientras el consumidor trabaja con el anterior, hasta `ahead` listos;
    sin `pool`, cada `next` lee en el hilo que lo pide. Mide en `meter` los bytes de cada
    tramo y la pared de cada lectura. Un error de lectura sale en el `next` que lo habría
    traído. `close()` espera a la tarea en curso y suelta lo que traía; hay que llamarlo
    antes de cerrar el archivo.

    Con `tap`, cada lote pasa por él en el hilo que lo leyó (dentro de la medición de la
    lectura) y lo que devuelve es lo que se entrega: así quien lee decodifica una sola vez y
    quien consume recibe solo lo que usa (`L2Events.tap`).
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
        tap: Callable[[pa.RecordBatch], pa.RecordBatch] | None = None,
    ) -> None:
        # Los valores por defecto se leen aquí, no al definir la clase: las pruebas los
        # reducen para recorrer muchos tramos y lotes con un lago chico.
        self._file = file
        self._meter = meter
        self._tap = tap
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
            batch = next(self._source, None)
            if batch is not None and self._tap is not None:
                batch = self._tap(batch)
            return batch

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
