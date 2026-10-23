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

Los lotes traen las siete columnas que el motor usa más `theta` (`EVENT_FILE_COLUMNS`) y se
decodifican una sola vez. El hilo que los lee los pasa por `L2Events.tap` (ADR-L3-16,
ITSC-365): anota la huella de las claves de cada row group (el esqueleto con que el escritor
comprueba lo que el motor selló, sin releer el archivo) y entrega al motor solo las siete que
usa. El linaje ya no se calcula aquí: L2 publica el `content_hash` junto al archivo y la
compuerta lo lee (ITSC-368); las otras cuatro columnas de L2 (tiempos y precios de la
referencia) no se decodifican.
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

from l3_dc_descriptors.event import EVENT_COLUMNS
from l3_dc_descriptors.timing import ReadMeter

# Bytes comprimidos por tramo: varios row groups de `events.parquet` (≈ 0,5 MB cada uno en
# 2021-01) por lectura. Lo que ITSC-362 midió: la latencia por viaje, no el ancho de banda,
# era el límite (1 335 lecturas de ≈ 126 ms para 610 MB).
L2_LOT_BYTES = 8 << 20
# Eventos por lote que se entrega al motor: ≈ 2 MiB decodificados (65 B por evento).
L2_BATCH_ROWS = 32_768
# Las columnas que se decodifican de `events.parquet`: las siete que usa el motor y `theta`,
# que el esqueleto compara contra la partición. `EventFile` agrega por su cuenta las que
# siempre lee (las tres fronteras y `confirm_time`, que ya están en `EVENT_COLUMNS`).
EVENT_FILE_COLUMNS = (*EVENT_COLUMNS, "theta")
# La escala de `theta` en L2 (`decimal128(9, 8)`): la de la partición.
THETA_SCALE = 8
# Los tipos de las columnas de L2 que el motor lee como bytes de Arrow sin convertirlas
# (las cuatro de ids y tiempo, `int64`, las comprueba `EventFile`).
L2_TYPES = MappingProxyType(
    {
        "direction": pa.int8(),
        "confirm_price": DECIMAL,
        "extreme_price": DECIMAL,
    }
)
# Bytes de las siete columnas de fronteras de un evento, que el motor retiene hasta cerrarlo: cuatro
# `int64`, un `int8` y dos `decimal128`.
BOUNDARY_ROW_BYTES = 4 * 8 + 1 + 2 * 16
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
    de su `FamilyWriter`. Por cada row group no vacío guarda la huella (SHA-256) de sus
    `confirm_agg_trade_id` y si su `theta` es el de la partición en todas las filas. El
    escritor compara la huella de lo que el motor selló, que copió esos ids de L2, con la
    del row group que L3 leyó, y su `theta` con el de la partición: mismas filas, mismo
    orden y mismas claves, en 33 B por row group y sin un viaje a GCS. Las filas por row
    group salen del pie del archivo.

    El `theta` se compara como bytes: los 16 del `decimal128` de la partición contra los de
    cada fila (`_first_other`), sin pasar por los kernels de decimales de Arrow.

    Un lote no cruza row groups (`EventFile.batches`); aun así `tap` reparte por filas, así
    que un lote que los cruzara tampoco movería ninguna huella. Un solo hilo a la vez llama
    a `tap` (el de `BatchStream`); el escritor consulta desde otros, de ahí el candado.
    """

    def __init__(self, file: EventFile, theta: Decimal) -> None:
        if "theta" not in file.schema.names:
            raise FramesInputError(f"{file.path}: falta la columna 'theta'")
        kind = file.schema.field("theta").type
        if not (pa.types.is_decimal128(kind) and kind.scale == THETA_SCALE):
            raise FramesInputError(
                f"{file.path}: theta es {kind} y debería ser decimal128 con escala 8"
            )
        self.path = file.path
        self.row_groups = [rows for rows in file.rows if rows]
        self._theta = theta_bytes(theta)
        self._lock = threading.Lock()
        self._done: dict[int, tuple[bytes, bool]] = {}
        self._group = 0  # el row group que se está llenando
        self._filled = 0
        self._ids = hashlib.sha256()
        self._same_theta = True

    def tap(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Observa un lote con las columnas de `EVENT_FILE_COLUMNS` y devuelve las siete que
        el motor usa.
        """
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


def theta_bytes(theta: Decimal) -> bytes:
    """Los 16 bytes con que un `decimal128(·, 8)` guarda a `theta`: el entero `θ × 10⁸`,
    complemento a dos, little-endian.
    """
    return int(theta.scaleb(THETA_SCALE)).to_bytes(16, "little", signed=True)


def _first_other(column: pa.Array, theta: bytes) -> int | None:
    """La primera fila que no es `theta` (un nulo cuenta como distinto), o `None`.

    Compara los bytes del buffer de valores contra los 16 de `theta`, dos palabras de 8 bytes
    por fila con NumPy sobre el buffer, sin copiarlo; un `theta` de otro ancho o escala no
    puede ser el de la partición y es distinto desde la fila 0.
    """
    if not len(column):
        return None
    if not (pa.types.is_decimal128(column.type) and column.type.scale == THETA_SCALE):
        return 0
    words = np.frombuffer(
        column.buffers()[1],
        dtype="<u8",
        count=2 * len(column),
        offset=16 * column.offset,
    ).reshape(-1, 2)
    low, high = np.frombuffer(theta, dtype="<u8")
    same = (words[:, 0] == low) & (words[:, 1] == high)
    if column.null_count:  # el buffer de un nulo no está definido
        same &= column.is_valid().to_numpy(zero_copy_only=False)
    if same.all():
        return None
    return int(np.argmin(same))


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
