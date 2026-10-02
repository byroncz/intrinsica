"""Lector de la landing de L1: `consolidated.parquet`, un row group por vez.

Contrato de entrada en `docs/data-contracts.md` ("Salida Parquet de L1") y
TRD-L2 §7.1. L2 lee solo el canónico (ADR-L2-09) y solo las tres columnas que
usa el detector.
"""

import logging
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyutils import resolve_fs

from l2_dc_events.timing import Phases

logger = logging.getLogger(__name__)

CONSOLIDATED = "consolidated.parquet"
COLUMNS = ("price", "transact_time", "agg_trade_id")
PRICE_TYPE = pa.decimal128(18, 8)
# Row groups que el lector pide por delante del que se consume (`read_batches`).
READ_AHEAD = 2


class LandingError(Exception):
    """La entrada de la unidad no existe o no cumple el contrato de L1.

    `check_type` es el del hallazgo de TRD-L2 §9.3 que la describe (`None` si
    el error no tiene uno) y `details` su payload.
    """

    def __init__(
        self,
        message: str,
        check_type: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.check_type = check_type
        self.details = details or {}


@dataclass(frozen=True)
class Ticks:
    """Un lote de ticks como buffers de Arrow sin copia, listos para el fan-out.

    `prices` es el `decimal128` crudo (16 bytes por tick); `times` y
    `agg_trade_ids`, `int64`. Retienen la memoria del lote: soltarlos (y el
    `RecordBatch` de origen) la libera.
    """

    prices: memoryview
    times: memoryview
    agg_trade_ids: memoryview

    def __len__(self) -> int:
        return len(self.times)

    def chunks(self, size: int) -> Iterator[Ticks]:
        """Tramos de a lo más `size` ticks, como vistas del mismo buffer."""
        for start in range(0, len(self), size):
            stop = min(start + size, len(self))
            yield Ticks(
                self.prices[start * 16 : stop * 16],
                self.times[start:stop],
                self.agg_trade_ids[start:stop],
            )


def consolidated_path(
    root: str, provider: str, market: str, asset: str, year: int, month: int
) -> str:
    """Ruta hive del `consolidated.parquet` del mes bajo `root` (local o `gs://`)."""
    return (
        f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/year={year:04d}/month={month:02d}/{CONSOLIDATED}"
    )


def _missing(fs: pafs.FileSystem, path: str, resolved: str) -> LandingError:
    """Explica por qué falta el consolidado: solo provisionales o nada.

    `resolved` es la ruta sin esquema que entiende `fs` (un `FileSelector` con
    un URI `gs://` o `file://` lo rechaza); `path` solo va en el mensaje.
    """
    parent = resolved.rpartition("/")[0]
    selector = pafs.FileSelector(parent, allow_not_found=True)
    provisional = [
        info.base_name
        for info in fs.get_file_info(selector)
        if info.base_name.startswith("provisional-day=")
    ]
    if provisional:
        return LandingError(
            f"{path} no existe y solo hay {len(provisional)} provisionales: L2 "
            "lee solo el consolidado (ADR-L2-09); espera al monthly-close de L1",
            "input_provisional_only",
            {"expected_path": path, "provisionals": len(provisional)},
        )
    return LandingError(
        f"{path} no existe: L1 no ha publicado el mes",
        "input_missing",
        {"expected_path": path},
    )


def _validate(parquet: pq.ParquetFile, path: str) -> None:
    schema = parquet.schema_arrow
    for name in COLUMNS:
        if name not in schema.names:
            raise LandingError(f"{path}: falta la columna {name!r} del contrato de L1")
    if schema.field("price").type != PRICE_TYPE:
        raise LandingError(
            f"{path}: price es {schema.field('price').type}, se esperaba {PRICE_TYPE}"
        )


def open_consolidated(path: str) -> pq.ParquetFile:
    """Abre `path` sin leer datos y valida las columnas que L2 consume.

    Quien lo llama debe cerrarlo (`parquet.close()` o `with`).
    """
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type == pafs.FileType.NotFound:
        raise _missing(fs, path, resolved)
    # Con `filesystem=` el `ParquetFile` es dueño del archivo y `close()` lo
    # cierra (pasarle un `NativeFile` ya abierto lo dejaría abierto). En GCS
    # eso libera la conexión sin esperar al recolector.
    parquet = pq.ParquetFile(resolved, filesystem=fs)
    try:
        _validate(parquet, path)
    except LandingError:
        parquet.close()
        raise
    return parquet


def _buffer(array: pa.Array, width: int, fmt: str) -> memoryview:
    """Vista sin copia de los valores de `array`, respetando su desplazamiento."""
    if array.null_count:
        raise LandingError("el contrato de L1 no admite nulos en price, tiempo ni id")
    values = memoryview(array.buffers()[1]).cast("B")
    start = array.offset * width
    view = values[start : start + len(array) * width]
    return view if fmt == "B" else view.cast(fmt)


def ticks_of(batch: pa.RecordBatch) -> Ticks:
    """Las tres columnas del lote como buffers, sin copiar."""
    return Ticks(
        prices=_buffer(batch.column("price"), 16, "B"),
        times=_buffer(batch.column("transact_time"), 8, "q"),
        agg_trade_ids=_buffer(batch.column("agg_trade_id"), 8, "q"),
    )


def _compressed_bytes(parquet: pq.ParquetFile, index: int) -> int:
    """Bytes comprimidos que ocupan en el archivo las columnas de `COLUMNS` del row group."""
    group = parquet.metadata.row_group(index)
    columns = (group.column(i) for i in range(group.num_columns))
    return sum(c.total_compressed_size for c in columns if c.path_in_schema in COLUMNS)


def _read_group(parquet: pq.ParquetFile, index: int) -> tuple[pa.Table, float]:
    """El row group `index` y la CPU que costó decodificarlo (`thread_time`)."""
    cpu = time.thread_time()
    group = parquet.read_row_group(index, columns=list(COLUMNS), use_threads=False)
    return group, time.thread_time() - cpu


def read_batches(
    parquet: pq.ParquetFile,
    phases: Phases | None = None,
    in_flight: int = READ_AHEAD,
) -> Iterator[pa.RecordBatch]:
    """Los ticks del archivo, un row group a la vez, en el orden del archivo.

    Con `in_flight` = k > 0, un hilo lector pide hasta k row groups por delante
    del que se consume: en GCS cada row group cuesta ~0,3 s de latencia con la
    CPU ociosa, y en serie eran 70 s de `read_s` sobre 237 row groups. En RAM
    hay a lo más 1 + k row groups (el que se consume y los k pedidos), que es
    lo que RNF-L2-01 permite crecer. Con k = 0 se lee en serie en el hilo que
    llama. El lector decodifica sin hilos de Arrow (`use_threads=False`): la
    decodificación de tres columnas es ínfima frente al fan-out, y con hilos el
    lector retiene una cantidad variable de buffers de más (~1,2 MB por row
    group de 1,6 MB medidos).

    Quien consume debe soltar cada lote antes de pedir el próximo, o el row
    group no se libera, y cerrar el generador (`contextlib.closing`) para que
    el hilo lector termine antes de cerrar el archivo.

    Con `phases`, `read_s` es lo que el hilo principal esperó por cada row
    group (con k = 0, lo que tardó en leerlo) y `decode_s`, la CPU que costó
    decodificarlo en el hilo lector. Si la cuota de CPU estrangula al proceso,
    esa espera también cae en `read_s`: `cpu_throttled_s` (ver
    `cpu.throttled_s`) la distingue.
    """
    count = parquet.num_row_groups
    pool = ThreadPoolExecutor(max_workers=1) if in_flight > 0 else None
    pending: deque[Future[tuple[pa.Table, float]]] = deque()
    requested = 0
    try:
        for index in range(count):
            wall = time.perf_counter()
            if pool is None:
                group, cpu = _read_group(parquet, index)
            else:
                while requested < min(count, index + in_flight + 1):
                    pending.append(pool.submit(_read_group, parquet, requested))
                    requested += 1
                group, cpu = pending.popleft().result()
            if phases is not None:
                phases.read_s += time.perf_counter() - wall
                phases.decode_s += cpu
                phases.row_groups += 1
                phases.bytes_in += _compressed_bytes(parquet, index)
            # Un row group puede venir en varios trozos; cada uno es un lote.
            batches = group.to_batches()
            del group
            # `pop(0)` saca el lote de la lista: sin esto la lista lo retendría
            # mientras el consumidor trabaja con el siguiente trozo.
            while batches:
                batch = batches.pop(0)
                if batch.num_rows:
                    yield batch
                del batch
            del batches
    finally:
        if pool is not None:
            # Lo pedido y no consumido se descarta; esperar al lector en curso
            # evita que lea de un archivo que el llamador ya cerró.
            pool.shutdown(wait=True, cancel_futures=True)
            pending.clear()
