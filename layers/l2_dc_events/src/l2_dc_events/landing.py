"""Lector de la landing de L1: `consolidated.parquet`, un row group por vez.

Contrato de entrada en `docs/data-contracts.md` ("Salida Parquet de L1") y
TRD-L2 §7.1. L2 lee solo el canónico (ADR-L2-09) y solo las tres columnas que
usa el detector.
"""

import logging
import time
from collections.abc import Iterator
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


def read_batches(
    parquet: pq.ParquetFile, phases: Phases | None = None
) -> Iterator[pa.RecordBatch]:
    """Los ticks del archivo, un row group a la vez, en el orden del archivo.

    En RAM hay un solo row group: el anterior se suelta antes de leer el
    siguiente. Quien consume debe soltar cada lote antes de pedir el próximo,
    o el row group no se libera.

    Con `phases`, separa cada lectura en decodificar y esperar, sin tocar cómo
    se lee. Sin hilos, el hilo que llama decodifica (Parquet a Arrow): su tiempo
    de CPU es `decode_s` y el resto de la pared es `read_s`, espera de I/O (la
    lectura de GCS va en hilos propios de Arrow). Si la cuota de CPU estrangula
    al proceso, esa espera también cae en `read_s`: `cpu_throttled_s` (ver
    `cpu.throttled_s`) la distingue.
    """
    for index in range(parquet.num_row_groups):
        wall, cpu = time.perf_counter(), time.thread_time()
        # Sin hilos: la decodificación de tres columnas es ínfima frente al
        # fan-out, y con hilos el lector retiene una cantidad variable de
        # buffers de más (~1,2 MB por row group de 1,6 MB medidos).
        group = parquet.read_row_group(index, columns=list(COLUMNS), use_threads=False)
        if phases is not None:
            wall = time.perf_counter() - wall
            cpu = min(time.thread_time() - cpu, wall)
            phases.decode_s += cpu
            phases.read_s += wall - cpu
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
