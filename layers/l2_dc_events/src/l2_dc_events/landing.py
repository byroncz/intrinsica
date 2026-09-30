"""Lector de la landing de L1: `consolidated.parquet`, un row group por vez.

Contrato de entrada en `docs/data-contracts.md` ("Salida Parquet de L1") y
TRD-L2 §7.1. L2 lee solo el canónico (ADR-L2-09) y solo las tres columnas que
usa el detector.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CONSOLIDATED = "consolidated.parquet"
COLUMNS = ("price", "transact_time", "agg_trade_id")
PRICE_TYPE = pa.decimal128(18, 8)


class LandingError(Exception):
    """La entrada de la unidad no existe o no cumple el contrato de L1."""


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


def _resolve_fs(path: str) -> tuple[pafs.FileSystem, str]:
    if "://" not in path:
        return pafs.LocalFileSystem(), path
    return pafs.FileSystem.from_uri(path)


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
            "lee solo el consolidado (ADR-L2-09); espera al monthly-close de L1"
        )
    return LandingError(f"{path} no existe: L1 no ha publicado el mes")


def open_consolidated(path: str) -> pq.ParquetFile:
    """Abre `path` sin leer datos y valida las columnas que L2 consume."""
    fs, resolved = _resolve_fs(path)
    if fs.get_file_info(resolved).type == pafs.FileType.NotFound:
        raise _missing(fs, path, resolved)
    parquet = pq.ParquetFile(fs.open_input_file(resolved))
    schema = parquet.schema_arrow
    for name in COLUMNS:
        if name not in schema.names:
            raise LandingError(f"{path}: falta la columna {name!r} del contrato de L1")
    if schema.field("price").type != PRICE_TYPE:
        raise LandingError(
            f"{path}: price es {schema.field('price').type}, se esperaba {PRICE_TYPE}"
        )
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


def read_batches(parquet: pq.ParquetFile) -> Iterator[pa.RecordBatch]:
    """Los ticks del archivo, un row group a la vez, en el orden del archivo.

    En RAM hay un solo row group: el anterior se suelta antes de leer el
    siguiente. Quien consume debe soltar cada lote antes de pedir el próximo,
    o el row group no se libera.
    """
    for index in range(parquet.num_row_groups):
        # Sin hilos: la decodificación de tres columnas es ínfima frente al
        # fan-out, y con hilos el lector retiene una cantidad variable de
        # buffers de más (~1,2 MB por row group de 1,6 MB medidos).
        group = parquet.read_row_group(index, columns=list(COLUMNS), use_threads=False)
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
