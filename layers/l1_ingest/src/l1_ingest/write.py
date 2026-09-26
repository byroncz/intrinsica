"""Escritura del Parquet de salida de L1 (§7.2 del TRD-L1, paso 9 del §8.1)."""

import hashlib
import os
import uuid
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from l1_ingest.manifest import resolve_fs
from l1_ingest.schema import OUTPUT_SCHEMA

CONSOLIDATED = "consolidated.parquet"
ROW_GROUP_SIZE = 1_000_000
SORT_ORDER = [("transact_time", "ascending"), ("agg_trade_id", "ascending")]

_WRITE_OPTIONS = {
    "compression": "zstd",
    "compression_level": 3,
    "write_statistics": True,
    "sorting_columns": pq.SortingColumn.from_ordering(OUTPUT_SCHEMA, SORT_ORDER),
}


def day_filename(day: int) -> str:
    """Nombre del archivo provisional de un día del mes en curso."""
    return f"provisional-day={day:02d}.parquet"


def partition_path(
    root: str | Path,
    provider: str,
    market: str,
    asset: str,
    year: int,
    month: int,
    filename: str,
) -> str:
    """Ruta hive del archivo bajo `root` (local o `gs://<bucket>/l1`)."""
    return (
        f"{str(root).rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/year={year:04d}/month={month:02d}/{filename}"
    )


def _is_sorted(table: pa.Table) -> bool:
    """Verifica el orden con una pasada lineal sobre las dos claves, sin ordenar."""
    if table.num_rows < 2:
        return True
    time = table["transact_time"].combine_chunks()
    trade_id = table["agg_trade_id"].combine_chunks()
    prev_time, next_time = time[:-1], time[1:]
    out_of_order = pc.or_(
        pc.less(next_time, prev_time),
        pc.and_(
            pc.equal(next_time, prev_time),
            pc.less(trade_id[1:], trade_id[:-1]),
        ),
    )
    return not pc.any(out_of_order).as_py()


class PartitionWriter:
    """Escribe el Parquet de salida por lotes, con las propiedades físicas de §7.2.

    El archivo se escribe a un temporal del mismo directorio (o del mismo
    prefijo en GCS) y `commit` lo renombra sobre el destino: nunca queda un
    archivo a medias ni se toca el anterior si algo falla. Sin `commit`, salir
    del bloque `with` borra el temporal. El llamador garantiza el orden.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._fs, self._target = resolve_fs(path)
        parent, _, name = self._target.rpartition("/")
        self._tmp = f"{parent}/.{name}.{uuid.uuid4().hex}.tmp"
        self._sink: pa.NativeFile | None = None
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> Self:
        if not isinstance(self._fs, pafs.GcsFileSystem):
            Path(self._target).parent.mkdir(parents=True, exist_ok=True)
        self._sink = self._fs.open_output_stream(self._tmp)
        self._writer = pq.ParquetWriter(self._sink, OUTPUT_SCHEMA, **_WRITE_OPTIONS)
        return self

    def write_table(self, table: pa.Table) -> None:
        self._check_schema(table.schema)
        self._writer.write_table(table, row_group_size=ROW_GROUP_SIZE)

    def write_batch(self, batch: pa.RecordBatch) -> None:
        self._check_schema(batch.schema)
        self._writer.write_batch(batch, row_group_size=ROW_GROUP_SIZE)

    def commit(self) -> str:
        """Cierra el archivo y lo deja en el destino. Devuelve `path`."""
        self._close()
        if isinstance(self._fs, pafs.GcsFileSystem):
            self._fs.move(self._tmp, self._target)
        else:
            os.replace(self._tmp, self._target)
        return self.path

    @staticmethod
    def _check_schema(schema: pa.Schema) -> None:
        if not schema.equals(OUTPUT_SCHEMA):
            raise ValueError(
                f"el esquema no es OUTPUT_SCHEMA:\n{schema}\n!=\n{OUTPUT_SCHEMA}"
            )

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._close()
        try:
            self._fs.delete_file(self._tmp)
        except FileNotFoundError:
            pass  # ya se renombró en `commit`


def write_partition(table: pa.Table, path: str) -> str:
    """Escribe `table` en `path` con las propiedades físicas de §7.2.

    Sobrescribe de forma atómica (ver `PartitionWriter`). Devuelve `path`.
    """
    if not table.schema.equals(OUTPUT_SCHEMA):
        raise ValueError(
            f"el esquema no es OUTPUT_SCHEMA:\n{table.schema}\n!=\n{OUTPUT_SCHEMA}"
        )
    if not _is_sorted(table):
        raise ValueError(f"la tabla no está ordenada por {SORT_ORDER}")
    with PartitionWriter(path) as writer:
        writer.write_table(table)
        return writer.commit()


def _raw_bytes(column: pa.Array) -> memoryview:
    """Bytes de valores de una columna de ancho fijo, respetando el offset del slice."""
    if pa.types.is_boolean(column.type):
        # El bitmap de un slice arrastra bits vecinos: se pasa a uint8.
        column = pc.cast(column, pa.uint8())
    if column.null_count:
        raise ValueError("content_hash no admite nulos: OUTPUT_SCHEMA no los permite")
    width = column.type.bit_width // 8
    start = column.offset * width
    return memoryview(column.buffers()[1])[start : start + len(column) * width]


class ContentHasher:
    """SHA-256 incremental del contenido lógico: esquema más filas en orden.

    Un hash por columna sobre sus bytes de valores; el digest final combina el
    esquema con los digests de columna. No depende del chunking ni de los
    bytes del Parquet: dos escrituras con las mismas filas dan el mismo hash
    aunque el archivo difiera, y cada lote se ve una sola vez.
    """

    def __init__(self, schema: pa.Schema = OUTPUT_SCHEMA) -> None:
        # La metadata del esquema (clave-valor) no es contenido lógico.
        self._schema = schema.remove_metadata()
        self._columns = [hashlib.sha256() for _ in self._schema]

    def update(self, batch: pa.RecordBatch) -> None:
        for digest, column in zip(self._columns, batch.columns, strict=True):
            digest.update(_raw_bytes(column))

    def hexdigest(self) -> str:
        digest = hashlib.sha256(self._schema.serialize().to_pybytes())
        for column in self._columns:
            digest.update(column.digest())
        return digest.hexdigest()


def content_hash(table: pa.Table) -> str:
    """SHA-256 del contenido lógico de `table` (ver `ContentHasher`)."""
    hasher = ContentHasher(table.schema)
    for batch in table.to_batches():
        hasher.update(batch)
    return hasher.hexdigest()
