"""Escritura de la salida de L2: particiones `events` y `carry_over` (§7.2 y §7.4).

El patrón es el de `l1_ingest.write` (escritura atómica y `content_hash`
incremental), copiado y no importado: L2 no depende de `l1_ingest` (RNF-14).
La extracción a `shared/` es una card aparte.
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from l2_dc_events.landing import resolve_fs

logger = logging.getLogger(__name__)

EVENTS = "events.parquet"
CARRY_OVER = "carry_over.parquet"
THETA_DIGITS = 8


def format_theta(theta: int) -> str:
    """`theta = round(θ × 10⁸)` como el `theta=<t>` de la ruta: `0.00010000`.

    Ancho fijo, sin recortar ceros (§7.2): ordena bien como texto y deja
    a la vista la escala que comparte con el precio.
    """
    if not 0 < theta < 10**THETA_DIGITS:
        raise ValueError(f"theta={theta} fuera de (0, 10^{THETA_DIGITS})")
    return f"0.{theta:0{THETA_DIGITS}d}"


def partition_path(
    root: str | Path,
    provider: str,
    market: str,
    asset: str,
    theta: int,
    year: int,
    month: int,
    filename: str,
) -> str:
    """Ruta hive del archivo bajo `root` (local o `gs://<bucket>/<prefijo>`)."""
    return (
        f"{str(root).rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/theta={format_theta(theta)}/year={year:04d}/month={month:02d}/{filename}"
    )


class PartitionWriter:
    """Escribe un Parquet por lotes con las propiedades físicas de §7.2.

    ZSTD-3, estadísticas por columna y, si hay `sort_order`, sus
    `sorting_columns` en los metadatos. El archivo se escribe a un temporal
    del mismo directorio (o prefijo, en GCS) y `commit` lo renombra sobre el
    destino: nunca queda un archivo a medias ni se toca el anterior si algo
    falla. Sin `commit`, salir del bloque `with` borra el temporal. El
    llamador garantiza el orden.

    Escribir directo sobre el destino no es seguro: `ParquetWriter.__del__` y
    los destructores de los streams de GCS finalizan la subida aunque
    `close()` nunca se llame (ITSC-234, H1 de L1).
    """

    def __init__(
        self,
        path: str,
        schema: pa.Schema,
        sort_order: list[tuple[str, str]] | None = None,
    ) -> None:
        self.path = path
        self.schema = schema
        self._fs, self._target = resolve_fs(path)
        parent, _, name = self._target.rpartition("/")
        self._tmp = f"{parent}/.{name}.{uuid.uuid4().hex}.tmp"
        self._options: dict = {
            "compression": "zstd",
            "compression_level": 3,
            "write_statistics": True,
        }
        if sort_order:
            self._options["sorting_columns"] = pq.SortingColumn.from_ordering(
                schema, sort_order
            )
        self._sink: pa.NativeFile | None = None
        self._writer: pq.ParquetWriter | None = None
        self._committed = False

    def __enter__(self) -> Self:
        if not isinstance(self._fs, pafs.GcsFileSystem):
            Path(self._target).parent.mkdir(parents=True, exist_ok=True)
        self._sink = self._fs.open_output_stream(self._tmp)
        self._writer = pq.ParquetWriter(self._sink, self.schema, **self._options)
        return self

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Escribe el lote como su propio row group (los lotes son ya de buen tamaño)."""
        if not batch.schema.equals(self.schema):
            raise ValueError(f"el esquema no es el del archivo:\n{batch.schema}")
        self._writer.write_batch(batch, row_group_size=max(batch.num_rows, 1))

    def commit(self) -> str:
        """Cierra el archivo y lo deja en el destino. Devuelve `path`."""
        self._close()
        if isinstance(self._fs, pafs.GcsFileSystem):
            self._fs.move(self._tmp, self._target)
        else:
            os.replace(self._tmp, self._target)
        self._committed = True
        return self.path

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
        if self._committed:
            return  # el temporal ya no existe: `commit` lo movió
        try:
            self._fs.delete_file(self._tmp)
        except OSError:
            # Sin excepción previa, un temporal que no se pudo borrar es un
            # fallo real; con ella, no debe ocultarla.
            if exc is None:
                raise
            logger.warning("no se pudo borrar el temporal %s", self._tmp)


def _raw_bytes(column: pa.Array) -> memoryview:
    """Bytes de valores de una columna de ancho fijo, respetando el offset del slice."""
    if pa.types.is_boolean(column.type):
        # El bitmap de un slice arrastra bits vecinos: se pasa a uint8.
        column = pc.cast(column, pa.uint8())
    width = column.type.bit_width // 8
    start = column.offset * width
    return memoryview(column.buffers()[1])[start : start + len(column) * width]


class ContentHasher:
    """SHA-256 incremental del contenido lógico: esquema más filas en orden.

    Un hash por columna; el digest final combina el esquema con los digests de
    columna. No depende del chunking ni de los bytes del Parquet: dos
    escrituras con las mismas filas dan el mismo hash aunque el archivo
    difiera, y cada lote se ve una sola vez.

    Las columnas de ancho fijo y no nulables (todas las de `events`) se
    hashean sobre los bytes de sus valores. Las demás (texto o nulables: el
    carry-over, de una fila) valor a valor, porque el buffer de un valor nulo
    no está definido. Se decide por el esquema y no por cada lote, para que el
    hash no dependa de cómo se parta la serie.
    """

    def __init__(self, schema: pa.Schema) -> None:
        # La metadata del esquema (clave-valor) no es contenido lógico.
        self._schema = schema.remove_metadata()
        self._columns = [hashlib.sha256() for _ in self._schema]
        self._by_value = [not _is_raw(field) for field in self._schema]

    def update(self, batch: pa.RecordBatch) -> None:
        for digest, by_value, column in zip(
            self._columns, self._by_value, batch.columns, strict=True
        ):
            if by_value:
                for value in column.to_pylist():
                    digest.update(repr(value).encode() + b"\0")
            else:
                digest.update(_raw_bytes(column))

    def hexdigest(self) -> str:
        digest = hashlib.sha256(self._schema.serialize().to_pybytes())
        for column in self._columns:
            digest.update(column.digest())
        return digest.hexdigest()


def _is_raw(field: pa.Field) -> bool:
    """Si los bytes de valores de la columna bastan para hashearla."""
    kind = field.type
    fixed = (
        pa.types.is_integer(kind)
        or pa.types.is_decimal(kind)
        or pa.types.is_boolean(kind)
    )
    return fixed and not field.nullable
