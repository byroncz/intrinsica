"""Escritor Parquet por lotes con sobrescritura atómica (§7.2 de los TRD de L1 y L2)."""

import logging
import os
import uuid
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from pyutils.fs import resolve_fs

logger = logging.getLogger(__name__)


class PartitionWriter:
    """Escribe un Parquet por lotes con las propiedades físicas de §7.2.

    ZSTD-3, estadísticas por columna y, si hay `sort_order`, sus
    `sorting_columns` en los metadatos. Cada escritura debe traer exactamente
    `schema`. Con `row_group_size` fijo, una escritura mayor se parte en row
    groups de ese tamaño; sin él, cada lote o tabla es su propio row group
    (los lotes ya vienen del tamaño que se quiere).

    El archivo se escribe a un temporal del mismo directorio (o del mismo
    prefijo en GCS) y `commit` lo renombra sobre el destino: nunca queda un
    archivo a medias ni se toca el anterior si algo falla. Sin `commit`, salir
    del bloque `with` borra el temporal. El llamador garantiza el orden.

    En GCS, `commit` mueve el temporal con `fs.move` (copia más borrado): con
    el bucket versionado, el `.tmp` borrado queda como versión no vigente
    hasta que la regla de lifecycle de `landing` lo elimina
    (`docs/data-contracts.md`, "Sobrescritura atómica"). Escribir directo
    sobre `target` sin temporal no es seguro: `ParquetWriter.__del__` y los
    destructores de `GcsOutputStream`/`ObjectWriteStream` finalizan la subida
    aunque `close()` nunca se llame, así que abortar podía dejar el destino
    sustituido por un Parquet con solo las filas ya escritas (ITSC-234, H1).
    """

    def __init__(
        self,
        path: str,
        schema: pa.Schema,
        sort_order: list[tuple[str, str]] | None = None,
        row_group_size: int | None = None,
    ) -> None:
        self.path = path
        self.schema = schema
        self._row_group_size = row_group_size
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
        self._check_schema(batch.schema)
        self._writer.write_batch(batch, row_group_size=self._group(batch.num_rows))

    def write_table(self, table: pa.Table) -> None:
        self._check_schema(table.schema)
        self._writer.write_table(table, row_group_size=self._group(table.num_rows))

    def commit(self) -> str:
        """Cierra el archivo y lo deja en el destino. Devuelve `path`."""
        self._close()
        if isinstance(self._fs, pafs.GcsFileSystem):
            self._fs.move(self._tmp, self._target)
        else:
            os.replace(self._tmp, self._target)
        self._committed = True
        return self.path

    def _group(self, num_rows: int) -> int:
        return self._row_group_size or max(num_rows, 1)

    def _check_schema(self, schema: pa.Schema) -> None:
        if not schema.equals(self.schema):
            raise ValueError(
                f"el esquema no es el del archivo:\n{schema}\n!=\n{self.schema}"
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
