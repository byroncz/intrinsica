"""Escritura del Parquet de salida de L1 (§7.2 del TRD-L1, paso 9 del §8.1)."""

import hashlib
import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from l1_ingest.manifest import resolve_fs
from l1_ingest.schema import OUTPUT_SCHEMA

CONSOLIDATED = "consolidated.parquet"
ROW_GROUP_SIZE = 1_000_000
SORT_ORDER = [("transact_time", "ascending"), ("agg_trade_id", "ascending")]


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


def write_partition(table: pa.Table, path: str) -> str:
    """Escribe `table` en `path` con las propiedades físicas de §7.2.

    Sobrescribe de forma atómica: en local escribe a un temporal del mismo
    directorio y lo renombra; en GCS reemplazar el objeto ya es atómico.
    Devuelve `path`.
    """
    if not table.schema.equals(OUTPUT_SCHEMA):
        raise ValueError(
            f"el esquema no es OUTPUT_SCHEMA:\n{table.schema}\n!=\n{OUTPUT_SCHEMA}"
        )
    if not _is_sorted(table):
        raise ValueError(f"la tabla no está ordenada por {SORT_ORDER}")
    fs, target = resolve_fs(path)
    options = {
        "compression": "zstd",
        "compression_level": 3,
        "write_statistics": True,
        "row_group_size": ROW_GROUP_SIZE,
        "sorting_columns": pq.SortingColumn.from_ordering(OUTPUT_SCHEMA, SORT_ORDER),
    }
    if isinstance(fs, pafs.GcsFileSystem):
        # Se escribe a memoria primero: si falla, el objeto anterior no se toca.
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, **options)
        with fs.open_output_stream(target) as out:
            out.write(sink.getvalue())
        return path

    directory = Path(target).parent
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{Path(target).name}.{uuid.uuid4().hex}.tmp"
    try:
        pq.write_table(table, str(tmp), **options)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def content_hash(table: pa.Table) -> str:
    """SHA-256 del contenido lógico: esquema más filas en orden.

    No depende del chunking ni de los bytes del Parquet: dos escrituras con las
    mismas filas dan el mismo hash aunque el archivo difiera.
    """
    # Los bool se pasan a uint8: el escritor IPC serializa el bitmap de un slice
    # con los bits vecinos, y esos bits no son parte del contenido lógico.
    columns = [
        pc.cast(col, pa.uint8()) if pa.types.is_boolean(col.type) else col
        for col in table.columns
    ]
    flat = pa.table(columns, names=table.schema.names).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, flat.schema) as writer:
        writer.write_table(flat)
    digest = hashlib.sha256(table.schema.serialize().to_pybytes())
    digest.update(sink.getvalue())
    return digest.hexdigest()
