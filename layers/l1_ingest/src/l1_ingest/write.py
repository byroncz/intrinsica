"""Escritura del Parquet de salida de L1 (§7.2 del TRD-L1, paso 9 del §8.1).

El escritor atómico y el hash de contenido son de `shared/pyutils`; aquí solo
queda lo propio de L1: las rutas, el esquema y el orden de la salida.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from pyutils import PartitionWriter

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


def output_writer(path: str) -> PartitionWriter:
    """`PartitionWriter` de `path` con el esquema, el orden y los row groups de §7.2."""
    return PartitionWriter(
        path, OUTPUT_SCHEMA, SORT_ORDER, row_group_size=ROW_GROUP_SIZE
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

    Sobrescribe de forma atómica (ver `pyutils.PartitionWriter`). Devuelve `path`.
    """
    if not table.schema.equals(OUTPUT_SCHEMA):
        raise ValueError(
            f"el esquema no es OUTPUT_SCHEMA:\n{table.schema}\n!=\n{OUTPUT_SCHEMA}"
        )
    if not _is_sorted(table):
        raise ValueError(f"la tabla no está ordenada por {SORT_ORDER}")
    with output_writer(path) as writer:
        writer.write_table(table)
        return writer.commit()
