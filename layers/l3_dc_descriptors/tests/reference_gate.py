"""La compuerta de entrada y el linaje por separado, como los hacía el orquestador antes de
ITSC-365: la referencia contra la que se compara la pasada única de `summarize_month`.

`gate_theta` comprueba la entrada de L2 de un θ y lee los dos carry-overs con
`read_carry_over`; `file_hash` relee un Parquet entero, un row group a la vez, y es el
`content_hash` que L2 deja en `events_summary`. `summarize_month` no los usa: saca el
mismo hash de los bytes que ya leyó.
"""

from contextlib import closing

import pyarrow.parquet as pq
from dc_frames import l2_path
from dc_frames.lake import Month, month_of, month_ordinal, partition_path
from l3_dc_descriptors.l2 import CARRY_OVER, EVENTS, CarryOver, read_carry_over, require
from pyutils import ContentHasher, resolve_fs


def gate_theta(
    l2_root: str,
    key: tuple[str, str, str],
    month: Month,
    series_start: Month,
    text: str,
) -> tuple[CarryOver, int]:
    """El carry-over del mes y la dirección del mes anterior (0 en el primer mes)."""
    require(l2_path(l2_root, key, text, month), text, month, EVENTS)
    carry = read_carry_over(l2_root, key, text, month)
    if month == series_start:
        return carry, 0
    before = month_of(month_ordinal(month) - 1)
    return carry, read_carry_over(l2_root, key, text, before).direction


def file_hash(path: str) -> str:
    """El `content_hash` de un Parquet leyéndolo de nuevo, un row group a la vez."""
    fs, resolved = resolve_fs(path)
    with closing(pq.ParquetFile(resolved, filesystem=fs)) as parquet:
        hasher = ContentHasher(parquet.schema_arrow)
        for i in range(parquet.num_row_groups):
            table = parquet.read_row_group(i)
            for batch in table.to_batches():
                hasher.update(batch)
            del table
        return hasher.hexdigest()


def lineage(
    l2_root: str, key: tuple[str, str, str], text: str, month: Month
) -> tuple[str, str]:
    """`(events, carry_over)`: los `content_hash` de las dos entradas de L2 del θ."""
    return (
        file_hash(l2_path(l2_root, key, text, month)),
        file_hash(partition_path(l2_root, key, text, month, CARRY_OVER)),
    )
