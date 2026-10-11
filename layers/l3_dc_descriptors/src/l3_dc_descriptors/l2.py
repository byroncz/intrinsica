"""Entradas de L2 que no pasan por el lector: `carry_over.parquet` y la compuerta (§9.2).

Solo se lee por nombre de columna: L3 no importa `l2_dc_events` (RNF-L3-04). El carry-over
no es estado de L3: sirve para conciliar los ticks del mes (§9.1) y para detectar un L2
incoherente.
"""

from contextlib import closing
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.fs as pafs
from dc_frames.lake import Month, check_columns, open_parquet, partition_path
from pyutils import resolve_fs

from l3_dc_descriptors.errors import L2InputMissing, L2StateInconsistent

EVENTS = "events.parquet"
CARRY_OVER = "carry_over.parquet"

_COLUMNS = {
    "direction": pa.int8(),
    "has_pending_event": pa.bool_(),
    "pending_reference_agg_trade_id": pa.int64(),
    "ext_high_agg_trade_id": pa.int64(),
    "ext_low_agg_trade_id": pa.int64(),
}


@dataclass(frozen=True, slots=True)
class CarryOver:
    """Lo que de `carry_over.parquet` de L2 usa la conciliación."""

    direction: int
    has_pending: bool
    pending_reference: int | None
    ext_high: int
    ext_low: int


def l2_file(
    l2_root: str, key: tuple[str, str, str], theta: str, month: Month, name: str
) -> str:
    return partition_path(l2_root, key, theta, month, name)


def require(path: str, theta: str, month: Month, name: str) -> None:
    """`L2InputMissing` si el archivo de L2 no existe."""
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        raise L2InputMissing(theta, month, name, path)


def read_carry_over(
    l2_root: str, key: tuple[str, str, str], theta: str, month: Month
) -> CarryOver:
    """El carry-over de `(θ, mes)`, o `L2InputMissing` si no existe.

    Con `direction ≠ 0` y sin evento pendiente (o sin su referencia) L2 es incoherente
    y lanza `L2StateInconsistent` (ADR-L3-05): la conciliación supone que una dirección
    fijada implica un evento pendiente. `direction = 0` es un estado válido.
    """
    path = l2_file(l2_root, key, theta, month, CARRY_OVER)
    require(path, theta, month, CARRY_OVER)
    with closing(open_parquet(path, "el carry-over de L2")) as parquet:
        check_columns(parquet.schema_arrow, _COLUMNS, path)
        row = parquet.read(columns=list(_COLUMNS)).to_pylist()
    if len(row) != 1:
        raise L2StateInconsistent.mismatch(
            theta, month, "filas de carry_over.parquet", len(row), 1
        )
    row = row[0]
    carry = CarryOver(
        row["direction"],
        row["has_pending_event"],
        row["pending_reference_agg_trade_id"],
        row["ext_high_agg_trade_id"],
        row["ext_low_agg_trade_id"],
    )
    if carry.direction != 0 and (
        not carry.has_pending or carry.pending_reference is None
    ):
        raise L2StateInconsistent.no_pending(
            theta, month, carry.direction, carry.has_pending
        )
    return carry
