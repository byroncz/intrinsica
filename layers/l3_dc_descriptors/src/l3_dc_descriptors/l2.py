"""Entradas de L2 que no pasan por el lector: `carry_over.parquet`, su huella y la compuerta (§9.2).

Solo se lee por nombre de columna: L3 no importa `l2_dc_events` (RNF-L3-04). El carry-over
no es estado de L3: sirve para conciliar los ticks del mes (§9.1) y para detectar un L2
incoherente. La huella (`content_hash.json`) la publica L2 junto a sus dos Parquet y L3 la
lee sin recalcularla (ITSC-368, ADR-L3-16).
"""

from contextlib import closing
from dataclasses import dataclass
from types import MappingProxyType

import pyarrow as pa
import pyarrow.fs as pafs
from dc_frames import FramesInputError
from dc_frames.lake import Month, check_columns, open_parquet, partition_path
from pyutils import lineage, resolve_fs

from l3_dc_descriptors.errors import (
    L2InputMissing,
    L2LineageMissing,
    L2StateInconsistent,
)

EVENTS = "events.parquet"
CARRY_OVER = "carry_over.parquet"
LINEAGE = lineage.LINEAGE

_COLUMNS = MappingProxyType(
    {
        "direction": pa.int8(),
        "has_pending_event": pa.bool_(),
        "pending_reference_agg_trade_id": pa.int64(),
        "ext_high_agg_trade_id": pa.int64(),
        "ext_low_agg_trade_id": pa.int64(),
    }
)


@dataclass(frozen=True, slots=True)
class CarryOver:
    """Lo que de `carry_over.parquet` de L2 usa la conciliación."""

    direction: int
    has_pending: bool
    pending_reference: int | None
    ext_high: int
    ext_low: int


@dataclass(frozen=True, slots=True)
class Lineage:
    """Los `content_hash` de las dos entradas de L2 de un `(θ, mes)`, tal como L2 los publicó."""

    events: str
    carry_over: str


def l2_file(
    l2_root: str, key: tuple[str, str, str], theta: str, month: Month, name: str
) -> str:
    return partition_path(l2_root, key, theta, month, name)


def require(path: str, theta: str, month: Month, name: str) -> None:
    """`L2InputMissing` si el archivo de L2 no existe."""
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        raise L2InputMissing(theta, month, name, path)


def read_lineage(
    l2_root: str, key: tuple[str, str, str], theta: str, month: Month
) -> Lineage:
    """La huella que L2 publicó de `(θ, mes)`, o `L2LineageMissing` si falta o no es válida.

    No se recalcula nunca: una partición sin huella es una partición que el backfill de
    ITSC-368 no cubrió, y recalcularla en silencio escondería el hueco.
    """
    path = l2_file(l2_root, key, theta, month, LINEAGE)
    try:
        found = lineage.read(path)
    except FileNotFoundError:
        raise L2LineageMissing(theta, month, path, "no existe") from None
    except ValueError as error:
        raise L2LineageMissing(theta, month, path, str(error)) from error
    return Lineage(found["events_content_hash"], found["carry_over_content_hash"])


def read_carry_over(
    l2_root: str, key: tuple[str, str, str], theta: str, month: Month
) -> CarryOver:
    """El carry-over de `(θ, mes)`, o `L2InputMissing` si no existe.

    Con `direction ≠ 0` y sin evento pendiente (o sin su referencia) L2 es incoherente
    y lanza `L2StateInconsistent` (ADR-L3-05): la conciliación supone que una dirección
    fijada implica un evento pendiente. `direction = 0` es un estado válido.
    """
    path = l2_file(l2_root, key, theta, month, CARRY_OVER)
    try:
        parquet = open_parquet(path, "el carry-over de L2")
    except FramesInputError:
        # `open_parquet` ya comprobó que existe; si falta, es el hallazgo de la compuerta.
        require(path, theta, month, CARRY_OVER)
        raise
    with closing(parquet):
        check_columns(parquet.schema_arrow, _COLUMNS, path)
        table = parquet.read(columns=list(_COLUMNS))
    row = table.to_pylist()
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
