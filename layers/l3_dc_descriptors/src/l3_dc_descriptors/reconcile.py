"""Conciliación de ticks por mes, sin estado de otros meses (TRD-L3 §9.1).

    N_M = (Σ_cerrados − P_M) + Cola_M + U_M

`N_M`, `Cola_M` y `U_M` salen del pie de `consolidated.parquet` del mes y de los ids de
L2; `Σ_cerrados` y `P_M`, del resumidor y del lector. Son fuentes distintas: por eso la
suma comprueba algo.
"""

from dataclasses import dataclass

import numpy as np
from dc_frames.lake import Month, l1_path, open_parquet

from l3_dc_descriptors.errors import PhaseTicksMismatch
from l3_dc_descriptors.l2 import CarryOver

_BEFORE_ANY_TICK = -1


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Los términos de §9.1 de un θ y un mes."""

    n: int
    closed: int
    before_month: int
    tail: int
    unassigned: int

    @property
    def balanced(self) -> bool:
        return self.n == self.closed - self.before_month + self.tail + self.unassigned


class L1Month:
    """El pie de `consolidated.parquet` de un mes: cuántos ticks tiene y cuántos hay hasta un id.

    Cuenta con las estadísticas de `agg_trade_id` de cada row group; solo el row group
    que contiene el id se decodifica, y solo esa columna.
    """

    def __init__(self, l1_root: str, key: tuple[str, str, str], month: Month) -> None:
        path = l1_path(l1_root, key, month)
        self._parquet = open_parquet(path, f"L1 del mes {month[0]:04d}-{month[1]:02d}")
        meta = self._parquet.metadata
        index = self._parquet.schema_arrow.get_field_index("agg_trade_id")
        self._groups: list[tuple[int, int, int, int]] = []
        for i in range(meta.num_row_groups):
            group = meta.row_group(i)
            if not group.num_rows:
                continue
            stats = group.column(index).statistics
            if stats is None or not stats.has_min_max:
                self._parquet.close()
                raise ValueError(
                    f"{path}: el row group {i} no trae estadísticas de agg_trade_id"
                )
            self._groups.append((i, group.num_rows, stats.min, stats.max))
        self.rows = sum(rows for _, rows, _, _ in self._groups)

    def count_up_to(self, agg_trade_id: int) -> int:
        """Los ticks del mes con id ≤ `agg_trade_id`."""
        total = 0
        for i, rows, low, high in self._groups:
            if high <= agg_trade_id:
                total += rows
            elif low <= agg_trade_id:
                ids = self._parquet.read_row_group(i, columns=["agg_trade_id"])
                ids = ids.column(0).combine_chunks().to_numpy(zero_copy_only=False)
                total += int(np.searchsorted(ids, agg_trade_id, "right"))
        return total

    def close(self) -> None:
        self._parquet.close()


def first_reference_bound(
    previous_direction: int,
    carry: CarryOver,
    first_reference: int | None,
) -> int:
    """`τ`: el id hasta el cual los ticks del mes no son de ningún evento (§9.1).

    `first_reference` es la referencia de la primera fila de `events.parquet` del mes,
    o `None` si ningún evento cerró.
    """
    if previous_direction != 0:
        return _BEFORE_ANY_TICK  # la primera referencia quedó atrás: U_M = 0
    if carry.direction != 0:
        # La primera confirmación fue en este mes: τ es la referencia del primer evento.
        return (
            first_reference if first_reference is not None else carry.pending_reference
        )
    return min(carry.ext_high, carry.ext_low)


def reconcile(
    l1: L1Month,
    *,
    theta: str,
    month: Month,
    closed: int,
    before_month: int,
    last_extreme: int | None,
    first_reference: int | None,
    previous_direction: int,
    carry: CarryOver,
) -> Reconciliation:
    """Concilia los ticks del mes; `PhaseTicksMismatch` si no cuadran."""
    tau = first_reference_bound(previous_direction, carry, first_reference)
    boundary = tau if last_extreme is None else max(tau, last_extreme)
    result = Reconciliation(
        n=l1.rows,
        closed=closed,
        before_month=before_month,
        tail=l1.rows - l1.count_up_to(boundary),
        unassigned=l1.count_up_to(tau),
    )
    if not result.balanced:
        raise PhaseTicksMismatch(
            theta,
            month,
            result.n,
            result.closed,
            result.before_month,
            result.tail,
            result.unassigned,
        )
    return result
