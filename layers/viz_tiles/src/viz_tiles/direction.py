"""Estado de dirección por θ de cada columna: un bloque de `dir-<w>.u8`.

TRD-viz §7.5 y ADR-VZ-09. El estado de una columna es el de su último tick, y el
de un tick sale de comparar su `agg_trade_id` con los intervalos de los eventos
del θ. Se resuelve con `searchsorted` sobre los bordes de los eventos y los ids
de las columnas: no se recorre ningún tick.

Se usan ids y no tiempos: varios ticks comparten `transact_time`, pero
`agg_trade_id` es estrictamente creciente, y así un extremo que comparte instante
con un tick posterior no mezcla las fases.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from viz_tiles.contract import (
    STATE_CONFIRM_DOWN,
    STATE_CONFIRM_UP,
    STATE_NONE,
    STATE_OVERSHOOT_DOWN,
    STATE_OVERSHOOT_UP,
)

_INT64_MAX = np.iinfo(np.int64).max

# Por dirección del evento: estado de confirmación y de overshoot.
_UP = (STATE_CONFIRM_UP, STATE_OVERSHOOT_UP)
_DOWN = (STATE_CONFIRM_DOWN, STATE_OVERSHOOT_DOWN)


@dataclass(frozen=True)
class PendingEvent:
    """Evento pendiente al cierre del mes, de `carry_over.parquet` (TRD-viz §7.6).

    Su referencia y su confirmación son conocidas; su extremo no. El extremo
    vigente (`extreme_agg_trade_id`, el `ext_high_*` si `direction = 1` o el
    `ext_low_*` si `direction = -1`) es un candidato que un tick posterior
    puede superar: hasta él los ticks son overshoot del pendiente, y después de él
    son la confirmación del sentido contrario (provisional).
    """

    reference_agg_trade_id: int
    confirm_agg_trade_id: int
    extreme_agg_trade_id: int
    direction: int

    @classmethod
    def from_carry_over(cls, row: Mapping[str, object]) -> PendingEvent | None:
        """El pendiente de una fila de `carry_over.parquet`, o `None` si no hay."""
        if not row["has_pending_event"]:
            return None
        direction = int(row["direction"])
        extreme = "ext_high" if direction == 1 else "ext_low"
        return cls(
            reference_agg_trade_id=int(row["pending_reference_agg_trade_id"]),
            confirm_agg_trade_id=int(row["pending_confirm_agg_trade_id"]),
            extreme_agg_trade_id=int(row[f"{extreme}_agg_trade_id"]),
            direction=direction,
        )


@dataclass(frozen=True)
class _Borders:
    """Intervalos `(reference, extreme]` de los eventos, ordenados, con su fase."""

    reference: np.ndarray
    confirm: np.ndarray
    extreme: np.ndarray
    states: np.ndarray  # (n, 2): estado de confirmación y de overshoot


def _borders(events: pa.Table | pa.RecordBatch, pending: PendingEvent | None):
    reference = events.column("reference_agg_trade_id").to_numpy()
    confirm = events.column("confirm_agg_trade_id").to_numpy()
    extreme = events.column("extreme_agg_trade_id").to_numpy()
    directions = events.column("direction").to_numpy()
    if pending is not None:
        d = pending.direction
        reference = np.append(
            reference, [pending.reference_agg_trade_id, pending.extreme_agg_trade_id]
        )
        confirm = np.append(confirm, [pending.confirm_agg_trade_id, _INT64_MAX])
        extreme = np.append(extreme, [pending.extreme_agg_trade_id, _INT64_MAX])
        # Después del candidato, todo es confirmación del sentido contrario.
        directions = np.append(directions, [d, -d])
    if np.any(np.diff(reference) <= 0):
        raise ValueError("los eventos deben venir ordenados por reference_agg_trade_id")
    if not np.all(np.isin(directions, (1, -1))):
        raise ValueError("direction debe ser 1 o -1")
    states = np.where(directions[:, None] == 1, _UP, _DOWN)
    return _Borders(reference, confirm, extreme, states)


def _state_of(last_ids: np.ndarray, borders: _Borders) -> np.ndarray:
    tile = np.full(len(last_ids), STATE_NONE, dtype=np.uint8)
    if not len(borders.reference):
        return tile
    # El evento candidato de un tick x es el de mayor referencia menor que x.
    event = np.searchsorted(borders.reference, last_ids, side="left") - 1
    found = (last_ids >= 0) & (event >= 0)
    event = np.where(found, event, 0)
    inside = found & (last_ids <= borders.extreme[event])
    overshoot = (last_ids > borders.confirm[event]).astype(np.intp)
    states = borders.states[event, overshoot]
    tile[inside] = states[inside]
    return tile


def direction_tile(
    last_ids: np.ndarray,
    events: pa.Table | pa.RecordBatch,
    pending: PendingEvent | None = None,
) -> np.ndarray:
    """Tile `uint8` de un nivel: el estado de dirección de cada columna.

    `last_ids` es el `agg_trade_id` del último tick de cada columna (-1 si está
    vacía, `DayReduction.last_ids`). `events` son las filas del θ con el esquema
    de `events.parquet` que tocan el día, ordenadas por referencia; `pending` es
    la cola del carry-over, si la hay.

    Para un tick con id `x`: está en el evento `e` si
    `e.reference < x <= e.extreme`; es confirmación si `x <= e.confirm` y
    overshoot si no. Fuera de todo evento, o en una columna vacía, el estado es 0.
    """
    return _state_of(last_ids, _borders(events, pending))


def direction_tiles(
    last_ids: Mapping[int, np.ndarray],
    events: pa.Table | pa.RecordBatch,
    pending: PendingEvent | None = None,
) -> dict[int, np.ndarray]:
    """`direction_tile` de cada nivel de `last_ids`, armando los bordes una vez."""
    borders = _borders(events, pending)
    return {w: _state_of(ids, borders) for w, ids in last_ids.items()}
