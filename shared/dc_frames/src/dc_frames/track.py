"""El avance de un θ sobre sus eventos y la validación de sus fronteras contra un row group.

Lo comparten el lector por evento y por chunks (`reader._Cursor`) y la entrega por
row group (`delivery.RowGroupCursor`): ambos buscan las fronteras por valor con
`searchsorted` y fallan igual (TRD-L3 §7.4, garantías 1 a 4).
"""

from collections.abc import Iterator, Sequence
from decimal import Decimal

import numpy as np

from dc_frames.lake import BOUNDARY_COLUMNS, EventGroup, RowGroup
from dc_frames.types import FrameBoundaryError, FramesInputError

_NAMES = ("referencia", "confirmación", "extremo")


class Track:
    """El avance de un θ: de qué row group de eventos va, cuál es el evento que sigue y qué vio.

    Solo el primer evento sin entregar (`_k`) puede estar a medias (`E_i ≤ R_{i+1}`).
    `_seen` dice cuáles de sus fronteras `(R, C, E)` ya se vieron como ticks de un row
    group de L1 anterior, y `_before` cuántos ticks de `(R, E]` ya contó en row groups
    de un mes anterior al de su partición.
    """

    def __init__(self, theta: Decimal, groups: Iterator[EventGroup]) -> None:
        self.theta = theta
        self._groups = groups
        self._group = next(groups, None)
        self._k = 0
        self._seen = np.zeros(3, bool)
        self._before = 0

    @property
    def done(self) -> bool:
        return self._group is None

    @property
    def reference(self) -> int:
        """Referencia del evento en curso. Solo si no está `done`."""
        return int(self._group.bounds[self._k, 0])

    def describe(self) -> str:
        r, c, e = (int(x) for x in self._group.bounds[self._k])
        return (
            f"θ {self.theta}, evento con referencia {r}, confirmación {c} y extremo {e}"
        )

    def _next_group(self) -> None:
        last_extreme = int(self._group.bounds[-1, 2])
        self._group = next(self._groups, None)
        self._k = 0
        self._seen = np.zeros(3, bool)
        if self._group is not None and self._group.bounds[0, 0] < last_extreme:
            raise FramesInputError(
                f"θ {self.theta}: un row group de eventos empieza en la referencia "
                f"{self._group.bounds[0, 0]}, antes del extremo {last_extreme} del anterior"
            )


def locate(
    theta: Decimal,
    rg: RowGroup,
    bounds: np.ndarray,
    confirm_times: np.ndarray,
    seen_first: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Las posiciones de las fronteras `bounds` (n, 3) en `rg` y cuáles se vieron ya.

    `right[i, j]` es `searchsorted(ids, frontera, "right")`: cuántos ticks del row group
    tienen id ≤ la frontera, así que 0 si cae antes y el largo del row group si cae
    después. Las fases son `[R, C)` y `[C, E)` en esas posiciones.
    `seen_first` son las fronteras del primer evento vistas en row groups anteriores.

    Falla con `FrameBoundaryError` si una frontera dentro del rango del row group no es
    un tick, si una anterior a él no se vio en un row group previo (cae en un hueco entre
    ambos) o si el tick `C` no tiene `transact_time = confirm_time`.
    """
    left = np.searchsorted(rg.ids, bounds, "left")
    right = np.searchsorted(rg.ids, bounds, "right")
    inside = (bounds >= rg.low) & (bounds <= rg.high)
    seen = np.zeros(bounds.shape, bool)
    seen[0] = seen_first
    bad = (inside & (left == right)) | ((bounds < rg.low) & ~seen)
    if bad.any():
        raise _boundary_error(theta, bounds, bad, rg)
    _check_confirm_times(theta, rg, bounds, confirm_times, left, inside[:, 1])
    seen |= inside
    return right, seen


def _check_confirm_times(
    theta: Decimal,
    rg: RowGroup,
    bounds: np.ndarray,
    confirm_times: np.ndarray,
    left: np.ndarray,
    inside: np.ndarray,
) -> None:
    """El tick `C` de cada evento cuya confirmación cae en este row group debe tener
    `transact_time = confirm_time` (TRD-L3 §7.4, garantía 1): si no, L1 y L2 no cuadran.
    """
    rows = np.flatnonzero(inside)
    if not rows.size:
        return
    found = rg.columns[0].to_numpy(zero_copy_only=False)[left[rows, 1]]
    wanted = confirm_times[rows]
    wrong = np.flatnonzero(found != wanted)
    if wrong.size:
        i = int(rows[wrong[0]])
        raise FrameBoundaryError(
            f"θ {theta}: el tick de confirmación {bounds[i, 1]} (evento con "
            f"referencia {bounds[i, 0]}) tiene transact_time {found[wrong[0]]} y L2 "
            f"dice confirm_time {wanted[wrong[0]]}: L1 y L2 no cuadran",
            theta=f"{theta:.8f}",
            field="confirm_time",
            got=found[wrong[0]],
            wanted=wanted[wrong[0]],
        )


def _boundary_error(
    theta: Decimal, bounds: np.ndarray, bad: np.ndarray, rg: RowGroup
) -> FrameBoundaryError:
    i, j = (int(x) for x in np.argwhere(bad)[0])
    event = bounds[i]
    return FrameBoundaryError(
        f"θ {theta}: el id de {_NAMES[j]} {event[j]} del evento "
        f"(referencia {event[0]}, confirmación {event[1]}, extremo {event[2]}) "
        f"no es un tick de L1: cae en un hueco del proveedor o L1 y L2 no cuadran "
        f"(row group de L1 con ids {rg.low}..{rg.high})",
        theta=f"{theta:.8f}",
        field=BOUNDARY_COLUMNS[j],
        got=event[j],
        wanted="un tick de L1",
    )


def unclosed_error(live: Sequence[Track]) -> FrameBoundaryError:
    """L1 terminó con eventos sin cerrar: el extremo de alguno no existe como tick."""
    return FrameBoundaryError(
        "L1 terminó antes de cerrar todos los eventos; el extremo no existe como tick. "
        + "; ".join(c.describe() for c in live),
        theta=f"{live[0].theta:.8f}",
        field="extreme_agg_trade_id",
        got="ausente",
        wanted="un tick de L1",
    )
