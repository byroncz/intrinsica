from datetime import date

import numpy as np
import pytest
from viz_helpers import DOWN, UP, events, ticks_batch
from viz_tiles.contract import LEVELS
from viz_tiles.direction import PendingEvent, direction_tile, direction_tiles
from viz_tiles.reduce import reduce_day


def states(ids, *rows, pending=None) -> list[int]:
    return direction_tile(
        np.array(ids, dtype=np.int64), events(*rows), pending
    ).tolist()


def test_phase_borders():
    ids = [10, 11, 20, 21, 30, 31, 40, 41, 55, 56]
    #      ref  c   c+1 ext  ref+ c   c+1 ext  tras el último
    assert states(ids, UP, DOWN) == [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]


def test_dtype_and_length():
    tile = direction_tile(np.array([11, 25]), events(UP))
    assert tile.dtype == np.uint8 and tile.shape == (2,)


def test_before_first_event_and_empty_column_are_zero():
    assert states([-1, 0, 5, 10, -1, 11], UP) == [0, 0, 0, 0, 0, 1]


def test_no_events_is_all_zero():
    assert states([1, 2, 3]) == [0, 0, 0]


def test_a_column_is_the_state_of_its_last_tick_not_of_its_first():
    # El último tick de la columna (id 25) es overshoot aunque haya ticks
    # anteriores de la confirmación: la columna se lee por su último id.
    assert states([25], UP) == [2]


def test_downward_event_uses_down_states():
    assert states([31, 41], DOWN) == [3, 4]
    assert states([20, 25], (10, 20, 30, -1)) == [3, 4]


PENDING_UP = PendingEvent(
    reference_agg_trade_id=55,
    confirm_agg_trade_id=70,
    extreme_agg_trade_id=90,
    direction=1,
)


def test_pending_tail_after_closed_events():
    ids = [55, 56, 70, 71, 90, 91, 10**9]
    # El pendiente: confirmación alza, overshoot alza hasta el candidato y,
    # después de él, la confirmación del sentido contrario (baja).
    assert states(ids, UP, DOWN, pending=PENDING_UP) == [4, 1, 1, 2, 2, 3, 3]


def test_pending_down_tail_ends_in_up_confirmation():
    pending = PendingEvent(55, 70, 90, -1)
    assert states([56, 71, 91], UP, DOWN, pending=pending) == [3, 4, 1]


def test_pending_without_closed_events():
    assert states([54, 55, 56, 71, 91], pending=PENDING_UP) == [0, 0, 1, 2, 3]


def test_pending_candidate_beyond_the_month_leaves_all_overshoot():
    pending = PendingEvent(55, 70, 10**12, 1)
    assert states([71, 10**9], UP, DOWN, pending=pending) == [2, 2]


def test_pending_from_carry_over_row():
    row = {
        "has_pending_event": True,
        "direction": 1,
        "pending_reference_agg_trade_id": 55,
        "pending_confirm_agg_trade_id": 70,
        "ext_high_agg_trade_id": 90,
        "ext_low_agg_trade_id": 12,
    }
    assert PendingEvent.from_carry_over(row) == PENDING_UP
    down = {**row, "direction": -1}
    assert PendingEvent.from_carry_over(down) == PendingEvent(55, 70, 12, -1)
    assert PendingEvent.from_carry_over({**row, "has_pending_event": False}) is None


def test_events_must_be_sorted_by_reference():
    with pytest.raises(ValueError, match="ordenados"):
        direction_tile(np.array([11]), events(DOWN, UP))


def test_direction_must_be_plus_or_minus_one():
    with pytest.raises(ValueError, match="direction"):
        direction_tile(np.array([11]), events((10, 20, 30, 0)))


def test_coarse_level_takes_the_state_of_the_last_non_empty_fine_column():
    day = date(2026, 8, 31)
    rows = [
        (11, 1, 100, 1),
        (12, 2, 101, 1),
        (21, 30, 102, 1),
        (31, 70, 103, 1),
    ]
    out = reduce_day([ticks_batch(day, rows)], day)
    tiles = direction_tiles(out.last_ids, events(UP, DOWN))
    # Nivel fino: ids 12, 21 y 31 en las columnas 0, 1 y 3; la 2 está vacía.
    assert tiles[4096][:4].tolist() == [1, 2, 0, 3]
    # Nivel 128: los cuatro ticks caen en la columna 0,
    # cuyo último tick es el id 31 (confirmación baja).
    assert tiles[128][0] == 3
    # Cada nivel coincide con resolver el estado directo desde su último id.
    for w in LEVELS:
        np.testing.assert_array_equal(
            tiles[w], direction_tile(out.last_ids[w], events(UP, DOWN))
        )
    # Una columna gruesa vacía queda en 0.
    assert tiles[128][1] == 0
