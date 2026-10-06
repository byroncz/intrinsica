import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from lake_fixture import (
    CARRY_OVER,
    DAY,
    EVENTS,
    MONTH,
    build_lake,
    events_path,
    landing_path,
    write_carry_over,
    write_consolidated,
)
from viz_tiles import cli
from viz_tiles.contract import DAY_US
from viz_tiles.write import find_index

KEY = {"provider": "binance", "market": "spot", "asset": "BTCUSDT"}
NEXT_DAY = "2017-08-19"
ID_SHIFT = 100_000


def run(roots, *args) -> int:
    return cli.main(["--mode", "tiles", *args], roots.env())


def two_day_lake(base):
    """El lago del fixture con los mismos ticks repetidos al día siguiente."""
    roots, ticks, pending = build_lake(base)
    second = [
        {**t, "id": t["id"] + ID_SHIFT, "time": t["time"] + DAY_US} for t in ticks
    ]
    write_consolidated(landing_path(roots), ticks + second, row_group_size=500)
    return roots, ticks + second, pending


def test_a_month_runs_one_pass_and_closes_each_day(tmp_path):
    roots, _, _ = two_day_lake(tmp_path)
    assert run(roots, "--from", "2017-08") == 0
    for day in (DAY, NEXT_DAY):
        index = find_index(roots.tiles, **KEY, day=date.fromisoformat(day))
        # El row group que cruza la medianoche reparte sus ticks entre los dos días.
        assert index["ticks"] == 4735
    assert (
        json.loads((Path(roots.tiles) / "latest.json").read_text())["day"] == NEXT_DAY
    )


def test_only_the_row_groups_of_the_requested_day_are_read(tmp_path, monkeypatch):
    roots, _, _ = two_day_lake(tmp_path)
    groups = pq.ParquetFile(landing_path(roots)).num_row_groups
    reads = []
    original = pq.ParquetFile.read_row_group

    def counting(self, i, *args, **kwargs):
        if "quantity" in kwargs.get("columns", ()):  # lectura de ticks, no de eventos
            reads.append(i)
        return original(self, i, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counting)
    assert run(roots, "--day", NEXT_DAY) == 0
    # Los row groups del primer día no se leen (estadísticas de transact_time).
    assert groups == 19
    assert reads and min(reads) >= 9
    assert len(reads) <= 10


def test_each_events_file_is_read_once_per_month_not_once_per_day(
    tmp_path, monkeypatch
):
    """TRD-viz §8.1: dos días del mes leen los row groups de cada θ una sola vez."""
    roots, _, _ = two_day_lake(tmp_path)
    expected = sum(
        pq.ParquetFile(events_path(roots, theta, MONTH, EVENTS)).num_row_groups
        for theta in (100_000, 250_000, 500_000, 1_000_000, 2_000_000)
    )
    reads = []
    original = pq.ParquetFile.read_row_group

    def counting(self, i, *args, **kwargs):
        if "confirm_time" in kwargs.get("columns", ()):  # lectura de eventos del mes
            reads.append(i)
        return original(self, i, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counting)
    assert run(roots, "--from", "2017-08") == 0
    assert expected > 5 and len(reads) == expected


def test_a_second_day_with_only_the_tail_has_the_provisional_state(tmp_path):
    roots, _, _ = two_day_lake(tmp_path)
    assert run(roots, "--from", "2017-08") == 0
    index = find_index(roots.tiles, **KEY, day=date(2017, 8, 19))
    # El candidato de cada θ cayó el día anterior: la cola del 19 es provisional desde 0.
    assert [t["provisional_from_s"] for t in index["thetas"]] == [0.0] * 5


def test_a_broken_chain_marks_the_theta_missing_and_writes_the_day(tmp_path):
    roots, _, pending = build_lake(tmp_path)
    # El carry-over de septiembre cerró el pendiente, pero L2 no escribió su events.parquet.
    info = pending[100_000]
    write_carry_over(
        events_path(roots, 100_000, (2017, 9), CARRY_OVER),
        100_000,
        (2017, 9),
        direction=-1,
        extreme=info["extreme"],
        pending=None,
    )
    assert run(roots, "--day", DAY) == 1
    index = find_index(roots.tiles, **KEY, day=date(2017, 8, 18))
    assert index["missing_thetas"] == ["0.00100000"]
    assert len(index["thetas"]) == 4


@pytest.mark.parametrize("flag", [[], ["--force"]])
def test_a_gap_day_inside_the_month_is_a_missing_input(tmp_path, flag):
    roots, ticks, _ = build_lake(tmp_path)
    # Ticks el 18 y el 20: el 19 es un hueco dentro de lo que L1 cubre.
    third = [
        {**t, "id": t["id"] + ID_SHIFT, "time": t["time"] + 2 * DAY_US} for t in ticks
    ]
    write_consolidated(landing_path(roots), ticks + third)
    assert run(roots, "--from", "2017-08", *flag) == 1
    assert find_index(roots.tiles, **KEY, day=date(2017, 8, 18)) is not None
    assert find_index(roots.tiles, **KEY, day=date(2017, 8, 19)) is None
    assert find_index(roots.tiles, **KEY, day=date(2017, 8, 20)) is not None
