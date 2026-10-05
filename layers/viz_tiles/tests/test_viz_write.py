import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
from viz_helpers import DOWN, UP, events, ticks_batch
from viz_tiles.contract import INDEX_FIELDS, LATEST_FIELDS, LEVELS, TILES_VERSION
from viz_tiles.direction import direction_tiles
from viz_tiles.reduce import day_start_us, reduce_day
from viz_tiles.write import ThetaTiles, day_dir, read_index, write_day

DAY = date(2026, 8, 31)
KEY = {"provider": "binance", "market": "spot", "asset": "BTCUSDT"}
ROWS = [(11, 1, 100, 1), (21, 30, 102, "0.5"), (31, 70, 103, 1), (56, 80, 104, 1)]
WHEN = datetime(2026, 10, 5, 17, 0, tzinfo=UTC)


def theta(name: str, reduction, pending=None) -> ThetaTiles:
    return ThetaTiles(
        theta=name,
        events=2,
        provisional_from_s=None,
        direction=direction_tiles(reduction.last_ids, events(UP, DOWN), pending),
    )


def write(root, day=DAY, rows=ROWS, **kwargs):
    reduction = reduce_day([ticks_batch(day, rows)], day)
    args = {
        **KEY,
        "day": day,
        "reduction": reduction,
        "thetas": [
            theta("0.00010000", reduction),
            theta("0.05000000", reduction),
        ],
        "input_hash": "ab" * 32,
        "image_version": "0.1.0+test",
        "generated_at": WHEN,
        **kwargs,
    }
    return reduction, write_day(root, **args)


def test_round_trip_returns_identical_arrays(tmp_path):
    reduction, index = write(tmp_path)
    directory = day_dir(tmp_path, **KEY, day=DAY)
    assert directory.endswith(
        "provider=binance/market=spot/asset=BTCUSDT/day=2026-08-31"
    )
    on_disk = read_index(tmp_path, **KEY, day=DAY)
    assert on_disk == index
    for w in LEVELS:
        price = np.fromfile(f"{directory}/{index['price'][str(w)]}", dtype="<f4")
        volume = np.fromfile(f"{directory}/{index['volume'][str(w)]}", dtype="<f4")
        np.testing.assert_array_equal(price, reduction.price[w])
        np.testing.assert_array_equal(volume, reduction.volume[w])
        for entry in index["thetas"]:
            dirs = np.fromfile(f"{directory}/{entry['dir'][str(w)]}", dtype="u1")
            np.testing.assert_array_equal(
                dirs, direction_tiles(reduction.last_ids, events(UP, DOWN))[w]
            )


def test_index_has_every_contract_field_in_order(tmp_path):
    _, index = write(tmp_path)
    assert list(index) == [name for name, _ in INDEX_FIELDS]
    types = {"string": str, "integer": int, "array": list, "object": dict}
    for name, kind in INDEX_FIELDS:
        assert isinstance(index[name], types[kind]), name
    assert index["tiles_version"] == TILES_VERSION
    assert index["t0"] == day_start_us(DAY)
    assert index["levels"] == list(LEVELS)
    assert index["ticks"] == len(ROWS)
    assert index["input_hash"] == "ab" * 32
    assert index["generated_at"] == "2026-10-05T17:00:00Z"
    assert [t["theta"] for t in index["thetas"]] == ["0.00010000", "0.05000000"]
    assert index["thetas"][0]["dir"]["2048"] == "dir-2048-0.00010000.u8"


def test_a_day_has_one_file_per_level_kind_and_theta(tmp_path):
    write(tmp_path)
    directory = day_dir(tmp_path, **KEY, day=DAY)
    names = sorted(p.name for p in Path(directory).iterdir())
    assert len(names) == 6 + 6 + 2 * 6 + 1
    assert "index.json" in names and "index.json.tmp" not in names
    assert "dir-128-0.05000000.u8" in names


def test_content_hash_is_stable_and_follows_the_arrays(tmp_path):
    _, first = write(tmp_path / "a")
    _, again = write(tmp_path / "b", generated_at=datetime(2030, 1, 1, tzinfo=UTC))
    assert first["content_hash"] == again["content_hash"]
    _, other = write(tmp_path / "c", rows=[*ROWS, (60, 90, 105, 1)])
    assert other["content_hash"] != first["content_hash"]


def test_latest_points_to_the_newest_day_and_never_goes_back(tmp_path):
    write(tmp_path)
    latest = tmp_path / "latest.json"
    assert json.loads(latest.read_text()) == {
        "tiles_version": TILES_VERSION,
        **KEY,
        "day": "2026-08-31",
    }
    assert list(json.loads(latest.read_text())) == list(LATEST_FIELDS)
    write(tmp_path, day=date(2026, 9, 1))
    assert json.loads(latest.read_text())["day"] == "2026-09-01"
    write(tmp_path, day=date(2026, 8, 15))
    assert json.loads(latest.read_text())["day"] == "2026-09-01"
    write(tmp_path, day=date(2026, 9, 1))
    assert json.loads(latest.read_text())["day"] == "2026-09-01"


def test_index_is_replaced_not_appended_on_rewrite(tmp_path):
    write(tmp_path)
    write(tmp_path, input_hash="cd" * 32)
    assert read_index(tmp_path, **KEY, day=DAY)["input_hash"] == "cd" * 32


def test_index_is_written_last(tmp_path, monkeypatch):
    """Si falla un tile, el día queda sin índice: no existe para el lector."""
    import viz_tiles.write as module

    calls = []
    real = module._put

    def failing(fs, path, data):
        calls.append(path)
        if len(calls) == 5:
            raise OSError("disco lleno")
        real(fs, path, data)

    write(tmp_path)
    monkeypatch.setattr(module, "_put", failing)
    with pytest.raises(OSError):
        write(tmp_path, input_hash="cd" * 32)
    assert not (Path(day_dir(tmp_path, **KEY, day=DAY)) / "index.json").exists()


def test_wrong_tile_length_is_rejected(tmp_path):
    reduction = reduce_day([], DAY)
    bad = ThetaTiles("0.00010000", 0, None, {w: np.zeros(3, "u1") for w in LEVELS})
    with pytest.raises(ValueError, match="se esperaban"):
        write_day(
            tmp_path,
            **KEY,
            day=DAY,
            reduction=reduction,
            thetas=[bad],
            input_hash="x",
            image_version="v",
        )
