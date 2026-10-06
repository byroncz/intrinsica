import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow.fs as pafs
import pytest
from viz_helpers import DOWN_T, UP, UP_T, events, ticks_batch, timed_events
from viz_tiles.contract import (
    INDEX_FIELDS,
    LATEST_FIELDS,
    LEVELS,
    TILES_VERSION,
    price_scale,
)
from viz_tiles.direction import direction_tiles
from viz_tiles.events import (
    ConfirmAccumulator,
    Confirmations,
    EventsBuffer,
    event_rows,
)
from viz_tiles.reduce import day_start_us, reduce_day
from viz_tiles.write import ThetaTiles, day_dir, read_index, write_day

DAY = date(2026, 8, 31)
SCALE = price_scale("BTCUSDT")
KEY = {"provider": "binance", "market": "spot", "asset": "BTCUSDT"}
ROWS = [(11, 1, 100, 1), (21, 30, 102, "0.5"), (31, 70, 103, 1), (56, 80, 104, 1)]
WHEN = datetime(2026, 10, 5, 17, 0, tzinfo=UTC)


def theta(
    name: str,
    reduction,
    acc: ConfirmAccumulator,
    buffer: EventsBuffer,
    pending=None,
) -> ThetaTiles:
    table = timed_events(DAY, UP_T, DOWN_T)
    rows, confirm_us = event_rows(table, pending, False, day_start_us(DAY))
    acc.add(confirm_us)
    buffer.add(rows)
    return ThetaTiles(
        theta=name,
        events=len(rows),
        provisional_from_s=None,
        direction=direction_tiles(reduction.last_ids, table, pending),
    )


def write(root, day=DAY, rows=ROWS, **kwargs):
    reduction = reduce_day([ticks_batch(day, rows)], day, SCALE)
    acc = ConfirmAccumulator()
    buffer = EventsBuffer()
    thetas = [
        theta("0.00010000", reduction, acc, buffer),
        theta("0.05000000", reduction, acc, buffer),
    ]
    args = {
        **KEY,
        "day": day,
        "reduction": reduction,
        "confirmations": acc.finish(),
        "events": buffer,
        "thetas": thetas,
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
    block = direction_tiles(reduction.last_ids, events(UP, DOWN_T[:4]))
    for w in LEVELS:
        price = np.fromfile(f"{directory}/{index['price'][str(w)]}", dtype="<i4")
        volume = np.fromfile(f"{directory}/{index['volume'][str(w)]}", dtype="<f4")
        dirs = np.fromfile(f"{directory}/{index['dir'][str(w)]}", dtype="u1")
        np.testing.assert_array_equal(price, reduction.price[w])
        np.testing.assert_array_equal(volume, reduction.volume[w])
        # Un bloque de w bytes por θ, en el orden de `thetas` del índice.
        assert len(dirs) == len(index["thetas"]) * w
        for k in range(len(index["thetas"])):
            np.testing.assert_array_equal(dirs[k * w : (k + 1) * w], block[w])


def test_price_file_is_two_integer_blocks(tmp_path):
    """`price-<w>.i32`: tiempos uint32 (ms) y precios int32 (1/price_scale)."""
    _, index = write(tmp_path)
    directory = day_dir(tmp_path, **KEY, day=DAY)
    w = 128
    raw = np.fromfile(f"{directory}/{index['price'][str(w)]}", dtype="u1")
    assert len(raw) == 32 * w
    t = np.frombuffer(raw[: 16 * w].tobytes(), dtype="<u4")
    p = np.frombuffer(raw[16 * w :].tobytes(), dtype="<i4")
    assert index["price_scale"] == SCALE
    # Columna 0 de w = 128 (675 s): primero, mínimo, máximo y último de ROWS.
    assert t[:4].tolist() == [1000, 1000, 80_000, 80_000]
    assert p[:4].tolist() == [10_000, 10_000, 10_400, 10_400]
    # Columna 1: vacía; el inicio de la columna y el centinela.
    assert t[4:8].tolist() == [675_000] * 4
    assert (p[4:8] == -(2**31)).all()


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
    assert index["price_scale"] == SCALE
    assert index["price"]["128"] == "price-128.i32"
    assert index["dir"]["2048"] == "dir-2048.u8"
    assert index["count"]["4096"] == "count-4096.u32"
    assert index["confirms"]["128"] == "confirms-128.u8"
    assert index["simul"]["128"] == "simul-128.u8"
    assert index["events"] == "events.bin"
    assert set(index["thetas"][0]) == {
        "theta",
        "events",
        "events_offset",
        "provisional_from_s",
    }
    assert [t["events_offset"] for t in index["thetas"]] == [0, 2]


def test_a_day_has_one_file_per_level_and_kind(tmp_path):
    write(tmp_path)
    directory = day_dir(tmp_path, **KEY, day=DAY)
    names = sorted(p.name for p in Path(directory).iterdir())
    # 39 objetos por día (arreglos, página e índice), con cualquier número de θ.
    assert len(names) == 6 * 6 + 1 + 2
    assert "index.json" in names and "index.json.tmp" not in names
    assert "dir-128.u8" in names and "dir-128-0.05000000.u8" not in names


def test_dir_file_with_no_thetas_is_empty_but_present(tmp_path):
    reduction = reduce_day([], DAY, SCALE)
    index = write_day(
        tmp_path,
        **KEY,
        day=DAY,
        reduction=reduction,
        confirmations=Confirmations.empty(),
        thetas=[],
        input_hash="x",
        image_version="v",
    )
    directory = day_dir(tmp_path, **KEY, day=DAY)
    assert (Path(directory) / index["dir"]["128"]).stat().st_size == 0


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


def test_latest_of_another_series_is_not_overwritten(tmp_path):
    """La raíz es mono-activo: otra serie falla sin escribir nada ni mover `latest.json`."""
    write(tmp_path)
    before = (tmp_path / "latest.json").read_text()
    reduction = reduce_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    with pytest.raises(ValueError, match="mono-activo"):
        write_day(
            tmp_path,
            **{**KEY, "asset": "ETHUSDT"},
            day=date(2026, 9, 2),
            reduction=reduction,
            confirmations=Confirmations.empty(),
            thetas=[],
            input_hash="ab" * 32,
            image_version="0.1.0+test",
        )
    assert (tmp_path / "latest.json").read_text() == before
    assert not (tmp_path / "provider=binance/market=spot/asset=ETHUSDT").exists()


class BucketFS:
    """Imita GCS con rol por prefijo: `create_dir` pide `storage.buckets.get` y falla;
    escribir un objeto crea su ruta implícita."""

    def __init__(self) -> None:
        self._fs = pafs.LocalFileSystem()

    def create_dir(self, path, **kwargs):
        raise PermissionError("storage.buckets.get denegado")

    def open_output_stream(self, path, **kwargs):
        self._fs.create_dir(str(Path(path).parent), recursive=True)
        return self._fs.open_output_stream(path)

    def __getattr__(self, name):
        return getattr(self._fs, name)


def test_write_day_does_not_create_directories_outside_local_disk(
    tmp_path, monkeypatch
):
    import viz_tiles.write as module

    real = module.resolve_fs
    monkeypatch.setattr(
        module, "resolve_fs", lambda root: (BucketFS(), real(root)[1]), raising=True
    )
    write(tmp_path)
    directory = Path(day_dir(tmp_path, **KEY, day=DAY))
    # 37 arreglos, `index.json` e `index.html`; `latest.*` van en la raíz.
    assert len(list(directory.iterdir())) == 39
    assert (tmp_path / "latest.json").exists() and (tmp_path / "latest.html").exists()


def test_index_is_replaced_not_appended_on_rewrite(tmp_path):
    write(tmp_path)
    write(tmp_path, input_hash="cd" * 32)
    assert read_index(tmp_path, **KEY, day=DAY)["input_hash"] == "cd" * 32


def test_index_is_written_last(tmp_path, monkeypatch):
    """Si falla un tile, el día queda sin índice: no existe para el lector."""
    import viz_tiles.write as module

    calls = []
    real = module._put

    def failing(fs, path, data, metadata=None):
        calls.append(path)
        if len(calls) == 5:
            raise OSError("disco lleno")
        real(fs, path, data, metadata)

    write(tmp_path)
    monkeypatch.setattr(module, "_put", failing)
    with pytest.raises(OSError):
        write(tmp_path, input_hash="cd" * 32)
    assert not (Path(day_dir(tmp_path, **KEY, day=DAY)) / "index.json").exists()


def test_wrong_tile_length_is_rejected(tmp_path):
    reduction = reduce_day([], DAY, SCALE)
    bad = ThetaTiles("0.00010000", 0, None, {w: np.zeros(3, "u1") for w in LEVELS})
    with pytest.raises(ValueError, match="se esperaban"):
        write_day(
            tmp_path,
            **KEY,
            day=DAY,
            reduction=reduction,
            confirmations=Confirmations.empty(),
            thetas=[bad],
            input_hash="x",
            image_version="v",
        )
