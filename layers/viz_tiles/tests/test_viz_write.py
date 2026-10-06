import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow.fs as pafs
import pytest
from viz_helpers import DOWN_T, UP_T, month_events, ticks_batch
from viz_tiles.contract import (
    EVENT_BYTES,
    INDEX_FIELDS,
    LATEST_FIELDS,
    TILES_VERSION,
    price_scale,
)
from viz_tiles.events import EventsBuffer, event_rows
from viz_tiles.ticks import day_start_us, decode_ticks, encode_day
from viz_tiles.write import ThetaEvents, day_dir, read_index, write_day

DAY = date(2026, 8, 31)
SCALE = price_scale("BTCUSDT")
KEY = {"provider": "binance", "market": "spot", "asset": "BTCUSDT"}
ROWS = [(11, 1, 100, 1), (21, 30, 102, "0.5"), (31, 70, 103, 1), (56, 80, 104, 1)]
WHEN = datetime(2026, 10, 5, 17, 0, tzinfo=UTC)


def theta(name: str, buffer: EventsBuffer, pending=None) -> ThetaEvents:
    rows = event_rows(
        month_events(DAY, UP_T, DOWN_T), pending, False, day_start_us(DAY)
    )
    buffer.add(rows)
    return ThetaEvents(theta=name, events=len(rows), provisional_from_s=None)


def write(root, day=DAY, rows=ROWS, **kwargs):
    ticks = encode_day([ticks_batch(day, rows)], day, SCALE)
    buffer = EventsBuffer()
    thetas = [theta("0.00010000", buffer), theta("0.05000000", buffer)]
    args = {
        **KEY,
        "day": day,
        "ticks": ticks,
        "events": buffer,
        "thetas": thetas,
        "input_hash": "ab" * 32,
        "image_version": "0.1.0+test",
        "generated_at": WHEN,
        **kwargs,
    }
    return ticks, write_day(root, **args)


def test_round_trip_returns_the_ticks_and_the_events_of_the_day(tmp_path):
    ticks, index = write(tmp_path)
    directory = day_dir(tmp_path, **KEY, day=DAY)
    assert directory.endswith(
        "provider=binance/market=spot/asset=BTCUSDT/day=2026-08-31"
    )
    assert read_index(tmp_path, **KEY, day=DAY) == index
    raw = Path(f"{directory}/{index['ticks_file']}").read_bytes()
    assert raw == ticks.to_bytes()
    got = decode_ticks(raw, index["ticks"])
    assert got.time_ms.tolist() == [1000, 30_000, 70_000, 80_000]
    assert got.price.tolist() == [10_000, 10_200, 10_300, 10_400]
    assert got.quantity.tolist() == [100_000_000, 50_000_000, 100_000_000, 100_000_000]
    # Dos θ de dos eventos: cuatro eventos de 13 bytes.
    events = Path(f"{directory}/{index['events']}").read_bytes()
    assert len(events) == 4 * EVENT_BYTES
    assert np.frombuffer(events, "<i4", 4, 0).tolist() == [500, 50_000, 500, 50_000]


def test_index_has_every_contract_field_in_order(tmp_path):
    _ticks, index = write(tmp_path)
    assert list(index) == [name for name, _ in INDEX_FIELDS]
    types = {"string": str, "integer": int, "array": list, "object": dict}
    for name, kind in INDEX_FIELDS:
        assert isinstance(index[name], types[kind]), name
    assert index["tiles_version"] == TILES_VERSION == "2.0.0"
    assert index["t0"] == day_start_us(DAY)
    assert index["ticks"] == len(ROWS)
    assert (index["first_agg_trade_id"], index["last_agg_trade_id"]) == (11, 56)
    assert index["price_scale"] == SCALE
    assert index["ticks_file"] == "ticks.bin" and index["events"] == "events.bin"
    assert index["input_hash"] == "ab" * 32
    assert index["generated_at"] == "2026-10-05T17:00:00Z"
    assert [t["theta"] for t in index["thetas"]] == ["0.00010000", "0.05000000"]
    assert set(index["thetas"][0]) == {
        "theta",
        "events",
        "events_offset",
        "provisional_from_s",
    }
    assert [t["events_offset"] for t in index["thetas"]] == [0, 2]
    # Ni niveles ni arreglos por columna: la 2.0.0 los eliminó.
    assert not {"levels", "price", "volume", "dir", "count", "confirms", "simul"} & set(
        index
    )


def test_a_day_has_four_objects_whatever_the_number_of_thetas(tmp_path):
    write(tmp_path)
    directory = day_dir(tmp_path, **KEY, day=DAY)
    names = sorted(p.name for p in Path(directory).iterdir())
    assert names == ["events.bin", "index.html", "index.json", "ticks.bin"]


def test_a_day_without_thetas_writes_empty_events(tmp_path):
    ticks = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    index = write_day(
        tmp_path,
        **KEY,
        day=DAY,
        ticks=ticks,
        thetas=[],
        input_hash="x",
        image_version="v",
    )
    directory = day_dir(tmp_path, **KEY, day=DAY)
    assert (Path(directory) / index["events"]).stat().st_size == 0
    assert index["thetas"] == []


def test_content_hash_is_stable_and_follows_the_files(tmp_path):
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
    ticks = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    with pytest.raises(ValueError, match="mono-activo"):
        write_day(
            tmp_path,
            **{**KEY, "asset": "ETHUSDT"},
            day=date(2026, 9, 2),
            ticks=ticks,
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
    # `ticks.bin`, `events.bin`, `index.json` e `index.html`; `latest.*` van en la raíz.
    assert len(list(directory.iterdir())) == 4
    assert (tmp_path / "latest.json").exists() and (tmp_path / "latest.html").exists()


def test_index_is_replaced_not_appended_on_rewrite(tmp_path):
    write(tmp_path)
    write(tmp_path, input_hash="cd" * 32)
    assert read_index(tmp_path, **KEY, day=DAY)["input_hash"] == "cd" * 32


def test_index_is_written_last(tmp_path, monkeypatch):
    """Si falla un archivo, el día queda sin índice: no existe para el lector."""
    import viz_tiles.write as module

    calls = []
    real = module._put

    def failing(fs, path, data, metadata=None):
        calls.append(path)
        if len(calls) == 2:  # `ticks.bin`, el segundo archivo del día
            raise OSError("disco lleno")
        real(fs, path, data, metadata)

    write(tmp_path)
    monkeypatch.setattr(module, "_put", failing)
    with pytest.raises(OSError):
        write(tmp_path, input_hash="cd" * 32)
    assert not (Path(day_dir(tmp_path, **KEY, day=DAY)) / "index.json").exists()


def test_write_day_rejects_rows_that_disagree_with_the_event_count(tmp_path):
    ticks = encode_day([ticks_batch(DAY, ROWS)], DAY, SCALE)
    bad = ThetaEvents("0.00010000", 3, None)
    with pytest.raises(ValueError, match="filas"):
        write_day(
            tmp_path,
            **KEY,
            day=DAY,
            ticks=ticks,
            thetas=[bad],
            input_hash="x",
            image_version="v",
        )


def test_ticks_are_written_section_by_section_without_joining_them(
    tmp_path, monkeypatch
):
    """`ticks.bin` sale de sus tres secciones tal cual: no se arma un buffer con el día."""
    import viz_tiles.write as module

    seen = {}
    real = module._put

    def spying(fs, path, data, metadata=None):
        seen[Path(path).name] = data
        real(fs, path, data, metadata)

    monkeypatch.setattr(module, "_put", spying)
    ticks, _ = write(tmp_path)
    assert len(seen["ticks.bin"]) == 3
    assert [bytes(p) for p in seen["ticks.bin"]] == [bytes(s) for s in ticks.sections]
