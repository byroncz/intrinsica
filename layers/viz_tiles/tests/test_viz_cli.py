import io
import json
import logging
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
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
    read_events,
    write_carry_over,
    write_events,
)
from viz_helpers import TICKS_SCHEMA
from viz_tiles import cli
from viz_tiles.context import RunContext
from viz_tiles.contract import (
    DAY_MS,
    DAY_US,
    EVENT_BYTES,
    FLAG_EXTREME_CLIPPED,
    FLAG_PROVISIONAL,
    TILES_VERSION,
    price_scale,
)
from viz_tiles.ticks import day_start_us, decode_ticks, encode_day
from viz_tiles.write import find_index, read_index

KEY = {"provider": "binance", "market": "spot", "asset": "BTCUSDT"}
DAY_DATE = date(2017, 8, 18)
# θ = 2 %: el pendiente sube.
THETA_UP = 2_000_000


def run(roots, *args) -> int:
    return cli.main(["--mode", "tiles", *args], roots.env())


def findings_of(roots) -> list[dict]:
    """Los hallazgos persistidos en el lago de DQ, con `details` ya decodificado."""
    rows = []
    for path in sorted(Path(roots.dq).rglob("*.parquet")):
        rows += pq.read_table(path).to_pylist()
    for row in rows:
        row["details"] = json.loads(row["details"])
    return rows


def summaries(roots) -> list[dict]:
    found = [f for f in findings_of(roots) if f["check_type"] == "tiles_summary"]
    return sorted(found, key=lambda f: f["detected_at"])


def day_directory(roots, day=DAY) -> Path:
    return Path(roots.tiles) / f"provider=binance/market=spot/asset=BTCUSDT/day={day}"


def tile_bytes(roots, name) -> bytes:
    return (day_directory(roots) / name).read_bytes()


def events_of(roots, k: int) -> list[tuple[int, int, int, int]]:
    """Las filas `(ref_ms, confirm_ms, extremo_ms, banderas)` del θ en la posición `k`."""
    index = read_index(roots.tiles, **KEY, day=DAY_DATE)
    raw = tile_bytes(roots, index["events"])
    total = sum(t["events"] for t in index["thetas"])
    doc = index["thetas"][k]
    lo, hi = doc["events_offset"], doc["events_offset"] + doc["events"]
    sections = [np.frombuffer(raw, "<i4", total, 4 * total * i) for i in range(3)]
    flags = np.frombuffer(raw, "u1", total, 24 * total)
    return [
        (int(sections[0][i]), int(sections[1][i]), int(sections[2][i]), int(flags[i]))
        for i in range(lo, hi)
    ]


@pytest.fixture
def lake(tmp_path):
    return build_lake(tmp_path)


# -- uso ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--mode", "export", "--day", DAY],
        ["--mode", "tiles", "--day", DAY, "--from", "2017-08"],
        ["--mode", "tiles", "--to", "2017-08"],
        ["--mode", "tiles", "--day", "2017-8-18"],
        ["--mode", "tiles", "--day", "2017-02-30"],
        ["--mode", "tiles", "--from", "2017-13"],
        ["--mode", "tiles", "--from", "2017-09", "--to", "2017-08"],
        ["--mode", "tiles", "--asset", "ETHUSDT"],
        ["--day", DAY],
    ],
)
def test_usage_errors_exit_2(lake, args):
    assert cli.main(args, lake[0].env()) == 2


@pytest.mark.parametrize(
    "missing",
    ["VIZ_LANDING_ROOT", "VIZ_EVENTS_ROOT", "VIZ_TILES_ROOT", "VIZ_DQ_ROOT"],
)
def test_missing_root_exits_2(lake, missing, capsys):
    env = {k: v for k, v in lake[0].env().items() if k != missing}
    assert cli.main(["--mode", "tiles", "--day", DAY], env) == 2
    assert missing in capsys.readouterr().err


# -- un día ------------------------------------------------------------------


def test_day_builds_tiles_from_the_fixture(lake, caplog):
    roots, ticks, _pending = lake
    with caplog.at_level(logging.INFO):
        assert run(roots, "--day", DAY) == 0

    index = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert index["ticks"] == len(ticks) == 4735
    assert index["tiles_version"] == TILES_VERSION
    assert (index["first_agg_trade_id"], index["last_agg_trade_id"]) == (
        ticks[0]["id"],
        ticks[-1]["id"],
    )
    assert [t["theta"] for t in index["thetas"]] == [
        "0.00100000",
        "0.00250000",
        "0.00500000",
        "0.01000000",
        "0.02000000",
    ]
    assert index["missing_thetas"] == []
    # El candidato de cada pendiente cae dentro del día: la cola es provisional.
    assert all(t["provisional_from_s"] is not None for t in index["thetas"])

    # `ticks.bin` decodifica a los ticks de L1, uno por uno.
    t0 = day_start_us(DAY_DATE)
    got = decode_ticks(tile_bytes(roots, "ticks.bin"), index["ticks"])
    assert got.time_ms.tolist() == [(t["time"] - t0) // 1000 for t in ticks]
    assert got.price.tolist() == [int(t["price"] * 100) for t in ticks]
    assert got.quantity.tolist() == [10**8] * len(ticks)

    (summary,) = summaries(roots)
    assert (summary["layer"], summary["mode"], summary["stage"]) == (
        "viz",
        "tiles",
        "canonical",
    )
    assert (summary["year"], summary["month"]) == MONTH
    assert summary["metric_value"] == 4735
    details = summary["details"]
    assert details["day"] == DAY and details["skipped"] is False
    assert details["objects"] == 4
    sizes = {p.name: p.stat().st_size for p in day_directory(roots).iterdir()}
    assert details["bytes"] == sum(sizes.values())
    assert details["ticks_bytes"] == sizes["ticks.bin"]
    assert details["events_bytes"] == sizes["events.bin"]
    assert details["page_bytes"] == sizes["index.html"]
    assert details["input_hash"] == index["input_hash"]
    assert details["content_hash"] == index["content_hash"]
    assert len(details["provisional_thetas"]) == 5
    assert details["provisional_tail"] is True
    probe = [m for m in caplog.messages if m.startswith("sonda: unit=2017-08-18")]
    assert len(probe) == 1
    assert "ticks=4735 wall_s=" in probe[0] and "rss_mib=" in probe[0]
    assert "ticks_bytes=" in probe[0] and "page_bytes=" in probe[0]
    assert any('"check_type": "tiles_summary"' in m for m in caplog.messages)


def test_ticks_file_matches_a_direct_encoding_of_l1(lake):
    roots, ticks, _ = lake
    assert run(roots, "--day", DAY) == 0
    batch = pa.record_batch(
        [
            pa.array([t["id"] for t in ticks], pa.int64()),
            pa.array([t["price"] for t in ticks], pa.decimal128(18, 8)),
            pa.array([Decimal(1)] * len(ticks), pa.decimal128(18, 8)),
            pa.array([t["time"] for t in ticks], pa.int64()),
        ],
        schema=TICKS_SCHEMA,
    )
    expected = io.BytesIO()
    encode_day([batch], DAY_DATE, price_scale("BTCUSDT"), expected)
    assert tile_bytes(roots, "ticks.bin") == expected.getvalue()


# -- idempotencia ------------------------------------------------------------


def test_second_run_skips_and_writes_nothing(lake, caplog):
    roots, _, _ = lake
    assert run(roots, "--day", DAY) == 0
    directory = day_directory(roots)
    before = {
        p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()
    }
    latest = (Path(roots.tiles) / "latest.json").stat().st_mtime_ns
    time.sleep(0.01)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert run(roots, "--day", DAY) == 0
    after = {
        p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()
    }
    assert after == before
    assert (Path(roots.tiles) / "latest.json").stat().st_mtime_ns == latest
    assert any("al día" in m for m in caplog.messages)
    assert any(
        m.startswith("sonda: unit=2017-08-18 ticks=4735") for m in caplog.messages
    )

    first, second = summaries(roots)
    assert first["details"]["skipped"] is False
    assert second["details"]["skipped"] is True
    assert second["details"]["content_hash"] == first["details"]["content_hash"]
    assert second["details"]["objects"] == 4


def test_force_rewrites_with_the_same_content(lake):
    roots, _, _ = lake
    assert run(roots, "--day", DAY) == 0
    first = read_index(roots.tiles, **KEY, day=DAY_DATE)
    (day_directory(roots) / "ticks.bin").unlink()

    # Sin --force el día está al día y no se toca, aunque falte un archivo.
    assert run(roots, "--day", DAY) == 0
    assert not (day_directory(roots) / "ticks.bin").exists()

    assert run(roots, "--day", DAY, "--force") == 0
    assert (day_directory(roots) / "ticks.bin").exists()
    again = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert again["content_hash"] == first["content_hash"]
    assert again["input_hash"] == first["input_hash"]
    assert [s["details"]["skipped"] for s in summaries(roots)] == [False, True, False]


def test_a_changed_input_file_rebuilds_the_day(lake):
    roots, _, _ = lake
    assert run(roots, "--day", DAY) == 0
    first = read_index(roots.tiles, **KEY, day=DAY_DATE)

    # Los mismos eventos con otro tamaño de row group: otro archivo, mismos tiles.
    theta = 100_000
    rows = read_events()[theta][:-1]
    write_events(
        events_path(roots, theta, MONTH, EVENTS), theta, rows, row_group_size=5
    )
    assert run(roots, "--day", DAY) == 0
    again = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert again["input_hash"] != first["input_hash"]
    assert again["content_hash"] == first["content_hash"]
    assert [s["details"]["skipped"] for s in summaries(roots)] == [False, False]


# -- entradas faltantes --------------------------------------------------------


def test_missing_consolidated_fails_without_writing(lake):
    roots, _, _ = lake
    landing_path(roots).unlink()
    assert run(roots, "--day", DAY) == 1
    assert not list(Path(roots.tiles).rglob("*"))
    (missing,) = findings_of(roots)
    assert missing["check_type"] == "input_missing"
    assert (missing["severity"], missing["status"]) == ("error", "fail")
    assert missing["details"]["what"] == "l1"
    assert missing["details"]["day"] == DAY
    assert (missing["layer"], missing["mode"], missing["stage"]) == (
        "viz",
        "tiles",
        "canonical",
    )


def test_missing_events_fails_without_writing(lake):
    roots, _, _ = lake
    for theta in read_events():
        events_path(roots, theta, MONTH, EVENTS).unlink()
    assert run(roots, "--day", DAY) == 1
    assert not list(Path(roots.tiles).rglob("*"))
    (missing,) = findings_of(roots)
    assert missing["check_type"] == "input_missing"
    assert missing["details"]["what"] == "events"
    assert missing["details"]["day"] == DAY


def test_day_without_ticks_fails(lake):
    roots, _, _ = lake
    assert run(roots, "--day", "2017-08-19") == 1
    (missing,) = findings_of(roots)
    assert missing["check_type"] == "input_missing"
    assert missing["details"]["what"] == "ticks"
    assert missing["details"]["day"] == "2017-08-19"
    assert not (Path(roots.tiles) / "latest.json").exists()


def test_theta_without_carry_over_is_a_missing_theta(lake):
    roots, _, _ = lake
    events_path(roots, 500_000, MONTH, CARRY_OVER).unlink()
    # El día se escribe con los demás θ y la corrida termina con código 1.
    assert run(roots, "--day", DAY) == 1
    index = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert index["missing_thetas"] == ["0.00500000"]
    assert [t["theta"] for t in index["thetas"]] == [
        "0.00100000",
        "0.00250000",
        "0.01000000",
        "0.02000000",
    ]
    # El θ sin carry-over no aporta eventos: `events.bin` trae los de los otros cuatro.
    assert sum(t["events"] for t in index["thetas"]) * EVENT_BYTES == len(
        tile_bytes(roots, "events.bin")
    )
    (missing,) = [f for f in findings_of(roots) if f["check_type"] == "input_missing"]
    assert missing["details"]["what"] == "carry_over"
    assert missing["details"]["theta"] == "0.00500000"


# -- rango -------------------------------------------------------------------


def test_month_range_builds_only_the_days_l1_covers(lake):
    roots, _, _ = lake
    # El fixture tiene un solo día de agosto de 2017: los demás días del mes
    # (anteriores y posteriores a los ticks) no se piden, así que el mes sale con 0.
    assert run(roots, "--from", "2017-08") == 0
    series = Path(roots.tiles) / "provider=binance/market=spot/asset=BTCUSDT"
    assert [p.name for p in series.iterdir()] == [f"day={DAY}"]
    assert read_index(roots.tiles, **KEY, day=DAY_DATE)["ticks"] == 4735
    assert run(roots, "--from", "2017-08", "--to", "2017-08") == 0
    assert [s["details"]["skipped"] for s in summaries(roots)] == [False, True]


def test_range_continues_after_a_month_that_fails(lake):
    roots, _, _ = lake
    assert run(roots, "--from", "2017-07", "--to", "2017-08") == 1
    assert find_index(roots.tiles, **KEY, day=DAY_DATE) is not None
    (missing,) = [f for f in findings_of(roots) if f["check_type"] == "input_missing"]
    assert (missing["month"], missing["details"]["what"]) == (7, "l1")


def test_default_mode_runs_the_previous_month(lake, monkeypatch):
    roots, _, _ = lake
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 9, 5))
    assert run(roots) == 0
    assert find_index(roots.tiles, **KEY, day=DAY_DATE)["ticks"] == 4735


# -- cola provisional y cadena de carry-overs ------------------------------------


def next_month_carry(roots, pending, extreme):
    """El carry-over de 2017-09 de cada θ: sigue el pendiente con ese candidato."""
    for theta, info in pending.items():
        write_carry_over(
            events_path(roots, theta, (2017, 9), CARRY_OVER),
            theta,
            (2017, 9),
            direction=info["direction"],
            extreme=extreme(info),
            pending=(info["row"], info["row"]),
        )


def test_chain_moving_the_candidate_out_of_the_month_rebuilds_the_day(lake):
    roots, ticks, pending = lake
    assert run(roots, "--day", DAY) == 0
    first = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert all(t["provisional_from_s"] is not None for t in first["thetas"])

    # La primera carry-over de la cadena ya mueve el candidato a septiembre:
    # el día pasa a ser overshoot certero y sus tiles cambian.
    def beyond(info):
        return {
            "id": ticks[-1]["id"] + 10_000,
            "price": info["extreme"]["price"],
            "time": ticks[-1]["time"] + 40 * DAY_US,
        }

    next_month_carry(roots, pending, beyond)
    assert run(roots, "--day", DAY) == 0
    again = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert again["input_hash"] != first["input_hash"]
    assert again["content_hash"] != first["content_hash"]
    assert all(t["provisional_from_s"] is None for t in again["thetas"])

    # Con el candidato fuera del mes, el pendiente (que sube) ya no es provisional y
    # su extremo cae después del día: sale recortado al borde.
    *_, (_, _, extreme_ms, flags) = events_of(roots, 4)
    assert extreme_ms == DAY_MS
    assert flags & FLAG_EXTREME_CLIPPED and not flags & FLAG_PROVISIONAL

    # Sin más cambios, la siguiente corrida salta el día.
    assert run(roots, "--day", DAY) == 0
    assert summaries(roots)[-1]["details"]["skipped"] is True


def test_chain_growth_with_the_candidate_inside_the_month_keeps_the_tiles(lake):
    roots, _, pending = lake
    assert run(roots, "--day", DAY) == 0
    first = read_index(roots.tiles, **KEY, day=DAY_DATE)

    # La misma carry-over que la del mes: el candidato no se movió.
    next_month_carry(roots, pending, lambda info: info["extreme"])
    assert run(roots, "--day", DAY) == 0
    again = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert again["input_hash"] != first["input_hash"]
    assert again["content_hash"] == first["content_hash"]
    assert [t["provisional_from_s"] for t in again["thetas"]] == [
        t["provisional_from_s"] for t in first["thetas"]
    ]


def test_closed_chain_uses_the_definitive_extreme(lake):
    roots, ticks, pending = lake
    assert run(roots, "--day", DAY) == 0
    first = read_index(roots.tiles, **KEY, day=DAY_DATE)

    # θ = 2 %: L2 cierra el pendiente en septiembre con un extremo posterior a
    # todos los ticks del día (y al candidato del mes).
    far = ticks[-1]["id"] + 500
    row = dict(pending[THETA_UP]["row"])
    row |= {
        "extreme_price": "5000.00000000",
        "extreme_time": str(ticks[-1]["time"] + 1_000_000),
        "extreme_agg_trade_id": str(far),
    }
    write_events(events_path(roots, THETA_UP, (2017, 9), EVENTS), THETA_UP, [row])
    write_carry_over(
        events_path(roots, THETA_UP, (2017, 9), CARRY_OVER),
        THETA_UP,
        (2017, 9),
        direction=-1,
        extreme={"id": far + 5, "price": "4900", "time": ticks[-1]["time"] + 2_000_000},
        pending=None,
    )
    assert run(roots, "--day", DAY) == 0
    again = read_index(roots.tiles, **KEY, day=DAY_DATE)
    assert again["input_hash"] != first["input_hash"]
    by_theta = {t["theta"]: t for t in again["thetas"]}
    assert by_theta["0.02000000"]["provisional_from_s"] is None
    # Los demás θ siguen con su cola provisional.
    assert by_theta["0.00100000"]["provisional_from_s"] is not None

    # El evento ya cerrado termina en su extremo definitivo, un segundo después del último tick.
    *_, (_, _, extreme_ms, flags) = events_of(roots, 4)
    assert (
        extreme_ms == (ticks[-1]["time"] + 1_000_000 - day_start_us(DAY_DATE)) // 1000
    )
    assert not flags & (FLAG_PROVISIONAL | FLAG_EXTREME_CLIPPED)
    assert again["content_hash"] != first["content_hash"]


# -- revisión hacia atrás ------------------------------------------------------


def stored_index(provisional: bool) -> dict:
    return {
        "thetas": [
            {"theta": "0.00100000", "provisional_from_s": 5.0 if provisional else None}
        ]
    }


def test_review_walks_back_over_provisional_months(monkeypatch):
    stored = {
        # Julio y agosto cierran con cola provisional; junio es definitivo.
        date(2017, 8, 31): True,
        date(2017, 8, 30): True,
        date(2017, 8, 29): False,
        date(2017, 7, 31): True,
        date(2017, 7, 30): False,
        date(2017, 6, 30): False,
    }

    def fake_find(root, provider, market, asset, day):
        return stored_index(stored[day]) if day in stored else None

    monkeypatch.setattr(cli, "find_index", fake_find)
    ctx = RunContext("r", "v", "l", "e", "t", "d")
    assert cli.review_months(ctx, (2017, 9)) == [
        ((2017, 7), [date(2017, 7, 31)]),
        ((2017, 8), [date(2017, 8, 30), date(2017, 8, 31)]),
    ]
    # Si el mes previo es definitivo, no se revisa nada.
    assert cli.review_months(ctx, (2017, 7)) == []
    assert cli.review_months(ctx, (2017, 8)) == [((2017, 7), [date(2017, 7, 31)])]


def test_review_stops_at_a_month_without_index(monkeypatch):
    monkeypatch.setattr(cli, "find_index", lambda *a, **k: None)
    ctx = RunContext("r", "v", "l", "e", "t", "d")
    assert cli.review_months(ctx, (2017, 9)) == []
