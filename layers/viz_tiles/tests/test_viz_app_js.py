"""La vista (`site/app.js`) con el uPlot real en Node y un DOM de mentira.

No dibuja: atrapa errores de la lógica (decodificación, niveles, θ, zoom, hooks)
sin un navegador. Se salta si no hay `node` (GitHub Actions lo trae). Lo que solo
un navegador comprueba (pintura, medidas, tiempos) queda en la Evaluación
ergonómica de la card.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from lake_fixture import DAY, build_lake
from viz_tiles import cli
from viz_tiles.contract import EMPTY_PRICE, PAGE_FILE

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "harness.cjs"

pytestmark = pytest.mark.skipif(NODE is None, reason="sin node en el PATH")


@pytest.fixture(scope="module")
def day_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("lake")
    roots, _, _ = build_lake(base)
    assert cli.main(["--mode", "tiles", "--day", DAY], roots.env()) == 0
    return (
        Path(roots.tiles) / "provider=binance/market=spot/asset=BTCUSDT" / f"day={DAY}"
    )


def view(day_dir, *flags) -> dict:
    done = subprocess.run(
        [NODE, str(HARNESS), str(day_dir / PAGE_FILE), *flags],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def normal(day_dir):
    return view(day_dir)


def page_with(day_dir, tmp_path, **changes) -> Path:
    """Una copia de la página del día con otros campos en el índice (los mismos arreglos)."""
    from viz_tiles import render

    index = json.loads((day_dir / "index.json").read_text()) | changes
    names = sorted(
        n for kind in ("price", "volume", "dir") for n in index[kind].values()
    )
    out = tmp_path / "page"
    out.mkdir()
    arrays = [(n, (day_dir / n).read_bytes()) for n in names]
    (out / PAGE_FILE).write_bytes(render.render_day(index, arrays))
    return out


def step(out, label):
    return next(s for s in out["steps"] if s["label"] == label)


def test_view_runs_without_errors_or_network(normal):
    # Un `fetch` o un XMLHttpRequest fallarían aquí: el arnés no los define.
    assert normal["errors"] == []
    assert not [m for m in normal["logs"] if m.startswith(("ERROR", "WARN"))]
    assert normal["day"] == DAY
    assert normal["dataScriptRemoved"] is True


def test_status_strip_shows_the_theta_catalog_in_normal_mode(normal):
    assert len(normal["options"]) == 5
    assert normal["options"][0].startswith("0.00100000")
    start = step(normal, "inicio")
    assert start["mode"] == "NORMAL" and start["reasons"] == ""
    assert normal["updated"].endswith("Z")


def test_decodes_the_level_that_covers_the_viewport(normal, day_dir):
    start = step(normal, "inicio")
    # 1 500 px de ancho: el nivel 2 048 (el primero con columnas ≥ píxeles).
    assert start["level"].startswith("nivel 2048")
    assert start["priceLen"] == 4 * 2048 and start["volLen"] == 2048
    # Solo se decodificó ese nivel: precio, volumen y los 5 θ de dirección.
    assert start["metrics"]["decoded_bytes"] == 65536 + 8192 + 5 * 2048
    assert start["x"] == [0, 86400] and start["vx"] == [0, 86400]


def test_decoded_series_equals_the_tile_files(normal, day_dir):
    w = 2048
    raw = np.fromfile(day_dir / f"price-{w}.i32", dtype="<i4")
    t, p = raw[: 4 * w].view("<u4"), raw[4 * w :]
    valid = p != EMPTY_PRICE
    data = step(normal, "inicio")["data"]
    assert data["points"] == 4 * w
    assert data["nulls"] == int((~valid).sum())
    assert data["yMin"] == pytest.approx(p[valid].min() / 100)
    assert data["yMax"] == pytest.approx(p[valid].max() / 100)
    assert data["xLast"] == pytest.approx(t[-1] / 1000)
    volume = np.fromfile(day_dir / f"volume-{w}.f32", dtype="<f4")
    assert data["volSum"] == pytest.approx(float(volume.sum()), rel=1e-6)


def test_volume_panel_takes_20_to_25_percent_of_the_height(normal):
    price, volume = normal["panelHeights"]
    assert 0.20 <= volume / (price + volume) <= 0.25


def test_first_paint_and_decoded_size_go_to_the_console(normal):
    start = step(normal, "inicio")
    assert start["metrics"]["first_paint_ms"] is not None
    assert any(m.startswith("viz: primer trazo") for m in normal["logs"])
    assert any("B decodificados" in m for m in normal["logs"])


def test_theta_change_repaints_without_moving_the_y_axis(normal):
    before, after = step(normal, "antes-θ"), step(normal, "después-θ")
    assert before["y"] == after["y"]
    assert before["x"] == after["x"]
    assert len(after["metrics"]["theta_change_ms"]) == 1
    assert any(m.startswith("viz: cambio de θ en") for m in normal["logs"])
    assert normal["regionFills"] > 0
    # El bloque ya estaba en memoria: no se decodificó nada más.
    assert after["metrics"]["decoded_bytes"] == before["metrics"]["decoded_bytes"]


def test_tooltip_shows_hour_min_max_and_state(normal):
    tip = normal["tooltip"]
    # Dos decimales: log10(price_scale) con price_scale = 100.
    assert re.search(r"^\d\d:\d\d:\d\d\.\d{3} – \d\d:\d\d:\d\d\.\d{3} UTC", tip)
    assert re.search(r"mín \d+\.\d{2}   máx \d+\.\d{2}", tip)
    assert "θ 0.00250000: " in tip
    assert normal["tooltipHidden"] is True


def test_zoom_uses_the_finest_level_and_reset_goes_back(normal):
    zoom = step(normal, "zoom")
    assert zoom["x"] == [36000, 39600] and zoom["vx"] == zoom["x"]
    assert zoom["level"].startswith("nivel 4096")
    assert zoom["priceLen"] == 4 * 4096 and zoom["volLen"] == 4096
    # Zoom explícito del usuario: es lo único que mueve el eje Y.
    assert zoom["y"] != step(normal, "antes-θ")["y"]
    reset = step(normal, "reset")
    assert reset["x"] == [0, 86400] and reset["level"].startswith("nivel 2048")
    assert reset["y"] == step(normal, "antes-θ")["y"]
    assert step(normal, "resize")["x"] == [0, 86400]


def test_missing_theta_is_a_visible_gap_with_text(day_dir, tmp_path):
    """Un θ de `missing_thetas` se ofrece, marca el modo degradado y nunca se rellena."""
    out = page_with(day_dir, tmp_path, missing_thetas=["0.10000000"])
    result = view(out, "--pick-last")
    assert result["errors"] == []
    assert result["options"][-1].startswith("⚠ 0.10000000")
    gap = step(result, "θ-sin-datos")
    assert gap["mode"] == "⚠ DEGRADADO"
    assert "0.10000000" in gap["reasons"] and "sin datos" in gap["reasons"]
    assert any("sin datos en L2" in str(m) for m in result["messages"])


def test_a_missing_tile_degrades_the_day_and_falls_back(day_dir):
    result = view(day_dir, "--drop", "price-2048.i32")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["mode"] == "⚠ DEGRADADO"
    assert "falta price-2048.i32" in start["reasons"]
    assert start["level"].startswith("nivel 4096")  # el nivel siguiente que sí hay


def test_unknown_major_version_is_rejected(day_dir, tmp_path):
    result = view(page_with(day_dir, tmp_path, tiles_version="2.0.0"))
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["mode"] == "⚠ DEGRADADO" and "no soportada" in start["reasons"]
    assert start["metrics"]["decoded_bytes"] == 0


def test_a_day_without_thetas_still_draws_price_and_volume(day_dir, tmp_path):
    result = view(page_with(day_dir, tmp_path, thetas=[]))
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["priceLen"] == 4 * 2048 and start["mode"] == "⚠ DEGRADADO"
    assert "sin θ" in start["reasons"]
