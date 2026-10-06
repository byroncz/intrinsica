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
APP_JS = Path(__file__).parents[1] / "site" / "app.js"
GLYPHS = ("▲", "▼")

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


def test_status_strip_shows_the_theta_catalog_and_complete_data(normal):
    assert len(normal["options"]) == 5
    assert normal["options"][0].startswith("0.00100000")
    start = step(normal, "inicio")
    assert start["status"] == "completos" and start["statusClass"] == "item"
    assert normal["updated"].endswith("Z")


def test_status_strip_says_datos_not_modo(day_dir):
    """El rótulo es "Datos" (integridad de la página), no un "Modo" que parezca un selector."""
    page = (day_dir / PAGE_FILE).read_text(encoding="utf-8")
    header = re.search(r'<header id="status">.*?</header>', page, re.DOTALL).group(0)
    assert '<span class="label">Datos</span>' in header
    assert "Modo" not in header and "NORMAL" not in page and "DEGRADADO" not in page


def test_incomplete_data_is_named_with_its_reason_in_amber(day_dir):
    result = view(day_dir, "--drop", "dir-2048.u8")
    start = step(result, "inicio")
    assert start["status"] == "incompletos: falta dir-2048.u8"
    assert start["statusClass"] == "item incomplete"
    # El ámbar lo pone la hoja de estilo sobre esa clase (la prueba lee la página real).
    page = (day_dir / PAGE_FILE).read_text(encoding="utf-8")
    assert re.search(r"#s-data-box\.incomplete b\s*\{\s*color:\s*var\(--amber\)", page)


LEGEND = (
    "tenue: confirmación · intenso: overshoot · franja arriba: alza · abajo: baja"
    " · fina: confirmación · gruesa: overshoot"
)


def test_legend_says_what_the_dark_background_shows(day_dir):
    """Sobre fondo oscuro, menos opacidad se ve más oscuro: "tenue" es la confirmación."""
    page = (day_dir / PAGE_FILE).read_text(encoding="utf-8")
    assert f">{LEGEND}<" in page
    assert "tono claro" not in page and "tono fuerte" not in page
    # La leyenda y el relleno dicen lo mismo: la confirmación es la menos opaca.
    app = APP_JS.read_text(encoding="utf-8")
    assert "(st.strong ? 0.42 : 0.18)" in app


def test_no_glyphs_mark_the_regions_or_the_tooltip(day_dir):
    """Dirección y fase las dan la franja del borde y el relleno; el tooltip, con texto."""
    result = view(day_dir, "--sweep")
    assert result["errors"] == []
    shown = result["texts"] + result["thetaLines"] + [result["tooltip"]]
    assert not [t for t in shown if any(g in t for g in GLYPHS)], shown
    # El recorrido sí pasó por regiones de las dos fases y los dos sentidos, dichas con texto.
    states = {line.split(": ", 1)[1] for line in result["thetaLines"]}
    assert {"confirmación alza", "overshoot alza"} <= states
    assert {"confirmación baja", "overshoot baja"} & states
    # Ni la plantilla ni la vista lo dibujan en ningún lado.
    assert not any(g in APP_JS.read_text(encoding="utf-8") for g in GLYPHS)
    assert not any(
        g in (day_dir / PAGE_FILE).read_text(encoding="utf-8") for g in GLYPHS
    )


AMBER = "#e3b341"


def test_a_gap_is_a_dashed_midline_no_dc_band_looks_like(normal):
    """Hueco y confirmación baja no comparten forma: uno es un trazo, la otra una franja."""
    draw = normal["firstDraw"]
    plot = step(normal, "inicio")["plot"]
    mid = plot["top"] + plot["height"] / 2
    fill, rects, gap_ys, i = None, [], [], 0
    while i < len(draw):
        name, *args = draw[i]
        if name == "fillStyle":
            fill = args[0]
        elif name == "fillRect":
            rects.append((fill, *args))
        elif name == "setLineDash" and args[0] == [4, 3]:  # marca del hueco
            assert draw[i - 1] == ["lineWidth", 1], draw[i - 1]  # 1 px
            assert draw[i + 1][0] == "moveTo"
            gap_ys.append(draw[i + 1][2])
        i += 1
    assert gap_ys, "la fixture tiene huecos y deben dibujarse"
    assert all(y == pytest.approx(mid, abs=1) for y in gap_ys)  # a media altura
    # Ninguna franja es ámbar: el ámbar solo marca lo anómalo, y con un trazo.
    assert not [r for r in rects if r[0] == AMBER]
    # Las franjas DC miden 3 o 8 px y van pegadas al borde de arriba o de abajo.
    bands = [r for r in rects if r[0].startswith("rgb(") and r[4] != plot["height"]]
    assert {r[4] for r in bands} == {3, 8}
    top, bottom = plot["top"], plot["top"] + plot["height"]
    assert all(r[2] == top or r[2] + r[4] == bottom for r in bands)


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
    assert gap["status"].startswith("incompletos: ")
    assert "0.10000000" in gap["status"] and "sin datos" in gap["status"]
    assert any("sin datos en L2" in str(m) for m in result["messages"])


def test_a_missing_tile_degrades_the_day_and_falls_back(day_dir):
    result = view(day_dir, "--drop", "price-2048.i32")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["status"].startswith("incompletos: ")
    assert "falta price-2048.i32" in start["status"]
    assert start["level"].startswith("nivel 4096")  # el nivel siguiente que sí hay


def test_a_corrupt_price_tile_falls_back_like_a_missing_one(day_dir):
    result = view(day_dir, "--corrupt", "price-2048.i32")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["status"].startswith("incompletos: ")
    assert "price-2048.i32: tamaño inesperado" in start["status"]
    assert start["level"].startswith("nivel 4096")  # el día se dibuja con el siguiente
    assert start["priceLen"] == 4 * 4096


def test_a_missing_direction_tile_is_said_not_filled_with_zeros(day_dir):
    result = view(day_dir, "--drop", "dir-2048.u8")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert (
        start["status"].startswith("incompletos: ")
        and "falta dir-2048.u8" in start["status"]
    )
    assert start["level"].startswith("nivel 2048")  # el precio sigue ahí
    assert "⚠ falta el tile de dirección de este nivel" in result["firstMessages"]
    assert "θ 0.00250000: ⚠ falta el tile de dirección" in result["tooltip"]
    assert "sin evento" not in result["tooltip"]


def test_a_missing_volume_tile_draws_no_bars_and_says_so(day_dir):
    result = view(day_dir, "--drop", "volume-2048.f32")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert (
        start["status"].startswith("incompletos: ")
        and "falta volume-2048.f32" in start["status"]
    )
    assert start["volLen"] == 2048
    assert start["data"]["volSum"] == 0  # nulls, no barras en cero
    assert "⚠ falta el tile de volumen de este nivel" in result["firstMessages"]
    assert "vol: ⚠ falta el tile" in result["tooltip"]
    assert "vol 0.0000" not in result["tooltip"]


def test_unknown_major_version_is_rejected(day_dir, tmp_path):
    result = view(page_with(day_dir, tmp_path, tiles_version="2.0.0"))
    assert result["errors"] == []
    start = step(result, "inicio")
    assert (
        start["status"].startswith("incompletos: ")
        and "no soportada" in start["status"]
    )
    assert start["metrics"]["decoded_bytes"] == 0


def test_a_day_without_thetas_still_draws_price_and_volume(day_dir, tmp_path):
    result = view(page_with(day_dir, tmp_path, thetas=[]))
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["priceLen"] == 4 * 2048 and start["status"].startswith("incompletos: ")
    assert "sin θ" in start["status"]


# ---- Línea o barras de rango por columna (umbral: 5 px por columna) -------------------

W = 4096  # el nivel más fino: el único donde una columna llega a ocupar 5 px
COL_S = 86400 / W


def span_for(plot_w, col_px):
    """Segundos visibles para que una columna del nivel 4 096 ocupe `col_px` píxeles."""
    return plot_w * COL_S / col_px


@pytest.fixture(scope="module")
def crossings(day_dir, normal):
    """Zooms que cruzan el umbral en los dos sentidos, sin cambiar de nivel."""
    plot_w = step(normal, "inicio")["plotW"]
    t0 = 36000
    spans = [span_for(plot_w, px) for px in (5.02, 4.98, 5.02, 1.0)]
    flags = []
    for span in spans:
        flags += ["--zoom", f"{t0},{t0 + span}"]
    result = view(day_dir, *flags)
    steps = [s for s in result["steps"] if s["label"].startswith("zoom:")]
    assert len(steps) == 4
    return result, steps


def test_the_full_day_is_the_m4_line(normal):
    start = step(normal, "inicio")
    assert start["metrics"]["price_draw"] == {"mode": "line", "columns": 0}
    assert start["pathOps"] is None and start["pathOpCount"] > 0
    assert "barras desde 5 px por columna" in start["level"]
    assert start["level"].endswith("dibujo: línea")


def test_zooming_in_past_5_px_per_column_swaps_the_line_for_bars(normal):
    zoom = step(normal, "zoom")  # una hora a 1 500 px: ~8 px por columna
    assert zoom["level"].startswith("nivel 4096")
    assert zoom["metrics"]["price_draw"]["mode"] == "bars"
    assert zoom["level"].endswith("dibujo: barras")
    assert any(m.startswith("viz: precio en barras") for m in normal["logs"])
    # Y al volver al día completo, la línea.
    reset = step(normal, "reset")
    assert reset["metrics"]["price_draw"] == {"mode": "line", "columns": 0}
    assert reset["pathOps"] is None
    assert any(m == "viz: precio en línea" for m in normal["logs"])


def test_the_threshold_is_crossed_by_pixels_in_both_directions(crossings):
    _, (barras, linea, barras_otra_vez, lejos) = crossings
    # Mismo nivel en los cuatro: lo que cambia es solo el ancho de la columna en pantalla.
    assert all(s["level"].startswith("nivel 4096") for s in (barras, linea, lejos))
    assert barras["metrics"]["price_draw"]["mode"] == "bars"  # 5,02 px: barras
    assert (
        linea["metrics"]["price_draw"]["mode"] == "line"
    )  # 4,98 px: de vuelta a la línea
    assert linea["pathOps"] is None and linea["pathOpCount"] > 0
    assert barras_otra_vez["metrics"]["price_draw"]["mode"] == "bars"  # y otra vez
    assert lejos["metrics"]["price_draw"]["mode"] == "line"


def test_each_bar_spans_min_to_max_with_first_left_and_last_right(
    normal, day_dir, crossings
):
    zoom = step(normal, "zoom")
    raw = np.fromfile(day_dir / f"price-{W}.i32", dtype="<i4")
    price = (raw[4 * W :].astype(float) / 100).reshape(W, 4)
    present = price[:, 0] != EMPTY_PRICE / 100
    lo_x, hi_x = zoom["x"]
    cols = [
        c for c in range(int(lo_x // COL_S), int(np.ceil(hi_x / COL_S))) if present[c]
    ]
    assert cols, "la hora del zoom no tiene ticks en la fixture"
    ops = zoom["pathOps"]
    # Tres tramos (dos operaciones cada uno) por columna con ticks; ninguno une dos columnas.
    assert len(ops) == 6 * len(cols) == 6 * zoom["metrics"]["price_draw"]["columns"]

    left, top = zoom["plot"]["left"], zoom["plot"]["top"]
    width, height = zoom["plot"]["width"], zoom["plot"]["height"]
    y_lo, y_hi = zoom["yScale"]

    def px_y(v):
        return top + (y_hi - v) / (y_hi - y_lo) * height

    for i, c in enumerate(cols):
        seg = ops[6 * i : 6 * i + 6]
        assert [o[0] for o in seg] == ["moveTo", "lineTo"] * 3
        (_, x0, ya), (_, x1, yb), (_, xf0, yf), (_, xf1, yf2) = seg[:4]
        (_, xl0, yl), (_, xl1, yl2) = seg[4:]
        first, mn, mx, last = price[c, 0], price[c].min(), price[c].max(), price[c, 3]
        assert x0 == x1 == xf1 == xl0  # barra y muescas salen del centro de la columna
        assert x0 == pytest.approx(
            left + ((c + 0.5) * COL_S - lo_x) / (hi_x - lo_x) * width, abs=1
        )
        assert (ya, yb) == pytest.approx((px_y(mx), px_y(mn)), abs=1)  # de máx a mín
        assert yf == yf2 == pytest.approx(px_y(first), abs=1)  # primero: a la izquierda
        assert xf0 < x0
        assert yl == yl2 == pytest.approx(px_y(last), abs=1)  # último: a la derecha
        assert xl1 > x0
        # Cada columna es un objeto discreto: sus muescas no tocan el borde de la vecina.
        half_col = width * COL_S / (hi_x - lo_x) / 2
        assert xf0 > x0 - half_col and xl1 < x0 + half_col
