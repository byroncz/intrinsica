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
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from lake_fixture import DAY, build_lake
from test_viz_events import write_flash_day
from viz_tiles import cli
from viz_tiles.contract import EMPTY_PRICE, EVENT_BYTES, PAGE_FILE
from viz_tiles.reduce import day_start_us

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
    names = render.expected_names(index)
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
    assert "(strong ? 0.42 : 0.18)" in app


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
    # Solo se decodificó ese nivel (precio, volumen, los 5 θ de dirección, ticks y las dos
    # confirmaciones) y los eventos exactos del día, una sola vez.
    index = json.loads((day_dir / "index.json").read_text())
    events = sum(t["events"] for t in index["thetas"]) * EVENT_BYTES
    assert start["metrics"]["decoded_bytes"] == (
        65536 + 8192 + 5 * 2048 + 4 * 2048 + 2 * 2048 + events
    )
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


def test_three_panels_split_the_height_65_15_20(normal):
    """Reparto de TRD-viz §6.13: precio 65 %, confirmaciones 15 %, volumen 20 %."""
    price, confirms, volume = normal["panelHeights"]
    total = price + confirms + volume
    assert price / total == pytest.approx(0.65, abs=0.01)
    assert confirms / total == pytest.approx(0.15, abs=0.01)
    assert volume / total == pytest.approx(0.20, abs=0.01)


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


# ---- Precio: puntos o segmentos, nunca velas ------------------------------------------

W = 4096  # el nivel más fino: el único donde una columna llega a ocupar 5 px
COL_S = 86400 / W
T0_US = day_start_us(date(2017, 8, 18))


def tile(day_dir, name, dtype):
    return np.fromfile(day_dir / name, dtype=dtype)


def m4_columns(day_dir, w):
    """`(tiempos s, precios, ticks)` por columna de un nivel, de sus tiles."""
    raw = tile(day_dir, f"price-{w}.i32", "<i4")
    t = raw[: 4 * w].view("<u4").reshape(w, 4) / 1000
    p = (raw[4 * w :].astype(float) / 100).reshape(w, 4)
    return t, p, tile(day_dir, f"count-{w}.u32", "<u4")


def px_x(step_, sec):
    plot, (lo, hi) = step_["plot"], step_["x"]
    return plot["left"] + (sec - lo) / (hi - lo) * plot["width"]


def px_y(step_, value):
    plot, (lo, hi) = step_["plot"], step_["yScale"]
    return plot["top"] + (hi - value) / (hi - lo) * plot["height"]


def test_the_template_draws_no_candles():
    """Sin muescas de primero y último: eso es una vela (TRD-viz §6.11)."""
    app = APP_JS.read_text(encoding="utf-8")
    assert "notch" not in app.lower() and "BAR_NOTCH" not in app
    assert "muesca" not in app.lower() and "vela" not in app.lower().replace(
        "velas", ""
    )


def test_the_full_day_draws_envelopes_and_ticks_never_a_joined_line(normal, day_dir):
    start = step(normal, "inicio")
    assert start["level"].startswith("nivel 2048")
    draw = start["metrics"]["price_draw"]
    assert draw["mode"] == "segments"
    t, p, count = m4_columns(day_dir, 2048)
    present = p[:, 0] > 0
    few = present & (count <= 2)
    many = present & (count >= 3)
    flat = many & (p.min(axis=1) == p.max(axis=1))
    # Un segmento por columna con varios ticks (la horizontal del plano no cuenta como segmento).
    assert draw["segments"] == int((many & ~flat).sum())
    expected_points = sum(
        len({(a, b) for a, b in zip(t[c], p[c], strict=True)})
        for c in np.flatnonzero(few)
    )
    assert draw["points"] == expected_points
    ops = start["pathOps"]
    assert ops is not None
    # Nada une una columna con la vecina: cada lineTo sigue a su moveTo y de ahí no sale otro.
    kinds = [o[0] for o in ops]
    for i, kind in enumerate(kinds):
        if kind == "lineTo":
            assert kinds[i - 1] == "moveTo"
        assert kind in ("moveTo", "lineTo", "rect")
    assert kinds.count("rect") == expected_points


def test_each_envelope_spans_the_exact_min_to_max_of_its_column(normal, day_dir):
    start = step(normal, "inicio")
    _t, p, count = m4_columns(day_dir, 2048)
    cols = [
        c
        for c in range(2048)
        if p[c, 0] > 0 and count[c] >= 3 and p[c].min() < p[c].max()
    ]
    segments = [
        (a, b)
        for a, b in zip(start["pathOps"], start["pathOps"][1:], strict=False)
        if a[0] == "moveTo" and b[0] == "lineTo" and a[1] == b[1]
    ]
    assert len(segments) == len(cols)
    for c, ((_, x0, ya), (_, x1, yb)) in zip(cols, segments, strict=True):
        assert x0 == x1 == pytest.approx(px_x(start, (c + 0.5) * 86400 / 2048), abs=1)
        # De máximo a mínimo, a lo sumo medio píxel más para que un rango nunca desaparezca.
        assert ya <= px_y(start, p[c].max()) + 1 and yb >= px_y(start, p[c].min()) - 1


def test_zoomed_in_columns_show_their_m4_ticks_over_their_envelope(normal, day_dir):
    zoom = step(normal, "zoom")  # una hora a 1 500 px: ~8 px por columna
    assert zoom["level"].startswith("nivel 4096")
    assert zoom["metrics"]["price_draw"]["mode"] == "points"
    t, p, count = m4_columns(day_dir, W)
    lo, hi = zoom["x"]
    ops = zoom["pathOps"]
    rects = [o for o in ops if o[0] == "rect"]
    centers = {(round(o[1] + o[3] / 2), round(o[2] + o[4] / 2)) for o in rects}
    expected = set()
    for c in range(int(lo // COL_S), int(np.ceil(hi / COL_S))):
        if p[c, 0] <= 0:
            continue
        for k in range(4):
            expected.add((round(px_x(zoom, t[c, k])), round(px_y(zoom, p[c, k]))))
    assert centers == expected
    assert len(rects) == zoom["metrics"]["price_draw"]["points"]
    assert zoom["level"].endswith("precio: segmentos y puntos")
    # Una columna con 3 ticks o más nunca queda en cuatro puntos sueltos: lleva su segmento.
    visible = range(int(lo // COL_S), int(np.ceil(hi / COL_S)))
    envelopes = [
        c for c in visible if p[c, 0] > 0 and count[c] >= 3 and p[c].min() < p[c].max()
    ]
    assert envelopes
    assert zoom["metrics"]["price_draw"]["segments"] == len(envelopes)


def test_the_threshold_between_segments_and_points_is_5_px(crossings):
    _, (a, b, c, far) = crossings
    assert [s["metrics"]["price_draw"]["mode"] for s in (a, b, c, far)] == [
        "points",
        "segments",
        "points",
        "segments",
    ]


@pytest.fixture(scope="module")
def crossings(day_dir, normal):
    plot_w = step(normal, "inicio")["plotW"]
    flags = []
    for px in (5.02, 4.98, 5.02, 1.0):
        flags += ["--zoom", f"36000,{36000 + plot_w * COL_S / px}"]
    result = view(day_dir, *flags)
    steps = [s for s in result["steps"] if s["label"].startswith("zoom:")]
    assert len(steps) == 4
    return result, steps


def test_tooltip_says_how_many_ticks_the_column_has(normal, day_dir):
    tip = normal["tooltip"]  # cursor en la columna de las 12:03 del nivel 2048
    start = re.search(r"^(\d\d):(\d\d):(\d\d)\.\d{3}", tip)
    sec = int(start[1]) * 3600 + int(start[2]) * 60 + int(start[3])
    count = tile(day_dir, "count-2048.u32", "<u4")
    col = round(sec * 2048 / 86400)
    assert f"\n{count[col]} tick" in tip  # incluida la columna vacía: "0 ticks"


def test_a_missing_count_tile_is_said_not_filled_with_zeros(day_dir):
    result = view(day_dir, "--drop", "count-2048.u32")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert "falta count-2048.u32" in start["status"]
    assert "ticks: ⚠ falta el tile" in result["tooltip"]
    assert " 0 ticks" not in result["tooltip"]


# ---- Franjas desde los eventos exactos y navegación -----------------------------------


def events_of(day_dir, theta_pos):
    index = json.loads((day_dir / "index.json").read_text())
    total = sum(t["events"] for t in index["thetas"])
    raw = (day_dir / index["events"]).read_bytes()
    doc = index["thetas"][theta_pos]
    sl = slice(doc["events_offset"], doc["events_offset"] + doc["events"])
    return {
        "ref": np.frombuffer(raw, "<i4", total, 0)[sl] / 1000,
        "confirm": np.frombuffer(raw, "<i4", total, 4 * total)[sl] / 1000,
        "extreme": np.frombuffer(raw, "<i4", total, 8 * total)[sl] / 1000,
        "flags": np.frombuffer(raw, "u1", total, 12 * total)[sl],
    }


@pytest.fixture(scope="module")
def target(day_dir):
    """Un evento del primer θ con confirmación y extremo distintos, de más de 30 s."""
    ev = events_of(day_dir, 0)
    k = next(
        i
        for i in range(len(ev["ref"]))
        if ev["ref"][i] > 40000
        and ev["confirm"][i] < ev["extreme"][i]
        and ev["extreme"][i] - ev["ref"][i] > 30
    )
    return k, float(ev["ref"][k]) - 0.0005


@pytest.fixture(scope="module")
def navigated(day_dir, target):
    # El primer θ, con la escala fijada en 600 s centrada justo antes del evento `k`.
    return view(day_dir, "--nav", "0", "--nav-at", str(target[1]))


def test_next_event_moves_the_window_and_keeps_the_scale(navigated, day_dir, target):
    ev = events_of(day_dir, 0)
    zero, one, two, back = (step(navigated, f"nav-{i}") for i in range(4))
    span = zero["x"][1] - zero["x"][0]
    assert span == pytest.approx(600)
    # Sin evento elegido, "siguiente" va al primero que arranca después del centro de la vista.
    k = target[0]
    assert int(np.argmax(ev["ref"] > target[1])) == k
    for s in (one, two, back):
        assert s["x"][1] - s["x"][0] == pytest.approx(
            span
        )  # la escala del humano no cambia
    for s, kk in ((one, k), (two, k + 1), (back, k)):
        a, z = max(kk - 1, 0), min(kk + 1, len(ev["ref"]) - 1)
        window = (ev["ref"][a], ev["extreme"][z])
        assert (s["x"][0] + s["x"][1]) / 2 == pytest.approx(sum(window) / 2, abs=1e-6)
        assert f"evento {kk + 1} de {len(ev['ref'])}" in s["nav"]["info"]
    assert one["x"] == pytest.approx(
        back["x"]
    )  # siguiente y anterior vuelven al mismo sitio


def test_fit_sets_the_scale_to_the_window_with_a_margin(navigated, day_dir, target):
    ev = events_of(day_dir, 0)
    k = target[0]
    fit = step(navigated, "nav-fit")
    start, end = ev["ref"][k - 1], ev["extreme"][k + 1]
    margin = (end - start) * 0.05
    assert fit["x"] == pytest.approx([start - margin, end + margin], abs=1e-6)
    # Es una acción aparte: la navegación no la había hecho.
    assert step(navigated, "nav-3")["x"][1] - step(navigated, "nav-3")["x"][
        0
    ] == pytest.approx(600)
    assert fit["nav"]["fit"] is False  # habilitado con un evento elegido
    assert step(navigated, "nav-0")["nav"]["fit"] is True  # sin evento, deshabilitado


def test_bands_are_drawn_at_the_exact_instants_of_the_event(navigated, day_dir, target):
    ev = events_of(day_dir, 0)
    k = target[0]
    fit = step(navigated, "nav-fit")
    rects, style = [], None
    for name, *args in navigated["fitDraw"]:
        if name == "fillStyle":
            style = args[0]
        elif name == "fillRect":
            rects.append((style, *args))
    up = int(ev["flags"][k]) & 1
    rgb = "63, 185, 80" if up else "248, 81, 73"
    confirm_band = [
        r
        for r in rects
        if r[0] == f"rgba({rgb}, 0.18)" and r[4] == fit["plot"]["height"]
    ]
    over_band = [
        r
        for r in rects
        if r[0] == f"rgba({rgb}, 0.42)" and r[4] == fit["plot"]["height"]
    ]
    x_ref, x_conf, x_ext = (px_x(fit, ev[n][k]) for n in ("ref", "confirm", "extreme"))
    assert any(
        abs(r[1] - x_ref) < 1 and abs(r[1] + r[3] - x_conf) < 1.5 for r in confirm_band
    )
    assert any(
        abs(r[1] - x_conf) < 1 and abs(r[1] + r[3] - x_ext) < 1.5 for r in over_band
    )
    # La línea de extremo (1 px) y la de confirmación (1 px) están en sus instantes.
    lines = [r for r in rects if r[3] == 1 and r[4] == fit["plot"]["height"]]
    assert any(abs(r[1] - x_ext) <= 1 for r in lines)
    assert any(abs(r[1] - x_conf) <= 1 for r in lines)


def test_the_selected_event_has_a_frame_and_the_info_says_its_times(navigated):
    two = step(navigated, "nav-2")
    assert re.search(
        r"evento \d+ de 573 · (alza|baja) · referencia \d\d:\d\d:\d\d\.\d{3} · confirmación "
        r"\d\d:\d\d:\d\d\.\d{3} · extremo \d\d:\d\d:\d\d\.\d{3} · ventana ",
        two["nav"]["info"],
    )
    assert two["metrics"]["nav"]["window"] is not None


def test_theta_change_forgets_the_selected_event(day_dir):
    result = view(day_dir, "--nav", "0", "--nav-at", "43000", "--nav-switch", "1")
    assert step(result, "nav-1")["nav"]["info"].startswith("evento ")
    assert step(result, "nav-fit")["metrics"]["nav"]["k"] >= 0
    after = step(result, "nav-switch")
    # Los índices de evento son de cada θ: al cambiar de θ no queda ninguno elegido.
    assert after["metrics"]["nav"]["k"] == -1
    assert after["nav"]["fit"]  # "Ajustar" queda deshabilitado
    assert re.fullmatch(r"\d+ eventos? en el día", after["nav"]["info"])


def test_a_missing_events_file_says_so_and_draws_no_bands(day_dir):
    result = view(day_dir, "--drop", "events.bin")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert "falta events.bin" in start["status"]
    assert (
        "⚠ faltan los eventos exactos: no se dibujan franjas" in result["firstMessages"]
    )
    assert start["nav"]["prev"] and start["nav"]["next"]  # sin eventos no se navega
    dc = ("63, 185, 80", "248, 81, 73")
    assert not [
        c
        for c in result["regionDraw"]
        if c[0] == "fillStyle" and any(t in c[1] for t in dc)
    ]


def test_a_corrupt_events_file_is_rejected_not_misread(day_dir):
    result = view(day_dir, "--corrupt", "events.bin")
    assert result["errors"] == []
    assert "events.bin: tamaño inesperado" in step(result, "inicio")["status"]


# ---- Panel de confirmaciones multiescala -----------------------------------------------


def test_the_confirmations_panel_shows_the_two_tiles_as_bars(normal, day_dir):
    start = step(normal, "inicio")
    assert start["confLen"] == 2048 and start["cx"] == start["x"]
    np.testing.assert_array_equal(
        start["confData"][0], tile(day_dir, "confirms-2048.u8", "u1")
    )
    np.testing.assert_array_equal(
        start["confData"][1], tile(day_dir, "simul-2048.u8", "u1")
    )
    assert max(start["confData"][0]) >= 2


def test_the_confirmations_tooltip_lists_which_theta_confirmed_and_when(day_dir):
    confirms = tile(day_dir, "confirms-2048.u8", "u1")
    col = int(np.argmax(confirms >= 2))
    sec = (col + 0.5) * 86400 / 2048
    result = view(day_dir, "--hover", str(sec))
    tip = result["confTooltip"]
    assert f"θ que confirman: {confirms[col]}" in tip
    assert (
        f"máx. en el mismo instante: {tile(day_dir, 'simul-2048.u8', 'u1')[col]}" in tip
    )
    listed = re.findall(
        r"^θ (0\.\d{8})  (\d\d:\d\d:\d\d\.\d{3})$", tip, flags=re.MULTILINE
    )
    index = json.loads((day_dir / "index.json").read_text())
    expected = []
    for pos, doc in enumerate(index["thetas"]):
        ev = events_of(day_dir, pos)
        for ms, flags in zip(ev["confirm"], ev["flags"], strict=True):
            if col * 86400 / 2048 <= ms < (col + 1) * 86400 / 2048:
                expected.append(doc["theta"])
    assert sorted(t for t, _ in listed) == sorted(expected)
    assert len(listed) >= confirms[col]


def test_missing_confirmation_tiles_are_said(day_dir):
    result = view(day_dir, "--drop", "confirms-2048.u8")
    assert "falta confirms-2048.u8" in step(result, "inicio")["status"]
    assert "⚠ falta el tile confirms de este nivel" in result["firstMessages"]


def test_legend_explains_lines_markers_and_the_confirmations_panel(day_dir):
    page = (day_dir / PAGE_FILE).read_text(encoding="utf-8")
    assert "línea clara: extremo, frontera entre eventos" in page
    assert "marca gris con número: eventos enteros dentro del mismo píxel" in page
    assert "panel central: barra, θ que confirman en la columna" in page
    assert "marca intensa, máximo de θ que confirman en el mismo instante" in page
    for label in ("Evento anterior", "Evento siguiente", "Ajustar a la ventana"):
        assert f">{label}<" in page


# ---- El caso de aceptación: cuatro eventos en menos de un segundo ---------------------

FLASH = 12 * 3600 + 40 * 60 + 26


@pytest.fixture(scope="module")
def flash(tmp_path_factory):
    return write_flash_day(tmp_path_factory.mktemp("flash"))


def test_the_density_marker_counts_the_four_events_of_the_flash_crash(flash):
    result = view(flash)
    assert result["errors"] == []
    start = step(result, "inicio")
    # Los cuatro eventos caen en un píxel: una marca con el número, no franjas indistinguibles.
    assert start["metrics"]["density"]["groups"] == 1
    assert start["metrics"]["density"]["max"] == 4
    assert "4 eventos" in result["texts"]


def test_navigating_to_the_flash_crash_and_fitting_shows_the_events_alternating(flash):
    result = view(flash, "--nav", "0", "--nav-at", str(FLASH + 0.2))
    one, fit = step(result, "nav-1"), step(result, "nav-fit")
    # "Siguiente" llega al segundo evento del flash: su ventana va del primero al tercero.
    assert "evento 3 de 6" in one["nav"]["info"]
    assert one["x"][1] - one["x"][0] == pytest.approx(600)
    assert (
        fit["x"][1] - fit["x"][0] < 5
    )  # la ventana de tres eventos del flash cabe en segundos
    # Con la ventana ajustada la marca de densidad se abre: cada evento tiene su franja.
    assert fit["metrics"]["density"]["groups"] == 0
    rects = [c for c in result["fitDraw"] if c[0] == "fillRect"]
    fills = [c[1] for c in result["fitDraw"] if c[0] == "fillStyle"]
    assert any("63, 185, 80" in f for f in fills) and any(
        "248, 81, 73" in f for f in fills
    )
    assert len(rects) >= 6


def test_the_first_event_of_the_day_has_no_previous_event_and_says_so(flash):
    result = view(flash, "--nav", "0", "--nav-at", "10")
    one = step(result, "nav-1")
    assert "evento 1 de 6" in one["nav"]["info"]
    assert "sin evento anterior en el día" in one["nav"]["info"]
    assert one["nav"]["prev"] is True  # no hay anterior que elegir
