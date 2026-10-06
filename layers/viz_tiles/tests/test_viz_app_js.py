"""La vista (`site/app.js`) con el uPlot real en Node y un DOM de mentira.

No dibuja: atrapa errores de la lógica (decodificación, marcos por píxel, θ, zoom,
hooks) sin un navegador y lee los rectángulos (`fillRect`) que cada panel dejó en su
lienzo, para compararlos con un oráculo hecho con NumPy desde los ticks de L1. Se salta
si no hay `node` (GitHub Actions lo trae). Lo que solo un navegador comprueba (pintura,
medidas, tiempos reales) queda en la Evaluación ergonómica de la card.
"""

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from lake_fixture import DAY, build_lake, read_ticks
from test_viz_events import (
    BURST,
    BURST_TICKS,
    FLASH,
    SCALE,
    flash_ticks,
    write_flash_day,
)
from viz_tiles import cli
from viz_tiles.contract import FLAG_CONFIRM_CLIPPED, PAGE_FILE, TICKS_CHUNK
from viz_tiles.events import EventRows, EventsBuffer
from viz_tiles.ticks import DayTicks, day_start_us, encode_ticks
from viz_tiles.write import ThetaEvents, write_day
from viz_tiles.write import day_dir as tiles_day_dir

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "harness.cjs"
APP_JS = Path(__file__).parents[1] / "site" / "app.js"
GLYPHS = ("▲", "▼")
T0 = day_start_us(date(2017, 8, 18))

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
    page = day_dir if str(day_dir).endswith(".html") else day_dir / PAGE_FILE
    done = subprocess.run(
        [NODE, str(HARNESS), str(page), *flags],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def normal(day_dir):
    return view(day_dir)


def page_with(day_dir, tmp_path, **changes) -> Path:
    """Una copia de la página del día con otros campos en el índice (los mismos archivos)."""
    from viz_tiles import render

    index = json.loads((day_dir / "index.json").read_text()) | changes
    out = tmp_path / "page"
    out.mkdir()
    arrays = [
        (n, [(day_dir / n).read_bytes()])
        for n in render.expected_names(json.loads((day_dir / "index.json").read_text()))
    ]
    (out / PAGE_FILE).write_bytes(render.render_day(index, arrays))
    return out


def step(out, label):
    return next(s for s in out["steps"] if s["label"] == label)


# ---- El oráculo: lo que cada píxel debe llevar, hecho con NumPy desde los ticks ----------


def smoke_ticks():
    """Los ticks del día del humo como `(ms, precio en unidades de tick, cantidad)`."""
    rows = read_ticks()
    return (
        np.array([(t["time"] - T0) // 1000 for t in rows]),
        np.array([int(t["price"] * 100) for t in rows]),
        np.ones(len(rows)),
    )


def frame_of(ticks, x, cols):
    """Los números por píxel de la ventana `x` (s): ticks, mínimo, máximo y volumen."""
    t, price, qty = ticks
    a, z = x[0] * 1000, x[1] * 1000
    k = cols / max(z - a, 1e-6)
    visible = (t >= np.ceil(a)) & (t <= z)
    px = np.minimum(np.floor((t[visible] - a) * k).astype(int), cols - 1)
    lo = np.full(cols, 2**40)
    hi = np.full(cols, -(2**40))
    np.minimum.at(lo, px, price[visible])
    np.maximum.at(hi, px, price[visible])
    return {
        "n": np.bincount(px, minlength=cols),
        "lo": lo,
        "hi": hi,
        "vol": np.bincount(px, weights=qty[visible], minlength=cols),
        "k": k,
    }


def events_of(day_dir, theta_pos):
    index = json.loads((day_dir / "index.json").read_text())
    total = sum(t["events"] for t in index["thetas"])
    raw = (day_dir / index["events"]).read_bytes()
    doc = index["thetas"][theta_pos]
    sl = slice(doc["events_offset"], doc["events_offset"] + doc["events"])
    return {
        "ref": np.frombuffer(raw, "<i4", total, 0)[sl],
        "confirm": np.frombuffer(raw, "<i4", total, 4 * total)[sl],
        "extreme": np.frombuffer(raw, "<i4", total, 8 * total)[sl],
        "flags": np.frombuffer(raw, "u1", total, 12 * total)[sl],
    }


def confirmations_frame(day_dir, x, cols):
    """`(θ que confirman, máximo en el mismo instante)` por píxel, de los eventos de todos los θ."""
    index = json.loads((day_dir / "index.json").read_text())
    a, z = x[0] * 1000, x[1] * 1000
    k = cols / max(z - a, 1e-6)
    pairs = set()
    for pos in range(len(index["thetas"])):
        ev = events_of(day_dir, pos)
        for ms, flags in zip(ev["confirm"], ev["flags"], strict=True):
            if not flags & FLAG_CONFIRM_CLIPPED and np.ceil(a) <= ms <= z:
                pairs.add((int(ms), pos))
    cc = np.zeros(cols, int)
    cs = np.zeros(cols, int)
    by_px: dict[int, set] = {}
    by_ms: dict[int, set] = {}
    for ms, pos in pairs:
        px = min(int(np.floor((ms - a) * k)), cols - 1)
        by_px.setdefault(px, set()).add(pos)
        by_ms.setdefault(ms, set()).add(pos)
        cs[px] = max(cs[px], 0)
    for px, thetas in by_px.items():
        cc[px] = len(thetas)
    for ms, thetas in by_ms.items():
        px = min(int(np.floor((ms - a) * k)), cols - 1)
        cs[px] = max(cs[px], len(thetas))
    return cc, cs


def y_of(geometry, y_range, value):
    """El píxel vertical (del lienzo) de un valor en la escala del panel."""
    lo, hi = y_range
    return geometry["top"] + (hi - value) / (hi - lo) * geometry["height"]


def price_marks(s):
    """Las marcas de precio por píxel: `{px: [("dot" | "segment", y, alto)]}`."""
    out: dict[int, list] = {}
    left = s["plot"]["left"]
    for x, y, w, h in s["marks"]["price"]:
        if w == 3 and h == 3:
            out.setdefault(x + 1 - left, []).append(("dot", y + 1.5, h))
        else:
            assert w == 1, (x, y, w, h)
            out.setdefault(x - left, []).append(("segment", y, h))
    return out


# ---- La página y la franja de estado ---------------------------------------------------


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
    result = view(day_dir, "--drop", "events.bin")
    start = step(result, "inicio")
    assert start["status"] == "incompletos: falta events.bin"
    assert start["statusClass"] == "item incomplete"
    # El ámbar lo pone la hoja de estilo sobre esa clase (la prueba lee la página real).
    page = (day_dir / PAGE_FILE).read_text(encoding="utf-8")
    assert re.search(r"#s-data-box\.incomplete b\s*\{\s*color:\s*var\(--amber\)", page)


def test_three_panels_split_the_height_65_15_20(normal):
    """Reparto de TRD-viz §6.13: precio 65 %, confirmaciones 15 %, volumen 20 %."""
    price, confirms, volume = normal["panelHeights"]
    total = price + confirms + volume
    assert price / total == pytest.approx(0.65, abs=0.01)
    assert confirms / total == pytest.approx(0.15, abs=0.01)
    assert volume / total == pytest.approx(0.20, abs=0.01)


# ---- La decodificación, una sola vez --------------------------------------------------


def test_ticks_are_decoded_once_on_opening_and_reported_to_the_console(normal, day_dir):
    start = step(normal, "inicio")
    index = json.loads((day_dir / "index.json").read_text())
    sizes = (day_dir / "ticks.bin").stat().st_size + (
        day_dir / "events.bin"
    ).stat().st_size
    assert start["metrics"]["ticks"] == index["ticks"] == 4735
    assert start["metrics"]["decoded_bytes"] == sizes
    (line,) = [
        m for m in normal["logs"] if m.startswith("viz: ticks decodificados en ")
    ]
    assert re.fullmatch(
        r"viz: ticks decodificados en [\d.]+ ms \(4735 ticks, \d+ B decodificados\)",
        line,
    )
    assert start["metrics"]["first_paint_ms"] is not None
    assert any(m.startswith("viz: primer trazo") for m in normal["logs"])
    # Zoom, cambio de θ y cambio de tamaño no decodifican nada más.
    assert step(normal, "resize")["metrics"]["decoded_bytes"] == sizes
    assert len([m for m in normal["logs"] if "decodificados en" in m]) == 1


# ---- Precio por píxel: puntos o segmentos, nunca velas ----------------------------------


def test_the_template_draws_no_candles_and_no_line_between_pixels():
    app = APP_JS.read_text(encoding="utf-8")
    assert "notch" not in app.lower() and "BAR_NOTCH" not in app
    assert "muesca" not in app.lower()
    price = app.split("function drawPrice(u)")[1].split("\n  function drawBars")[0]
    # El precio son rectángulos: ni caminos ni líneas que unan un píxel con el siguiente.
    assert "lineTo" not in price and "moveTo" not in price and "stroke(" not in price
    assert "Path2D" not in app


def test_the_full_day_draws_one_mark_per_pixel_from_the_ticks(normal):
    start = step(normal, "inicio")
    cols = round(start["plotW"])
    fr = frame_of(smoke_ticks(), start["x"], cols)
    marks = price_marks(start)
    n = fr["n"]
    # Un píxel con 1 o 2 ticks, puntos; con 3 o más, un segmento. Los vacíos, nada.
    assert sorted(marks) == [int(c) for c in np.flatnonzero(n)]
    expected_dots = int(
        ((n == 1) | ((n == 2) & (fr["lo"] == fr["hi"]))).sum()
    ) + 2 * int(((n == 2) & (fr["lo"] != fr["hi"])).sum())
    expected_segments = int((n >= 3).sum())
    assert start["metrics"]["price_draw"] == {
        "dots": expected_dots,
        "segments": expected_segments,
    }
    assert len(start["marks"]["price"]) == expected_dots + expected_segments
    assert expected_segments > 0 and expected_dots > 0
    scale = 100
    for c in np.flatnonzero(n):
        got = marks[int(c)]
        y_lo = y_of(start["plot"], start["yPrice"], fr["lo"][c] / scale)
        y_hi = y_of(start["plot"], start["yPrice"], fr["hi"][c] / scale)
        if n[c] >= 3:
            ((kind, y, h),) = got
            # Del máximo al mínimo, y nunca menos de un píxel de alto.
            assert kind == "segment"
            assert y == pytest.approx(y_hi, abs=1) and h == pytest.approx(
                max(1, y_lo - y_hi), abs=1.01
            )
        else:
            assert {k for k, *_ in got} == {"dot"}
            ys = sorted(y for _, y, _ in got)
            assert ys[0] == pytest.approx(min(y_lo, y_hi), abs=1.01)
            assert ys[-1] == pytest.approx(max(y_lo, y_hi), abs=1.01)


def test_a_price_axis_fits_the_ticks_in_view(normal):
    start = step(normal, "inicio")
    _, price, _ = smoke_ticks()
    lo, hi = price.min() / 100, price.max() / 100
    pad = (hi - lo) * 0.05
    assert start["yPrice"] == pytest.approx([lo - pad, hi + pad])
    # Con zoom el eje sigue a lo que se ve, no al día.
    zoom = step(normal, "zoom")
    t, p, _ = smoke_ticks()
    seen = p[(t >= 36_000_000) & (t <= 39_600_000)] / 100
    assert zoom["yPrice"] == pytest.approx(
        [
            seen.min() - (seen.max() - seen.min()) * 0.05,
            seen.max() + (seen.max() - seen.min()) * 0.05,
        ]
    )


def test_zoomed_in_every_tick_is_a_point_and_the_view_matches_the_oracle(day_dir):
    t, _, _ = smoke_ticks()
    start_ms = int(t[len(t) // 2])
    x = (start_ms / 1000 - 30, start_ms / 1000 + 30)
    result = view(day_dir, "--zoom", f"{x[0]},{x[1]}")
    zoom = step(result, f"zoom:{x[0]},{x[1]}")
    cols = round(zoom["plotW"])
    fr = frame_of(smoke_ticks(), x, cols)
    assert fr["n"].sum() >= 3
    assert sorted(price_marks(zoom)) == [int(c) for c in np.flatnonzero(fr["n"])]
    # Ticks sueltos: ningún segmento, solo puntos del tamaño de un tick.
    if fr["n"].max() <= 2:
        assert zoom["metrics"]["price_draw"]["segments"] == 0
    assert zoom["metrics"]["frame"]["ticks"] == int(fr["n"].sum())
    assert (
        f"{int(fr['n'].sum())} ticks en la vista" in zoom["view"]
        or "1 tick en la vista" in zoom["view"]
    )


# ---- Volumen y confirmaciones por píxel ------------------------------------------------


def test_volume_is_one_bar_per_pixel_with_the_sum_of_its_ticks(normal):
    start = step(normal, "inicio")
    cols = round(start["plotW"])
    fr = frame_of(smoke_ticks(), start["x"], cols)
    bars = start["marks"]["volume"]
    assert sorted(x - start["plotVol"]["left"] for x, *_ in bars) == [
        int(c) for c in np.flatnonzero(fr["n"])
    ]
    assert start["yVol"] == pytest.approx([0, fr["vol"].max() * 1.08])
    plot = start["plotVol"]
    base = plot["top"] + plot["height"]
    for x, y, w, h in bars:
        value = fr["vol"][x - plot["left"]]
        assert w == 1
        assert y == pytest.approx(y_of(plot, start["yVol"], value), abs=1)
        assert h == pytest.approx(
            max(1, base - y_of(plot, start["yVol"], value)), abs=1.01
        )


def test_volume_of_a_pixel_is_the_sum_of_the_quantity_of_its_ticks(tmp_path):
    """Con cantidades distintas por tick, la barra es la suma, no el conteo."""
    flash = write_flash_day(tmp_path)
    result = view(flash, "--zoom", f"{BURST - 0.001},{BURST + 0.001}")
    zoom = step(result, f"zoom:{BURST - 0.001},{BURST + 0.001}")
    plot = zoom["plotVol"]
    # 4 090 ticks de 0,01 en un mismo milisegundo: una sola barra con la suma, 40,9.
    ((_x, y, _w, _h),) = zoom["marks"]["volume"]
    assert zoom["yVol"][1] == pytest.approx(BURST_TICKS * 0.01 * 1.08)
    assert y == pytest.approx(y_of(plot, zoom["yVol"], BURST_TICKS * 0.01), abs=1)


def test_confirmations_are_the_theta_that_confirm_in_each_pixel(normal, day_dir):
    start = step(normal, "inicio")
    cols = round(start["plotW"])
    cc, cs = confirmations_frame(day_dir, start["x"], cols)
    assert cc.max() >= 2 and (cs <= cc).all()
    plot = start["plotConf"]
    base = plot["top"] + plot["height"]
    for name, expected in (("confirms", cc), ("simul", cs)):
        bars = start["marks"][name]
        assert sorted(x - plot["left"] for x, *_ in bars) == [
            int(c) for c in np.flatnonzero(expected)
        ]
        for x, y, w, h in bars:
            assert y == pytest.approx(
                y_of(plot, start["yConf"], expected[x - plot["left"]]), abs=1
            )
            assert h == pytest.approx(
                base - y_of(plot, start["yConf"], expected[x - plot["left"]]), abs=1.01
            )
    # El eje parte de 0 y llega al máximo con un poco de aire.
    assert start["yConf"] == [0, int(np.ceil(cc.max() * 1.15))]


def test_the_confirmations_tooltip_lists_which_theta_confirmed_and_when(day_dir):
    cc, _ = confirmations_frame(day_dir, (0, 86400), 1412)
    px = int(np.argmax(cc >= 2))
    result = view(day_dir, "--hover", f"confirms@{(px + 0.5) * 86400 / 1412}")
    (hover,) = result["hovers"]
    tip = hover["tooltip"]
    assert f"θ que confirman: {cc[px]}" in tip
    index = json.loads((day_dir / "index.json").read_text())
    expected = []
    for pos, doc in enumerate(index["thetas"]):
        ev = events_of(day_dir, pos)
        for ms, flags in zip(ev["confirm"], ev["flags"], strict=True):
            if not flags & FLAG_CONFIRM_CLIPPED and px == min(
                int(ms) * 1412 // 86_400_000, 1411
            ):
                expected.append(doc["theta"])
    listed = re.findall(
        r"^θ (0\.\d{8})  (\d\d:\d\d:\d\d\.\d{3})$", tip, flags=re.MULTILINE
    )
    assert sorted(t for t, _ in listed) == sorted(expected)
    assert len(listed) >= cc[px]


# ---- El tooltip por píxel ---------------------------------------------------------------


def test_tooltip_shows_hour_ticks_min_max_volume_and_the_event(normal):
    start = step(normal, "inicio")
    fr = frame_of(smoke_ticks(), start["x"], round(start["plotW"]))
    c = 700  # el cursor del arnés está a 700 px
    tip = normal["tooltip"]
    assert re.search(r"^\d\d:\d\d:\d\d – \d\d:\d\d:\d\d UTC", tip)
    n = int(fr["n"][c])
    if n:
        assert f"\n{n} tick" in tip
        if fr["lo"][c] != fr["hi"][c]:
            assert f"mín {fr['lo'][c] / 100:.2f}   máx {fr['hi'][c] / 100:.2f}" in tip
        assert f"vol {fr['vol'][c]:.4f}" in tip
    else:
        assert "\nsin ticks" in tip
    assert "θ 0.00250000" in tip
    assert normal["tooltipHidden"] is True


def test_theta_change_repaints_without_moving_the_axes(normal):
    before, after = step(normal, "antes-θ"), step(normal, "después-θ")
    assert before["yPrice"] == after["yPrice"] and before["x"] == after["x"]
    assert len(after["metrics"]["theta_change_ms"]) == 1
    assert any(m.startswith("viz: cambio de θ en") for m in normal["logs"])
    assert normal["regionFills"] > 0
    # Los ticks y los eventos ya estaban en memoria: no se decodificó nada más.
    assert after["metrics"]["decoded_bytes"] == before["metrics"]["decoded_bytes"]
    # El precio no depende de θ: las mismas marcas.
    assert before["marks"]["price"] == after["marks"]["price"]


def test_zoom_and_reset_move_every_axis_together(normal):
    zoom = step(normal, "zoom")
    assert (
        zoom["x"] == [36000, 39600]
        and zoom["vx"] == zoom["x"]
        and zoom["cx"] == zoom["x"]
    )
    assert zoom["yPrice"] != step(normal, "antes-θ")["yPrice"]
    reset = step(normal, "reset")
    assert reset["x"] == [0, 86400]
    assert reset["yPrice"] == step(normal, "antes-θ")["yPrice"]
    assert step(normal, "resize")["x"] == [0, 86400]
    assert step(normal, "resize")["marks"]["price"] == reset["marks"]["price"]


# ---- Datos faltantes o dañados ----------------------------------------------------------


def test_missing_theta_is_a_visible_gap_with_text(day_dir, tmp_path):
    """Un θ de `missing_thetas` se ofrece, marca los datos incompletos y nunca se rellena."""
    out = page_with(day_dir, tmp_path, missing_thetas=["0.10000000"])
    result = view(out, "--pick-last")
    assert result["errors"] == []
    assert result["options"][-1].startswith("⚠ 0.10000000")
    gap = step(result, "θ-sin-datos")
    assert gap["status"].startswith("incompletos: ")
    assert "0.10000000" in gap["status"] and "sin datos" in gap["status"]
    assert any("sin datos en L2" in str(m) for m in result["messages"])


def test_a_missing_ticks_file_says_so_and_draws_nothing(day_dir):
    result = view(day_dir, "--drop", "ticks.bin")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["status"] == "incompletos: falta ticks.bin"
    assert start["metrics"]["ticks"] == 0
    assert "x" not in start  # ningún panel: el día no se puede dibujar sin ticks


def test_a_corrupt_ticks_file_is_rejected_not_misread(day_dir):
    result = view(day_dir, "--corrupt", "ticks.bin")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert "ticks.bin: tamaño inesperado" in start["status"]
    assert "x" not in start


def test_a_missing_events_file_says_so_and_draws_no_bands(day_dir):
    result = view(day_dir, "--drop", "events.bin")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert "falta events.bin" in start["status"]
    assert (
        "⚠ faltan los eventos exactos: no se dibujan franjas" in result["firstMessages"]
    )
    assert "⚠ faltan los eventos exactos" in start["panelTexts"]["confirms"]
    assert start["nav"]["prev"] and start["nav"]["next"]  # sin eventos no se navega
    assert start["marks"]["confirms"] == []  # nunca barras en cero: ninguna
    assert start["marks"]["price"]  # el precio sigue ahí
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


@pytest.mark.parametrize("version", ["1.2.0", "3.0.0"])
def test_other_major_versions_are_rejected(day_dir, tmp_path, version):
    result = view(page_with(day_dir, tmp_path, tiles_version=version))
    assert result["errors"] == []
    start = step(result, "inicio")
    assert (
        start["status"].startswith("incompletos: ")
        and "no soportada" in start["status"]
    )
    assert start["metrics"]["decoded_bytes"] == 0


def test_a_day_without_thetas_still_draws_price_volume(day_dir, tmp_path):
    result = view(page_with(day_dir, tmp_path, thetas=[]))
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["marks"]["price"] and start["marks"]["volume"]
    assert start["status"].startswith("incompletos: ") and "sin θ" in start["status"]


# ---- Franjas desde los eventos exactos y navegación -----------------------------------


@pytest.fixture(scope="module")
def target(day_dir):
    """Un evento del primer θ con confirmación y extremo distintos, de más de 30 s."""
    ev = events_of(day_dir, 0)
    k = next(
        i
        for i in range(len(ev["ref"]))
        if ev["ref"][i] > 40_000_000
        and ev["confirm"][i] < ev["extreme"][i]
        and ev["extreme"][i] - ev["ref"][i] > 30_000
    )
    return k, float(ev["ref"][k]) / 1000 - 0.0005


@pytest.fixture(scope="module")
def navigated(day_dir, target):
    # El primer θ, con la escala fijada en 600 s centrada justo antes del evento `k`.
    return view(day_dir, "--nav", "0", "--nav-at", str(target[1]))


def test_next_event_moves_the_window_and_keeps_the_scale(navigated, day_dir, target):
    ev = {k: v / 1000 for k, v in events_of(day_dir, 0).items() if k != "flags"}
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
    assert one["x"] == pytest.approx(
        back["x"]
    )  # siguiente y anterior vuelven al mismo sitio


def test_the_status_strip_says_only_which_event_it_is(navigated, day_dir, target):
    """La franja de estado: "evento k / n". El detalle del evento va al tooltip."""
    n = len(events_of(day_dir, 0)["ref"])
    k = target[0]
    for label, kk in (("nav-1", k), ("nav-2", k + 1), ("nav-3", k)):
        assert step(navigated, label)["nav"]["info"] == f"evento {kk + 1} / {n}"
    assert step(navigated, "nav-0")["nav"]["info"] == f"{n} eventos"
    for s in navigated["steps"]:
        info = s["nav"]["info"]
        assert not re.search(
            r"referencia|confirmación|extremo|ventana|provisional|recorte", info
        )


def test_fit_sets_the_scale_to_the_window_with_a_margin(navigated, day_dir, target):
    ev = {k: v / 1000 for k, v in events_of(day_dir, 0).items() if k != "flags"}
    k = target[0]
    fit = step(navigated, "nav-fit")
    start, end = ev["ref"][k - 1], ev["extreme"][k + 1]
    margin = (end - start) * 0.05
    assert fit["x"] == pytest.approx([start - margin, end + margin], abs=1e-6)
    # Es una acción aparte: la navegación no la había hecho.
    s3 = step(navigated, "nav-3")
    assert s3["x"][1] - s3["x"][0] == pytest.approx(600)
    assert fit["nav"]["fit"] is False  # habilitado con un evento elegido
    assert step(navigated, "nav-0")["nav"]["fit"] is True  # sin evento, deshabilitado


def test_bands_are_drawn_at_the_exact_instants_of_the_event(navigated, day_dir, target):
    ev = {k: v / 1000 for k, v in events_of(day_dir, 0).items() if k != "flags"}
    flags = events_of(day_dir, 0)["flags"]
    k = target[0]
    fit = step(navigated, "nav-fit")
    rects, style = [], None
    for name, *args in navigated["fitDraw"]:
        if name == "fillStyle":
            style = args[0]
        elif name == "fillRect":
            rects.append((style, *args))
    rgb = "63, 185, 80" if int(flags[k]) & 1 else "248, 81, 73"
    full = fit["plot"]["height"]
    confirm_band = [r for r in rects if r[0] == f"rgba({rgb}, 0.18)" and r[4] == full]
    over_band = [r for r in rects if r[0] == f"rgba({rgb}, 0.42)" and r[4] == full]

    def px(sec):
        plot, (lo, hi) = fit["plot"], fit["x"]
        return plot["left"] + (sec - lo) / (hi - lo) * plot["width"]

    x_ref, x_conf, x_ext = (px(ev[n][k]) for n in ("ref", "confirm", "extreme"))
    assert any(
        abs(r[1] - x_ref) < 1 and abs(r[1] + r[3] - x_conf) < 1.5 for r in confirm_band
    )
    assert any(
        abs(r[1] - x_conf) < 1 and abs(r[1] + r[3] - x_ext) < 1.5 for r in over_band
    )
    # La línea de confirmación (1 px, del color del evento) está en su instante; la de extremo ya no existe.
    lines = [r for r in rects if r[3] == 1 and r[4] == full]
    assert any(r[0] == f"rgb({rgb})" and abs(r[1] - x_conf) <= 1 for r in lines)


def test_no_light_vertical_line_marks_the_extreme_of_an_event(navigated):
    """El cambio de color entre franjas ya marca el extremo: ninguna línea clara de 1 px."""
    app = APP_JS.read_text(encoding="utf-8")
    assert "F_EXTREME_CLIPPED | F_PROVISIONAL" not in app
    light = ("#d7dee6", "rgb(215, 222, 230)", "#e6edf3")
    full = step(navigated, "nav-fit")["plot"]["height"]
    style, tall = None, []
    for name, *args in navigated["fitDraw"]:
        if name == "fillStyle":
            style = args[0]
        elif name == "fillRect" and args[3] == full and style in light:
            tall.append(args)
    assert not tall  # ninguna línea vertical clara de todo el alto del panel
    # Ni el marco del evento elegido es claro: el precio es el único trazo claro del panel.
    assert "#d7dee6" not in APP_JS.read_text(encoding="utf-8")


def test_only_the_price_is_light_in_the_price_panel(normal):
    styles = {
        c[1] for c in normal["regionDraw"] if c[0] in ("fillStyle", "strokeStyle")
    }
    assert not [s for s in styles if s in ("#d7dee6", "#ffffff", "white", "#fff")]


def test_the_event_tooltip_says_its_times_window_and_notes(day_dir, target):
    ev = events_of(day_dir, 0)
    k = target[0]
    mid = (ev["ref"][k] + ev["extreme"][k]) / 2 / 1000
    span = 2 * (ev["extreme"][k] - ev["ref"][k]) / 1000
    result = view(
        day_dir, "--zoom", f"{mid - span},{mid + span}", "--hover", f"price@{mid}@0"
    )
    (hover,) = result["hovers"]
    tip = hover["tooltip"]
    up = "alza" if int(ev["flags"][k]) & 1 else "baja"
    n = len(ev["ref"])
    assert f"θ 0.00100000 · evento {k + 1} / {n} · {up}" in tip
    for label in ("referencia", "confirmación", "extremo"):
        assert re.search(rf"^{label} \d\d:\d\d:\d\d\.\d{{3}}$", tip, flags=re.MULTILINE)
    assert re.search(
        r"^ventana \d\d:\d\d:\d\d\.\d{3} a \d\d:\d\d:\d\d\.\d{3}$",
        tip,
        flags=re.MULTILINE,
    )


def test_the_provisional_tail_is_noted_in_the_tooltip_and_marked_on_the_panel(day_dir):
    ev = events_of(day_dir, 0)
    last = len(ev["ref"]) - 1
    assert ev["flags"][last] & 2  # la cola del día es provisional
    mid = (ev["ref"][last] + ev["extreme"][last]) / 2 / 1000
    span = max(2 * (ev["extreme"][last] - ev["ref"][last]) / 1000, 1)
    result = view(
        day_dir,
        "--zoom",
        f"{mid - span},{min(86400, mid + span)}",
        "--hover",
        f"price@{mid}@0",
    )
    tip = result["hovers"][0]["tooltip"]
    assert "el extremo es provisional (candidato vigente)" in tip
    assert "sin evento siguiente en el día" in tip
    assert "provisional ▸" in result["texts"]


def test_theta_change_forgets_the_selected_event(day_dir):
    result = view(day_dir, "--nav", "0", "--nav-at", "43000", "--nav-switch", "1")
    assert re.fullmatch(r"evento \d+ / \d+", step(result, "nav-1")["nav"]["info"])
    assert step(result, "nav-fit")["metrics"]["nav"]["k"] >= 0
    after = step(result, "nav-switch")
    # Los índices de evento son de cada θ: al cambiar de θ no queda ninguno elegido.
    assert after["metrics"]["nav"]["k"] == -1
    assert after["nav"]["fit"]  # "Ajustar" queda deshabilitado
    assert re.fullmatch(r"\d+ eventos?", after["nav"]["info"])


def test_no_glyphs_mark_the_regions_or_the_tooltip(day_dir):
    """Dirección y fase las dan la franja del borde y el relleno; el tooltip, con texto."""
    result = view(day_dir, "--sweep")
    assert result["errors"] == []
    shown = [*result["texts"], *result["thetaLines"], result["tooltip"]]
    assert not [t for t in shown if any(g in t for g in GLYPHS)], shown
    assert any("alza" in t or "baja" in t for t in result["thetaLines"])
    assert not any(g in APP_JS.read_text(encoding="utf-8") for g in GLYPHS)
    assert not any(
        g in (day_dir / PAGE_FILE).read_text(encoding="utf-8") for g in GLYPHS
    )


# ---- Rótulos y leyenda con muestras -----------------------------------------------------


def test_the_lower_panels_carry_a_short_label_and_the_price_none(normal):
    start = step(normal, "inicio")
    assert "Volumen" in start["panelTexts"]["volume"]
    assert "θ que confirman" in start["panelTexts"]["confirms"]
    assert "Volumen" not in start["panelTexts"]["price"]
    assert "θ que confirman" not in start["panelTexts"]["price"]
    assert not [
        t for t in start["panelTexts"]["price"] if t.strip() in ("Precio", "Price")
    ]


def legend_of(page: str) -> str:
    return re.search(r'<span id="legend".*?</footer>', page, re.DOTALL).group(0)


def test_legend_is_made_of_samples_not_sentences(day_dir):
    page = (day_dir / PAGE_FILE).read_text(encoding="utf-8")
    legend = legend_of(page)
    # Un recuadro por franja (alza/baja × confirmación/overshoot), un trazo por línea, la marca gris con número.
    for sample in ("sw up conf", "sw up over", "sw down conf", "sw down over"):
        assert f'class="{sample}"' in legend
    for sample in (
        "ln up",
        "ln down",
        "mk price",
        "sw dense",
        "ln prov",
        "bar both",
        "bar vol",
    ):
        assert f'class="{sample}"' in legend
    assert legend.count('class="key"') == 11
    # El único texto visible es el número de la marca gris: nada obliga a leer una frase.
    samples = legend.split('<span id="f-help">')[0]
    visible = re.sub(
        r"<[^>]*>", " ", re.sub(r'\s(title|aria-label)="[^"]*"', "", samples)
    )
    assert visible.split() == ["12"], visible.split()
    assert "tenue: confirmación" not in page and "línea clara" not in page
    assert "marca gris con número" not in page
    # Cada muestra se explica al pasar el mouse.
    assert legend.count("title=") == 11
    # Los botones de navegación siguen en la franja.
    for label in ("Evento anterior", "Evento siguiente", "Ajustar a la ventana"):
        assert f">{label}<" in page


def test_swatches_use_the_colors_and_intensities_of_the_drawing(day_dir):
    css = (Path(__file__).parents[1] / "site" / "style.css").read_text()
    app = APP_JS.read_text()
    assert "(strong ? 0.42 : 0.18)" in app
    assert ".sw.conf { background: rgba(var(--rgb), 0.18); }" in css
    assert ".sw.over { background: rgba(var(--rgb), 0.42); }" in css
    assert "63, 185, 80" in css and "248, 81, 73" in css
    assert "border-top: 3px" in css and "border-top: 8px" in css


# ---- El caso de aceptación: cuatro eventos en menos de un segundo ---------------------


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


def test_with_a_one_second_zoom_the_four_events_show_with_their_ticks(flash):
    window = f"{FLASH - 0.05},{FLASH + 1.05}"
    result = view(flash, "--theta", "0", "--zoom", window)
    zoom = step(result, f"zoom:{window}")
    # Con la ventana de un segundo la marca de densidad se abre: cada evento tiene su franja.
    assert zoom["metrics"]["density"]["groups"] == 0
    assert zoom["metrics"]["density"]["events"] >= 5
    # Los ticks de esa ventana: los 10 de cada 0,1 s y los 4 090 del milisegundo, todos presentes.
    x = (FLASH - 0.05, FLASH + 1.05)
    t = np.array([round(r[1] * 1000) for r in flash_ticks()])
    p = np.array([round(float(r[2]) * 100) for r in flash_ticks()])
    q = np.array([float(r[3]) for r in flash_ticks()])
    fr = frame_of((t, p, q), x, round(zoom["plotW"]))
    # Los 10 de 0,1 s, el tick de fondo que cae en `FLASH` y los 4 090 del milisegundo: ninguno se pierde.
    assert (
        zoom["metrics"]["frame"]["ticks"] == int(fr["n"].sum()) == 10 + 1 + BURST_TICKS
    )
    assert sorted(price_marks(zoom)) == [int(c) for c in np.flatnonzero(fr["n"])]
    # Los 4 090 ticks del milisegundo son UN segmento del mínimo al máximo, no 4 090 puntos.
    assert zoom["metrics"]["price_draw"]["segments"] == 1
    n = fr["n"]
    dots = int(((n == 1) | ((n == 2) & (fr["lo"] == fr["hi"]))).sum()) + 2 * int(
        ((n == 2) & (fr["lo"] != fr["hi"])).sum()
    )
    assert (
        zoom["metrics"]["price_draw"]["dots"] == dots == 11
    )  # 9 instantes sueltos y 2 ticks en `FLASH`


def test_at_the_millisecond_of_the_burst_the_price_is_one_min_max_segment_and_the_tooltip_counts(
    flash,
):
    window = f"{BURST - 0.0005},{BURST + 0.0015}"
    result = view(flash, "--theta", "0", "--zoom", window, "--hover", f"price@{BURST}")
    zoom = step(result, f"zoom:{window}")
    plot = zoom["plot"]
    ((_x, y, w, h),) = zoom["marks"]["price"]
    assert w == 1
    # De 85 100 a 84 900: el segmento ocupa casi todo el alto del panel.
    assert y == pytest.approx(y_of(plot, zoom["yPrice"], 85_100), abs=1)
    assert y + h == pytest.approx(y_of(plot, zoom["yPrice"], 84_900), abs=1.5)
    (hover,) = result["hovers"]
    tip = hover["tooltip"]
    assert f"{BURST_TICKS} ticks en este ms" in tip
    assert "mín 84900.00   máx 85100.00" in tip
    assert f"vol {BURST_TICKS * 0.01:.4f}" in tip
    assert "12:40:26.350" in tip


def test_the_confirmations_of_two_theta_at_the_same_instant_are_an_intense_mark(flash):
    window = f"{FLASH + 0.04},{FLASH + 0.06}"
    result = view(flash, "--zoom", window, "--hover", f"confirms@{FLASH + 0.0510}")
    zoom = step(result, f"zoom:{window}")
    ((x, y, _w, h),) = zoom["marks"]["confirms"]
    ((sx, sy, _sw, sh),) = zoom["marks"]["simul"]
    # Dos θ confirman en 12:40:26.051: la barra y la marca intensa miden lo mismo.
    assert (x, y, h) == (sx, sy, sh)
    assert zoom["yConf"][1] == 3  # ceil(2 · 1,15)
    tip = result["hovers"][0]["tooltip"]
    assert "θ que confirman: 2" in tip and "máx. en el mismo instante: 2" in tip
    assert "θ 0.00509931  12:40:26.051" in tip and "θ 0.01000000  12:40:26.051" in tip


def test_navigating_to_the_flash_crash_and_fitting_shows_the_events_alternating(flash):
    result = view(flash, "--nav", "0", "--nav-at", str(FLASH + 0.2))
    one, fit = step(result, "nav-1"), step(result, "nav-fit")
    # "Siguiente" llega al tercer evento del θ: su ventana va del segundo al cuarto.
    assert one["nav"]["info"] == "evento 3 / 6"
    assert one["x"][1] - one["x"][0] == pytest.approx(600)
    assert fit["x"][1] - fit["x"][0] < 5  # la ventana de tres eventos cabe en segundos
    assert fit["metrics"]["density"]["groups"] == 0
    fills = [c[1] for c in result["fitDraw"] if c[0] == "fillStyle"]
    assert any("63, 185, 80" in f for f in fills) and any(
        "248, 81, 73" in f for f in fills
    )
    assert len([c for c in result["fitDraw"] if c[0] == "fillRect"]) >= 6


def test_the_first_event_of_the_day_has_no_previous_event_and_says_so(flash):
    result = view(flash, "--nav", "0", "--nav-at", "10")
    one = step(result, "nav-1")
    assert one["nav"]["info"] == "evento 1 / 6"
    assert one["nav"]["prev"] is True  # no hay anterior que elegir
    # La nota va al tooltip del evento, no a la franja de estado.
    zoom = f"{44_000 - 600},{44_000 + 600}"
    hover = view(flash, "--zoom", zoom, "--hover", "price@44250@0")["hovers"][0][
        "tooltip"
    ]
    assert "sin evento anterior en el día" in hover


# ---- Rendimiento: el día real, un recorrido lineal ---------------------------------------


def write_big_day(
    root: Path, ticks: int = 950_000, events: int = 600, thetas: int = 50
):
    """Un día del tamaño del 2026-09-30 (948 740 ticks) con 50 θ de `events` eventos cada uno."""
    rng = np.random.default_rng(30)
    day = date(2026, 9, 30)
    time_ms = np.sort(rng.integers(0, 86_400_000, ticks))
    price = 6_000_000 + np.cumsum(rng.integers(-4, 5, ticks))
    quantity = rng.integers(1, 400_000_000, ticks)
    # `ticks.bin` se escribe donde `write_day` lo espera (el job lo escribe al leer L1).
    directory = Path(tiles_day_dir(root, "binance", "spot", "BTCUSDT", day))
    directory.mkdir(parents=True)
    data = encode_ticks(time_ms, price, quantity)
    (directory / "ticks.bin").write_bytes(data)
    day_ticks = DayTicks(
        ticks=ticks,
        chunk=TICKS_CHUNK,
        price_scale=SCALE,
        first_agg_trade_id=1,
        last_agg_trade_id=ticks,
        rounded=0,
        max_abs_delta_int=0,
        nbytes=len(data),
    )
    buffer = EventsBuffer()
    docs = []
    for k in range(thetas):
        ref = np.sort(rng.integers(0, 86_000_000, events + 1)).astype("<i4")
        rows = EventRows(
            ref[:-1],
            ((ref[:-1].astype(np.int64) + ref[1:]) // 2).astype("<i4"),
            ref[1:],
            (np.arange(events) % 2).astype("u1"),
        )
        buffer.add(rows)
        docs.append(ThetaEvents(f"0.{k + 1:08d}", events, None))
    write_day(
        root,
        provider="binance",
        market="spot",
        asset="BTCUSDT",
        day=day,
        ticks=day_ticks,
        events=buffer,
        thetas=docs,
        input_hash="ab" * 32,
        image_version="0.1.0+test",
    )
    return Path(root) / "provider=binance/market=spot/asset=BTCUSDT/day=2026-09-30"


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    return write_big_day(tmp_path_factory.mktemp("big"))


def test_a_real_sized_day_decodes_and_redraws_well_inside_the_budget(big):
    """Criterio de aceptación: apertura < 5 s y redibujo < 100 ms con el día real.

    Lo mide el humano en su navegador; aquí el mismo motor de JavaScript (V8) con 948 740
    ticks y 50 θ de 600 eventos dice que el recorrido lineal no se acerca al límite.
    """
    result = view(big, "--bench", "10")
    assert result["errors"] == []
    start = step(result, "inicio")
    assert start["metrics"]["ticks"] == 950_000
    (line,) = [
        m for m in result["logs"] if m.startswith("viz: ticks decodificados en ")
    ]
    decode_ms = float(re.search(r"en ([\d.]+) ms", line)[1])
    assert decode_ms < 5000
    draws = [b["draw_ms"] for b in result["bench"]]
    assert len(draws) == 10 and max(draws) < 100, draws
    assert max(b["frame_ms"] for b in result["bench"]) < 100
    # El día entero recorre los 950 000 ticks; un tramo del medio, solo los visibles.
    assert result["bench"][1]["ticks"] == 950_000
    assert result["bench"][0]["ticks"] < 950_000 / 5
    assert start["metrics"]["decoded_bytes"] > 5_000_000


def test_the_tooltip_names_the_event_that_closes_at_its_extreme_millisecond(flash):
    """El tick extremo pertenece al evento que cierra, (referencia, extremo] (ADR-VZ-12)."""
    # FLASH + 0,300 es el extremo del evento 2 y la referencia del 3: ese ms es del 2.
    ms = FLASH + 0.300
    window = f"{ms - 0.0005},{ms + 0.0015}"
    result = view(
        flash, "--theta", "0", "--zoom", window, "--hover", f"price@{ms + 0.0005}"
    )
    tip = result["hovers"][0]["tooltip"]
    assert "evento 2 / 6" in tip and "extremo 12:40:26.300" in tip
    # Un ms después ya es del evento 3.
    window = f"{ms + 0.0005},{ms + 0.0025}"
    result = view(
        flash, "--theta", "0", "--zoom", window, "--hover", f"price@{ms + 0.0015}"
    )
    assert "evento 3 / 6" in result["hovers"][0]["tooltip"]


def test_a_day_in_several_chunks_draws_exactly_like_the_same_day_in_one(
    flash, tmp_path
):
    """El decodificador recorre los tramos en orden: partir `ticks.bin` no cambia ningún píxel."""
    many = write_flash_day(tmp_path, chunk=1_000)  # 17 tramos
    zoom = f"{BURST - 0.001},{BURST + 0.001}"
    flags = ("--zoom", zoom, "--hover", f"price@{BURST}")
    one, parts = view(flash, *flags), view(many, *flags)
    assert parts["errors"] == []
    for name in ("inicio", f"zoom:{zoom}"):
        assert step(parts, name)["marks"] == step(one, name)["marks"]
    assert parts["hovers"] == one["hovers"]
    assert step(parts, "inicio")["metrics"]["ticks"] == len(flash_ticks())
