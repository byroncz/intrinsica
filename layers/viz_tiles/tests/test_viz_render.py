"""La página HTML de un día (TRD-viz §6.5, §7.9) y el modo `render`."""

import base64
import gzip
import hashlib
import json
import re
import shutil
import time
from datetime import date
from pathlib import Path

import pyarrow.fs as pafs
import pytest
from lake_fixture import DAY, build_lake
from test_viz_write import DAY as WRITE_DAY
from test_viz_write import KEY, write
from viz_tiles import cli, render
from viz_tiles.contract import LATEST_PAGE_FILE, LEVELS, PAGE_FILE, TILES_VERSION
from viz_tiles.write import (
    BINARY_META,
    GZIP_META,
    JSON_META,
    PAGE_META,
    day_dir,
)

SITE = render.template_dir()
DATA = re.compile(r"window\.VIZ_DATA=(\{.*\});\n</script>", re.DOTALL)


def embedded(html: str) -> dict:
    """El `window.VIZ_DATA` de una página, ya como diccionario."""
    return json.loads(DATA.search(html).group(1))


def content_hash(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + files[name])
    return digest.hexdigest()


# -- la plantilla ------------------------------------------------------------


def test_template_has_the_files_of_the_card():
    for name in render.TEMPLATE_FILES:
        assert (SITE / name).is_file(), name
    assert not list(SITE.rglob("*.map"))


def test_uplot_is_pinned_and_licensed():
    banner = (SITE / "vendor/uPlot.iife.min.js").read_text().splitlines()[0]
    assert "(v1.6.32)" in banner
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "uPlot 1.6.32" in readme and "MIT" in readme


def test_own_code_makes_no_network_requests():
    """Sin CDN, sin fuentes externas, sin fetch: la página se abre por `file://`."""
    forbidden = re.compile(
        r"https?://|//cdn|fetch\s*\(|XMLHttpRequest|sendBeacon|WebSocket|EventSource|"
        r"import\s*\(|@import|<link\b|\bsrc\s*=|\bhref\s*=",
        re.IGNORECASE,
    )
    for name in ("index.html", "app.js", "style.css"):
        assert not forbidden.search((SITE / name).read_text()), name


def test_template_hash_changes_with_any_file(tmp_path):
    copy = tmp_path / "site"
    shutil.copytree(SITE, copy)
    before = render.load_template(copy).hash
    assert before == render.load_template(SITE).hash
    (copy / "style.css").write_text((copy / "style.css").read_text() + "\n/* x */\n")
    assert render.load_template.__wrapped__(copy).hash != before


# -- render_day --------------------------------------------------------------


def day_files(tmp_path, **kwargs) -> tuple[dict, dict[str, bytes], Path]:
    _, index = write(tmp_path, **kwargs)
    directory = Path(day_dir(tmp_path, **KEY, day=WRITE_DAY))
    names = [n for kind in ("price", "volume", "dir") for n in index[kind].values()]
    return index, {n: (directory / n).read_bytes() for n in names}, directory


def test_page_embeds_the_arrays_byte_for_byte(tmp_path):
    index, files, directory = day_files(tmp_path)
    html = (directory / PAGE_FILE).read_text()
    data = embedded(html)
    decoded = {n: base64.b64decode(b64) for n, b64 in data["files"].items()}
    assert decoded == files
    assert content_hash(decoded) == index["content_hash"]
    assert data["index"] == json.loads((directory / "index.json").read_text())
    assert data["tiles_version"] == index["tiles_version"] == TILES_VERSION
    assert data["generated_at"] == index["generated_at"]
    assert len(files) == 18 and index["page"] == PAGE_FILE


def test_page_is_one_self_contained_document(tmp_path):
    _, _, directory = day_files(tmp_path)
    html = (directory / PAGE_FILE).read_text()
    # Los marcadores de la plantilla se reemplazaron todos.
    assert "@@" not in html
    # Lo único con una URL es el rótulo de licencia de uPlot, un comentario.
    outside = re.sub(r"/\*!.*?\*/", "", html, flags=re.DOTALL)
    assert not re.search(
        r"https?://|fetch\s*\(|XMLHttpRequest|<link\b|\bsrc\s*=", outside
    )
    assert html.count("<script") == 3 and html.count("<style") == 1
    assert html.lstrip().lower().startswith("<!doctype html>")


def test_page_closing_tag_cannot_appear_inside_the_data(tmp_path):
    index, files, _ = day_files(tmp_path)
    tricky = {**index, "image_version": "</script><b>"}
    page = render.render_day(tricky, sorted(files.items())).decode()
    assert page.count("</script>") == 3
    assert embedded(page)["index"]["image_version"] == "</script><b>"


def test_gzip_page_is_deterministic_and_equals_the_plain_one(tmp_path):
    index, files, _ = day_files(tmp_path)
    arrays = sorted(files.items())
    plain = render.render_day(index, arrays)
    zipped = render.render_day(index, arrays, compress=True)
    assert gzip.decompress(zipped) == plain
    assert render.render_day(index, arrays, compress=True) == zipped
    assert len(zipped) < len(plain)


def test_arrays_must_match_the_index(tmp_path):
    index, files, _ = day_files(tmp_path)
    arrays = sorted(files.items())
    with pytest.raises(ValueError, match="no son los que el índice lista"):
        render.render_day(index, arrays[:-1])


def test_page_meta_reads_plain_and_gzip_heads(tmp_path):
    index, files, _ = day_files(tmp_path)
    expected = (TILES_VERSION, render.load_template().hash)
    arrays = sorted(files.items())
    plain = render.render_day(index, arrays)
    zipped = render.render_day(index, arrays, compress=True)
    assert render.page_meta(plain[: render.META_PROBE]) == expected
    # Un gzip cortado en el comienzo basta: no hace falta bajar la página.
    assert render.page_meta(zipped[:2048]) == expected
    assert render.page_meta(b"<html>sin huella</html>") is None


# -- metadatos de los objetos ------------------------------------------------


class RecordingFS:
    """Un sistema de archivos que no es local: anota los metadatos de cada escritura."""

    def __init__(self) -> None:
        self._fs = pafs.LocalFileSystem()
        self.metadata: dict[str, dict | None] = {}

    def open_output_stream(self, path, **kwargs):
        self.metadata[Path(path).name] = kwargs.get("metadata")
        return self._fs.open_output_stream(path)

    def __getattr__(self, name):
        return getattr(self._fs, name)


def test_every_object_is_written_with_its_metadata(tmp_path, monkeypatch):
    import viz_tiles.write as module

    recorder = RecordingFS()
    real = module.resolve_fs
    monkeypatch.setattr(
        module, "resolve_fs", lambda root: (recorder, real(root)[1]), raising=True
    )
    _, index = write(tmp_path)
    meta = recorder.metadata
    for kind in ("price", "volume", "dir"):
        for name in index[kind].values():
            assert meta[name] == BINARY_META, name
    assert meta["index.json"] == JSON_META
    assert meta["latest.json"] == JSON_META
    # Un bucket recibe la página en gzip, con su tipo y sin caché.
    assert meta[PAGE_FILE] == PAGE_META | GZIP_META
    assert meta[LATEST_PAGE_FILE] == PAGE_META | GZIP_META
    # Y lo que quedó escrito es, de verdad, un gzip con la huella en la cabecera.
    stored = (Path(day_dir(tmp_path, **KEY, day=WRITE_DAY)) / PAGE_FILE).read_bytes()
    assert stored[:2] == b"\x1f\x8b"
    assert render.page_meta(stored[:2048])[0] == TILES_VERSION


def test_local_disk_pages_are_plain_so_file_urls_open(tmp_path):
    _, _, directory = day_files(tmp_path)
    assert (directory / PAGE_FILE).read_bytes()[:15].lower() == b"<!doctype html>"
    assert (tmp_path / LATEST_PAGE_FILE).read_bytes() == (
        directory / PAGE_FILE
    ).read_bytes()


def test_latest_page_follows_latest_json_and_never_goes_back(tmp_path):
    write(tmp_path)
    first = (tmp_path / LATEST_PAGE_FILE).read_bytes()
    write(tmp_path, day=date(2026, 9, 1))
    second = (tmp_path / LATEST_PAGE_FILE).read_bytes()
    assert embedded(second.decode())["index"]["day"] == "2026-09-01"
    assert second != first
    write(tmp_path, day=date(2026, 8, 15))
    assert (tmp_path / LATEST_PAGE_FILE).read_bytes() == second


def test_index_lists_the_page_and_the_page_exists_before_it(tmp_path):
    index, _, directory = day_files(tmp_path)
    assert index["page"] == PAGE_FILE
    assert (directory / index["page"]).is_file()
    assert sorted(p.name for p in directory.iterdir()).count(PAGE_FILE) == 1
    assert len(list(directory.iterdir())) == 3 * len(LEVELS) + 2


# -- modo render -------------------------------------------------------------


@pytest.fixture
def lake(tmp_path):
    roots, _, _ = build_lake(tmp_path)
    assert cli.main(["--mode", "tiles", "--day", DAY], roots.env()) == 0
    return roots


def render_env(roots) -> dict[str, str]:
    """El entorno de `render`: sin L1 ni L2."""
    env = roots.env()
    del env["VIZ_LANDING_ROOT"], env["VIZ_EVENTS_ROOT"]
    return env


def day_path(roots) -> Path:
    return (
        Path(roots.tiles) / "provider=binance/market=spot/asset=BTCUSDT" / f"day={DAY}"
    )


def snapshot(directory: Path) -> dict[str, tuple[int, bytes]]:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}


def render_summaries(roots) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for path in sorted(Path(roots.dq).rglob("*.parquet")):
        rows += pq.read_table(path).to_pylist()
    found = [r for r in rows if r["check_type"] == "render_summary"]
    for row in found:
        row["details"] = json.loads(row["details"])
    return sorted(found, key=lambda r: r["detected_at"])


@pytest.fixture
def fresh_template(monkeypatch):
    """`load_template` guarda en caché: cada prueba empieza y termina limpia."""
    render.load_template.cache_clear()
    yield monkeypatch
    render.load_template.cache_clear()


def test_render_needs_only_the_tiles_and_dq_roots(lake, fresh_template):
    assert cli.main(["--mode", "render", "--day", DAY], render_env(lake)) == 0


@pytest.mark.parametrize("missing", ["VIZ_TILES_ROOT", "VIZ_DQ_ROOT"])
def test_render_missing_root_exits_2(lake, missing, capsys):
    env = {k: v for k, v in render_env(lake).items() if k != missing}
    assert cli.main(["--mode", "render", "--day", DAY], env) == 2
    assert missing in capsys.readouterr().err


def test_render_skips_a_page_that_is_up_to_date(lake, fresh_template):
    before = snapshot(day_path(lake))
    latest = (Path(lake.tiles) / LATEST_PAGE_FILE).stat().st_mtime_ns
    time.sleep(0.01)
    assert cli.main(["--mode", "render", "--day", DAY], render_env(lake)) == 0
    assert snapshot(day_path(lake)) == before
    assert (Path(lake.tiles) / LATEST_PAGE_FILE).stat().st_mtime_ns == latest
    (summary,) = render_summaries(lake)
    assert summary["details"]["skipped"] is True
    assert summary["mode"] == "render"
    assert summary["details"]["template_hash"] == render.load_template().hash


def test_render_rebuilds_when_the_template_changes(lake, tmp_path, fresh_template):
    site = tmp_path / "site"
    shutil.copytree(SITE, site)
    (site / "style.css").write_text((site / "style.css").read_text() + "\n/* v2 */\n")
    fresh_template.setenv("VIZ_TEMPLATE_DIR", str(site))
    render.load_template.cache_clear()

    before = snapshot(day_path(lake))
    assert cli.main(["--mode", "render", "--day", DAY], render_env(lake)) == 0
    after = snapshot(day_path(lake))

    # Solo cambia la página; los tiles y el índice quedan intactos.
    changed = {n for n in after if after[n] != before[n]}
    assert changed == {PAGE_FILE}
    page = after[PAGE_FILE][1].decode()
    assert "/* v2 */" in page
    assert render.page_meta(page.encode()) == (
        TILES_VERSION,
        render.load_template().hash,
    )
    # El último día arrastra también `latest.html`.
    assert (Path(lake.tiles) / LATEST_PAGE_FILE).read_text() == page
    # Y la página nueva sigue reproduciendo el content_hash del índice.
    files = {n: base64.b64decode(b) for n, b in embedded(page)["files"].items()}
    index = json.loads(after["index.json"][1])
    assert content_hash(files) == index["content_hash"]
    assert render_summaries(lake)[-1]["details"]["skipped"] is False

    # Segunda corrida: ya está al día.
    again = snapshot(day_path(lake))
    assert cli.main(["--mode", "render", "--day", DAY], render_env(lake)) == 0
    assert snapshot(day_path(lake)) == again


def test_render_force_rewrites_the_same_bytes(lake, fresh_template):
    before = (day_path(lake) / PAGE_FILE).read_bytes()
    assert (
        cli.main(["--mode", "render", "--day", DAY, "--force"], render_env(lake)) == 0
    )
    assert (day_path(lake) / PAGE_FILE).read_bytes() == before
    assert render_summaries(lake)[-1]["details"]["skipped"] is False


def test_render_restores_a_missing_latest_page(lake, fresh_template):
    (Path(lake.tiles) / LATEST_PAGE_FILE).unlink()
    assert cli.main(["--mode", "render", "--day", DAY], render_env(lake)) == 0
    assert (Path(lake.tiles) / LATEST_PAGE_FILE).read_bytes() == (
        day_path(lake) / PAGE_FILE
    ).read_bytes()


def test_render_does_not_read_l1_or_l2(lake, fresh_template, monkeypatch):
    import shutil as sh

    sh.rmtree(lake.landing)
    sh.rmtree(lake.events)
    assert (
        cli.main(["--mode", "render", "--day", DAY, "--force"], render_env(lake)) == 0
    )


def test_render_month_range_renders_only_days_with_tiles(lake, fresh_template):
    (day_path(lake) / PAGE_FILE).unlink()
    assert cli.main(["--mode", "render", "--from", "2017-08"], render_env(lake)) == 0
    assert (day_path(lake) / PAGE_FILE).is_file()
    assert [d.name for d in day_path(lake).parent.iterdir()] == [f"day={DAY}"]


def test_render_without_arguments_takes_the_previous_month(
    lake, fresh_template, monkeypatch
):
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 9, 5))
    (day_path(lake) / PAGE_FILE).unlink()
    assert cli.main(["--mode", "render"], render_env(lake)) == 0
    assert (day_path(lake) / PAGE_FILE).is_file()


def test_render_of_a_day_without_tiles_is_input_missing(lake, fresh_template):
    import pyarrow.parquet as pq

    assert cli.main(["--mode", "render", "--day", "2017-08-19"], render_env(lake)) == 1
    rows = [
        r
        for path in Path(lake.dq).rglob("*.parquet")
        for r in pq.read_table(path).to_pylist()
        if r["check_type"] == "input_missing"
    ]
    (row,) = rows
    assert row["mode"] == "render"
    assert json.loads(row["details"]) == {"day": "2017-08-19", "what": "tiles"}


def test_render_of_a_month_without_tiles_is_input_missing(lake, fresh_template):
    assert cli.main(["--mode", "render", "--from", "2017-10"], render_env(lake)) == 1


def test_render_refuses_tiles_that_do_not_match_their_content_hash(
    lake, fresh_template
):
    page = day_path(lake) / PAGE_FILE
    before = page.read_bytes()
    tile = day_path(lake) / "price-128.i32"
    raw = bytearray(tile.read_bytes())
    raw[0] ^= 0xFF
    tile.write_bytes(bytes(raw))
    assert (
        cli.main(["--mode", "render", "--day", DAY, "--force"], render_env(lake)) == 1
    )
    assert page.read_bytes() == before


def test_render_refuses_a_day_with_a_missing_tile(lake, fresh_template):
    (day_path(lake) / "dir-4096.u8").unlink()
    assert (
        cli.main(["--mode", "render", "--day", DAY, "--force"], render_env(lake)) == 1
    )


def test_tiles_mode_still_rejects_export(lake):
    assert cli.main(["--mode", "export", "--day", DAY], lake.env()) == 2


def test_page_is_served_unchanged_by_a_static_http_server(tmp_path):
    """`python -m http.server` entrega el mismo documento: no depende de `file://`."""
    import functools
    import http.server
    import threading
    import urllib.request

    _, _, directory = day_files(tmp_path)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/{PAGE_FILE}"
        with urllib.request.urlopen(url, timeout=10) as response:
            served = response.read()
            assert response.headers["Content-Type"].startswith("text/html")
        server.shutdown()
    assert served == (directory / PAGE_FILE).read_bytes()
