"""Qué pide el `GcsFileSystem` de Arrow al publicar `latest.html` (ITSC-318).

La cuenta del job solo tiene `storage.objects.*` bajo el prefijo `tiles/`; consultar
el objeto `tiles` (sin barra), padre de `tiles/latest.html`, da 403. Aquí un GCS falso
por HTTP lo niega y registra las peticiones, para que la decisión de ADR-VZ-10 se apoye
en lo que hace Arrow y no en un doble de prueba que la suponga.
"""

import gzip

import pytest
import viz_tiles.write as write_module
from gcs_fake import BUCKET, FakeGcs
from test_viz_write import write
from viz_tiles.contract import EVENT_BYTES
from viz_tiles.write import stream_latest_page, stream_page

DAY_PREFIX = "tiles/provider=binance/market=spot/asset=BTCUSDT/day=2026-08-31"


def parts():
    yield "events.bin", [b"\0" * (4 * EVENT_BYTES)]
    yield "ticks.bin", [b"\0" * 16]


@pytest.fixture
def gcs():
    with FakeGcs(denied=("tiles",)) as fake:
        yield fake


@pytest.fixture
def index(tmp_path):
    return write(tmp_path)[1]


@pytest.mark.parametrize("step", ["move", "copy_file"])
def test_moving_or_copying_into_the_root_asks_for_its_parent(gcs, step):
    """Por eso `latest.html` no se publica con `move` ni con `copy_file`."""
    fs = gcs.filesystem
    with fs.open_output_stream(f"{BUCKET}/{DAY_PREFIX}/latest.html.tmp") as out:
        out.write(b"x")
    with pytest.raises(OSError):
        getattr(fs, step)(
            f"{BUCKET}/{DAY_PREFIX}/latest.html.tmp", f"{BUCKET}/tiles/latest.html"
        )
    assert gcs.asked_for("tiles")


def test_writing_a_stream_into_the_root_does_not_ask_for_its_parent(gcs):
    fs = gcs.filesystem
    with fs.open_output_stream(f"{BUCKET}/tiles/latest.html") as out:
        out.write(b"x")
    assert gcs.objects["tiles/latest.html"][0] == b"x"
    assert not gcs.asked_for("tiles")


def test_latest_page_reaches_the_bucket_root_without_asking_for_its_parent(gcs, index):
    fs = gcs.filesystem
    directory = f"{BUCKET}/{DAY_PREFIX}"
    stream_page(fs, f"{directory}/index.html", index, parts(), None)
    stream_latest_page(fs, f"{BUCKET}/tiles", directory, index, parts(), None)

    page, meta = gcs.objects["tiles/latest.html"]
    # Los mismos bytes y los mismos metadatos que la página del día.
    assert (page, meta) == gcs.objects[f"{DAY_PREFIX}/index.html"]
    assert meta["contentEncoding"] == "gzip"
    assert gzip.decompress(page).startswith(b"<!")
    assert not [name for name in gcs.objects if name.endswith(".tmp")]
    assert not gcs.asked_for("tiles")


def test_a_failed_copy_to_the_root_leaves_no_temporary(gcs, index, monkeypatch):
    fs = gcs.filesystem
    directory = f"{BUCKET}/{DAY_PREFIX}"

    def broken(*args, **kwargs):
        raise OSError("se cortó la lectura del temporal")

    monkeypatch.setattr("viz_tiles.write.read_blocks", broken)
    with pytest.raises(OSError, match="se cortó"):
        stream_latest_page(fs, f"{BUCKET}/tiles", directory, index, parts(), None)
    assert not [name for name in gcs.objects if name.endswith(".tmp")]


def test_latest_page_has_the_bytes_of_the_day_page_despite_transcoding(gcs, index):
    """GCS descomprime al releer un objeto gzip (ITSC-320): el temporal va plano."""
    fs = gcs.filesystem
    directory = f"{BUCKET}/{DAY_PREFIX}"
    stream_page(fs, f"{directory}/index.html", index, parts(), None)
    stream_latest_page(fs, f"{BUCKET}/tiles", directory, index, parts(), None)

    page, page_meta = gcs.objects[f"{DAY_PREFIX}/index.html"]
    latest, latest_meta = gcs.objects["tiles/latest.html"]
    assert page[:2] == b"\x1f\x8b" and latest[:2] == b"\x1f\x8b"
    assert gzip.decompress(latest) == gzip.decompress(page)
    assert latest_meta == page_meta
    assert len(latest) == len(page)
    assert (
        fs.get_file_info(f"{directory}/index.html").size
        == fs.get_file_info(f"{BUCKET}/tiles/latest.html").size
    )


def test_the_temporary_is_written_plain_and_unlabeled(gcs, index, monkeypatch):
    fs = gcs.filesystem
    seen = {}
    real = write_module.read_blocks

    def spy(fs, path):
        seen["encoding"] = gcs.objects[path.removeprefix(f"{BUCKET}/")][1].get(
            "contentEncoding"
        )
        seen["head"] = gcs.objects[path.removeprefix(f"{BUCKET}/")][0][:2]
        return real(fs, path)

    monkeypatch.setattr("viz_tiles.write.read_blocks", spy)
    stream_latest_page(
        fs, f"{BUCKET}/tiles", f"{BUCKET}/{DAY_PREFIX}", index, parts(), None
    )
    assert seen["encoding"] is None
    assert seen["head"] != b"\x1f\x8b"


def _stage_day(gcs, index, monkeypatch):
    """El día del índice en el bucket falso, listo para `advance_latest_from_files`."""

    fs = gcs.filesystem
    directory = f"{BUCKET}/{DAY_PREFIX}"
    for name, blocks in parts():
        with fs.open_output_stream(f"{directory}/{name}") as out:
            for block in blocks:
                out.write(block)
    stream_page(fs, f"{directory}/index.html", index, parts(), None)
    monkeypatch.setattr(
        write_module, "resolve_fs", lambda root: (fs, f"{BUCKET}/tiles")
    )
    monkeypatch.setattr(write_module, "day_dir", lambda *a, **k: directory)
    return fs


def test_a_latest_page_that_arrived_plain_is_rewritten(gcs, index, monkeypatch):
    """El `latest.html` roto de ITSC-320 pesa más que la página: se rehace sin borrar nada."""
    from viz_tiles.write import advance_latest_from_files

    _stage_day(gcs, index, monkeypatch)
    advance_latest_from_files("gs://bucket/tiles", index)
    good = gcs.objects["tiles/latest.html"]
    assert good == gcs.objects[f"{DAY_PREFIX}/index.html"]

    plain = gzip.decompress(good[0])
    gcs.objects["tiles/latest.html"] = (plain, good[1])
    advance_latest_from_files("gs://bucket/tiles", index)
    assert gcs.objects["tiles/latest.html"] == good

    # Ya es copia: una corrida más no la vuelve a escribir.
    before = len([r for r in gcs.requests if r[0] == "POST"])
    advance_latest_from_files("gs://bucket/tiles", index)
    assert len([r for r in gcs.requests if r[0] == "POST"]) == before
