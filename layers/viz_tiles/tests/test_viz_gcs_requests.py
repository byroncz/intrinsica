"""Qué pide el `GcsFileSystem` de Arrow al publicar `latest.html` (ITSC-318).

La cuenta del job solo tiene `storage.objects.*` bajo el prefijo `tiles/`; consultar
el objeto `tiles` (sin barra), padre de `tiles/latest.html`, da 403. Aquí un GCS falso
por HTTP lo niega y registra las peticiones, para que la decisión de ADR-VZ-10 se apoye
en lo que hace Arrow y no en un doble de prueba que la suponga.
"""

import gzip

import pytest
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
