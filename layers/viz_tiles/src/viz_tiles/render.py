"""La página HTML autocontenida de un día (TRD-viz §6.5 y §7.9).

Una plantilla (`layers/viz_tiles/site/`: HTML, JS y CSS propios más uPlot) se
rellena con los dos archivos del día (`ticks.bin` y `events.bin`) en base64, bajo
su nombre, dentro de un `<script>` (`window.VIZ_DATA`). El documento no pide nada a
la red: Google sirve cada archivo privado de un bucket desde un dominio bloqueado
de un solo uso, así que una página que descarga sus datos con `fetch` no funciona
(RVZ-06).

Memoria: los archivos entran de uno en uno y por tramos de unos cientos de KB, se
codifican y se sueltan; nunca vive más de un tramo y su base64, y la salida es un
solo documento.
"""

import base64
import functools
import gzip
import hashlib
import json
import os
import re
import zlib
from collections.abc import Buffer, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from viz_tiles.contract import PAGE_FILE

# Archivos de la plantilla, en el orden en que entran al hash.
TEMPLATE_FILES = (
    PAGE_FILE,
    "app.js",
    "style.css",
    "vendor/uPlot.iife.min.js",
    "vendor/uPlot.min.css",
)

# Marcadores de `index.html` y el archivo con que se reemplaza cada uno.
_SOURCES = {
    "UPLOT_JS": "vendor/uPlot.iife.min.js",
    "UPLOT_CSS": "vendor/uPlot.min.css",
    "APP_JS": "app.js",
    "STYLE_CSS": "style.css",
}
_MARKER = re.compile(r"@@([A-Z_]+)@@")
_DATA_MARKER = "@@VIZ_DATA@@"

# La huella de la plantilla vive en el propio HTML, en un `<meta>` al principio
# del documento: el modo `render` la lee sin bajar la página entera.
_META = re.compile(
    r'<meta name="viz-render" content="tiles_version=([^;"]+);template=([0-9a-f]{64})">'
)
# Bytes iniciales que bastan para encontrar la huella.
META_PROBE = 8192


@dataclass(frozen=True)
class Template:
    """La plantilla leída del disco: sus fuentes y la huella de todos sus archivos."""

    sources: Mapping[str, str]
    hash: str


def template_dir() -> Path:
    """Dónde está `site/`: `VIZ_TEMPLATE_DIR` en la imagen, el repo en desarrollo."""
    given = os.environ.get("VIZ_TEMPLATE_DIR")
    if given:
        return Path(given)
    return Path(__file__).resolve().parents[2] / "site"


@functools.cache
def load_template(directory: Path | None = None) -> Template:
    """Lee la plantilla y calcula su huella: SHA-256 de `nombre, byte nulo, bytes`."""
    root = directory or template_dir()
    digest = hashlib.sha256()
    sources: dict[str, str] = {}
    for name in TEMPLATE_FILES:
        raw = (root / name).read_bytes()
        digest.update(name.encode() + b"\0")
        digest.update(raw)
        sources[name] = raw.decode()
    return Template(sources=sources, hash=digest.hexdigest())


def render_meta(tiles_version: str, template: Template) -> str:
    return f"tiles_version={tiles_version};template={template.hash}"


def _fill(source: str, values: Mapping[str, str]) -> str:
    """Una sola pasada: lo que se inserta no se vuelve a escanear en busca de marcadores."""
    return _MARKER.sub(lambda m: values[m.group(1)], source)


def _json(doc: object) -> str:
    """JSON apto para ir dentro de un `<script>`: sin `<` que cierre la etiqueta."""
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def expected_names(index: Mapping) -> list[str]:
    """Los archivos de datos que el índice lista, en orden de nombre."""
    return sorted([index["events"], index["ticks_file"]])


# Bytes por tramo al codificar en base64: múltiplo de 3, así los tramos no dejan
# relleno (`=`) en medio del texto.
B64_STEP = 3 << 18


def _base64(parts: Sequence[Buffer]) -> Iterator[str]:
    """El base64 de los tramos de un archivo, sin juntarlos en un solo buffer."""
    carry = b""
    for part in parts:
        view = memoryview(part)
        for off in range(0, len(view), B64_STEP):
            chunk = carry + view[off : off + B64_STEP]
            cut = len(chunk) - len(chunk) % 3
            yield base64.b64encode(chunk[:cut]).decode("ascii")
            carry = chunk[cut:]
    if carry:
        yield base64.b64encode(carry).decode("ascii")


def page_chunks(
    index: Mapping,
    arrays: Iterable[tuple[str, Sequence[Buffer]]],
    template: Template,
) -> Iterator[str]:
    """El documento en pedazos: cada archivo se codifica cuando llega y se suelta."""
    values = {key: template.sources[name] for key, name in _SOURCES.items()} | {
        "RENDER_META": render_meta(index["tiles_version"], template)
    }
    head, tail = template.sources[PAGE_FILE].split(_DATA_MARKER)
    yield _fill(head, values)
    yield (
        f'{{"tiles_version":{_json(index["tiles_version"])},'
        f'"generated_at":{_json(index["generated_at"])},"files":{{'
    )
    names: list[str] = []
    for name, parts in arrays:
        names.append(name)
        yield f'{"," if len(names) > 1 else ""}{_json(name)}:"'
        yield from _base64(parts)
        yield '"'
    yield f'}},"index":{_json(index)}}}'
    yield _fill(tail, values)
    if names != expected_names(index):
        raise ValueError(
            f"los arreglos recibidos {names} no son los que el índice lista "
            f"{expected_names(index)}"
        )


def render_day(
    index: Mapping,
    arrays: Iterable[tuple[str, Sequence[Buffer]]],
    *,
    template: Template | None = None,
    compress: bool = False,
) -> bytes:
    """El `index.html` de un día: la plantilla con `index` y sus dos archivos dentro.

    `arrays` entrega `(nombre, tramos)` en orden de nombre (el de `content_hash`);
    los tramos de un archivo, puestos uno tras otro, son sus bytes.
    Con `compress` el resultado es un gzip determinista (sin marca de tiempo): es
    lo que se guarda en un bucket con `Content-Encoding: gzip`; en disco local el
    archivo va sin comprimir para que abra por `file://`.
    """
    template = template or load_template()
    chunks = (chunk.encode() for chunk in page_chunks(index, arrays, template))
    if not compress:
        return b"".join(chunks)
    out = bytearray()
    with gzip.GzipFile(fileobj=_Sink(out), mode="wb", mtime=0, compresslevel=9) as gz:
        for chunk in chunks:
            gz.write(chunk)
    return bytes(out)


class _Sink:
    """Destino de `GzipFile` que acumula en un `bytearray`."""

    def __init__(self, out: bytearray) -> None:
        self._out = out

    def write(self, data: bytes) -> int:
        self._out += data
        return len(data)

    def flush(self) -> None:
        pass


def page_meta(head: bytes) -> tuple[str, str] | None:
    """`(tiles_version, hash de la plantilla)` guardados en el comienzo de una página.

    `head` son los primeros bytes tal como los entregó el almacenamiento: texto
    plano o gzip (según si el sistema de archivos descomprime al leer).
    """
    if head[:2] == b"\x1f\x8b":
        head = zlib.decompressobj(31).decompress(head, META_PROBE)
    match = _META.search(head[:META_PROBE].decode("utf-8", errors="replace"))
    return (match.group(1), match.group(2)) if match else None
