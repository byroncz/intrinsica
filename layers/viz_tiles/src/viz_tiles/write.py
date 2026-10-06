"""Escritura de un día de tiles: arreglos planos, la página, `index.json` y `latest.json`.

TRD-viz §7.2, §7.7 y ADR-VZ-10. El `index.json` se escribe al final y es la
marca de commit: un día sin él no existe para el tablero. Si el job muere a
medias, el día queda sin índice y la siguiente corrida lo rehace. La página
`index.html` entra antes que el índice, así que un índice implica su página.
"""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow.fs as pafs
from pyutils.fs import resolve_fs

from viz_tiles.contract import (
    EVENTS_FILE,
    FILE_BY_KIND,
    INDEX_FIELDS,
    INDEX_FILE,
    KINDS,
    LATEST_FIELDS,
    LATEST_FILE,
    LATEST_PAGE_FILE,
    LEVELS,
    PAGE_FILE,
    TILES_VERSION,
    tile_name,
)
from viz_tiles.events import Confirmations, EventsBuffer
from viz_tiles.reduce import DayReduction, day_start_us
from viz_tiles.render import Template, render_day

# Metadatos de cada clase de objeto (TRD-viz §7.2). Los binarios son inmutables;
# el índice, `latest.json` y las páginas se revalidan en cada apertura.
BINARY_META = {
    "Content-Type": "application/octet-stream",
    "Cache-Control": "private, max-age=31536000, immutable",
}
JSON_META = {"Content-Type": "application/json", "Cache-Control": "no-cache"}
PAGE_META = {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"}
GZIP_META = {"Content-Encoding": "gzip"}


class SeriesMismatch(ValueError):
    """La raíz de tiles es de otra serie (activo, proveedor o mercado)."""


@dataclass(frozen=True)
class ThetaTiles:
    """Tiles de un θ: dirección por nivel y cuántos eventos suyos tocan el día.

    `direction[w]` es el arreglo uint8 del nivel `w`. Los eventos mismos no viajan
    aquí: van al `EventsBuffer` del día, `events` por θ y en el orden de `thetas`.
    """

    theta: str
    events: int
    provisional_from_s: float | None
    direction: Mapping[int, np.ndarray]  # un bloque de `w` bytes por nivel


def day_dir(root: str | Path, provider: str, market: str, asset: str, day: date) -> str:
    """Directorio del día bajo `root`: `provider=…/market=…/asset=…/day=YYYY-MM-DD`."""
    base = resolve_fs(root)[1]
    return (
        f"{base}/provider={provider}/market={market}/asset={asset}"
        f"/day={day.isoformat()}"
    )


def _tile_view(kind: str, w: int, array: np.ndarray, blocks: int = 1) -> memoryview:
    """Vista de los bytes del archivo: el arreglo con el tipo y el largo del contrato.

    No copia: con el `dtype` del contrato `astype(copy=False)` devuelve el mismo
    arreglo y `memoryview` lo expone sin duplicarlo.

    `blocks` es el número de θ de un archivo `dir` (un bloque de `w` por θ).
    """
    spec = FILE_BY_KIND[kind]
    expected = spec.per_column * w * blocks
    if array.shape != (expected,):
        raise ValueError(
            f"{kind}-{w}: se esperaban {expected} valores, hay {array.shape}"
        )
    return memoryview(np.ascontiguousarray(array.astype(spec.dtype, copy=False)))


def _put(
    fs: pafs.FileSystem,
    path: str,
    data: bytes | memoryview,
    metadata: dict[str, str] | None = None,
) -> None:
    """Escribe un objeto con sus metadatos; el disco local los ignora."""
    with fs.open_output_stream(path, metadata=metadata) as out:
        out.write(data)


def _put_whole(
    fs: pafs.FileSystem, path: str, data: bytes, metadata: dict[str, str]
) -> None:
    """Escribe un archivo entero o nada: en disco local, temporal y renombre."""
    if isinstance(fs, pafs.LocalFileSystem):
        tmp = f"{path}.tmp"
        _put(fs, tmp, data, metadata)
        fs.move(tmp, path)
    else:
        _put(fs, path, data, metadata)


def _put_json(fs: pafs.FileSystem, path: str, doc: dict) -> None:
    data = (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode()
    _put_whole(fs, path, data, JSON_META)


def compresses_pages(fs: pafs.FileSystem) -> bool:
    """Si las páginas van en gzip: en un bucket sí; en disco no, para abrir por `file://`."""
    return not isinstance(fs, pafs.LocalFileSystem)


def write_page(fs: pafs.FileSystem, path: str, page: bytes) -> None:
    """Escribe una página ya renderizada (con `compress` igual a `compresses_pages(fs)`)."""
    meta = PAGE_META | (GZIP_META if compresses_pages(fs) else {})
    _put_whole(fs, path, page, meta)


def _exists(fs: pafs.FileSystem, path: str) -> bool:
    return fs.get_file_info(path).type == pafs.FileType.File


def write_day(
    root: str | Path,
    *,
    provider: str,
    market: str,
    asset: str,
    day: date,
    reduction: DayReduction,
    confirmations: Confirmations,
    events: EventsBuffer | None = None,
    thetas: Sequence[ThetaTiles],
    missing_thetas: Sequence[str] = (),
    input_hash: str,
    image_version: str,
    generated_at: datetime | None = None,
    template: Template | None = None,
) -> dict:
    """Escribe los tiles del día y devuelve el `index.json` que dejó.

    Orden: se borra el índice previo, se escriben los arreglos, la página
    `index.html` y al final el índice (marca de commit); después `latest.html` y
    `latest.json`, que solo avanzan. El `input_hash` lo calcula quien llama
    (TRD-viz §7.8); `content_hash` sale de los bytes de los arreglos y no depende
    de `generated_at`.
    """
    fs, base = resolve_fs(root)
    # Antes de escribir nada: un día de otra serie no se deja a medias en la raíz.
    _read_latest(fs, base, {"provider": provider, "market": market, "asset": asset})
    directory = day_dir(root, provider, market, asset, day)
    if isinstance(fs, pafs.LocalFileSystem):
        # En GCS no hay directorios y `create_dir` pide `storage.buckets.get`, que
        # el rol por prefijo no da; `open_output_stream` crea la ruta al escribir.
        fs.create_dir(directory)
    index_path = f"{directory}/{INDEX_FILE}"
    if _exists(fs, index_path):
        fs.delete_file(index_path)

    # Nombre de archivo -> (tipo, nivel). Los arreglos se hashean y se escriben
    # desde una vista del que ya está en RAM; la dirección de un nivel se arma al
    # escribirlo, un nivel a la vez. `events.bin` (nivel 0) va aparte: no es por nivel.
    names = {kind: {str(w): tile_name(kind, w) for w in LEVELS} for kind in KINDS}
    files: dict[str, tuple[str, int]] = {EVENTS_FILE: ("events", 0)}
    for kind, by_level in names.items():
        for w in LEVELS:
            files[by_level[str(w)]] = (kind, w)

    events = events if events is not None else EventsBuffer()
    declared = sum(t.events for t in thetas)
    if len(events) != declared:
        raise ValueError(
            f"los θ declaran {declared} eventos pero el buffer trae {len(events)} filas"
        )
    offsets = np.cumsum([0] + [t.events for t in thetas])

    def array_of(kind: str, w: int) -> np.ndarray:
        if kind == "count":
            return reduction.count[w]
        if kind in ("confirms", "simul"):
            return getattr(confirmations, kind)[w]
        if kind == "dir":
            # n bloques de w bytes, en el orden de `thetas` del índice.
            blocks = [t.direction[w] for t in thetas]
            for theta, block in zip(thetas, blocks, strict=True):
                if block.shape != (w,):
                    raise ValueError(
                        f"dir-{w} de {theta.theta}: se esperaban {w} valores, "
                        f"hay {block.shape}"
                    )
            return np.concatenate(blocks) if blocks else np.empty(0, "u1")
        return getattr(reduction, kind)[w]

    theta_docs = [
        {
            "theta": theta.theta,
            "events": theta.events,
            "events_offset": int(offset),
            "provisional_from_s": theta.provisional_from_s,
        }
        for theta, offset in zip(thetas, offsets[:-1], strict=True)
    ]

    def views() -> Iterator[tuple[str, memoryview]]:
        """Los arreglos en orden de nombre (el de `content_hash`), de uno en uno."""
        for name in sorted(files):
            kind, w = files[name]
            if kind == "events":
                yield name, events.packed()
                continue
            yield (
                name,
                _tile_view(
                    kind, w, array_of(kind, w), len(thetas) if kind == "dir" else 1
                ),
            )

    digest = hashlib.sha256()
    for name, view in views():
        digest.update(name.encode() + b"\0")
        digest.update(view)
        _put(fs, f"{directory}/{name}", view, BINARY_META)

    when = generated_at or datetime.now(UTC)
    values = {
        "tiles_version": TILES_VERSION,
        "provider": provider,
        "market": market,
        "asset": asset,
        "day": day.isoformat(),
        "t0": day_start_us(day),
        "price_scale": reduction.price_scale,
        "ticks": reduction.ticks,
        "levels": list(LEVELS),
        "price": names["price"],
        "volume": names["volume"],
        "dir": names["dir"],
        "count": names["count"],
        "confirms": names["confirms"],
        "simul": names["simul"],
        "events": EVENTS_FILE,
        "page": PAGE_FILE,
        "thetas": theta_docs,
        "missing_thetas": list(missing_thetas),
        "input_hash": input_hash,
        "content_hash": digest.hexdigest(),
        "generated_at": when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_version": image_version,
    }
    index = {name: values[name] for name, _ in INDEX_FIELDS}
    # La página lleva los mismos arreglos: se vuelven a armar de uno en uno en vez
    # de retenerlos, y se escribe antes del índice (marca de commit).
    page = render_day(index, views(), template=template, compress=compresses_pages(fs))
    write_page(fs, f"{directory}/{PAGE_FILE}", page)
    _put_json(fs, index_path, index)
    _advance_latest(fs, base, index, page)
    return index


def _read_latest(fs: pafs.FileSystem, base: str, series: dict) -> dict | None:
    """El `latest.json` de la raíz, o `None` si no existe.

    La raíz es mono-activo (TRD-viz §7.7): si apunta a otra serie, lanza
    `ValueError` en vez de pisarla o dejarla retroceder.
    """
    path = f"{base}/{LATEST_FILE}"
    if not _exists(fs, path):
        return None
    with fs.open_input_stream(path) as src:
        current = json.loads(src.read())
    if any(current.get(k) != v for k, v in series.items()):
        other = {k: current.get(k) for k in series}
        raise SeriesMismatch(
            f"{LATEST_FILE} es de otra serie {other}: la raíz de tiles es "
            f"mono-activo y no admite {series}"
        )
    return current


def _advance_latest(
    fs: pafs.FileSystem, base: str, index: dict, page: bytes | None = None
) -> None:
    """Escribe `latest.json` y `latest.html` si el día no es anterior al que apunta.

    `latest.html` es la copia de la página del día; va primero para que el puntero
    no apunte a una página que todavía no existe.
    """
    series = {k: index[k] for k in ("provider", "market", "asset")}
    current = _read_latest(fs, base, series)
    if current is not None and current["day"] > index["day"]:
        return
    if page is not None:
        write_page(fs, f"{base}/{LATEST_PAGE_FILE}", page)
    _put_json(
        fs, f"{base}/{LATEST_FILE}", {name: index[name] for name in LATEST_FIELDS}
    )


def read_index(
    root: str | Path, provider: str, market: str, asset: str, day: date
) -> dict:
    """El `index.json` de un día (local)."""
    path = Path(day_dir(root, provider, market, asset, day)) / INDEX_FILE
    return json.loads(path.read_text())


def find_index(
    root: str | Path, provider: str, market: str, asset: str, day: date
) -> dict | None:
    """El `index.json` del día en `root` (local o `gs://`), o `None` si no existe."""
    fs, _ = resolve_fs(root)
    path = f"{day_dir(root, provider, market, asset, day)}/{INDEX_FILE}"
    if not _exists(fs, path):
        return None
    with fs.open_input_stream(path) as src:
        return json.loads(src.read())


def day_objects(
    root: str | Path, provider: str, market: str, asset: str, day: date
) -> tuple[int, int]:
    """`(objetos, bytes)` que el día ocupa bajo `root` (un solo listado)."""
    fs, _ = resolve_fs(root)
    selector = pafs.FileSelector(
        day_dir(root, provider, market, asset, day), allow_not_found=True
    )
    files = [i for i in fs.get_file_info(selector) if i.type == pafs.FileType.File]
    return len(files), sum(i.size or 0 for i in files)
