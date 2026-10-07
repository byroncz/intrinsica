"""Escritura de un día de tiles: arreglos planos, la página, `index.json` y `latest.json`.

TRD-viz §7.2, §7.7 y ADR-VZ-10. El `index.json` se escribe al final y es la
marca de commit: un día sin él no existe para el tablero. Si el job muere a
medias, el día queda sin índice y la siguiente corrida lo rehace. La página
`index.html` entra antes que el índice, así que un índice implica su página.
"""

import hashlib
import json
import logging
from collections.abc import Buffer, Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.fs as pafs
from pyutils.fs import resolve_fs

from viz_tiles.contract import (
    EVENTS_FILE,
    INDEX_FIELDS,
    INDEX_FILE,
    LATEST_FIELDS,
    LATEST_FILE,
    LATEST_PAGE_FILE,
    PAGE_FILE,
    TICKS_FILE,
    TILES_VERSION,
)
from viz_tiles.events import EventsBuffer
from viz_tiles.render import Template, expected_names, gzip_to, render_day_to
from viz_tiles.ticks import DayTicks, TicksReader, day_start_us

logger = logging.getLogger(__name__)

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
class ThetaEvents:
    """Un θ del día: cuántos eventos suyos tocan el día y desde cuándo son provisionales.

    Los eventos mismos no viajan aquí: van al `EventsBuffer` del día, `events` por θ
    y en el orden de `thetas`.
    """

    theta: str
    events: int
    provisional_from_s: float | None


def day_dir(root: str | Path, provider: str, market: str, asset: str, day: date) -> str:
    """Directorio del día bajo `root`: `provider=…/market=…/asset=…/day=YYYY-MM-DD`."""
    base = resolve_fs(root)[1]
    return (
        f"{base}/provider={provider}/market={market}/asset={asset}"
        f"/day={day.isoformat()}"
    )


def _put(
    fs: pafs.FileSystem,
    path: str,
    data: bytes | memoryview | Sequence[bytes | memoryview],
    metadata: dict[str, str] | None = None,
) -> None:
    """Escribe un objeto con sus metadatos; el disco local los ignora.

    `data` puede ser una secuencia de tramos (`ticks.bin` son tres secciones): se
    escriben uno tras otro, sin juntarlos en un solo buffer.
    """
    parts = [data] if isinstance(data, bytes | memoryview) else data
    with fs.open_output_stream(path, metadata=metadata) as out:
        for part in parts:
            out.write(part)


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


def _publish_page(
    fs: pafs.FileSystem,
    tmp: str,
    index: dict,
    arrays: Iterator[tuple[str, Iterable[Buffer]]],
    template: Template | None,
    publish: Callable[[str], None],
    *,
    plain_tmp: bool = False,
) -> None:
    """Renderiza la página en `tmp` y la publica con `publish(tmp)`, que consume `tmp`.

    Mismos metadatos que `write_page`, salvo con `plain_tmp`: el temporal va sin
    comprimir y sin `Content-Encoding`, para quien lo relee con Arrow (GCS descomprime
    al servir un objeto etiquetado gzip; ver `stream_latest_page`). El temporal va
    también en un bucket: si el render falla a medias, el `with` cierra el flujo y GCS
    publica lo escrito (pyarrow no tiene `abort`); así cae en el temporal, que se
    borra, y no sobre la página vigente. El temporal se borra también si falla
    `publish`, no solo el render; ese borrado no tapa el error original.
    """
    compress = compresses_pages(fs) and not plain_tmp
    meta = PAGE_META | (GZIP_META if compress else {})
    try:
        with fs.open_output_stream(tmp, metadata=meta) as out:
            render_day_to(out, index, arrays, template=template, compress=compress)
        publish(tmp)
    except BaseException:
        _discard_tmp(fs, tmp)
        raise


def stream_page(
    fs: pafs.FileSystem,
    path: str,
    index: dict,
    arrays: Iterator[tuple[str, Iterable[Buffer]]],
    template: Template | None,
) -> None:
    """Renderiza la página de un día en un temporal, sin armarla en RAM, y lo renombra.

    En GCS `move` es copia más borrado y conserva los metadatos; consulta el padre
    del destino, así que solo sirve para rutas bajo `day=…/` (ver `stream_latest_page`).
    """
    _publish_page(
        fs, f"{path}.tmp", index, arrays, template, lambda tmp: fs.move(tmp, path)
    )


def stream_latest_page(
    fs: pafs.FileSystem,
    base: str,
    directory: str,
    index: dict,
    arrays: Iterator[tuple[str, Iterable[Buffer]]],
    template: Template | None,
) -> None:
    """Escribe `latest.html` en la raíz sin que ninguna operación consulte la raíz.

    En el `GcsFileSystem` de Arrow, `move` y `copy_file` pasan por el mismo
    `CopyFile`, que pregunta por el padre del destino: el objeto `tiles` (sin
    barra), fuera del prefijo `tiles/` en que la cuenta tiene permiso (ITSC-318;
    `test_viz_gcs_requests.py` lo mide). Solo `open_output_stream` no lo consulta.
    El render va a un temporal en el directorio del día, cuyo padre sí cae bajo el
    prefijo, y de ahí se copia por bloques a la raíz (O(bloque) en RAM); el temporal
    se borra siempre. En disco local no hay permisos por prefijo y `move` es atómico.

    Restricción (ITSC-320): GCS descomprime al servir un objeto con `Content-Encoding:
    gzip` si el cliente no pide gzip, y Arrow no lo pide; releerlo para copiarlo
    entrega bytes planos. Por eso el temporal va plano y sin etiqueta, y la copia a la
    raíz comprime al vuelo con los mismos parámetros que la página del día: el
    resultado es byte a byte el `index.html` del día, sin depender de cómo sirva GCS.
    """

    def publish(tmp: str) -> None:
        dest = f"{base}/{LATEST_PAGE_FILE}"
        if isinstance(fs, pafs.LocalFileSystem):
            fs.move(tmp, dest)
            return
        with (
            fs.open_output_stream(dest, metadata=PAGE_META | GZIP_META) as out,
            gzip_to(out) as gz,
        ):
            for block in read_blocks(fs, tmp):
                gz.write(block)
        fs.delete_file(tmp)

    _publish_page(
        fs,
        f"{directory}/{LATEST_PAGE_FILE}.tmp",
        index,
        arrays,
        template,
        publish,
        plain_tmp=True,
    )


def _discard_tmp(fs: pafs.FileSystem, path: str) -> None:
    """Borra un temporal que quedó tras un fallo; si no puede, avisa y no tapa el error."""
    try:
        if _exists(fs, path):
            fs.delete_file(path)
    except OSError as exc:
        logger.warning("no se pudo borrar el temporal %s: %s", path, exc)


def _exists(fs: pafs.FileSystem, path: str) -> bool:
    return fs.get_file_info(path).type == pafs.FileType.File


# Bytes por lectura al volver a leer un archivo del día: ni el archivo ni su
# base64 están enteros en RAM, solo un bloque a la vez.
READ_BLOCK = 1 << 20


def read_blocks(fs: pafs.FileSystem, path: str) -> Iterator[bytes]:
    """Los bytes de un archivo, de a `READ_BLOCK`, soltando cada bloque al entregarlo."""
    with fs.open_input_stream(path) as src:
        while block := src.read(READ_BLOCK):
            yield block


def day_blocks(
    fs: pafs.FileSystem, directory: str, index: dict
) -> Iterator[tuple[str, Iterator[bytes]]]:
    """Los archivos del día en orden de nombre, de uno en uno y por bloques."""
    for name in expected_names(index):
        yield name, read_blocks(fs, f"{directory}/{name}")


class TicksFile:
    """`ticks.bin` de un día, abierto para escribirlo tramo a tramo mientras se lee L1.

    El objeto se abre con el primer tramo: un día sin ticks no deja nada. Al abrirlo
    se borra el `index.json` previo del día (la marca de commit, TRD-viz §7.8): un
    `ticks.bin` a medias nunca queda bajo un índice que lo declara completo. Antes
    se comprueba que la raíz sea de la misma serie.
    """

    def __init__(
        self, root: str | Path, provider: str, market: str, asset: str, day: date
    ) -> None:
        self._fs, self._base = resolve_fs(root)
        self._series = {"provider": provider, "market": market, "asset": asset}
        self.directory = day_dir(root, provider, market, asset, day)
        self.path = f"{self.directory}/{TICKS_FILE}"
        self._out = None

    def _open(self) -> None:
        fs = self._fs
        _read_latest(fs, self._base, self._series)
        if isinstance(fs, pafs.LocalFileSystem):
            # En GCS no hay directorios (ver `write_day`).
            fs.create_dir(self.directory)
        index_path = f"{self.directory}/{INDEX_FILE}"
        if _exists(fs, index_path):
            fs.delete_file(index_path)
        self._out = fs.open_output_stream(self.path, metadata=BINARY_META)

    def write(self, data: bytes | memoryview | bytearray) -> int:
        if self._out is None:
            self._open()
        self._out.write(data)
        return len(data)

    def close(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None

    def discard(self) -> None:
        """Cierra y borra lo escrito: un día que no se escribe no deja su `ticks.bin`."""
        self.close()
        if _exists(self._fs, self.path):
            self._fs.delete_file(self.path)


@contextmanager
def open_ticks_reader(
    root: str | Path,
    provider: str,
    market: str,
    asset: str,
    day: date,
    ticks: DayTicks,
) -> Iterator[TicksReader]:
    """Un `TicksReader` sobre el `ticks.bin` ya escrito del día, abierto mientras dure el `with`.

    Lee el objeto por rangos: un tramo por lectura, nunca el archivo entero.
    """
    fs = resolve_fs(root)[0]
    path = f"{day_dir(root, provider, market, asset, day)}/{TICKS_FILE}"
    with fs.open_input_file(path) as src:
        yield TicksReader(ticks, lambda offset, size: src.read_at(size, offset))


def write_day(
    root: str | Path,
    *,
    provider: str,
    market: str,
    asset: str,
    day: date,
    ticks: DayTicks,
    events: EventsBuffer | None = None,
    thetas: Sequence[ThetaEvents],
    missing_thetas: Sequence[str] = (),
    input_hash: str,
    image_version: str,
    generated_at: datetime | None = None,
    template: Template | None = None,
) -> dict:
    """Escribe los archivos del día y devuelve el `index.json` que dejó.

    `ticks.bin` ya está escrito (`TicksFile`, tramo a tramo mientras se leía L1, y
    con él se borró el índice previo). Orden: se escribe `events.bin`, la página
    `index.html` y al final el índice (marca de commit); después `latest.html` y
    `latest.json`, que solo avanzan. El `input_hash` lo calcula quien llama
    (TRD-viz §7.8); `content_hash` sale de los bytes de los dos archivos y no
    depende de `generated_at`.
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

    events = events if events is not None else EventsBuffer()
    declared = sum(t.events for t in thetas)
    if len(events) != declared:
        raise ValueError(
            f"los θ declaran {declared} eventos pero el buffer trae {len(events)} filas"
        )
    offsets = [0]
    for theta in thetas:
        offsets.append(offsets[-1] + theta.events)

    theta_docs = [
        {
            "theta": theta.theta,
            "events": theta.events,
            "events_offset": offset,
            "provisional_from_s": theta.provisional_from_s,
        }
        for theta, offset in zip(thetas, offsets[:-1], strict=True)
    ]

    ticks_path = f"{directory}/{TICKS_FILE}"
    if not _exists(fs, ticks_path):
        raise ValueError(f"falta {TICKS_FILE}: se escribe mientras se leen los ticks")

    def parts() -> Iterator[tuple[str, Iterable[Buffer]]]:
        """Los archivos en orden de nombre (el de `content_hash`), de uno en uno.

        `ticks.bin` ya está en el objeto (se escribió por tramos al leer L1): se vuelve
        a leer por bloques, así que nunca está entero en RAM.
        """
        yield EVENTS_FILE, [events.packed()]
        yield TICKS_FILE, read_blocks(fs, ticks_path)

    _put(fs, f"{directory}/{EVENTS_FILE}", [events.packed()], BINARY_META)
    digest = hashlib.sha256()
    sizes = {}
    for name, chunks in parts():
        digest.update(name.encode() + b"\0")
        sizes[name] = 0
        for chunk in chunks:
            digest.update(chunk)
            sizes[name] += len(chunk)
    if sizes[TICKS_FILE] != ticks.nbytes:
        raise ValueError(
            f"{TICKS_FILE} pesa {sizes[TICKS_FILE]} B y el acumulador escribió "
            f"{ticks.nbytes} B"
        )

    when = generated_at or datetime.now(UTC)
    values = {
        "tiles_version": TILES_VERSION,
        "provider": provider,
        "market": market,
        "asset": asset,
        "day": day.isoformat(),
        "t0": day_start_us(day),
        "price_scale": ticks.price_scale,
        "ticks": ticks.ticks,
        "ticks_chunk": ticks.chunk,
        "first_agg_trade_id": ticks.first_agg_trade_id,
        "last_agg_trade_id": ticks.last_agg_trade_id,
        "ticks_file": TICKS_FILE,
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

    # La página lleva los mismos bytes: `ticks.bin` se vuelve a leer por bloques y
    # cada archivo se codifica y sale directo al objeto, sin armar la página entera;
    # se escribe antes del índice (marca de commit).
    stream_page(fs, f"{directory}/{PAGE_FILE}", index, parts(), template)
    _put_json(fs, index_path, index)
    _advance_latest(
        fs,
        base,
        index,
        lambda: stream_latest_page(fs, base, directory, index, parts(), template),
    )
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
    fs: pafs.FileSystem,
    base: str,
    index: dict,
    write_latest_page: Callable[[], None] | None = None,
    *,
    only_if_stale: bool = False,
    page_is_copy: bool = True,
) -> None:
    """Escribe `latest.json` y `latest.html` si el día no es anterior al que apunta.

    `latest.html` es la copia de la página del día (`write_latest_page` la vuelve a
    renderizar: así no se guarda la página entre una escritura y la otra); va
    primero para que el puntero no apunte a una página que todavía no existe. Con
    `only_if_stale` no hace nada si `latest.json` ya dice lo mismo que el índice y
    `latest.html` es copia de la página del día (`page_is_copy`, ITSC-320).
    """
    series = {k: index[k] for k in ("provider", "market", "asset")}
    current = _read_latest(fs, base, series)
    latest = {name: index[name] for name in LATEST_FIELDS}
    if current is not None and current["day"] > index["day"]:
        return
    if only_if_stale and current == latest and page_is_copy:
        return
    _discard_legacy_tmp(fs, base)
    if write_latest_page is not None:
        write_latest_page()
    _put_json(fs, f"{base}/{LATEST_FILE}", latest)


def _discard_legacy_tmp(fs: pafs.FileSystem, base: str) -> None:
    """Borra el `latest.html.tmp` de la raíz que dejó el `move` fallido (ITSC-318).

    Es limpieza de mejor esfuerzo: si no se puede borrar, se avisa y el job sigue.
    """
    path = f"{base}/{LATEST_PAGE_FILE}.tmp"
    try:
        if _exists(fs, path):
            fs.delete_file(path)
            logger.info("se borró el temporal huérfano %s", path)
    except OSError as exc:
        logger.warning("no se pudo borrar el temporal huérfano %s: %s", path, exc)


def advance_latest_from_files(root: str | Path, index: dict) -> None:
    """Avanza `latest.*` al día de `index`, ya escrito, si están atrasados.

    Para un día que el job salta por estar al día: su página y sus archivos ya
    existen, pero `latest.*` pudo quedar atrás (un fallo a mitad de un run). La
    página se renderiza desde los archivos del día, como en `write_day`.
    """
    fs, base = resolve_fs(root)
    day = date.fromisoformat(index["day"])
    directory = day_dir(root, index["provider"], index["market"], index["asset"], day)
    _advance_latest(
        fs,
        base,
        index,
        lambda: stream_latest_page(
            fs, base, directory, index, day_blocks(fs, directory, index), None
        ),
        only_if_stale=True,
        page_is_copy=latest_page_is_copy(fs, base, directory),
    )


def latest_page_is_copy(fs: pafs.FileSystem, base: str, directory: str) -> bool:
    """Si `latest.html` pesa lo mismo que la página del día (y por tanto es su copia).

    Ambas se comprimen igual (`gzip_to`), así que la misma plantilla y los mismos
    archivos dan los mismos bytes. Un `latest.html` que llegó plano con etiqueta gzip
    (ITSC-320) pesa el doble: no coincide y `latest` se rehace. `get_file_info` da el
    tamaño almacenado, sin transcodificar. Falta alguna de las dos: no es copia.
    """
    latest = fs.get_file_info(f"{base}/{LATEST_PAGE_FILE}")
    page = fs.get_file_info(f"{directory}/{PAGE_FILE}")
    both = latest.type == page.type == pafs.FileType.File
    return both and latest.size == page.size


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


def day_sizes(
    root: str | Path, provider: str, market: str, asset: str, day: date
) -> dict[str, int]:
    """Los bytes de cada objeto del día bajo `root`, por nombre (un solo listado)."""
    fs, _ = resolve_fs(root)
    selector = pafs.FileSelector(
        day_dir(root, provider, market, asset, day), allow_not_found=True
    )
    return {
        i.base_name: i.size or 0
        for i in fs.get_file_info(selector)
        if i.type == pafs.FileType.File
    }
