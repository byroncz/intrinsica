"""Modo `render`: vuelve a armar `index.html` desde los tiles ya escritos.

TRD-viz §8.3. No lee L1 ni L2: para cuando cambia la plantilla (HTML, JS, CSS o
uPlot) y hay que regenerar las páginas sin repetir la reducción. Un día se salta
si su página ya trae la misma `tiles_version` y la misma huella de plantilla
(guardadas en su `<meta name="viz-render">`).

Memoria: los 37 arreglos de un día se leen, se codifican y se sueltan de uno en
uno; nunca hay más de un día a la vez.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from datetime import date

import pyarrow.fs as pafs
from dq import Finding, Severity
from pyutils.fs import resolve_fs

from viz_tiles import findings
from viz_tiles.context import RunContext
from viz_tiles.contract import LATEST_FILE, LATEST_PAGE_FILE, PAGE_FILE
from viz_tiles.lake import Month
from viz_tiles.pipeline import days_of_month
from viz_tiles.render import (
    META_PROBE,
    Template,
    expected_names,
    load_template,
    page_meta,
    render_day,
)
from viz_tiles.write import (
    compresses_pages,
    day_dir,
    find_index,
    write_page,
)

logger = logging.getLogger(__name__)

GZIP_MAGIC = b"\x1f\x8b"
_BLOCK = 1 << 20  # bloque con el que se cuentan los bytes de una página plana


class TilesCorrupt(Exception):
    """Un arreglo del día falta o no coincide con el `content_hash` de su índice."""


def _read(fs: pafs.FileSystem, path: str, size: int | None = None) -> bytes:
    with fs.open_input_stream(path) as src:
        return src.read(size) if size else src.read()


def _exists(fs: pafs.FileSystem, path: str) -> bool:
    return fs.get_file_info(path).type == pafs.FileType.File


def _isize(stored: int, head: bytes, tail: bytes) -> int:
    """El tamaño del HTML plano de una página recién renderizada con `stored` bytes.

    Si es un gzip, su tamaño plano son los últimos 4 bytes (ISIZE, módulo 2^32:
    sobra para una página de unos MB); si va plana, es lo guardado.
    """
    return int.from_bytes(tail[-4:], "little") if head[:2] == GZIP_MAGIC else stored


def _skipped_decoded_bytes(fs: pafs.FileSystem, path: str, size: int) -> int:
    """El tamaño plano de una página guardada con `size` bytes, sin cargarla entera.

    En un bucket la página es un gzip y se lee el ISIZE del final. Pero si el
    almacenamiento la descomprime al leer (transcodificación de GCS), llega
    plana y `size` es el tamaño guardado: se cuentan los bytes por bloques.
    """
    with fs.open_input_file(path) as src:
        head = src.read(len(GZIP_MAGIC))
        if head == GZIP_MAGIC:
            src.seek(max(size - 4, 0))
            return _isize(size, head, src.read(4))
        count = len(head)
        while block := src.read(_BLOCK):
            count += len(block)
        return count


def _stored_meta(fs: pafs.FileSystem, path: str) -> tuple[str, str] | None:
    """La huella guardada en una página, o `None` si no existe o no la trae."""
    if not _exists(fs, path):
        return None
    return page_meta(_read(fs, path, META_PROBE))


def _arrays(
    fs: pafs.FileSystem, directory: str, index: dict, update: Callable[[bytes], None]
) -> Iterator[tuple[str, bytes]]:
    """Los arreglos del día en orden de nombre, de uno en uno, hasheándolos al pasar."""
    for name in expected_names(index):
        path = f"{directory}/{name}"
        if not _exists(fs, path):
            raise TilesCorrupt(f"falta {name}")
        data = _read(fs, path)
        update(name.encode() + b"\0")
        update(data)
        yield name, data
        del data


def _latest_day(fs: pafs.FileSystem, base: str) -> str | None:
    path = f"{base}/{LATEST_FILE}"
    if not _exists(fs, path):
        return None
    return json.loads(_read(fs, path)).get("day")


def render_month(
    ctx: RunContext,
    month: Month,
    days: list[date] | None,
    *,
    template: Template | None = None,
) -> bool:
    """Regenera las páginas de `days` del mes (los que tengan tiles, si es `None`).

    Un día pedido explícitamente sin `index.json` deja `input_missing` con
    `what = "tiles"`. Los hallazgos del mes se emiten juntos, en una llamada.
    Devuelve `False` si dejó un error.
    """
    out: list[Finding] = []
    try:
        _run(ctx, month, days, template or load_template(), out)
    finally:
        findings.emit(ctx, out)
    return not any(f.severity is Severity.ERROR for f in out)


def _run(
    ctx: RunContext,
    month: Month,
    days: list[date] | None,
    template: Template,
    out: list[Finding],
) -> None:
    key = (ctx.tiles_root, ctx.provider, ctx.market, ctx.asset)
    fs, base = resolve_fs(ctx.tiles_root)
    latest = _latest_day(fs, base)
    explicit = days is not None
    done = 0
    for day in days or days_of_month(month):
        index = find_index(*key, day)
        if index is None:
            if explicit:
                out.append(findings.input_missing(ctx, day, "tiles"))
            continue
        try:
            _render_day(ctx, fs, base, day, index, latest, template, out)
            done += 1
        except TilesCorrupt as exc:
            logger.error("día %s: %s", day, exc)
            out.append(findings.input_missing(ctx, day, "tiles", reason=str(exc)))
    if not explicit and not done:
        first = days_of_month(month)[0]
        logger.error("unidad %d-%02d: ningún día con index.json", *month)
        out.append(
            findings.input_missing(
                ctx, first, "tiles", month=f"{month[0]:04d}-{month[1]:02d}"
            )
        )


def _render_day(
    ctx: RunContext,
    fs: pafs.FileSystem,
    base: str,
    day: date,
    index: dict,
    latest: str | None,
    template: Template,
    out: list[Finding],
) -> None:
    directory = day_dir(ctx.tiles_root, ctx.provider, ctx.market, ctx.asset, day)
    try:
        expected_names(index)
    except KeyError as exc:
        # Tiles de una versión anterior: no traen todos los archivos de la actual.
        raise TilesCorrupt(
            f"el índice no trae {exc} (tiles_version {index.get('tiles_version')}): "
            "el día se rehace con --mode tiles, no con render"
        ) from exc
    page_path = f"{directory}/{PAGE_FILE}"
    latest_path = f"{base}/{LATEST_PAGE_FILE}"
    is_latest = latest == day.isoformat()
    want = (index["tiles_version"], template.hash)

    fresh = _stored_meta(fs, page_path) == want and (
        not is_latest or _stored_meta(fs, latest_path) == want
    )
    if fresh and not ctx.force:
        logger.info(
            "unidad %s: página al día (plantilla %s), se salta", day, template.hash[:12]
        )
        size = fs.get_file_info(page_path).size or 0
        decoded = (
            _skipped_decoded_bytes(fs, page_path, size)
            if compresses_pages(fs)
            else size
        )
        out.append(_summary(ctx, day, index, template, True, size, decoded))
        return

    digest = hashlib.sha256()
    page = render_day(
        index,
        _arrays(fs, directory, index, digest.update),
        template=template,
        compress=compresses_pages(fs),
    )
    if digest.hexdigest() != index["content_hash"]:
        raise TilesCorrupt(
            "los arreglos no coinciden con el content_hash del índice "
            f"({digest.hexdigest()[:12]} ≠ {index['content_hash'][:12]})"
        )
    write_page(fs, page_path, page)
    if is_latest:
        write_page(fs, latest_path, page)
    logger.info("unidad %s: página regenerada (%d B)", day, len(page))
    decoded = _isize(len(page), page[:2], page[-4:])
    out.append(_summary(ctx, day, index, template, False, len(page), decoded))


def _summary(
    ctx: RunContext,
    day: date,
    index: dict,
    template: Template,
    skipped: bool,
    size: int,
    decoded: int,
) -> Finding:
    return findings.render_summary(
        ctx,
        day,
        skipped=skipped,
        tiles_version=index["tiles_version"],
        template_hash=template.hash,
        page_bytes=size,
        decoded_bytes=decoded,
        content_hash=index["content_hash"],
    )
