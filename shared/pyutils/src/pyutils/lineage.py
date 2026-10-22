"""`content_hash.json`: la huella que L2 publica junto a `events.parquet` y `carry_over.parquet`.

Quien produce un archivo calcula su `content_hash` una sola vez, mientras lo escribe, y lo
deja en un objeto pequeño de la misma partición; quien lo consume (L3) lo lee en vez de
recalcularlo (ITSC-368). El valor es el de `ContentHasher`, el mismo que L2 emite en
`events_summary`: cambia quién lo calcula, no el algoritmo.
"""

import json
import os
import uuid
from datetime import UTC, datetime

import pyarrow.fs as pafs

from pyutils.fs import resolve_fs
from pyutils.hashing import HASHER_VERSION

LINEAGE = "content_hash.json"
KEYS = (
    "events_content_hash",
    "carry_over_content_hash",
    "hasher_version",
    "written_at",
)


def document(
    events_content_hash: str,
    carry_over_content_hash: str,
    written_at: datetime | None = None,
) -> dict[str, str]:
    """El contenido de `content_hash.json`; `written_at` es UTC, en ISO 8601."""
    moment = written_at or datetime.now(UTC)
    return {
        "events_content_hash": events_content_hash,
        "carry_over_content_hash": carry_over_content_hash,
        "hasher_version": HASHER_VERSION,
        "written_at": moment.astimezone(UTC).isoformat(timespec="seconds"),
    }


def parse(data: bytes) -> dict[str, str]:
    """El documento de `data`; `ValueError` si no es JSON o le falta o le sobra algo."""
    try:
        found = json.loads(data)
    except ValueError as error:  # incluye UnicodeDecodeError
        raise ValueError(f"no es JSON: {error}") from error
    if not isinstance(found, dict) or set(found) != set(KEYS):
        raise ValueError(f"se esperaban las claves {list(KEYS)}")
    if not all(isinstance(found[key], str) and found[key] for key in KEYS):
        raise ValueError("todos los valores deben ser texto no vacío")
    return found


def read(path: str) -> dict[str, str]:
    """Lee y valida `content_hash.json`. `FileNotFoundError` si no existe, `ValueError` si
    no tiene el formato."""
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        raise FileNotFoundError(path)
    with fs.open_input_stream(resolved) as stream:
        return parse(stream.read())


def write(path: str, content: dict[str, str]) -> str:
    """Publica `content_hash.json` de una vez. Devuelve `path`.

    El documento se serializa antes de abrir el destino, así que nunca queda a medias: en
    GCS el objeto aparece completo al cerrar la subida y en disco se renombra un temporal.
    """
    data = json.dumps(content, sort_keys=True).encode()
    fs, target = resolve_fs(path)
    if isinstance(fs, pafs.GcsFileSystem):
        with fs.open_output_stream(target) as stream:
            stream.write(data)
        return path
    parent, _, name = target.rpartition("/")
    tmp = f"{parent}/.{name}.{uuid.uuid4().hex}.tmp"
    os.makedirs(parent, exist_ok=True)
    with fs.open_output_stream(tmp) as stream:
        stream.write(data)
    os.replace(tmp, target)
    return path


def retract(path: str) -> None:
    """Borra `content_hash.json` si existe.

    L2 lo llama antes de reescribir los Parquet de la partición: si la corrida muere entre
    los Parquet y el nuevo `content_hash.json`, la partición queda sin linaje publicado (L3
    falla con `l2_lineage_missing`) y no con el de la versión anterior, que ya no describe
    los archivos.
    """
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type == pafs.FileType.File:
        fs.delete_file(resolved)
