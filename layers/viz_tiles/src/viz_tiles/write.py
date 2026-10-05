"""Escritura de un día de tiles: arreglos planos, `index.json` y `latest.json`.

TRD-viz §7.2, §7.7 y ADR-VZ-10. El `index.json` se escribe al final y es la
marca de commit: un día sin él no existe para el tablero. Si el job muere a
medias, el día queda sin índice y la siguiente corrida lo rehace.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow.fs as pafs
from pyutils.fs import resolve_fs

from viz_tiles.contract import (
    FILE_BY_KIND,
    INDEX_FIELDS,
    INDEX_FILE,
    LATEST_FIELDS,
    LATEST_FILE,
    LEVELS,
    TILES_VERSION,
    tile_name,
)
from viz_tiles.reduce import DayReduction, day_start_us


@dataclass(frozen=True)
class ThetaTiles:
    """Tiles de dirección de un θ: `direction[w]` es el arreglo uint8 del nivel `w`."""

    theta: str
    events: int
    provisional_from_s: float | None
    direction: Mapping[int, np.ndarray]


def day_dir(root: str | Path, provider: str, market: str, asset: str, day: date) -> str:
    """Directorio del día bajo `root`: `provider=…/market=…/asset=…/day=YYYY-MM-DD`."""
    base = resolve_fs(root)[1]
    return (
        f"{base}/provider={provider}/market={market}/asset={asset}"
        f"/day={day.isoformat()}"
    )


def _tile_bytes(kind: str, w: int, array: np.ndarray) -> bytes:
    """Bytes del archivo: el arreglo con el tipo y el largo que fija el contrato."""
    spec = FILE_BY_KIND[kind]
    expected = spec.per_column * w
    if array.shape != (expected,):
        raise ValueError(
            f"{kind}-{w}: se esperaban {expected} valores, hay {array.shape}"
        )
    return array.astype(spec.dtype, copy=False).tobytes()


def _put(fs: pafs.FileSystem, path: str, data: bytes) -> None:
    with fs.open_output_stream(path) as out:
        out.write(data)


def _put_json(fs: pafs.FileSystem, path: str, doc: dict) -> None:
    """Escribe un JSON; en disco local, entero o nada (temporal y renombre)."""
    data = (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode()
    if isinstance(fs, pafs.LocalFileSystem):
        tmp = f"{path}.tmp"
        _put(fs, tmp, data)
        fs.move(tmp, path)
    else:
        _put(fs, path, data)


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
    thetas: Sequence[ThetaTiles],
    missing_thetas: Sequence[str] = (),
    input_hash: str,
    image_version: str,
    generated_at: datetime | None = None,
) -> dict:
    """Escribe los tiles del día y devuelve el `index.json` que dejó.

    Orden: se borra el índice previo, se escriben los arreglos y al final el
    índice (marca de commit); después `latest.json`, que solo avanza. El
    `input_hash` lo calcula quien llama (TRD-viz §7.8); `content_hash` sale de
    los bytes de los arreglos y no depende de `generated_at`.
    """
    fs, base = resolve_fs(root)
    directory = day_dir(root, provider, market, asset, day)
    fs.create_dir(directory)
    index_path = f"{directory}/{INDEX_FILE}"
    if _exists(fs, index_path):
        fs.delete_file(index_path)

    # Nombre de archivo -> (tipo, nivel, arreglo). Son vistas de lo que ya está
    # en RAM: los bytes de cada archivo se generan de uno en uno al escribir.
    arrays: dict[str, tuple[str, int, np.ndarray]] = {}
    price = {str(w): tile_name("price", w) for w in LEVELS}
    volume = {str(w): tile_name("volume", w) for w in LEVELS}
    for w in LEVELS:
        arrays[price[str(w)]] = ("price", w, reduction.price[w])
        arrays[volume[str(w)]] = ("volume", w, reduction.volume[w])
    theta_docs = []
    for theta in thetas:
        names = {str(w): tile_name("dir", w, theta.theta) for w in LEVELS}
        for w in LEVELS:
            arrays[names[str(w)]] = ("dir", w, theta.direction[w])
        theta_docs.append(
            {
                "theta": theta.theta,
                "events": theta.events,
                "provisional_from_s": theta.provisional_from_s,
                "dir": names,
            }
        )
    # El resumen de contenido recorre los archivos en orden de nombre.
    digest = hashlib.sha256()
    for name in sorted(arrays):
        data = _tile_bytes(*arrays[name])
        digest.update(name.encode() + b"\0" + data)
        _put(fs, f"{directory}/{name}", data)

    when = generated_at or datetime.now(UTC)
    values = {
        "tiles_version": TILES_VERSION,
        "provider": provider,
        "market": market,
        "asset": asset,
        "day": day.isoformat(),
        "t0": day_start_us(day),
        "ticks": reduction.ticks,
        "levels": list(LEVELS),
        "price": price,
        "volume": volume,
        "thetas": theta_docs,
        "missing_thetas": list(missing_thetas),
        "input_hash": input_hash,
        "content_hash": digest.hexdigest(),
        "generated_at": when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_version": image_version,
    }
    index = {name: values[name] for name, _ in INDEX_FIELDS}
    _put_json(fs, index_path, index)
    _advance_latest(fs, base, index)
    return index


def _advance_latest(fs: pafs.FileSystem, base: str, index: dict) -> None:
    """Escribe `latest.json` si el día es posterior al que apunta (o no existe)."""
    path = f"{base}/{LATEST_FILE}"
    if _exists(fs, path):
        with fs.open_input_stream(path) as src:
            current = json.loads(src.read())
        same_series = all(
            current.get(k) == index[k] for k in ("provider", "market", "asset")
        )
        if same_series and current["day"] > index["day"]:
            return
    _put_json(fs, path, {name: index[name] for name in LATEST_FIELDS})


def read_index(
    root: str | Path, provider: str, market: str, asset: str, day: date
) -> dict:
    """El `index.json` de un día (local)."""
    path = Path(day_dir(root, provider, market, asset, day)) / INDEX_FILE
    return json.loads(path.read_text())
