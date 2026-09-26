"""Descarga en RAM de un archivo de Binance con verificación SHA-256."""

import hashlib
import http.client
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

BASE_URL = "https://data.binance.vision"
ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
TIMEOUT_SECONDS = 60


class L1DownloadError(Exception):
    """Base de los fallos de descarga de L1."""


class ChecksumError(L1DownloadError):
    """El SHA-256 calculado no coincide con el publicado."""


class DownloadError(L1DownloadError):
    """Fallo de red, del servidor o un .CHECKSUM ilegible."""


@dataclass(frozen=True)
class Download:
    data: bytes
    sha256: str
    file_bytes: int
    source_url: str
    downloaded_at: datetime


def source_url(
    asset: str,
    year: int,
    month: int,
    day: int | None = None,
    base_url: str = BASE_URL,
) -> str:
    """URL del ZIP de aggTrades: mensual si `day` es None, diario si no."""
    if day is None:
        return (
            f"{base_url}/data/spot/monthly/aggTrades/{asset}/"
            f"{asset}-aggTrades-{year:04d}-{month:02d}.zip"
        )
    return (
        f"{base_url}/data/spot/daily/aggTrades/{asset}/"
        f"{asset}-aggTrades-{year:04d}-{month:02d}-{day:02d}.zip"
    )


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _parse_checksum(url: str, checksum: bytes) -> str:
    # Formato de sha256sum: "<sha256>  <nombre>"
    fields = checksum.decode("ascii", errors="replace").split()
    if not fields:
        raise DownloadError(f"{url}.CHECKSUM está vacío")
    published = fields[0].lower()
    if len(published) != 64 or any(c not in "0123456789abcdef" for c in published):
        raise DownloadError(f"{url}.CHECKSUM no contiene un SHA-256 válido")
    return published


def _get_checksum(url: str) -> str:
    try:
        return _parse_checksum(url, _get(url + ".CHECKSUM"))
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadError(f"{url}.CHECKSUM: {exc}") from exc


def _attempt(url: str) -> Download:
    try:
        data = _get(url)
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadError(f"{url}: {exc}") from exc
    published = _get_checksum(url)
    actual = hashlib.sha256(data).hexdigest()
    if actual != published:
        raise ChecksumError(f"{url}: SHA-256 {actual} != publicado {published}")
    return Download(data, actual, len(data), url, datetime.now(UTC))


def _retry[T](
    call: Callable[[], T], attempts: int, backoff: float, sleep: Callable[[float], None]
) -> T:
    for n in range(1, attempts + 1):
        try:
            return call()
        except L1DownloadError:
            if n == attempts:
                raise
            sleep(backoff * 2 ** (n - 1))
    raise AssertionError("attempts debe ser >= 1")


def fetch_checksum(
    url: str,
    attempts: int = ATTEMPTS,
    backoff: float = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """SHA-256 publicado en el .CHECKSUM de `url`, sin descargar el ZIP.

    Mismos reintentos que `fetch`.
    """
    return _retry(lambda: _get_checksum(url), attempts, backoff, sleep)


def fetch(
    url: str,
    attempts: int = ATTEMPTS,
    backoff: float = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Download:
    """Descarga `url` y su .CHECKSUM y verifica el SHA-256.

    Reintenta hasta `attempts` intentos con espera creciente (backoff, 2 x
    backoff, ...). Si el último falla, lanza su excepción sin devolver datos.
    """
    return _retry(lambda: _attempt(url), attempts, backoff, sleep)
