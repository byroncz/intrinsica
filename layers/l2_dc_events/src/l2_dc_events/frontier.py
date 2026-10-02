"""La frontera de cada θ: hasta qué mes llegó (TRD-L2 §8.2).

Es el último mes de la cadena de carry-over de ese θ: contigua desde el primer
mes de la serie y con un `carry_over.parquet` válido al final. Contigua porque
el carry-over de un mes sale del anterior; un hueco rompe la cadena aunque
existan meses después. Válido es lo que `read_carry_over` acepta.

No hay manifiesto de L2 y los hallazgos `events_summary` no dicen si el archivo
sigue ahí, así que la frontera sale de un solo listado recursivo de la raíz de
eventos del activo (una llamada paginada, no un `get_file_info` por partición)
más la lectura del carry-over de la frontera de cada θ.
"""

import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import pyarrow.fs as pafs
from pyutils import resolve_fs

from l2_dc_events.carry import CarryOverError, read_carry_over
from l2_dc_events.context import Unit
from l2_dc_events.write import CARRY_OVER, THETA_DIGITS, partition_path

logger = logging.getLogger(__name__)

# Lecturas simultáneas del carry-over de la frontera (un archivo de una fila por θ).
READ_WORKERS = 8

_CARRY_OVER = re.compile(
    rf"/theta=0\.(\d{{{THETA_DIGITS}}})/year=(\d{{4}})/month=(\d{{2}})/"
    rf"{re.escape(CARRY_OVER)}$"
)

Month = tuple[int, int]


def ordinal(month: Month) -> int:
    """El mes como un entero consecutivo (`año * 12 + mes - 1`)."""
    year, number = month
    return year * 12 + number - 1


def month_of(n: int) -> Month:
    year, number = divmod(n, 12)
    return year, number + 1


def label(n: int | None) -> str | None:
    """`YYYY-MM` de un ordinal (`None` si no hay)."""
    if n is None:
        return None
    year, month = month_of(n)
    return f"{year:04d}-{month:02d}"


def list_carry_overs(events_root: str, asset: Unit) -> dict[int, set[int]]:
    """Los meses con `carry_over.parquet` de cada θ del activo, por ordinal.

    Incluye todos los θ que hay en el lago, también los que ya no están en el
    catálogo. No valida el contenido: solo dice qué archivos existen.
    """
    prefix = (
        f"{str(events_root).rstrip('/')}/provider={asset.provider}"
        f"/market={asset.market}/asset={asset.asset}"
    )
    fs, resolved = resolve_fs(prefix)
    selector = pafs.FileSelector(resolved, recursive=True, allow_not_found=True)
    found: dict[int, set[int]] = {}
    for info in fs.get_file_info(selector):
        match = _CARRY_OVER.search(info.path)
        if match is None:
            continue
        theta, year, month = (int(g) for g in match.groups())
        found.setdefault(theta, set()).add(ordinal((year, month)))
    return found


def _frontier(
    theta: int, months: set[int], start: int, events_root: str, asset: Unit
) -> int | None:
    """La frontera de un θ: el fin de su cadena contigua con carry-over válido."""
    last = start
    while last in months:
        last += 1
    # `last` es el primer mes que falta: la cadena llega hasta el anterior.
    for n in range(last - 1, start - 1, -1):
        year, month = month_of(n)
        path = partition_path(
            events_root,
            asset.provider,
            asset.market,
            asset.asset,
            theta,
            year,
            month,
            CARRY_OVER,
        )
        try:
            read_carry_over(path, theta)
        except CarryOverError as exc:
            logger.warning("θ=%d %s: carry-over inválido (%s)", theta, label(n), exc)
            continue
        return n
    return None


def frontiers(
    thetas: Iterable[int],
    listed: dict[int, set[int]],
    series_start: Month,
    events_root: str,
    asset: Unit,
) -> dict[int, int | None]:
    """La frontera (ordinal del último mes cerrado) de cada θ; `None` si no tiene.

    Un carry-over inválido en la frontera la retrocede al mes anterior, que se
    verifica a su vez: el θ se reprocesa desde el último estado utilizable.
    """
    thetas = list(thetas)
    start = ordinal(series_start)

    def one(theta: int) -> int | None:
        return _frontier(theta, listed.get(theta, set()), start, events_root, asset)

    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        return dict(zip(thetas, pool.map(one, thetas), strict=True))
