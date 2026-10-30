"""La frontera de cada θ y la disponibilidad de L2: dos listados recursivos (TRD-L3 §8.2).

La frontera de un θ y una familia es el último mes de su serie de `<familia>.parquet`,
**contigua desde `series_start`**, cuyo archivo es **válido**: existe, se lee, su esquema, su
`theta` y su `state_version` son los de la familia en la imagen y su número de filas por row
group es el de `events.parquet` de L2 del mismo mes. La frontera de una familia distinta de S
nunca supera la de S del θ: su archivo repite el esqueleto de `summaries.parquet` (§7.3). Contigua porque cada mes se escribe en orden y un
hueco es un mes que falta; sin carry-over (ADR-L3-12) los meses no dependen unos de otros,
pero la frontera dice "hasta aquí está todo" y eso solo vale sin huecos.

No hay manifiesto: la frontera sale de **un listado recursivo** de la raíz de L3 del activo
(una llamada paginada, no un `get_file_info` por partición) más la lectura del pie del
archivo de la frontera de cada θ y del de L2 del mismo mes. La disponibilidad de L2 sale de
**otro listado** de la raíz de eventos. Ninguna de las dos funciones guarda estado: reciben
los listados y devuelven valores.
"""

import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
from types import MappingProxyType

import pyarrow as pa
import pyarrow.fs as pafs
from dc_frames import FramesError, family_path, l2_path
from dc_frames.lake import Month, month_of, month_ordinal, open_parquet
from pyutils import resolve_fs

from l3_dc_descriptors.l2 import CARRY_OVER, EVENTS
from l3_dc_descriptors.schema import (
    FAMILY,
    PATH_FAMILY,
    PATH_SCHEMA,
    PATH_STATE_VERSION,
    STATE_VERSION,
    SUMMARIES_SCHEMA,
)

logger = logging.getLogger(__name__)

# Lecturas simultáneas de los pies de la frontera (un archivo de familia y uno de L2 por θ).
READ_WORKERS = 8
THETA_DIGITS = 8
# Por familia: su esquema y su `state_version`, lo que hace válido el archivo de la frontera.
_CONTRACTS = MappingProxyType(
    {
        FAMILY: (SUMMARIES_SCHEMA, STATE_VERSION),
        PATH_FAMILY: (PATH_SCHEMA, PATH_STATE_VERSION),
    }
)

_FILE = re.compile(
    rf"/theta=0\.(\d{{{THETA_DIGITS}}})/year=(\d{{4}})/month=(\d{{2}})/([^/]+)$"
)

# θ (× 10⁸) → ordinal del mes → nombres de los archivos que tiene la partición.
Listing = dict[int, dict[int, set[str]]]


def label(n: int | None) -> str | None:
    """`YYYY-MM` de un ordinal (`None` si no hay)."""
    if n is None:
        return None
    year, month = month_of(n)
    return f"{year:04d}-{month:02d}"


def theta_text(theta: int) -> str:
    """`theta = round(θ × 10⁸)` como el `theta=<t>` de la ruta: `0.00010000`."""
    return f"0.{theta:0{THETA_DIGITS}d}"


def list_partitions(
    root: str, key: tuple[str, str, str], names: Iterable[str]
) -> Listing:
    """Los archivos `names` de cada partición `theta/year/month` del activo bajo `root`.

    Incluye todos los θ que hay en el lago, también los que ya no están en el catálogo.
    No valida el contenido: solo dice qué archivos existen.
    """
    wanted = set(names)
    provider, market, asset = key
    prefix = (
        f"{str(root).rstrip('/')}/provider={provider}/market={market}/asset={asset}"
    )
    fs, resolved = resolve_fs(prefix)
    selector = pafs.FileSelector(resolved, recursive=True, allow_not_found=True)
    found: Listing = {}
    for info in fs.get_file_info(selector):
        match = _FILE.search(info.path)
        if match is None or match[4] not in wanted:
            continue
        theta, year, month = (int(g) for g in match.groups()[:3])
        n = month_ordinal((year, month))
        found.setdefault(theta, {}).setdefault(n, set()).add(match[4])
    return found


def family_file(family: str) -> str:
    """El archivo de la familia en la partición: `summaries.parquet`, `path.parquet`."""
    return f"{family}.parquet"


def list_summaries(
    summaries_root: str, key: tuple[str, str, str], family: str = FAMILY
) -> Listing:
    """Los archivos de la `family` de cada partición (por defecto `summaries.parquet`)."""
    return list_partitions(summaries_root, key, [family_file(family)])


def list_l2(events_root: str, key: tuple[str, str, str]) -> Listing:
    return list_partitions(events_root, key, [EVENTS, CARRY_OVER])


def l2_missing(listed: Listing, theta: int, n: int) -> list[str]:
    """Los archivos de L2 de `(θ, mes)` que faltan (vacío: L2 tiene el mes)."""
    have = listed.get(theta, {}).get(n, set())
    return [name for name in (EVENTS, CARRY_OVER) if name not in have]


def skeleton_missing(summaries: Listing, theta: int, n: int) -> bool:
    """¿Falta `summaries.parquet` de `(θ, mes)`, el esqueleto de las demás familias (§7.3)?"""
    return family_file(FAMILY) not in summaries.get(theta, {}).get(n, set())


def row_group_sizes(parquet) -> list[int]:
    meta = parquet.metadata
    sizes = (meta.row_group(i).num_rows for i in range(meta.num_row_groups))
    # Los row groups vacíos no se escriben (`FamilyWriter`): no cuentan.
    return [rows for rows in sizes if rows]


def invalid_reason(
    summaries_root: str,
    events_root: str,
    key: tuple[str, str, str],
    theta: int,
    month: Month,
    family: str = FAMILY,
) -> str | None:
    """Por qué el archivo de la `family` de `(θ, mes)` no es una frontera válida, o `None`.

    Lee solo los pies del archivo de la familia y de `events.parquet` del mes.
    """
    expected, version_wanted = _CONTRACTS[family]
    text = theta_text(theta)
    path = family_path(summaries_root, key, text, month, family)
    events = l2_path(events_root, key, text, month)
    try:
        with (
            closing(open_parquet(path, "la familia")) as table,
            closing(open_parquet(events, "L2")) as skeleton,
        ):
            schema = table.schema_arrow
            if not schema.remove_metadata().equals(expected):
                return "su esquema no es el de la imagen"
            version = (schema.metadata or {}).get(b"state_version")
            if version != version_wanted.encode():
                got = version.decode() if version else None
                return f"state_version {got} y la imagen es {version_wanted}"
            wanted = Decimal(theta).scaleb(-THETA_DIGITS)
            index = schema.get_field_index("theta")
            for i in range(table.metadata.num_row_groups):
                group = table.metadata.row_group(i)
                if not group.num_rows:
                    continue
                stats = group.column(index).statistics
                if (
                    stats is None
                    or not stats.has_min_max
                    or stats.min != wanted
                    or stats.max != wanted
                ):
                    return f"el row group {i} no es del θ {text}"
            if row_group_sizes(table) != row_group_sizes(skeleton):
                return "sus filas por row group no son las de events.parquet de L2"
    except (FramesError, pa.ArrowException, OSError) as exc:
        return f"no se pudo leer ({exc})"
    return None


def _frontier(
    theta: int,
    months: set[int],
    start: int,
    summaries_root: str,
    events_root: str,
    key: tuple[str, str, str],
    family: str,
) -> int | None:
    """La frontera de un θ: el fin de su serie contigua con un archivo válido."""
    last = start
    while last in months:
        last += 1
    # `last` es el primer mes que falta: la serie llega hasta el anterior.
    for n in range(last - 1, start - 1, -1):
        problem = invalid_reason(
            summaries_root, events_root, key, theta, month_of(n), family
        )
        if problem is None:
            return n
        logger.warning(
            "θ=%d %s: archivo de %s inválido (%s)", theta, label(n), family, problem
        )
    return None


def frontiers(
    thetas: Iterable[int],
    listed: Listing,
    series_start: Month,
    summaries_root: str,
    events_root: str,
    key: tuple[str, str, str],
    family: str = FAMILY,
    summaries: Listing | None = None,
) -> dict[int, int | None]:
    """La frontera (ordinal del último mes cerrado) de cada θ; `None` si no tiene.

    `listed` son los archivos de la `family`. Un archivo inválido en la frontera la retrocede
    al mes anterior, que se verifica a su vez: el θ se reprocesa desde el último mes
    utilizable. La de una familia distinta de S nunca supera la de S: `summaries`, el listado
    de `summaries.parquet`, es obligatorio, porque su archivo repite ese esqueleto.
    """
    thetas = list(thetas)
    start = month_ordinal(series_start)

    def one(theta: int) -> int | None:
        months = set(listed.get(theta, {}))
        return _frontier(theta, months, start, summaries_root, events_root, key, family)

    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        found = dict(zip(thetas, pool.map(one, thetas), strict=True))
    if family == FAMILY:
        return found
    if summaries is None:
        raise ValueError(f"la frontera de {family!r} necesita el listado de {FAMILY!r}")
    ceiling = frontiers(
        thetas, summaries, series_start, summaries_root, events_root, key
    )
    return {
        theta: None
        if reached is None or ceiling[theta] is None
        else min(reached, ceiling[theta])
        for theta, reached in found.items()
    }
