"""La frontera de cada θ y la disponibilidad de L2: dos listados recursivos (TRD-L3 §8.2).

La frontera de un θ y una familia es el fin del **prefijo contiguo de meses válidos** de su
serie de `<familia>.parquet`, contado desde `series_start`: ningún mes anterior a la frontera
puede ser inválido (ITSC-419). Un archivo es **válido** si existe, se lee, su esquema, su
`theta` y su `state_version` son los de la familia en la imagen y su número de filas por row
group es el de `events.parquet` de L2 del mismo mes. Con un inválido en el mes `m` la frontera
es `m − 1` (o ninguna si `m` es el primero), aunque haya archivos válidos después: una corrida
de un solo mes en medio de la serie deja un válido rodeado de archivos de otra versión, y una
regla que se queda con el último válido daría por buenos los meses anteriores. La frontera de
una familia distinta de S nunca supera la de S del θ: su archivo repite el esqueleto de
`summaries.parquet` (§7.3). Contigua porque cada mes se escribe en orden y un hueco es un mes
que falta; sin carry-over (ADR-L3-12) los meses no dependen unos de otros, pero la frontera
dice "hasta aquí está todo" y eso solo vale sin huecos.

No hay manifiesto: la frontera sale de **un listado recursivo** de la raíz de L3 del activo
(una llamada paginada, no un `get_file_info` por partición) más la lectura del pie de cada
archivo del prefijo y del de L2 del mismo mes (solo pies: ningún dato). La disponibilidad de
L2 sale de **otro listado** de la raíz de eventos. Ninguna de las dos funciones guarda estado:
reciben los listados y devuelven valores.

**Cuánto cuesta (ITSC-420).** Verificar cada mes midió 1 945,8 s en la nube (11 100 archivos,
22 200 pies, 8 lectores): ≈ 0,7 s por pie y lector, de la latencia de cada viaje y no de la
CPU. Se acota sin aflojar la comprobación: (a) una tarea por `(θ, mes)` en un pool de
`L3_FRONTIER_WORKERS` lectores (por defecto 64), en vez de una por θ, que con 50 θ dejaba
ociosos a todos los lectores de más; (b) el pie de `events.parquet` de un `(θ, mes)` se lee una
sola vez por corrida aunque lo pidan la familia y el techo de S; (c) sin el `get_file_info` de
existencia, porque el listado ya dijo que el archivo estaba; (d) un solo sistema de archivos
por raíz. Memoria: a lo sumo un pie por lector a la vez, más el caché de (b), que guarda solo
las filas por row group de cada `events.parquet` y se libera al terminar.
"""

import re
import threading
from array import array
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from dc_frames import FramesError, FramesInputError, family_path, l2_path
from dc_frames.lake import Month, month_of, month_ordinal
from pyutils import resolve_fs

from l3_dc_descriptors.l2 import CARRY_OVER, EVENTS
from l3_dc_descriptors.schema import (
    FAMILY,
    FIT_FAMILY,
    FIT_SCHEMA,
    FIT_STATE_VERSION,
    PATH_FAMILY,
    PATH_SCHEMA,
    PATH_STATE_VERSION,
    SPECTRUM_FAMILY,
    SPECTRUM_SCHEMA,
    SPECTRUM_STATE_VERSION,
    STATE_VERSION,
    SUMMARIES_SCHEMA,
)

# Lectores simultáneos de la verificación (`L3_FRONTIER_WORKERS`). Es E/S de pies, no CPU: lo que
# manda es cuántos viajes a la vez, y 8 lectores medían ≈ 0,7 s por pie y lector (ITSC-420).
FRONTIER_WORKERS_ENV = "L3_FRONTIER_WORKERS"
DEFAULT_FRONTIER_WORKERS = 64
# Tope: `pool.map` encola todas las tareas y cada lector es un hilo; más de esto no acelera (E/S de
# un viaje) y solo gasta hilos.
MAX_FRONTIER_WORKERS = 256
THETA_DIGITS = 8
# Por familia: su esquema y su `state_version`, lo que hace válido el archivo de la frontera.
_CONTRACTS = MappingProxyType(
    {
        FAMILY: (SUMMARIES_SCHEMA, STATE_VERSION),
        PATH_FAMILY: (PATH_SCHEMA, PATH_STATE_VERSION),
        FIT_FAMILY: (FIT_SCHEMA, FIT_STATE_VERSION),
        SPECTRUM_FAMILY: (SPECTRUM_SCHEMA, SPECTRUM_STATE_VERSION),
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
    """El archivo de la familia en la partición: `summaries`, `path`, `fit` o `spectrum` más `.parquet`."""
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


class FooterReader:
    """Lee los pies de la verificación de una corrida (ITSC-420): solo pies, nunca datos.

    Lo que lo distingue de `open_parquet` (que no cambia, lo usan otros lectores):
    - un solo sistema de archivos por raíz (`resolve_fs` crea uno por llamada y, en `gs://`,
      cada uno es una conexión y unas credenciales nuevas);
    - sin el `get_file_info` de existencia: el listado ya dijo que el archivo estaba. Uno que
      desapareció entre el listado y la lectura falla al abrirlo y es «no se pudo leer»;
    - con `share_events`, el pie de `events.parquet` de un `(θ, mes)` se lee una sola vez por
      corrida aunque lo pidan la familia y el techo de S. Del pie se guarda solo lo que la
      comprobación usa, las filas por row group (un `array` de enteros de 8 B por row group
      no vacío), no el pie; si la lectura falló, solo el texto del motivo, nunca la excepción
      (su traceback ancla los marcos y con ellos el pie). Cada entrada cuesta además su clave
      (la ruta) y el objeto `array` o `str`, del orden de 200 B. El caché vive lo que vive el
      lector: se libera con él.

    `opened` cuenta los pies que se pidieron al almacenamiento de verdad, también los que
    fallaron. Es seguro desde varios hilos.
    """

    def __init__(self, share_events: bool = True) -> None:
        self._lock = threading.Lock()
        self._roots: dict[str, tuple[pafs.FileSystem, str]] = {}
        self._rows: dict[str, array | str] | None = {} if share_events else None
        self.opened = 0

    def _locate(self, root: str, path: str) -> tuple[pafs.FileSystem, str]:
        """El sistema de archivos de `root` (uno por raíz) y `path` dentro de él."""
        prefix = str(root).rstrip("/")
        with self._lock:
            found = self._roots.get(prefix)
            if found is None:
                fs, base = resolve_fs(prefix)
                found = self._roots[prefix] = (fs, base.rstrip("/"))
        fs, base = found
        return fs, base + path[len(prefix) :]

    def open(self, root: str, path: str, what: str) -> pq.ParquetFile:
        """Abre `path`, que cuelga de `root`, y lee su pie; quien llama lo cierra.

        `FramesInputError` si no existe.
        """
        fs, resolved = self._locate(root, path)
        with self._lock:
            self.opened += 1
        try:
            return pq.ParquetFile(resolved, filesystem=fs)
        except FileNotFoundError:
            raise FramesInputError(f"falta {what}: {path}") from None

    def _read_rows(self, root: str, path: str) -> array | str:
        """Las filas por row group de `path`, o el motivo (texto) de que no se pudo leer."""
        try:
            with closing(self.open(root, path, "L2")) as events:
                return array("q", row_group_sizes(events))
        except (FramesError, pa.ArrowException, OSError) as exc:
            return f"no se pudo leer ({exc})"

    def event_rows(self, root: str, path: str) -> array | str:
        """Las filas por row group de `events.parquet` en `path`, o el motivo si no se leyó.

        El pie se lee una vez por corrida. No hace falta que dos hilos esperen la misma ruta: las
        pasadas de la familia y del techo de S corren en serie y, dentro de una, cada `(θ, mes)`
        se pide una sola vez. Si alguna vez coincidieran, el costo sería una lectura repetida.
        """
        if self._rows is None:
            return self._read_rows(root, path)
        with self._lock:
            found = self._rows.get(path)
        if found is None:
            found = self._read_rows(root, path)
            with self._lock:
                self._rows[path] = found
        return found


def invalid_reason(
    summaries_root: str,
    events_root: str,
    key: tuple[str, str, str],
    theta: int,
    month: Month,
    family: str = FAMILY,
    reader: FooterReader | None = None,
) -> str | None:
    """Por qué el archivo de la `family` de `(θ, mes)` no es una frontera válida, o `None`.

    Lee solo los pies del archivo de la familia y de `events.parquet` del mes. Con el `reader`
    de la corrida comparte el sistema de archivos y el pie de `events.parquet` (ver
    `FooterReader`); sin él, lee todo por su cuenta.
    """
    reader = FooterReader(share_events=False) if reader is None else reader
    expected, version_wanted = _CONTRACTS[family]
    text = theta_text(theta)
    path = family_path(summaries_root, key, text, month, family)
    events = l2_path(events_root, key, text, month)
    try:
        with closing(reader.open(summaries_root, path, "la familia")) as table:
            # Antes de cualquier comprobación, como siempre: si L2 no se lee, eso manda.
            skeleton = reader.event_rows(events_root, events)
            if isinstance(skeleton, str):
                return skeleton
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
            if array("q", row_group_sizes(table)) != skeleton:
                return "sus filas por row group no son las de events.parquet de L2"
    except (FramesError, pa.ArrowException, OSError) as exc:
        return f"no se pudo leer ({exc})"
    return None


@dataclass(frozen=True)
class Frontier:
    """La frontera de un θ y lo que la verificación encontró.

    `reached` es el ordinal del último mes del prefijo contiguo de meses válidos (`None` si no
    hay). `invalid` son los meses de la serie contigua cuyo archivo **de la familia** es
    inválido, ordinal → motivo (con el archivo delante: `path.parquet: ...`); solo trae los que
    la verificación leyó, es decir, hasta el primer hueco de la serie. `ceiling_invalid` son los
    de `summaries.parquet` cuando la familia no es S (su techo, §7.3): una pasada de la familia
    no los repara, por eso van aparte (vacío en S). `checked` cuenta los meses cuyo pie se leyó
    (dos pies cada uno, el de la familia y el de `events.parquet`, salvo que el de `events.parquet`
    ya estuviera leído: cuántos pies se pidieron de verdad lo dice `FooterReader.opened`).
    """

    reached: int | None
    invalid: Mapping[int, str] = field(default_factory=dict)
    checked: int = 0
    ceiling_invalid: Mapping[int, str] = field(default_factory=dict)

    @property
    def all_invalid(self) -> Mapping[int, str]:
        """Los inválidos de la familia y de su techo, con el motivo de la familia si coinciden."""
        return {**self.ceiling_invalid, **self.invalid}

    @property
    def first_invalid(self) -> int | None:
        """El primer mes inválido: el que retrocede la frontera (`None` si no hay)."""
        return min(self.all_invalid, default=None)


def _scan_all(
    thetas: list[int],
    listed: Listing,
    start: int,
    summaries_root: str,
    events_root: str,
    key: tuple[str, str, str],
    family: str,
    workers: int,
    reader: FooterReader,
) -> dict[int, Frontier]:
    """Verifica cada mes de la serie contigua de cada θ; la frontera es el fin de su prefijo válido.

    Lee todos los meses de la serie contigua, también los posteriores a un inválido: así la
    cuenta de inválidos es completa y ningún mes anterior a la frontera queda sin verificar.
    La tarea del pool es un `(θ, mes)`, no un θ: con 50 θ y 64 lectores, una tarea por θ dejaría
    14 lectores sin nada que hacer y cada θ recorrería sus 111 meses en serie. Cada tarea trae
    un motivo corto (o `None`), no un pie.
    """
    # `end` es el primer mes que falta: la serie contigua llega hasta el anterior.
    ends = {}
    for theta in thetas:
        months = listed.get(theta, {})
        end = start
        while end in months:
            end += 1
        ends[theta] = end
    tasks = [(theta, n) for theta in thetas for n in range(start, ends[theta])]

    def check(task: tuple[int, int]) -> str | None:
        theta, n = task
        return invalid_reason(
            summaries_root, events_root, key, theta, month_of(n), family, reader
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        problems = list(pool.map(check, tasks))
    invalid: dict[int, dict[int, str]] = {theta: {} for theta in thetas}
    for (theta, n), problem in zip(tasks, problems, strict=True):
        if problem is not None:
            invalid[theta][n] = f"{family_file(family)}: {problem}"
    result = {}
    for theta in thetas:
        first = min(invalid[theta], default=ends[theta])
        result[theta] = Frontier(
            reached=first - 1 if first > start else None,
            invalid=invalid[theta],
            checked=ends[theta] - start,
        )
    return result


def scan_frontiers(
    thetas: Iterable[int],
    listed: Listing,
    series_start: Month,
    summaries_root: str,
    events_root: str,
    key: tuple[str, str, str],
    family: str = FAMILY,
    summaries: Listing | None = None,
    workers: int | None = None,
    reader: FooterReader | None = None,
) -> dict[int, Frontier]:
    """La frontera de cada θ con lo que su verificación encontró (`frontiers` solo da el ordinal).

    `listed` son los archivos de la `family`. Se verifica **cada mes** de la serie contigua del
    θ desde `series_start` (solo pies, con un pool de `workers` lectores, 64 por defecto): la
    frontera es el fin del prefijo de meses válidos y un inválido en `m` la deja en `m − 1`.
    La de una familia distinta de S nunca supera la de S: `summaries`, el listado de
    `summaries.parquet`, es obligatorio, porque su archivo repite ese esqueleto; sus inválidos
    quedan aparte, en `ceiling_invalid`.

    El `reader` es el de la corrida (ver `FooterReader`): quien llama puede pasarlo para leer
    después cuántos pies se pidieron de verdad (`opened`). Sin él se crea uno, que comparte el
    pie de `events.parquet` solo si la familia no es S (en S nadie lo pide dos veces).
    """
    workers = DEFAULT_FRONTIER_WORKERS if workers is None else workers
    if workers < 1:
        raise ValueError(f"workers {workers} debe ser al menos 1")
    if family != FAMILY and summaries is None:
        raise ValueError(f"la frontera de {family!r} necesita el listado de {FAMILY!r}")
    reader = FooterReader(share_events=family != FAMILY) if reader is None else reader
    thetas = list(thetas)
    start = month_ordinal(series_start)
    own = _scan_all(
        thetas, listed, start, summaries_root, events_root, key, family, workers, reader
    )
    if family == FAMILY:
        return own
    ceiling = _scan_all(
        thetas,
        summaries or {},
        start,
        summaries_root,
        events_root,
        key,
        FAMILY,
        workers,
        reader,
    )
    result = {}
    for theta, found in own.items():
        top = ceiling[theta]
        reached = (
            None
            if found.reached is None or top.reached is None
            else min(found.reached, top.reached)
        )
        result[theta] = Frontier(
            reached=reached,
            invalid=found.invalid,
            checked=found.checked + top.checked,
            ceiling_invalid=top.invalid,
        )
    return result


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

    Es `scan_frontiers` sin el detalle de la verificación.
    """
    found = scan_frontiers(
        thetas,
        listed,
        series_start,
        summaries_root,
        events_root,
        key,
        family,
        summaries,
    )
    return {theta: frontier.reached for theta, frontier in found.items()}
