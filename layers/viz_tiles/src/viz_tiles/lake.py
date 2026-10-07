"""Las entradas de viz en el lago: rutas de L1 y L2, listado, huella y lectores.

Las raíces son locales o `gs://`. Una ruta *relativa* a su raíz
(`provider=…/consolidated.parquet`) es la identidad de un archivo de entrada en
el `input_hash` (TRD-viz §7.8); la absoluta se arma con `join`.

Aquí no se importa nada de L1 ni de L2 (`l2_dc_events` arrastra `dc_pyo3`, que
la imagen de viz no trae): solo se leen sus archivos Parquet por nombre de columna.
"""

import base64
import functools
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import google_crc32c
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyutils import resolve_fs

from viz_tiles.ticks import scaled_ints

CONSOLIDATED = "consolidated.parquet"
EVENTS = "events.parquet"
CARRY_OVER = "carry_over.parquet"

TICK_COLUMNS = ("agg_trade_id", "price", "quantity", "transact_time")
# Lo que un día necesita de cada evento: los ids de sus tres puntos (qué eventos
# tocan el día y la posición de cada tick en `events.bin`), los tres tiempos, el
# sentido y el precio de la confirmación (con él se elige el tick de la confirmación
# dentro de su instante, TRD-viz §7.5). El orden es el de los campos de `MonthEvents`.
EVENT_COLUMNS = (
    "reference_agg_trade_id",
    "confirm_agg_trade_id",
    "extreme_agg_trade_id",
    "reference_time",
    "confirm_time",
    "extreme_time",
    "direction",
    "confirm_price",
)
CARRY_COLUMNS = (
    "direction",
    "has_pending_event",
    "pending_reference_agg_trade_id",
    "pending_confirm_agg_trade_id",
    "pending_reference_time",
    "pending_confirm_time",
    "pending_confirm_price",
    "ext_high_agg_trade_id",
    "ext_high_time",
    "ext_low_agg_trade_id",
    "ext_low_time",
)
DECIMAL_TYPE = pa.decimal128(18, 8)

Month = tuple[int, int]

_CHUNK = 1 << 20


class InputError(Exception):
    """Una entrada existe pero no se puede usar, o falta.

    `what` es el valor de `details.what` del hallazgo `input_missing`
    (TRD-viz §9.3) y `details` el resto de su payload (`path`, `theta`, `reason`).
    """

    def __init__(self, what: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.what = what
        self.details = details


def ordinal(month: Month) -> int:
    """El mes como un entero consecutivo (`año * 12 + mes - 1`)."""
    return month[0] * 12 + month[1] - 1


def month_of(n: int) -> Month:
    year, number = divmod(n, 12)
    return year, number + 1


def join(root: str | Path, rel: str) -> str:
    return f"{str(root).rstrip('/')}/{rel}"


def landing_rel(provider: str, market: str, asset: str, month: Month) -> str:
    """Ruta del `consolidated.parquet` del mes, relativa a la raíz de L1."""
    year, number = month
    return (
        f"provider={provider}/market={market}/asset={asset}"
        f"/year={year:04d}/month={number:02d}/{CONSOLIDATED}"
    )


def events_rel(
    provider: str, market: str, asset: str, theta: str, month: Month, filename: str
) -> str:
    """Ruta de un archivo de L2 del θ y el mes, relativa a la raíz de L2.

    `theta` es el texto de la partición (`0.00010000`).
    """
    year, number = month
    return (
        f"provider={provider}/market={market}/asset={asset}/theta={theta}"
        f"/year={year:04d}/month={number:02d}/{filename}"
    )


_THETA_FILE = re.compile(
    r"/theta=(0\.\d{8})/year=(\d{4})/month=(\d{2})/"
    rf"({re.escape(EVENTS)}|{re.escape(CARRY_OVER)})$"
)


class EventsIndex:
    """Qué archivos de L2 hay en el lago, de un solo listado recursivo del activo."""

    def __init__(self, files: dict[Month, dict[str, set[str]]]) -> None:
        self._files = files

    @classmethod
    def list(
        cls, events_root: str | Path, provider: str, market: str, asset: str
    ) -> EventsIndex:
        prefix = join(events_root, f"provider={provider}/market={market}/asset={asset}")
        fs, resolved = resolve_fs(prefix)
        selector = pafs.FileSelector(resolved, recursive=True, allow_not_found=True)
        files: dict[Month, dict[str, set[str]]] = {}
        for info in fs.get_file_info(selector):
            match = _THETA_FILE.search(info.path)
            if match is None or info.type != pafs.FileType.File:
                continue
            month = (int(match[2]), int(match[3]))
            files.setdefault(month, {}).setdefault(match[1], set()).add(match[4])
        return cls(files)

    def thetas(self, month: Month) -> list[str]:
        """Los θ con algún archivo de L2 en el mes, de menor a mayor."""
        return sorted(self._files.get(month, {}))

    def has(self, month: Month, theta: str, filename: str) -> bool:
        return filename in self._files.get(month, {}).get(theta, set())


def exists(path: str) -> bool:
    fs, resolved = resolve_fs(path)
    return fs.get_file_info(resolved).type == pafs.FileType.File


@functools.cache
def _gcs_client():
    from google.cloud import storage

    return storage.Client()


def _split_gs(path: str) -> tuple[str, str]:
    bucket, _, name = path.removeprefix("gs://").partition("/")
    return bucket, name


def crc32c_of_file(path: str) -> str:
    """CRC32C del archivo local, en hexadecimal de 8 dígitos (como lo da GCS)."""
    checksum = google_crc32c.Checksum()
    with open(path, "rb") as source:
        while chunk := source.read(_CHUNK):
            checksum.update(chunk)
    return checksum.digest().hex()


def object_stat(path: str) -> tuple[int, str]:
    """`(tamaño, CRC32C)` del archivo, sin descargarlo si es un objeto de GCS.

    En GCS salen de los metadatos del objeto (el CRC32C llega en base64 y se
    pasa a hexadecimal); en local, de leer el archivo. Lanza `FileNotFoundError`
    si no existe.
    """
    if not path.startswith("gs://"):
        local = Path(path)
        return local.stat().st_size, crc32c_of_file(path)
    bucket, name = _split_gs(path)
    blob = _gcs_client().bucket(bucket).get_blob(name)
    if blob is None:
        raise FileNotFoundError(path)
    return int(blob.size), base64.b64decode(blob.crc32c).hex()


def input_hash(
    tiles_version: str, day: date, stats: Mapping[str, tuple[int, str]]
) -> str:
    """El `input_hash` del día (TRD-viz §7.8).

    SHA-256 del texto canónico: `tiles_version`, `day` y una línea
    `<ruta relativa>␉<tamaño>␉<CRC32C>` por archivo de entrada, ordenadas por ruta.
    """
    lines = [tiles_version, day.isoformat()]
    lines += [f"{rel}\t{size}\t{crc}" for rel, (size, crc) in sorted(stats.items())]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def open_parquet(path: str) -> pq.ParquetFile:
    """Abre `path` sin leer datos. Quien lo llama lo cierra."""
    fs, resolved = resolve_fs(path)
    return pq.ParquetFile(resolved, filesystem=fs)


def column_bounds(parquet: pq.ParquetFile, name: str) -> list[tuple[int, int] | None]:
    """`(mínimo, máximo)` de la columna en cada row group, o `None` sin estadísticas."""
    index = parquet.schema_arrow.get_field_index(name)
    bounds: list[tuple[int, int] | None] = []
    for i in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(i).column(index).statistics
        has = stats is not None and stats.has_min_max
        bounds.append((stats.min, stats.max) if has else None)
    return bounds


def validate_ticks(parquet: pq.ParquetFile, path: str) -> None:
    """Las columnas que viz lee de L1 y su tipo (contrato de L1)."""
    schema = parquet.schema_arrow
    for name in TICK_COLUMNS:
        if name not in schema.names:
            raise InputError(
                "l1",
                f"{path}: falta la columna {name!r} del contrato de L1",
                path=path,
                reason=f"falta la columna {name}",
            )
    for name in ("price", "quantity"):
        if schema.field(name).type != DECIMAL_TYPE:
            raise InputError(
                "l1",
                f"{path}: {name} es {schema.field(name).type}, se esperaba {DECIMAL_TYPE}",
                path=path,
                reason=f"{name} no es {DECIMAL_TYPE}",
            )


def read_carry_over(path: str, theta: str) -> dict:
    """La única fila del `carry_over.parquet`, con las columnas que viz usa."""
    fs, resolved = resolve_fs(path)
    try:
        table = pq.read_table(resolved, columns=list(CARRY_COLUMNS), filesystem=fs)
        if table.num_rows != 1:
            raise ValueError(f"trae {table.num_rows} filas y se esperaba una")
    except (pa.ArrowException, ValueError) as exc:
        raise InputError(
            "carry_over",
            f"{path}: no se pudo leer el carry-over ({type(exc).__name__}: {exc})",
            theta=theta,
            path=path,
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    return table.to_pylist()[0]


def _overlaps(
    bounds: tuple[int, int] | None, low: int | None, high: int | None
) -> bool:
    """¿Un row group con esos `(mín, máx)` puede tener un valor en `[low, high]`?"""
    if bounds is None:
        return True
    return (high is None or bounds[0] <= high) and (low is None or bounds[1] >= low)


@dataclass(frozen=True)
class MonthEvents:
    """Los eventos de un θ en un mes, en arreglos de NumPy y ordenados por referencia.

    Se leen una sola vez por mes y de ahí salen los de cada día (`touching`): son
    49 bytes por evento, y un mes de 50 θ trae decenas de miles. Los tiempos son
    µs UTC; `direction` es 1 (alza) o -1 (baja); `confirm_price` es el precio de la
    confirmación en enteros de 10⁻⁸ (el `decimal128(18, 8)` de L2, sin perder nada).
    """

    reference_id: np.ndarray
    confirm_id: np.ndarray
    extreme_id: np.ndarray
    reference_time: np.ndarray
    confirm_time: np.ndarray
    extreme_time: np.ndarray
    direction: np.ndarray
    confirm_price: np.ndarray

    def __len__(self) -> int:
        return len(self.reference_id)

    @classmethod
    def empty(cls) -> MonthEvents:
        return cls(
            *(np.empty(0, np.int64) for _ in range(6)),
            np.empty(0, np.int8),
            np.empty(0, np.int64),
        )

    def touching(self, first_id: int, last_id: int) -> MonthEvents:
        """Los eventos que tocan los ticks de id `[first_id, last_id]`.

        Un evento toca el rango si `reference < last_id` y `extreme >= first_id`.
        """
        keep = (self.reference_id < last_id) & (self.extreme_id >= first_id)
        return MonthEvents(*(column[keep] for column in vars(self).values()))


def _event_column(column: pa.ChunkedArray, dtype: type) -> np.ndarray:
    """Una columna de `events.parquet` como arreglo de NumPy; un decimal, en enteros de 10⁻⁸."""
    if pa.types.is_decimal(column.type):
        if not len(column):
            return np.empty(0, dtype)
        return np.ascontiguousarray(scaled_ints(column.combine_chunks()), dtype)
    return column.to_numpy().astype(dtype, copy=False)


def read_month_events(path: str) -> MonthEvents:
    """Todos los eventos de `events.parquet`, de una sola lectura, row group a row group.

    Cada row group se pasa a arreglos de NumPy y se suelta antes de leer el que
    sigue. Lanza `ValueError` si las referencias no son estrictamente crecientes
    una vez ordenadas (dos eventos con el mismo tick de referencia).
    """
    with open_parquet(path) as parquet:
        parts: list[MonthEvents] = []
        for i in range(parquet.num_row_groups):
            group = parquet.read_row_group(
                i, columns=list(EVENT_COLUMNS), use_threads=False
            )
            parts.append(
                MonthEvents(
                    *(
                        _event_column(group[name], dtype)
                        for name, dtype in zip(
                            EVENT_COLUMNS,
                            (np.int64,) * 6 + (np.int8, np.int64),
                            strict=True,
                        )
                    )
                )
            )
            del group
    if not parts:
        return MonthEvents.empty()
    events = MonthEvents(
        *(np.concatenate([getattr(p, name) for p in parts]) for name in vars(parts[0]))
    )
    del parts
    if np.any(np.diff(events.reference_id) <= 0):
        order = np.argsort(events.reference_id, kind="stable")
        events = MonthEvents(*(column[order] for column in vars(events).values()))
        if np.any(np.diff(events.reference_id) <= 0):
            raise ValueError(f"{path}: hay eventos con la misma referencia")
    return events


def find_extreme(path: str, reference_agg_trade_id: int) -> tuple[int, int] | None:
    """El extremo `(agg_trade_id, time en µs)` del evento de esa referencia.

    Es el evento que estaba pendiente al cierre del mes anterior y ya cerró:
    TRD-viz §7.6. `None` si `events.parquet` no lo trae.
    """
    with open_parquet(path) as parquet:
        references = column_bounds(parquet, "reference_agg_trade_id")
        for i in range(parquet.num_row_groups):
            if not _overlaps(
                references[i], reference_agg_trade_id, reference_agg_trade_id
            ):
                continue
            group = parquet.read_row_group(
                i,
                columns=[
                    "reference_agg_trade_id",
                    "extreme_agg_trade_id",
                    "extreme_time",
                ],
                use_threads=False,
            )
            hit = group.filter(
                pc.equal(group["reference_agg_trade_id"], reference_agg_trade_id)
            )
            if hit.num_rows:
                return (
                    int(hit["extreme_agg_trade_id"][0].as_py()),
                    int(hit["extreme_time"][0].as_py()),
                )
    return None
