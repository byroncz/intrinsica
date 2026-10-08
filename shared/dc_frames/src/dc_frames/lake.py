"""Lectura de L1 (`consolidated.parquet`) y L2 (`events.parquet`) por row group.

Solo se leen archivos Parquet por nombre de columna: aquí no se importa nada de L1
ni de L2 (las capas no se importan entre sí, RNF-14 del maestro).
"""

from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyutils import resolve_fs

from dc_frames.types import Frame, FramesInputError

CONSOLIDATED = "consolidated.parquet"
EVENTS = "events.parquet"

DECIMAL = pa.decimal128(18, 8)
# Lo que el lector pide de L1: la clave y las cuatro columnas de una trama.
# `first_trade_id`, `last_trade_id` e `is_best_match` no se leen (TRD-L3 §7.4).
L1_COLUMNS = {
    "agg_trade_id": pa.int64(),
    "transact_time": pa.int64(),
    "price": DECIMAL,
    "quantity": DECIMAL,
    "is_buyer_maker": pa.bool_(),
}
BOUNDARY_COLUMNS = (
    "reference_agg_trade_id",
    "confirm_agg_trade_id",
    "extreme_agg_trade_id",
)
CONFIRM_TIME = "confirm_time"

Month = tuple[int, int]


def month_ordinal(month: Month) -> int:
    return month[0] * 12 + month[1] - 1


def month_of(ordinal: int) -> Month:
    year, number = divmod(ordinal, 12)
    return year, number + 1


def months(first: Month, last: Month) -> Iterator[Month]:
    """Los meses de `first` a `last`, ambos inclusivos."""
    for n in range(month_ordinal(first), month_ordinal(last) + 1):
        yield month_of(n)


def _month_dir(provider: str, market: str, asset: str, month: Month) -> str:
    year, number = month
    return (
        f"provider={provider}/market={market}/asset={asset}"
        f"/year={year:04d}/month={number:02d}"
    )


def l1_path(root: str | Path, key: tuple[str, str, str], month: Month) -> str:
    return f"{str(root).rstrip('/')}/{_month_dir(*key, month)}/{CONSOLIDATED}"


def l2_path(
    root: str | Path, key: tuple[str, str, str], theta: str, month: Month
) -> str:
    """`theta` es el texto de la partición (`0.00010000`)."""
    provider, market, asset = key
    year, number = month
    return (
        f"{str(root).rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/theta={theta}/year={year:04d}/month={number:02d}/{EVENTS}"
    )


def open_parquet(path: str, what: str) -> pq.ParquetFile:
    """Abre `path` sin leer datos; `FramesInputError` si no existe. Quien llama lo cierra."""
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        raise FramesInputError(f"falta {what}: {path}")
    return pq.ParquetFile(resolved, filesystem=fs)


def check_columns(
    schema: pa.Schema, expected: dict[str, pa.DataType], path: str
) -> None:
    for name, kind in expected.items():
        if name not in schema.names:
            raise FramesInputError(f"{path}: falta la columna {name!r}")
        if schema.field(name).type != kind:
            raise FramesInputError(
                f"{path}: {name} es {schema.field(name).type}, se esperaba {kind}"
            )


def _single(column: pa.ChunkedArray) -> pa.Array:
    """El array de un row group; sin copiar cuando viene en un solo chunk."""
    if column.num_chunks == 1:
        return column.chunk(0)
    return column.combine_chunks()


@dataclass(slots=True)
class RowGroup:
    """Un row group de L1 decodificado: `ids` más las cuatro columnas de una trama.

    `ids` es una vista de NumPy sobre el buffer de Arrow, sin copia.
    """

    ids: np.ndarray
    columns: tuple[pa.Array, pa.Array, pa.Array, pa.Array]

    @property
    def low(self) -> int:
        return int(self.ids[0])

    @property
    def high(self) -> int:
        return int(self.ids[-1])

    def frame(self, start: int, stop: int) -> Frame:
        """Los ticks en las posiciones `[start, stop)`, como slices sin copia."""
        return Frame(*(column.slice(start, stop - start) for column in self.columns))


def _bounds(parquet: pq.ParquetFile, path: str) -> list[tuple[int, int] | None]:
    """`(mínimo, máximo)` de `agg_trade_id` por row group; `None` si el row group es vacío."""
    index = parquet.schema_arrow.get_field_index("agg_trade_id")
    out: list[tuple[int, int] | None] = []
    for i in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(i)
        if group.num_rows == 0:
            out.append(None)
            continue
        stats = group.column(index).statistics
        if stats is None or not stats.has_min_max:
            raise FramesInputError(
                f"{path}: el row group {i} no trae estadísticas de agg_trade_id"
            )
        out.append((stats.min, stats.max))
    return out


class L1:
    """L1 de un activo: entrega sus row groups en orden de id, de uno en uno."""

    def __init__(self, root: str | Path, key: tuple[str, str, str]) -> None:
        self._root = root
        self._key = key

    def _open(self, month: Month) -> tuple[pq.ParquetFile, str]:
        path = l1_path(self._root, self._key, month)
        parquet = open_parquet(path, f"L1 del mes {month[0]:04d}-{month[1]:02d}")
        try:
            check_columns(parquet.schema_arrow, L1_COLUMNS, path)
        except FramesInputError:
            parquet.close()
            raise
        return parquet, path

    def _first_month(self, month: Month, reference: int) -> Month:
        """El mes más cercano a `month` hacia atrás cuyo primer id es ≤ `reference`.

        Un evento del primer mes puede tener su referencia en meses anteriores.
        """
        while True:
            parquet, path = self._open(month)
            with closing(parquet):
                first = next((b[0] for b in _bounds(parquet, path) if b), None)
            if first is not None and first <= reference:
                return month
            month = month_of(month_ordinal(month) - 1)

    def row_groups(
        self, first: Month, last: Month, reference: int, needed: Callable[[int], bool]
    ) -> Iterator[RowGroup]:
        """Los row groups de L1 desde el que contiene `reference` hasta `last`.

        Un row group solo se decodifica si `needed(máximo)` lo pide; los demás se
        saltan con las estadísticas del pie del archivo. Un row group se suelta
        cuando el llamador pide el siguiente. Si falta un mes intermedio, o los ids
        no crecen entre row groups, falla.
        """
        previous = -1
        for month in months(self._first_month(first, reference), last):
            parquet, path = self._open(month)
            with closing(parquet):
                for i, bounds in enumerate(_bounds(parquet, path)):
                    if bounds is None:
                        continue
                    if bounds[0] <= previous:
                        raise FramesInputError(
                            f"{path}: el row group {i} empieza en el id {bounds[0]}, "
                            f"que no supera el id {previous} anterior"
                        )
                    previous = bounds[1]
                    if not needed(bounds[1]):
                        continue
                    table = parquet.read_row_group(i, columns=list(L1_COLUMNS))
                    columns = [_single(table.column(name)) for name in L1_COLUMNS]
                    del table
                    yield RowGroup(
                        columns[0].to_numpy(zero_copy_only=True), tuple(columns[1:])
                    )
                    del columns


@dataclass(slots=True)
class EventGroup:
    """Un row group de `events.parquet`: sus filas y sus tres ids de frontera.

    `bounds[i] = (R, C, E)` del evento `i` y `confirm_times[i]` es su `confirm_time`, que
    el tick `C` debe tener como `transact_time`. `batch` conserva las 11 columnas.
    """

    batch: pa.RecordBatch
    bounds: np.ndarray
    confirm_times: np.ndarray

    def __len__(self) -> int:
        return len(self.bounds)


def event_group(batch: pa.RecordBatch, where: str) -> EventGroup:
    """Valida el orden de los eventos de `batch` y extrae sus fronteras.

    Un evento cumple `R < C ≤ E`, y el siguiente empieza donde el anterior termina o
    después (`E_i ≤ R_{i+1}`). La búsqueda por prefijos del lector depende de eso.
    """
    bounds = np.column_stack(
        [batch.column(name).to_numpy(zero_copy_only=False) for name in BOUNDARY_COLUMNS]
    )
    reference, confirm, extreme = bounds.T
    if not (
        np.all(reference < confirm)
        and np.all(confirm <= extreme)
        and np.all(reference[1:] >= extreme[:-1])
    ):
        raise FramesInputError(
            f"{where}: los eventos no cumplen R < C <= E ni E[i] <= R[i+1]"
        )
    confirm_times = batch.column(CONFIRM_TIME).to_numpy(zero_copy_only=False)
    return EventGroup(batch, bounds, confirm_times)


def event_groups(
    root: str | Path,
    key: tuple[str, str, str],
    theta: str,
    first: Month,
    last: Month,
) -> Iterator[EventGroup]:
    """Los row groups de eventos de un θ, mes a mes, de uno en uno y sin vacíos."""
    expected = {name: pa.int64() for name in (*BOUNDARY_COLUMNS, CONFIRM_TIME)}
    for month in months(first, last):
        path = l2_path(root, key, theta, month)
        parquet = open_parquet(
            path, f"L2 del θ {theta} en {month[0]:04d}-{month[1]:02d}"
        )
        with closing(parquet):
            check_columns(parquet.schema_arrow, expected, path)
            for i in range(parquet.metadata.num_row_groups):
                if parquet.metadata.row_group(i).num_rows == 0:
                    continue
                table = parquet.read_row_group(i).combine_chunks()
                yield event_group(table.to_batches()[0], f"{path}, row group {i}")
