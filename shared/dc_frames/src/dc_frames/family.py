"""Escritor común de archivos de familia de L3 (TRD-L3 §7.3, ADR-L3-11, RL3-11).

Todo archivo de familia de una partición `theta/year/month` repite el esqueleto del
archivo que lo precede: las mismas filas, en el mismo orden y con los mismos límites de
row group. Para `summaries.parquet` ese archivo es `events.parquet` de L2; para las
demás familias, `summaries.parquet`. L4 los une por posición, así que un esqueleto
roto daría resultados falsos sin error: este escritor lo comprueba fila a fila y, si no
coincide, aborta con `FamilySkeletonMismatch` (`family_skeleton_mismatch`) sin publicar.
"""

import os
import re
from contextlib import closing
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyutils import ContentHasher, PartitionWriter

from dc_frames.lake import Month, ParquetSource, check_columns, partition_path
from dc_frames.types import FamilySkeletonMismatch, FramesInputError

KEYS = ("theta", "confirm_agg_trade_id")
SUMMARIES = "summaries"
SORT_ORDER = [("confirm_agg_trade_id", "ascending")]
_VERSION = re.compile(r"\d+\.\d+\.\d+")


def family_path(
    root: str | Path,
    key: tuple[str, str, str],
    theta: str,
    month: Month,
    family: str,
) -> str:
    """`<raíz de L3>/.../theta=<t>/year=YYYY/month=MM/<familia>.parquet`.

    `theta` es el texto de la partición (`0.00010000`); la familia de resumen se llama
    `summaries`.
    """
    return partition_path(root, key, theta, month, f"{family}.parquet")


class Skeleton(Protocol):
    """Lo que un `FamilyWriter` necesita saber del archivo cuyo esqueleto repite.

    `path` es el archivo (para los mensajes de error) y `row_groups`, las filas de cada
    row group no vacío, en orden. `differing_key` compara las claves (`theta` y
    `confirm_agg_trade_id`) del row group `n` que se escribe, cuya primera fila es la
    `first_row` del archivo, contra las del esqueleto: devuelve el detalle de la primera
    diferencia, o `None` si son las mismas. `close` suelta lo que el esqueleto tenga abierto.

    El esqueleto por defecto (`_FileSkeleton`) relee las claves del archivo de cada row
    group; quien ya tiene esas claves en RAM (L3 las vio al leer `events.parquet`) pasa el
    suyo y el escritor no vuelve al almacenamiento.
    """

    path: str
    row_groups: list[int]

    def differing_key(
        self, n: int, first_row: int, batch: pa.RecordBatch
    ) -> str | None:
        """Ver la clase."""
        ...

    def close(self) -> None:
        """Ver la clase."""
        ...


class _FileSkeleton:
    """El esqueleto de un archivo Parquet: sus claves se leen de él, un row group a la vez."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._source = ParquetSource(path, "el esqueleto de la familia")
        try:
            check_columns(
                self._source.schema, {"confirm_agg_trade_id": pa.int64()}, path
            )
            if "theta" not in self._source.schema.names:
                raise FramesInputError(f"{path}: falta la columna 'theta'")
            meta = self._source.metadata
            # Los row groups vacíos no se escriben: no hay fila que ponerles.
            self._groups = [
                (i, meta.row_group(i).num_rows)
                for i in range(meta.num_row_groups)
                if meta.row_group(i).num_rows
            ]
        except Exception:
            self._source.close()
            raise
        self.row_groups = [rows for _, rows in self._groups]

    def differing_key(
        self, n: int, first_row: int, batch: pa.RecordBatch
    ) -> str | None:
        # Un lector de vida corta con `pre_buffer` sobre el archivo ya abierto: las dos
        # columnas viajan juntas, sin abrir el archivo de nuevo en cada row group.
        with closing(self._source.reader()) as reader:
            wanted = reader.read_row_group(self._groups[n][0], columns=list(KEYS))
        for name in KEYS:
            same = pc.equal(batch.column(name), wanted.column(name).combine_chunks())
            # Una clave nula compara a null: sin `fill_null` pasaría la comprobación.
            same = same.fill_null(False)
            if not pc.all(same).as_py():
                row = pc.index(same, False).as_py()
                return (
                    f"row group {n}, fila {first_row + row}: "
                    f"{name} no es el del esqueleto"
                )
        return None

    def close(self) -> None:
        self._source.close()


class FamilyWriter:
    """Escribe un archivo de familia, un row group a la vez, contra su esqueleto.

    `skeleton` es el esqueleto que se repite: la ruta del archivo (`events.parquet` de L2
    para `summaries.parquet` y `summaries.parquet` de la partición para las demás
    familias) o un objeto `Skeleton`. Con una ruta, el escritor la abre y la cierra; si no
    existe, falla con `FramesInputError`: una familia distinta de `summaries` de un mes no
    se escribe sin su `summaries.parquet`. Un `Skeleton` que se pasa lo cierra quien lo creó.

    `schema` trae las claves `theta` y `confirm_agg_trade_id` (sin nulos) y las columnas
    de la familia, y no copia otras columnas de L2. Cada llamada a `write_row_group`
    recibe el siguiente row group del esqueleto, con el mismo número de filas y las
    mismas claves, en el mismo orden. Cualquier diferencia (filas, orden, límites de
    row group, o row groups que faltan o sobran al `commit`) lanza `FamilySkeletonMismatch`.

    El archivo lleva ZSTD-3, estadísticas, `sorting_columns` por `confirm_agg_trade_id`,
    la `state_version` de la familia en los metadatos y ninguna columna con diccionario
    (`PartitionWriter(compact_encoding=True)`). Se escribe a un temporal y `commit` lo
    renombra sobre el destino: un fallo, o salir del bloque `with` sin `commit`, deja el
    archivo anterior intacto. `commit` devuelve el `content_hash` del contenido lógico.

    Memoria: un row group de la familia y, con una ruta, las dos columnas de claves del
    esqueleto.
    """

    def __init__(
        self,
        path: str,
        schema: pa.Schema,
        state_version: str,
        skeleton: str | os.PathLike[str] | Skeleton,
    ) -> None:
        if not _VERSION.fullmatch(state_version):
            raise ValueError(f"state_version {state_version!r} no es semver (X.Y.Z)")
        _require_keys(schema)
        self.path = path
        self._owned = isinstance(skeleton, str | os.PathLike)
        self._skeleton: Skeleton = (
            _FileSkeleton(str(skeleton)) if self._owned else skeleton  # type: ignore[arg-type]
        )
        self.skeleton_path = self._skeleton.path
        try:
            self._groups = self._skeleton.row_groups
            self._schema = schema.remove_metadata()
            self._writer = PartitionWriter(
                path,
                self._schema.with_metadata({"state_version": state_version}),
                SORT_ORDER,
                compact_encoding=True,
            )
            self._hasher = ContentHasher(self._schema)
        except Exception:
            self._close_skeleton()
            raise
        self._written = 0
        self._rows = 0
        self._last_id: int | None = None

    def _close_skeleton(self) -> None:
        if self._owned:
            self._skeleton.close()

    @property
    def row_groups(self) -> list[int]:
        """Filas de cada row group del esqueleto: lo que `write_row_group` espera."""
        return list(self._groups)

    def __enter__(self) -> Self:
        self._writer.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self._writer.__exit__(exc_type, exc, tb)
        finally:
            self._close_skeleton()

    def write_row_group(self, batch: pa.RecordBatch) -> None:
        """Escribe el siguiente row group; debe repetir el del esqueleto."""
        if not batch.schema.equals(self._schema):
            raise ValueError(
                f"el esquema no es el de la familia:\n{batch.schema}\n!=\n{self._schema}"
            )
        if not batch.num_rows:
            return  # un lote vacío no es un row group: lo que falte lo dice `commit`
        n = self._written
        if n >= len(self._groups):
            raise self._mismatch(
                f"el esqueleto tiene {len(self._groups)} row groups y se escribe otro"
            )
        rows = self._groups[n]
        if batch.num_rows != rows:
            raise self._mismatch(
                f"el row group {n} tiene {batch.num_rows} filas y el del esqueleto {rows}"
            )
        self._check_order(batch, n)
        self._check_keys(batch, n)
        self._hasher.update(batch)
        self._writer.write_batch(batch)
        self._written += 1
        self._rows += rows

    def commit(self) -> str:
        """Publica el archivo y devuelve su `content_hash`."""
        if self._written != len(self._groups):
            raise self._mismatch(
                f"se escribieron {self._written} row groups y el esqueleto tiene "
                f"{len(self._groups)}"
            )
        self._writer.commit()
        return self._hasher.hexdigest()

    def _mismatch(self, detail: str) -> FamilySkeletonMismatch:
        return FamilySkeletonMismatch(
            f"family_skeleton_mismatch: {self.path} no repite el esqueleto de "
            f"{self.skeleton_path}: {detail}",
            detail,
        )

    def _check_order(self, batch: pa.RecordBatch, n: int) -> None:
        ids = batch.column("confirm_agg_trade_id").to_numpy(zero_copy_only=False)
        broken = np.flatnonzero(np.diff(ids) <= 0)
        if broken.size:
            raise self._mismatch(
                f"row group {n}: confirm_agg_trade_id no crece en la fila {broken[0] + 1}"
            )
        if self._last_id is not None and ids[0] <= self._last_id:
            raise self._mismatch(
                f"row group {n}: confirm_agg_trade_id {ids[0]} no supera el {self._last_id} "
                "del row group anterior"
            )
        self._last_id = int(ids[-1])

    def _check_keys(self, batch: pa.RecordBatch, n: int) -> None:
        detail = self._skeleton.differing_key(n, self._rows, batch)
        if detail is not None:
            raise self._mismatch(detail)


def _require_keys(schema: pa.Schema) -> None:
    for name in KEYS:
        if name not in schema.names:
            raise ValueError(f"el esquema de la familia no trae la clave {name!r}")
        if schema.field(name).nullable:
            raise ValueError(f"la clave {name!r} no puede ser nulable")
    if schema.field("confirm_agg_trade_id").type != pa.int64():
        raise ValueError("confirm_agg_trade_id debe ser int64")
    if not pa.types.is_decimal(schema.field("theta").type):
        raise ValueError("theta debe ser decimal")


__all__ = ["KEYS", "SUMMARIES", "FamilyWriter", "Skeleton", "family_path"]
