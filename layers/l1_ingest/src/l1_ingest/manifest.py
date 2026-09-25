"""Manifiesto de checksums: una fila por archivo ingerido (§7.4 del TRD-L1)."""

import uuid
from collections import defaultdict
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

GRANULARITIES = ("monthly", "daily")

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("provider", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset", pa.string(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("month", pa.int32(), nullable=False),
        pa.field("granularity", pa.string(), nullable=False),
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("downloaded_at", pa.int64(), nullable=False),
        pa.field("file_bytes", pa.int64(), nullable=False),
        pa.field("image_version", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ManifestEntry:
    """Procedencia de un archivo ingerido. `downloaded_at` va en µs UTC."""

    provider: str
    market: str
    asset: str
    year: int
    month: int
    granularity: str
    source_url: str
    sha256: str
    downloaded_at: int
    file_bytes: int
    image_version: str

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(
                f"granularity debe ser una de {GRANULARITIES}, no {self.granularity!r}"
            )


def resolve_fs(root: str | Path) -> tuple[pafs.FileSystem, str]:
    """Devuelve el sistema de archivos y la ruta base dentro de él."""
    if isinstance(root, str) and root.startswith("gs://"):
        return pafs.FileSystem.from_uri(root)
    return pafs.LocalFileSystem(), str(Path(root).resolve())


def _to_table(entries: list[ManifestEntry]) -> pa.Table:
    names = [f.name for f in fields(ManifestEntry)]
    columns = list(zip(*(astuple(e) for e in entries), strict=True))
    return pa.Table.from_arrays(
        [
            pa.array(col, type=MANIFEST_SCHEMA.field(n).type)
            for n, col in zip(names, columns, strict=True)
        ],
        schema=MANIFEST_SCHEMA,
    )


def write_manifest(
    entries: list[ManifestEntry], root: str | Path, run_id: str
) -> list[str]:
    """Agrega las entradas al manifiesto y devuelve las rutas escritas.

    `root` es una ruta local o `gs://<bucket>/<prefijo>`. Escribe un Parquet
    (ZSTD-3) por scope bajo
    `<root>/provider=<p>/market=<m>/asset=<a>/year=YYYY/month=MM/`, con nombre
    `<run_id>-<uuid4>.parquet` único, así que nunca sobrescribe (append-only).
    En GCS las rutas devueltas van sin el esquema `gs://`.
    """
    if not entries:
        return []
    fs, base = resolve_fs(root)

    by_scope: dict[tuple, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        scope = (entry.provider, entry.market, entry.asset, entry.year, entry.month)
        by_scope[scope].append(entry)

    paths = []
    for (provider, market, asset, year, month), group in by_scope.items():
        directory = (
            f"{base}/provider={provider}/market={market}/asset={asset}"
            f"/year={year:04d}/month={month:02d}"
        )
        if not isinstance(fs, pafs.GcsFileSystem):
            # En un almacén de objetos los directorios no existen.
            fs.create_dir(directory, recursive=True)
        path = f"{directory}/{run_id}-{uuid.uuid4()}.parquet"
        pq.write_table(
            _to_table(group),
            path,
            filesystem=fs,
            compression="zstd",
            compression_level=3,
            write_statistics=True,
        )
        paths.append(path)
    return paths
