"""Vistas SQL versionadas de L3 sobre `summaries` unido a `events`."""

from importlib.resources import files
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from pyutils import resolve_fs

DERIVED_VERSION = "1.0.0"

__all__ = ["DERIVED_VERSION", "load_derived"]

# year y month son las únicas claves de partición que llegan a la vista; los
# segmentos provider=, market=, asset= y theta= se ignoran (theta es la columna
# DECIMAL(9,8) del archivo).
_PARTITIONING = ds.partitioning(
    pa.schema([("year", pa.int32()), ("month", pa.int32())]), flavor="hive"
)


def _dataset(name: str, source: pa.Table | str | Path) -> pa.Table | ds.Dataset:
    """Tabla en memoria tal cual; una ruta, como dataset que DuckDB lee por lotes."""
    if isinstance(source, pa.Table):
        return source
    fs, path = resolve_fs(source)
    if path.endswith(".parquet"):
        return ds.dataset(path, filesystem=fs, format="parquet")
    # Una raíz de L2 trae también carry_over.parquet y content_hash.json en cada
    # partición: solo entran los archivos `<name>.parquet`.
    selector = pafs.FileSelector(path, recursive=True)
    paths = sorted(
        info.path
        for info in fs.get_file_info(selector)
        if info.is_file and info.base_name == f"{name}.parquet"
    )
    return ds.dataset(
        paths,
        filesystem=fs,
        format="parquet",
        partitioning=_PARTITIONING,
        partition_base_dir=path,
    )


def load_derived(
    con: duckdb.DuckDBPyConnection,
    summaries: pa.Table | str | Path,
    events: pa.Table | str | Path,
) -> str:
    """Registra `summaries` y `events`, crea la vista `derived` y devuelve `DERIVED_VERSION`.

    Cada argumento es una tabla en memoria o una ruta local o `gs://`: un
    archivo `.parquet` o una raíz con particionado hive. Las rutas se registran
    como datasets de PyArrow; DuckDB los lee por lotes, sin credenciales ni
    `httpfs`.
    """
    con.register("summaries", _dataset("summaries", summaries))
    con.register("events", _dataset("events", events))
    con.execute(files("dc_views").joinpath("derived.sql").read_text(encoding="utf-8"))
    return DERIVED_VERSION
