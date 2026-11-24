"""Vista `derived` de L3: familia D, M y N sobre `summaries`, `events` y `path`."""

from importlib.resources import files
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from pyutils import resolve_fs

DERIVED_VERSION = "1.1.0"
# Umbral de silencio (µs) de T-09 y T-10: un hueco cuenta si es estrictamente mayor.
# Es el de `l3_dc_descriptors.schema.SILENCE_US`; una prueba en L3 exige que sean
# iguales. `derived.sql` lo repite como literal marcado con `/* SILENCE_US */`
# porque la sonda `l3_derived_check.py` ejecuta el `.sql` sin importar este módulo;
# otra prueba exige que ese literal sea este valor.
SILENCE_US = 10_000_000

__all__ = ["DERIVED_VERSION", "SILENCE_US", "load_derived", "open_relation"]

# Sin `path`, la vista une contra esta tabla vacía y `net_duration` sale nula.
_NO_PATH = pa.table(
    {
        "theta": pa.array([], pa.decimal128(9, 8)),
        "confirm_agg_trade_id": pa.array([], pa.int64()),
        "c_idle_fraction": pa.array([], pa.float64()),
        "os_idle_fraction": pa.array([], pa.float64()),
    }
)

# year y month son las únicas claves de partición que llegan a la vista; los
# segmentos provider=, market=, asset= y theta= se ignoran (theta es la columna
# DECIMAL(9,8) del archivo).
_PARTITIONING = ds.partitioning(
    pa.schema([("year", pa.int32()), ("month", pa.int32())]), flavor="hive"
)


def open_relation(name: str, source: pa.Table | str | Path) -> pa.Table | ds.Dataset:
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
    path: pa.Table | str | Path | None = None,
) -> str:
    """Registra `summaries`, `events` y `path`, crea `derived` y devuelve `DERIVED_VERSION`.

    Cada argumento es una tabla en memoria o una ruta local o `gs://`: un
    archivo `.parquet` o una raíz con particionado hive. Las rutas se registran
    como datasets de PyArrow; DuckDB los lee por lotes, sin credenciales ni
    `httpfs`. `path` solo aporta `net_duration`: sin él, esa columna es nula.
    """
    con.register("summaries", open_relation("summaries", summaries))
    con.register("events", open_relation("events", events))
    con.register("path", _NO_PATH if path is None else open_relation("path", path))
    con.execute(files("dc_views").joinpath("derived.sql").read_text(encoding="utf-8"))
    return DERIVED_VERSION
