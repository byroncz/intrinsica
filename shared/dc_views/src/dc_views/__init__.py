"""Vistas SQL versionadas de L3 sobre `summaries` unido a `events`."""

from importlib.resources import files

import duckdb
import pyarrow as pa

DERIVED_VERSION = "1.0.0"

__all__ = ["DERIVED_VERSION", "load_derived"]


def load_derived(
    con: duckdb.DuckDBPyConnection, summaries: pa.Table, events: pa.Table
) -> str:
    """Registra las tablas, crea la vista `derived` y devuelve `DERIVED_VERSION`."""
    con.register("summaries", summaries)
    con.register("events", events)
    con.execute(files("dc_views").joinpath("derived.sql").read_text(encoding="utf-8"))
    return DERIVED_VERSION
