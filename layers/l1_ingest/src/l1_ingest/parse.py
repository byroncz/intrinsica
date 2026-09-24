"""Paso 4 y 5 del núcleo: ZIP en RAM a tabla Arrow con RAW_SCHEMA."""

import io
import zipfile

import duckdb
import pyarrow as pa
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.schema import RAW_SCHEMA

FIRST_LINE_MAX = 80

_DUCKDB_TYPES = {
    pa.int64(): "BIGINT",
    pa.string(): "VARCHAR",
    pa.bool_(): "BOOLEAN",
}


def _has_header(first_line: str) -> bool:
    """Sin header, el primer campo (agg_trade_id) es un entero."""
    try:
        int(first_line.split(",", 1)[0])
    except ValueError:
        return True
    return False


def read_zip(data: bytes) -> tuple[pa.Table, CheckResult]:
    """Decodifica un ZIP de aggTrades en RAM; no escribe archivos."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
        if len(members) != 1 or not members[0].lower().endswith(".csv"):
            raise ValueError(f"el ZIP debe traer exactamente un .csv, trae {members}")
        csv_bytes = zf.read(members[0])

    # Solo la primera línea se mira en Python; el resto lo decodifica DuckDB.
    first_line = csv_bytes.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    if not first_line:
        raise ValueError("el CSV está vacío")
    header = _has_header(first_line)

    # Con un objeto en memoria DuckDB necesita fsspec.
    table = duckdb.read_csv(
        io.BytesIO(csv_bytes),
        header=False,
        skiprows=1 if header else 0,
        names=RAW_SCHEMA.names,
        dtype={f.name: _DUCKDB_TYPES[f.type] for f in RAW_SCHEMA},
    ).to_arrow_table()

    check = CheckResult(
        check_type="header_detected",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=1.0 if header else 0.0,
        details={"first_line": first_line[:FIRST_LINE_MAX]},
    )
    return table.cast(RAW_SCHEMA), check
