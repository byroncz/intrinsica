"""Paso 4 y 5 del núcleo: ZIP en RAM a tabla Arrow con RAW_SCHEMA."""

import io
import zipfile
from collections.abc import Iterator

import duckdb
import pyarrow as pa
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.schema import RAW_SCHEMA

FIRST_LINE_MAX = 80
BATCH_ROWS = 1_000_000

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


def extract_csv(data: bytes) -> bytes:
    """Extrae el único .csv de un ZIP en RAM; el llamador suelta el ZIP después."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
        if len(members) != 1 or not members[0].lower().endswith(".csv"):
            raise ValueError(f"el ZIP debe traer exactamente un .csv, trae {members}")
        return zf.read(members[0])


def detect_header(csv_bytes: bytes) -> tuple[bool, CheckResult]:
    """Mira solo la primera línea; el resto lo decodifica DuckDB."""
    first_line = csv_bytes.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    if not first_line:
        raise ValueError("el CSV está vacío")
    header = _has_header(first_line)
    check = CheckResult(
        check_type="header_detected",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=1.0 if header else 0.0,
        details={"first_line": first_line[:FIRST_LINE_MAX]},
    )
    return header, check


def iter_batches(
    csv_bytes: bytes, header: bool, batch_rows: int = BATCH_ROWS
) -> Iterator[pa.RecordBatch]:
    """Decodifica el CSV en lotes de a lo más `batch_rows` filas con RAW_SCHEMA.

    Nunca materializa la tabla completa: en RAM viven el CSV y un lote.
    """
    # Con un objeto en memoria DuckDB necesita fsspec.
    relation = duckdb.read_csv(
        io.BytesIO(csv_bytes),
        header=False,
        skiprows=1 if header else 0,
        names=RAW_SCHEMA.names,
        dtype={f.name: _DUCKDB_TYPES[f.type] for f in RAW_SCHEMA},
    )
    for batch in relation.to_arrow_reader(batch_rows):
        yield batch.cast(RAW_SCHEMA)


def read_zip(data: bytes) -> tuple[pa.Table, CheckResult]:
    """Decodifica un ZIP de aggTrades a una tabla completa (ruta materializada)."""
    csv_bytes = extract_csv(data)
    header, check = detect_header(csv_bytes)
    table = pa.Table.from_batches(iter_batches(csv_bytes, header), schema=RAW_SCHEMA)
    return table, check
