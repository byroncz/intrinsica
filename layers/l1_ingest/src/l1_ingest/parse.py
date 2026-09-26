"""Paso 4 y 5 del núcleo: ZIP en RAM a lotes Arrow con RAW_SCHEMA."""

import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

import pyarrow as pa
import pyarrow.csv as pacsv
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.schema import RAW_SCHEMA

FIRST_LINE_MAX = 80
# Unos 70 bytes por fila de aggTrades: 64 MiB dan lotes de cerca de 1M de filas,
# el tamaño del row group de §7.2.
BLOCK_SIZE = 64 * 1024 * 1024


def _has_header(first_line: str) -> bool:
    """Sin header, el primer campo (agg_trade_id) es un entero."""
    try:
        int(first_line.split(",", 1)[0])
    except ValueError:
        return True
    return False


def _only_csv(zf: zipfile.ZipFile) -> str:
    members = zf.namelist()
    if len(members) != 1 or not members[0].lower().endswith(".csv"):
        raise ValueError(f"el ZIP debe traer exactamente un .csv, trae {members}")
    return members[0]


def detect_header(first_line: bytes) -> tuple[bool, CheckResult]:
    """Mira solo la primera línea; el resto lo decodifica pyarrow."""
    line = first_line.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    if not line:
        raise ValueError("el CSV está vacío")
    header = _has_header(line)
    check = CheckResult(
        check_type="header_detected",
        severity=Severity.INFO,
        status=Status.PASS,
        metric_value=1.0 if header else 0.0,
        details={"first_line": line[:FIRST_LINE_MAX]},
    )
    return header, check


def iter_batches(
    source: IO[bytes], header: bool, block_size: int = BLOCK_SIZE
) -> Iterator[pa.RecordBatch]:
    """Decodifica el CSV en lotes de unos `block_size` bytes con RAW_SCHEMA.

    Lee `source` de forma incremental: en RAM vive un bloque, no el CSV.
    """
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(
            block_size=block_size,
            skip_rows=1 if header else 0,
            column_names=RAW_SCHEMA.names,
        ),
        convert_options=pacsv.ConvertOptions(
            column_types=RAW_SCHEMA,
            # Un campo vacío es un error, no un nulo: OUTPUT_SCHEMA no admite nulos.
            null_values=[],
            strings_can_be_null=False,
        ),
    )
    for batch in reader:
        yield batch.cast(RAW_SCHEMA)


@contextmanager
def open_zip_batches(
    data: bytes, block_size: int = BLOCK_SIZE
) -> Iterator[tuple[CheckResult, Iterator[pa.RecordBatch]]]:
    """Abre el único .csv de un ZIP en RAM y lo entrega en lotes, sin descomprimirlo entero.

    Devuelve el hallazgo del header y los lotes; en RAM solo viven el ZIP
    comprimido y un bloque. Los lotes se consumen dentro del bloque `with`.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        member = _only_csv(zf)
        with zf.open(member) as head:
            header, check = detect_header(head.readline())
        with zf.open(member) as source:
            yield check, iter_batches(source, header, block_size)


def read_zip(data: bytes) -> tuple[pa.Table, CheckResult]:
    """Decodifica un ZIP de aggTrades a una tabla completa (ruta materializada)."""
    with open_zip_batches(data) as (check, batches):
        table = pa.Table.from_batches(batches, schema=RAW_SCHEMA)
    return table, check
