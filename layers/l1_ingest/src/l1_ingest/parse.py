"""Paso 4 y 5 del núcleo: ZIP en RAM a lotes Arrow con RAW_SCHEMA."""

import io
import zipfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import IO

import pyarrow as pa
import pyarrow.csv as pacsv
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.schema import RAW_SCHEMA

FIRST_LINE_MAX = 80
# Cota de la lectura del header: un CSV sin saltos de línea no se descomprime entero.
HEADER_READ_MAX = 4096
# Una fila de aggTrades ocupa ~83 bytes: 64 MiB dan lotes de ~800k filas, y cada
# lote es un row group. El tamaño varía de un mes a otro.
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
) -> Generator[pa.RecordBatch]:
    """Decodifica el CSV en lotes de unos `block_size` bytes con RAW_SCHEMA.

    Lee `source` de forma incremental: en RAM no vive el CSV, sino el bloque en
    curso más los que el lector de Arrow deja leídos por adelantado en un hilo de fondo.

    Medido con un ZIP sintético de 152 MB (10M filas, ~830 MB de CSV), bloques
    de 64 MiB y sin escritura: RSS máximo de 1.2 GiB, es decir ~1 GiB sobre el
    ZIP y el intérprete. Es la cota del readahead más el lote; el consumidor
    (conform, zstd y hash) suma encima.
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
    try:
        for batch in reader:
            yield batch.cast(RAW_SCHEMA)
    finally:
        reader.close()


@contextmanager
def open_zip_batches(
    data: bytes, block_size: int = BLOCK_SIZE
) -> Iterator[tuple[CheckResult, Generator[pa.RecordBatch]]]:
    """Abre el único .csv de un ZIP en RAM y lo entrega en lotes, sin descomprimirlo entero.

    Devuelve el hallazgo del header y los lotes; en RAM solo viven el ZIP
    comprimido y un bloque. Los lotes se consumen dentro del bloque `with`; al
    salir se cierra el generador (y con él el lector) antes que el miembro del ZIP.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        member = _only_csv(zf)
        with zf.open(member) as head:
            header, check = detect_header(head.readline(HEADER_READ_MAX))
        with zf.open(member) as source:
            batches = iter_batches(source, header, block_size)
            try:
                yield check, batches
            finally:
                batches.close()


def read_zip(data: bytes) -> tuple[pa.Table, CheckResult]:
    """Decodifica un ZIP de aggTrades a una tabla completa (ruta materializada)."""
    with open_zip_batches(data) as (check, batches):
        table = pa.Table.from_batches(batches, schema=RAW_SCHEMA)
    return table, check
