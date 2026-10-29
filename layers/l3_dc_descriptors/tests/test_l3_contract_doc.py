import re
from pathlib import Path

import pyarrow as pa
from l3_dc_descriptors import SUMMARIES_SCHEMA
from l3_dc_descriptors.schema import (
    PATH_SCHEMA,
    PATH_STATE_VERSION,
    SILENCE_US,
)

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


def _l3_section() -> str:
    return DOC.read_text().split("## Salida Parquet de L3")[1].split("\n## ")[0]


def _table(title: str) -> str:
    """El texto de la subsección `### <title>` de la salida de L3."""
    return _l3_section().split(f"### {title}")[1].split("\n### ")[0]


def _doc_rows(table: str) -> list[tuple[str, str, str]]:
    return re.findall(
        r"^\| `(\w+)` \| ([^|]+?) \| (sí|no) \|", table, flags=re.MULTILINE
    )


def _schema_rows(schema: pa.Schema) -> list[tuple[str, str, str]]:
    return [(f.name, str(f.type), "sí" if f.nullable else "no") for f in schema]


def test_summaries_doc_matches_schema():
    table = _table("Esquema de `summaries.parquet`")
    assert _doc_rows(table) == _schema_rows(SUMMARIES_SCHEMA)


def test_path_doc_matches_schema():
    table = _table("Esquema de `path.parquet`")
    assert _doc_rows(table) == _schema_rows(PATH_SCHEMA)


def test_path_doc_names_the_thirteen_registers_in_order():
    table = _table("Esquema de `path.parquet`")
    registers = re.findall(
        r"^\| `\w+` \| [^|]+ \| (?:sí|no) \| (T-\d\d)\.", table, re.MULTILINE
    )
    assert registers == [f"T-{i:02d}" for i in range(1, 14)]


def test_path_doc_metadata_matches_the_code():
    metadata = _table("Metadatos y lectura de `path.parquet`")
    assert f'`state_version = "{PATH_STATE_VERSION}"`' in metadata
    assert PATH_STATE_VERSION == "1.0.0"
    assert f'`silence_us = "{SILENCE_US}"`' in metadata
    assert SILENCE_US == 10_000_000
