import re
from pathlib import Path

from l1_ingest.schema import OUTPUT_SCHEMA

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


def _table_columns() -> list[str]:
    section = DOC.read_text().split("## Salida Parquet de L1")[1]
    section = section.split("\n## ")[0]
    return re.findall(r"^\| `(\w+)` \|", section, flags=re.MULTILINE)


def test_doc_columns_match_schema():
    assert _table_columns() == OUTPUT_SCHEMA.names
