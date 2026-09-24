import re
from pathlib import Path

from dq.schema import FINDING_SCHEMA

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


def _table_columns() -> list[str]:
    section = DOC.read_text().split("## Lago de hallazgos de calidad de datos")[1]
    section = section.split("\n## ")[0]
    return re.findall(r"^\| `(\w+)` \|", section, flags=re.MULTILINE)


def test_doc_columns_match_schema():
    assert _table_columns() == FINDING_SCHEMA.names
