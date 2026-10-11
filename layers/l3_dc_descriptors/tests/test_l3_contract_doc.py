import re
from pathlib import Path

from l3_dc_descriptors import SUMMARIES_SCHEMA

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


def test_summaries_doc_matches_schema():
    section = DOC.read_text().split("## Salida Parquet de L3")[1].split("\n## ")[0]
    table = section.split("### Esquema de `summaries.parquet`")[1].split("\n### ")[0]
    rows = re.findall(
        r"^\| `(\w+)` \| ([^|]+?) \| (sí|no) \|", table, flags=re.MULTILINE
    )
    expected = [
        (f.name, str(f.type), "sí" if f.nullable else "no") for f in SUMMARIES_SCHEMA
    ]
    assert rows == expected
