import re
from pathlib import Path

from l2_dc_events.schema import CARRY_OVER_SCHEMA, EVENTS_SCHEMA

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


def _table_rows(heading: str) -> list[tuple[str, str, str]]:
    """Columnas, tipo y nulabilidad de la tabla bajo `### <heading>` de la sección de L2."""
    section = DOC.read_text().split("## Salida Parquet de L2")[1].split("\n## ")[0]
    table = section.split(f"### {heading}")[1].split("\n### ")[0]
    return re.findall(
        r"^\| `(\w+)` \| ([^|]+?) \| (sí|no) \|", table, flags=re.MULTILINE
    )


def _expected(schema) -> list[tuple[str, str, str]]:
    return [(f.name, str(f.type), "sí" if f.nullable else "no") for f in schema]


def test_events_doc_matches_schema():
    assert _table_rows("Esquema de `events.parquet`") == _expected(EVENTS_SCHEMA)


def test_carry_over_doc_matches_schema():
    assert _table_rows("Esquema de `carry_over.parquet`") == _expected(
        CARRY_OVER_SCHEMA
    )
