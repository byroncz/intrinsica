import re
from pathlib import Path

import duckdb
from dc_views import DERIVED_VERSION, load_derived
from test_dc_views_derived import EVENTS_SCHEMA, ROWS, SUMMARIES_SCHEMA, _table

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


def _section() -> str:
    section = DOC.read_text(encoding="utf-8").split("## Salida Parquet de L3")[1]
    return section.split("\n## ")[0].split("### Vista `derived`")[1].split("\n### ")[0]


def test_derived_doc_matches_view():
    rows = re.findall(r"^\| `(\w+)` \| ([^|]+?) \|", _section(), flags=re.MULTILINE)
    con = duckdb.connect()
    load_derived(
        con,
        _table(SUMMARIES_SCHEMA, [s for s, _ in ROWS.values()]),
        _table(EVENTS_SCHEMA, [e for _, e in ROWS.values()]),
    )
    described = con.execute("DESCRIBE derived").fetchall()
    assert rows == [(name, dtype) for name, dtype, *_ in described]
    assert len(rows) == 21


def test_derived_doc_names_the_version():
    assert f"(hoy `{DERIVED_VERSION}`)" in _section()
