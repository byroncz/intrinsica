import json
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
from viz_tiles.contract import (
    DAY_S,
    INDEX_FIELDS,
    LATEST_FIELDS,
    LEVELS,
    PRICE_SCALE_BY_ASSET,
    STATES,
    TILE_FILES,
)

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"
TRD = Path(__file__).parents[3] / "docs" / "TRD" / "viz.md"


def _section() -> str:
    return DOC.read_text().split("\n## Tiles de viz")[1].split("\n## ")[0]


def _table(heading: str, pattern: str) -> list[tuple[str, ...]]:
    """Las filas de la tabla bajo `### <heading>` que casan con `pattern`."""
    table = _section().split(f"### {heading}")[1].split("\n### ")[0]
    return re.findall(pattern, table, flags=re.MULTILINE)


def test_files_doc_matches_code():
    expected = [
        (
            f.template.replace("{w}", "<w>").replace("{theta}", "<theta>"),
            np.dtype(f.dtype).name,
            str(f.per_column),
        )
        for f in TILE_FILES
    ]
    assert _table("Archivos", r"^\| `([^`]+)` \| (\w+) \| (\d+) \|") == expected


def test_levels_doc_matches_code():
    def duration(w: int) -> str:
        return format(Decimal(DAY_S) / w, "f").replace(".", ",")

    files = {f.kind: f for f in TILE_FILES}

    def size(kind: str, w: int) -> str:
        return str(files[kind].per_column * np.dtype(files[kind].dtype).itemsize * w)

    expected = [
        (
            str(w),
            duration(w),
            size("price", w),
            size("volume", w),
            size("dir", w),
        )
        for w in LEVELS
    ]
    pattern = r"^\| (\d+) \| ([\d,]+) \| (\d+) \| (\d+) \| (\d+) \|"
    assert _table("Niveles", pattern) == expected


def test_states_doc_matches_code():
    expected = [(str(code), label) for code, label in STATES.items()]
    assert _table("Estados", r"^\| `(\d+)` \| ([^|]+?) \|") == expected


def test_index_fields_doc_matches_code():
    expected = [(name, kind) for name, kind in INDEX_FIELDS]
    rows = _table("Campos de `index.json`", r"^\| `(\w+)` \| (\w+) \|")
    assert rows == expected


def test_index_fields_trd_matches_code():
    # El índice de ejemplo de TRD-viz §7.2 es JSON válido: sus llaves, en orden, y
    # el tipo de cada valor deben ser los de INDEX_FIELDS.
    text = TRD.read_text().split("### 7.2 ")[1].split("\n### ")[0]
    example = re.search(r"```json\n(.*?)\n```", text, flags=re.DOTALL)
    assert example, "TRD-viz §7.2 ya no trae el ejemplo de index.json"
    index = json.loads(example.group(1))
    json_types = {
        str: "string",
        int: "integer",
        list: "array",
        dict: "object",
    }
    assert [(name, json_types[type(v)]) for name, v in index.items()] == [
        (name, kind) for name, kind in INDEX_FIELDS
    ]


def test_latest_fields_are_listed_in_the_doc():
    text = _section().split("### Campos de `index.json`")[1]
    for name in LATEST_FIELDS:
        assert f"`{name}`" in text.split("`latest.json` lleva")[1]


def test_file_names_in_the_doc_tree_match_code():
    tree = _section().split("### Disposición")[1].split("### Archivos")[0]
    for f in TILE_FILES:
        assert f.template.replace("{w}", "<w>") in tree
    assert "index.json" in tree
    assert "latest.json" in tree


def test_price_scale_doc_matches_code():
    expected = [(asset, str(scale)) for asset, scale in PRICE_SCALE_BY_ASSET.items()]
    assert _table("Escala de precio", r"^\| `(\w+)` \| (\d+) \|") == expected
