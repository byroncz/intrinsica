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
    tile_name,
)

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"


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

    sizes = {f.kind: f for f in TILE_FILES}
    expected = [
        (
            str(w),
            duration(w),
            str(sizes["price"].per_column * 4 * w),
            str(sizes["volume"].per_column * 4 * w),
            str(sizes["dir"].per_column * w),
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


def test_latest_fields_are_listed_in_the_doc():
    text = _section().split("### Campos de `index.json`")[1]
    for name in LATEST_FIELDS:
        assert f"`{name}`" in text.split("`latest.json` lleva")[1]


def test_file_names_in_the_doc_tree_match_code():
    tree = _section().split("### Disposición")[1].split("### Archivos")[0]
    for name in (
        tile_name("price", 0).replace("0", "<w>"),
        tile_name("volume", 0).replace("0", "<w>"),
        "index.json",
        "latest.json",
    ):
        assert name in tree
    assert "dir-<w>-<theta>" not in tree


def test_dir_file_is_one_per_level_in_the_doc_tree():
    tree = _section().split("### Disposición")[1].split("### Archivos")[0]
    assert tile_name("dir", 0).replace("0", "<w>") in tree


def test_price_scale_doc_matches_code():
    expected = [(asset, str(scale)) for asset, scale in PRICE_SCALE_BY_ASSET.items()]
    assert _table("Escala de precio", r"^\| `(\w+)` \| (\d+) \|") == expected
