import json
import re
from pathlib import Path

from viz_tiles.contract import (
    EVENT_BYTES,
    EVENTS_FILE,
    FLAGS,
    INDEX_FIELDS,
    LATEST_FIELDS,
    PRICE_SCALE_BY_ASSET,
    TICK_SECTIONS,
    TICKS_FILE,
    TILES_VERSION,
)

DOC = Path(__file__).parents[3] / "docs" / "data-contracts.md"
TRD = Path(__file__).parents[3] / "docs" / "TRD" / "viz.md"


def _section() -> str:
    return DOC.read_text().split("\n## Tiles de viz")[1].split("\n## ")[0]


def _table(heading: str, pattern: str) -> list[tuple[str, ...]]:
    """Las filas de la tabla bajo `### <heading>` que casan con `pattern`."""
    table = _section().split(f"### {heading}")[1].split("\n### ")[0]
    return re.findall(pattern, table, flags=re.MULTILINE)


def test_ticks_sections_doc_matches_code():
    expected = [(str(n), name) for n, name in enumerate(TICK_SECTIONS, start=1)]
    assert _table("Archivos", r"^\| (\d) \| `(\w+)` \|") == expected


def test_ticks_file_doc_describes_the_encoding():
    text = " ".join(_section().split("### Archivos")[1].split("\n### ")[0].split())
    assert f"`{TICKS_FILE}`" in text
    assert "LEB128" in text and "zigzag" in text
    assert "(d << 1) ^ (d >> 63)" in text
    assert len(TICK_SECTIONS) == 3 and "tres secciones" in text


def test_tiles_version_is_the_one_of_the_doc():
    assert TILES_VERSION == "2.0.0"
    assert f"`tiles_version` es `{TILES_VERSION}`" in _section()


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
    assert index["tiles_version"] == TILES_VERSION
    assert index["ticks_file"] == TICKS_FILE and index["events"] == EVENTS_FILE


def test_latest_fields_are_listed_in_the_doc():
    text = _section().split("### Campos de `index.json`")[1]
    for name in LATEST_FIELDS:
        assert f"`{name}`" in text.split("`latest.json` lleva")[1]


def test_file_names_in_the_doc_tree_match_code():
    tree = _section().split("### Disposición")[1].split("### Página del día")[0]
    assert TICKS_FILE in tree and EVENTS_FILE in tree
    assert "index.json" in tree and "index.html" in tree
    assert "latest.json" in tree
    # Los arreglos por nivel de la 1.x ya no están en la disposición.
    assert not re.search(r"(price|volume|dir|count|confirms|simul)-<w>", tree)
    assert "4 objetos" in " ".join(_section().split())


def test_price_scale_doc_matches_code():
    expected = [(asset, str(scale)) for asset, scale in PRICE_SCALE_BY_ASSET.items()]
    assert _table("Escala de precio", r"^\| `(\w+)` \| (\d+) \|") == expected


def test_event_flags_doc_matches_code():
    expected = [(str(bit), label) for bit, label in FLAGS.items()]
    assert _table("Eventos exactos", r"^\| `(\d+)` \| ([^|]+?) \|") == expected


def test_events_file_doc_matches_code():
    text = _section().split("### Eventos exactos")[1].split("\n### ")[0]
    assert f"`{EVENTS_FILE}`" in text
    assert f"{EVENT_BYTES} bytes por evento" in text
    # 3 secciones int32 y una de banderas de un byte.
    assert EVENT_BYTES == 3 * 4 + 1
    assert "`events_offset`" in _section().split("### Campos de `index.json`")[1]
