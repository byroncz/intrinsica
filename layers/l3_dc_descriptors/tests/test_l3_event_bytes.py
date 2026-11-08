"""La cifra de bytes por evento abierto de cada familia (ITSC-407).

La fuente de verdad son los `size_of` que fija el test de `engine.rs`; las constantes de
`summarize.py` y las cifras de la documentación tienen que coincidir con ellos.
"""

import re
import types
from pathlib import Path

import pytest
from l3_dc_descriptors import summarize
from l3_dc_descriptors.boundaries import BOUNDARY_ROW_BYTES
from l3_dc_descriptors.schema import FAMILY, FIT_FAMILY, PATH_FAMILY

ENGINE_RS = (
    Path(__file__).parents[3] / "shared" / "dc_descriptors" / "src" / "engine.rs"
)


def rust_size(type_name: str) -> int:
    """El valor de `assert_eq!(size_of::<type_name>(), N)` del test de `engine.rs`."""
    pattern = rf"size_of::<{re.escape(type_name)}>\(\),\s*(\d+)\)"
    found = re.findall(pattern, ENGINE_RS.read_text())
    assert len(found) == 1, f"engine.rs no fija el tamaño de {type_name}"
    return int(found[0])


def test_the_constants_are_the_sizes_the_rust_test_pins():
    assert summarize.FIT_STATE_BYTES == rust_size("Partial<FitAcc>") == 672
    assert summarize.PATH_STATE_BYTES == rust_size("Run") == 272


def test_each_family_pays_its_state_plus_the_boundary_row():
    assert BOUNDARY_ROW_BYTES == 97
    assert dict(summarize.EVENT_BYTES) == {
        FAMILY: 337,
        FIT_FAMILY: rust_size("Partial<FitAcc>") + BOUNDARY_ROW_BYTES,
        PATH_FAMILY: rust_size("Run") + BOUNDARY_ROW_BYTES,
    }
    assert summarize.EVENT_BYTES[FIT_FAMILY] == 769
    assert summarize.EVENT_BYTES[PATH_FAMILY] == 369


class FakeEngine:
    def needed(self, position):
        return 0


class FakeFeed:
    position = 0
    delivered = 1000
    ahead = 0


@pytest.mark.parametrize("family", [FAMILY, FIT_FAMILY, PATH_FAMILY])
def test_unclosed_charges_the_figure_of_the_family(family):
    events, held = summarize._unclosed(FakeEngine(), [FakeFeed()], family)
    assert events == 1000
    assert held == 1000 * summarize.EVENT_BYTES[family]


def test_unclosed_charges_the_boundaries_ahead_at_97_bytes_in_every_family():
    feed = types.SimpleNamespace(position=0, delivered=1000, ahead=400)
    for family in (FAMILY, FIT_FAMILY, PATH_FAMILY):
        _, held = summarize._unclosed(FakeEngine(), [feed], family)
        assert held == 600 * summarize.EVENT_BYTES[family] + 400 * BOUNDARY_ROW_BYTES
