"""La cifra de bytes por evento abierto de cada familia (ITSC-407).

La fuente de verdad de A, T y F son los `size_of` que fija el test de `engine.rs`; la de S es la fila
de salida de `ENGINE_SCHEMA`. Las constantes de `summarize.py` y las cifras de la documentación
tienen que coincidir con ellas. F (ITSC-412) cobra un costo fijo por evento abierto, como las
demás, y además un costo por cada retorno que guarda: `Engine.buffered_ticks()` por 4 B.
"""

import re
import types
from pathlib import Path

import pytest
from l3_dc_descriptors import native, summarize
from l3_dc_descriptors.boundaries import BOUNDARY_ROW_BYTES
from l3_dc_descriptors.schema import FAMILY, FIT_FAMILY, PATH_FAMILY, SPECTRUM_FAMILY

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
    assert summarize.SPECTRUM_STATE_BYTES == rust_size("SpectrumRun") == 120
    # Cada retorno guardado es un `f32`.
    assert summarize.SPECTRUM_RETURN_BYTES == rust_size("f32") == 4


def test_the_store_of_s_is_pinned_apart_from_the_figure_it_is_charged():
    # S se cobra por su fila de salida (240 B), no por su almacén: `Partial<EventAcc>` mide 272 B.
    # Las dos cifras quedan fijadas aquí para que el 272 de `engine.rs` no quede huérfano.
    assert rust_size("Partial<EventAcc>") == 272
    assert sum(width for _, width, _ in native.ENGINE_SCHEMA) == 240
    assert summarize.EVENT_BYTES[FAMILY] == 240 + BOUNDARY_ROW_BYTES == 337


def test_each_family_pays_its_state_plus_the_boundary_row():
    assert BOUNDARY_ROW_BYTES == 97
    assert dict(summarize.EVENT_BYTES) == {
        FAMILY: 337,
        FIT_FAMILY: rust_size("Partial<FitAcc>") + BOUNDARY_ROW_BYTES,
        PATH_FAMILY: rust_size("Run") + BOUNDARY_ROW_BYTES,
        SPECTRUM_FAMILY: rust_size("SpectrumRun") + BOUNDARY_ROW_BYTES,
    }
    assert summarize.EVENT_BYTES[FIT_FAMILY] == 769
    assert summarize.EVENT_BYTES[PATH_FAMILY] == 369
    assert summarize.EVENT_BYTES[SPECTRUM_FAMILY] == 217


class FakeEngine:
    """Un motor sin eventos cerrados y, en `spectrum`, con `buffered` retornos en vuelo."""

    def __init__(self, buffered: int = 0):
        self.buffered = buffered

    def needed(self, position):
        return 0

    def buffered_ticks(self):
        return self.buffered


class FakeFeed:
    position = 0
    delivered = 1000
    ahead = 0


@pytest.mark.parametrize("family", [FAMILY, FIT_FAMILY, PATH_FAMILY])
def test_unclosed_charges_the_figure_of_the_family(family):
    events, held = summarize._unclosed(FakeEngine(), [FakeFeed()], family)
    assert events == 1000
    assert held == 1000 * summarize.EVENT_BYTES[family]


@pytest.mark.parametrize("buffered", [0, 1, 250_000, 20_000_000])
def test_unclosed_charges_f_its_fixed_figure_plus_4_bytes_per_buffered_return(buffered):
    events, held = summarize._unclosed(
        FakeEngine(buffered), [FakeFeed()], SPECTRUM_FAMILY
    )
    assert events == 1000
    assert held == 1000 * 217 + 4 * buffered
    # Las demás familias no cobran los retornos aunque el motor los reporte.
    _, other = summarize._unclosed(FakeEngine(buffered), [FakeFeed()], FIT_FAMILY)
    assert other == 1000 * 769


def test_the_budget_bounds_the_returns_of_one_event_at_80_mb():
    # Un evento en el presupuesto (20 M de ticks) guarda 20 M de retornos de 4 B: 80 MB, la mitad
    # de los 160 MB de la decisión 6 de la Épica con 8 B por retorno (el veredicto fijó f32).
    _, held = summarize._unclosed(
        FakeEngine(20_000_000),
        [types.SimpleNamespace(position=0, delivered=1, ahead=0)],
        SPECTRUM_FAMILY,
    )
    assert held == 217 + 80_000_000


def test_unclosed_charges_the_boundaries_ahead_at_97_bytes_in_every_family():
    feed = types.SimpleNamespace(position=0, delivered=1000, ahead=400)
    for family in (FAMILY, FIT_FAMILY, PATH_FAMILY, SPECTRUM_FAMILY):
        _, held = summarize._unclosed(FakeEngine(), [feed], family)
        assert held == 600 * summarize.EVENT_BYTES[family] + 400 * BOUNDARY_ROW_BYTES
