"""Los nombres de columna de L3 contra la regla de `dc_views` (vista `compare`, ITSC-434).

`dc_views` descubre las parejas `c_X`/`os_X` de cada familia con `phase_pairs` y las
etiqueta con `COMPARE_OPS`. Estas pruebas rompen el CI si un esquema de L3 gana o pierde
una pareja sin que cambie `COMPARE_OPS`, o si las dos columnas de una pareja difieren
de tipo (la vista haría `os_X - c_X` entre tipos distintos).
"""

import pytest
from dc_views import COMPARE_OPS, SILENCE_US, phase_pairs
from l3_dc_descriptors import schema
from l3_dc_descriptors.schema import (
    FIT_SCHEMA,
    PATH_SCHEMA,
    SPECTRUM_SCHEMA,
    SUMMARIES_SCHEMA,
)

FAMILIES = {
    "summaries": (SUMMARIES_SCHEMA, ()),
    "path": (PATH_SCHEMA, ("c_t25", "c_t50", "c_t75")),
    "fit": (FIT_SCHEMA, ()),
    "spectrum": (SPECTRUM_SCHEMA, ()),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_pairs_are_the_labelled_measures(family):
    arrow, _ = FAMILIES[family]
    assert set(phase_pairs(arrow.names).paired) == set(COMPARE_OPS[family])


@pytest.mark.parametrize("family", FAMILIES)
def test_both_columns_of_a_pair_have_the_same_type(family):
    arrow, _ = FAMILIES[family]
    for measure in phase_pairs(arrow.names).paired:
        c, os = arrow.field(f"c_{measure}"), arrow.field(f"os_{measure}")
        assert c.type == os.type, measure


@pytest.mark.parametrize("family", FAMILIES)
def test_columns_without_a_pair(family):
    arrow, single = FAMILIES[family]
    # `summaries` y `fit` traen solo `ev_*` y `confirm_instant_*` fuera de las parejas.
    assert set(phase_pairs(arrow.names).single) == set(single)


def test_pairs_cover_the_37_measures_once():
    measures = [m for f in FAMILIES for m in phase_pairs(FAMILIES[f][0].names).paired]
    measures += list(
        COMPARE_OPS["derived"]
    )  # la familia D sale de la vista, no de un archivo
    assert len(measures) == len(set(measures)) == 37


def test_dc_views_silence_us_is_the_one_of_l3():
    assert SILENCE_US == schema.SILENCE_US == 10_000_000
