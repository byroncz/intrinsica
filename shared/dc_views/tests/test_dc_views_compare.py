"""`phase_pairs`, `COMPARE_OPS` y la vista `compare` con un fixture calculado a mano."""

from collections import Counter

import duckdb
import pyarrow as pa
import pytest
from dc_views import (
    COMPARE_OPS,
    COMPARE_VERSION,
    load_compare,
    load_derived,
    phase_pairs,
)
from test_dc_views_derived import (
    EVENTS_SCHEMA,
    ROWS,
    SUMMARIES_SCHEMA,
    _table,
)

KEYS = [("theta", pa.decimal128(9, 8)), ("confirm_agg_trade_id", pa.int64())]
F64 = pa.float64()
BAND_MEASURES = [
    *(f"fourier_band_{i}" for i in range(1, 8)),
    *(f"haar_level_{i}" for i in range(1, 8)),
    "energy_loc",
]


def _pairs_schema(measures, kinds=None, unpaired=()):
    fields = list(KEYS)
    for measure in measures:
        kind = (kinds or {}).get(measure, F64)
        fields += [(f"c_{measure}", kind), (f"os_{measure}", kind)]
    fields += [(name, F64) for name in unpaired]
    return pa.schema(fields)


def _rows(schema, values):
    """Una tabla con las claves de A y B y `values` = {columna: (valor de A, valor de B)}."""
    keys = {
        "theta": ["0.01", "0.01"],
        "confirm_agg_trade_id": [1000, 2000],
    }
    rows = []
    for i in range(2):
        rows.append(
            {
                name: keys[name][i] if name in keys else values[name][i]
                for name in schema.names
            }
        )
    return _table(schema, rows)


def _view(con, name="compare"):
    cursor = con.execute(f"SELECT * FROM {name} ORDER BY theta, confirm_agg_trade_id")
    names = [d[0] for d in cursor.description]
    return names, [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


# --- phase_pairs ---------------------------------------------------------------------


def test_phase_pairs_splits_pairs_from_single_columns():
    names = [
        "theta",
        "confirm_agg_trade_id",
        "c_ticks",
        "os_ticks",
        "c_t25",
        "os_only",
        "ev_slope",
        "atomic_sweep",
        "confirm_instant_ticks",
        "year",
    ]
    pairs = phase_pairs(names)
    assert pairs.paired == ("ticks",)
    assert pairs.single == ("c_t25",) + ("os_only",)
    assert tuple(pairs) == (("ticks",), ("c_t25", "os_only"))


def test_phase_pairs_never_crosses_relations():
    # c_x en una relación y os_x en otra no forman pareja: cada llamada ve una sola.
    left, right = phase_pairs(["c_x", "c_y", "os_y"]), phase_pairs(["os_x"])
    assert left.paired == ("y",) and left.single == ("c_x",)
    assert right.paired == () and right.single == ("os_x",)


def test_phase_pairs_empty():
    assert phase_pairs([]) == ((), ())
    assert phase_pairs(["theta", "ev_speed", "tmv"]) == ((), ())


def test_phase_pairs_keeps_measures_with_underscores():
    assert phase_pairs(["c_quantity_buy", "os_quantity_buy"]).paired == (
        "quantity_buy",
    )


# --- COMPARE_OPS ---------------------------------------------------------------------


def test_compare_ops_labels_and_counts():
    assert list(COMPARE_OPS) == ["summaries", "derived", "path", "fit", "spectrum"]
    counts = Counter(op for ops in COMPARE_OPS.values() for op in ops.values())
    assert counts == {"ratio": 15, "diff": 22}
    assert {len(ops) for ops in COMPARE_OPS.values()} == {7, 5, 3, 15}
    measures = [m for ops in COMPARE_OPS.values() for m in ops]
    assert len(measures) == len(set(measures)) == 37  # ninguna se repite entre familias


def test_compare_ops_exact_labels():
    ratio = {
        r: {m for m, op in ops.items() if op == "ratio"}
        for r, ops in COMPARE_OPS.items()
    }
    diff = {
        r: {m for m, op in ops.items() if op == "diff"}
        for r, ops in COMPARE_OPS.items()
    }
    assert ratio["summaries"] == {
        "ticks",
        "duration",
        "quantity_buy",
        "quantity_sell",
        "price_max",
        "price_min",
        "vwap",
    }
    assert ratio["derived"] == {
        "tick_rate",
        "quantity_rate",
        "mean_trade",
        "range",
        "speed",
    }
    assert diff["derived"] == {"flow_imbalance", "vwap_distance"}
    assert ratio["path"] == {"realized_var", "tortuosity", "max_gap"}
    assert diff["path"] == {"sign_autocorr", "idle_fraction"}
    assert diff["fit"] == {"slope", "accel", "jump_share"}
    assert diff["spectrum"] == set(BAND_MEASURES)
    assert diff["summaries"] == ratio["fit"] == ratio["spectrum"] == set()


# --- Los nombres de D ----------------------------------------------------------------


def _derived_connection():
    con = duckdb.connect()
    load_derived(
        con,
        _table(SUMMARIES_SCHEMA, [s for s, _ in ROWS.values()]),
        _table(EVENTS_SCHEMA, [e for _, e in ROWS.values()]),
    )
    return con


def test_derived_names_match_compare_ops():
    described = _derived_connection().execute("DESCRIBE derived").fetchall()
    types = {name: dtype for name, dtype, *_ in described}
    pairs = phase_pairs(types)
    assert set(pairs.paired) == set(COMPARE_OPS["derived"])
    assert len(pairs.paired) == 7
    for measure in pairs.paired:
        assert types[f"c_{measure}"] == types[f"os_{measure}"], measure
    assert pairs.single == ("c_instant_bias",)


# --- Intercambio en D ----------------------------------------------------------------

# Las filas con las dos fases no vacías: con overshoot vacío, intercambiar deja c_* nulas.
BOTH_PHASES = "ABCEGH"
SWAPPED = [
    "ticks",
    "quantity_buy",
    "quantity_sell",
    "price_max",
    "price_min",
    "duration",
    "vwap",
]


def _swapped(summary):
    out = dict(summary)
    for measure in SWAPPED:
        out[f"c_{measure}"], out[f"os_{measure}"] = (
            summary[f"os_{measure}"],
            summary[f"c_{measure}"],
        )
    return out


def _derived_rows(summaries):
    keep = [ROWS[r] for r in BOTH_PHASES]
    con = duckdb.connect()
    load_derived(
        con,
        _table(SUMMARIES_SCHEMA, [summaries(s) for s, _ in keep]),
        _table(EVENTS_SCHEMA, [e for _, e in keep]),
    )
    _, rows = _view(con, "derived")
    return {(r["theta"], r["confirm_agg_trade_id"]): r for r in rows}


def test_swapping_phases_in_summaries_swaps_every_pair_of_d():
    """La c_X de una entrada es la os_X de la otra, y al revés, salvo `speed`.

    Los valores de las dos fases son distintos en cada fila: un `os_X` escrito con la
    columna de la otra fase (p. ej. `os_range` con `c_price_min`) rompe la igualdad.
    `speed` queda fuera porque sus precios salen de `events`, no de `summaries`; lo cubre
    el fixture calculado a mano de `test_dc_views_derived.py`.
    """
    plain = _derived_rows(lambda s: s)
    swapped = _derived_rows(_swapped)
    assert plain.keys() == swapped.keys() and len(plain) == len(BOTH_PHASES)
    measures = [m for m in COMPARE_OPS["derived"] if m != "speed"]
    assert len(measures) == 6
    for key, row in plain.items():
        other = swapped[key]
        for measure in measures:
            assert row[f"c_{measure}"] == pytest.approx(
                other[f"os_{measure}"], rel=1e-12
            ), (
                key,
                measure,
            )
            assert row[f"os_{measure}"] == pytest.approx(
                other[f"c_{measure}"], rel=1e-12
            ), (
                key,
                measure,
            )
    # La entrada es de verdad distinta entre fases: algún valor cambia al intercambiar.
    assert any(
        plain[k][f"c_{m}"] != swapped[k][f"c_{m}"]
        for k in plain
        for m in measures
        if plain[k][f"c_{m}"] is not None
    )


# --- La vista, con valores a mano ------------------------------------------------------

SUMMARY_MEASURES = list(COMPARE_OPS["summaries"])
DERIVED_MEASURES = list(COMPARE_OPS["derived"])
PATH_MEASURES = list(COMPARE_OPS["path"])
FIT_MEASURES = list(COMPARE_OPS["fit"])
PATH_KINDS = {"max_gap": pa.int64()}
KEYS_OF = ["theta", "confirm_agg_trade_id"]

# A y B de `ROWS`; la confirmación y el overshoot de cada una tienen valores distintos.
PATH_SCHEMA = _pairs_schema(
    PATH_MEASURES, PATH_KINDS, unpaired=["c_t25", "c_t50", "c_t75"]
)
PATH_VALUES = {
    "c_realized_var": (2e-6, 4e-6),
    "os_realized_var": (6e-6, 2e-6),
    "c_tortuosity": (2.0, 1.0),
    "os_tortuosity": (3.0, 4.0),
    "c_sign_autocorr": (0.5, -0.5),
    "os_sign_autocorr": (0.25, None),
    "c_max_gap": (0, 1_000_000),  # A: denominador cero
    "os_max_gap": (5, 4_000_000),
    "c_idle_fraction": (0.5, 0.0),
    "os_idle_fraction": (0.75, None),  # B: overshoot sin fracción
    "c_t25": (0.1, 0.1),
    "c_t50": (0.5, 0.5),
    "c_t75": (0.9, 0.9),
}
FIT_SCHEMA = _pairs_schema(FIT_MEASURES, unpaired=["ev_slope"])
FIT_VALUES = {
    "c_slope": (1.0, 2.0),
    "os_slope": (4.0, 1.5),
    "c_accel": (0.5, 0.5),
    "os_accel": (0.0, -0.5),
    "c_jump_share": (0.25, 0.5),
    "os_jump_share": (0.75, None),
    "ev_slope": (9.0, 9.0),
}
SPECTRUM_SCHEMA = _pairs_schema(BAND_MEASURES)
SPECTRUM_VALUES = {
    **{f"c_{m}": (1.0, 2.0) for m in BAND_MEASURES},
    **{f"os_{m}": (3.5, 0.5) for m in BAND_MEASURES},
}


@pytest.fixture
def inputs():
    # ROWS sin G (otro θ): la prueba usa A, B, C y D; las filas C y D no tienen path.
    keep = [ROWS[r] for r in "ABCD"]
    return {
        "summaries": _table(SUMMARIES_SCHEMA, [s for s, _ in keep]),
        "events": _table(EVENTS_SCHEMA, [e for _, e in keep]),
    }


def _family(schema, values, ids=(1000, 2000)):
    # A y B de ROWS son θ 0.01 con ids 1000 y 2000.
    rows = [
        {
            name: ("0.01" if name == "theta" else ids[i])
            if name in KEYS_OF
            else values[name][i]
            for name in schema.names
        }
        for i in range(2)
    ]
    return _table(schema, rows)


def _expected_summaries(row):
    s = ROWS[row][0]
    out = {}
    for measure in SUMMARY_MEASURES:
        c, os = s[f"c_{measure}"], s[f"os_{measure}"]
        c = None if c is None else float(c)
        os = None if os is None else float(os)
        out[measure] = None if c in (None, 0) or os is None else os / c
    return out


def test_compare_with_only_summaries_and_events(inputs):
    con = duckdb.connect()
    assert load_compare(con, **inputs) == COMPARE_VERSION == "1.0.0"
    names, rows = _view(con)
    # Una familia no entregada no aporta columnas: solo S y D.
    assert names[:2] == KEYS_OF
    assert sorted(names[2:]) == sorted(
        f"cmp_{m}" for m in SUMMARY_MEASURES + DERIVED_MEASURES
    )
    assert len(rows) == 4
    types = {n: t for n, t, *_ in con.execute("DESCRIBE compare").fetchall()}
    assert all(types[n] == "DOUBLE" for n in names[2:])
    by_id = {r["confirm_agg_trade_id"]: r for r in rows}

    # Una ratio: A, 80 ticks de overshoot contra 100 de confirmación, y 5 s contra 2 s.
    assert by_id[1000]["cmp_ticks"] == pytest.approx(0.8, rel=1e-12)
    assert by_id[1000]["cmp_duration"] == pytest.approx(2.5, rel=1e-12)
    assert by_id[1000]["cmp_price_max"] == pytest.approx(103 / 101.5, rel=1e-12)
    assert by_id[1000]["cmp_tick_rate"] == pytest.approx(0.32, rel=1e-12)  # 16 / 50
    assert by_id[1000]["cmp_speed"] == pytest.approx(0.8, rel=1e-12)  # 0.004 / 0.005
    # Una diff: B, (3 - 5) / 8 contra (1 - 3) / 4.
    assert by_id[2000]["cmp_flow_imbalance"] == pytest.approx(0.25, rel=1e-12)
    assert by_id[1000]["cmp_flow_imbalance"] == pytest.approx(-0.25, rel=1e-12)
    assert by_id[1000]["cmp_vwap_distance"] == pytest.approx(0.015, rel=1e-12)
    # Denominador cero: C tiene c_duration = 0, así que la razón de duraciones y las que
    # dependen de la tasa de la confirmación son nulas.
    assert by_id[3000]["cmp_duration"] is None
    assert by_id[3000]["cmp_tick_rate"] is None
    assert by_id[3000]["cmp_ticks"] == pytest.approx(1.5, rel=1e-12)
    # Overshoot vacío: D, los os_* nulos dan comparaciones nulas; las cuentas dan 0.
    assert by_id[4000]["cmp_price_max"] is None
    assert by_id[4000]["cmp_vwap"] is None
    assert by_id[4000]["cmp_duration"] is None
    assert by_id[4000]["cmp_flow_imbalance"] is None
    assert by_id[4000]["cmp_range"] is None
    assert by_id[4000]["cmp_ticks"] == 0.0
    assert by_id[4000]["cmp_quantity_buy"] == 0.0
    for row_name, ident in zip("ABCD", (1000, 2000, 3000, 4000), strict=True):
        for measure, want in _expected_summaries(row_name).items():
            have = by_id[ident][f"cmp_{measure}"]
            if want is None:
                assert have is None, (row_name, measure)
            else:
                assert have == pytest.approx(want, rel=1e-12), (row_name, measure)


def test_compare_with_every_family(inputs):
    con = duckdb.connect()
    load_compare(
        con,
        **inputs,
        path=_family(PATH_SCHEMA, PATH_VALUES),
        fit=_family(FIT_SCHEMA, FIT_VALUES),
        spectrum=_family(SPECTRUM_SCHEMA, SPECTRUM_VALUES),
    )
    names, rows = _view(con)
    measures = SUMMARY_MEASURES + DERIVED_MEASURES + PATH_MEASURES + FIT_MEASURES
    measures += BAND_MEASURES
    assert names[:2] == KEYS_OF
    assert sorted(names[2:]) == sorted(f"cmp_{m}" for m in measures)
    assert len(names) - 2 == 37
    # Nunca entran las columnas sin pareja.
    assert not {"cmp_t25", "cmp_t50", "cmp_t75", "cmp_instant_bias"} & set(names)
    by_id = {r["confirm_agg_trade_id"]: r for r in rows}
    a, b, c = by_id[1000], by_id[2000], by_id[3000]
    # path: ratios y diffs, el denominador cero (c_max_gap = 0 en A) y el tipo INT64.
    assert a["cmp_realized_var"] == pytest.approx(3.0, rel=1e-12)
    assert a["cmp_tortuosity"] == pytest.approx(1.5, rel=1e-12)
    assert a["cmp_max_gap"] is None
    assert b["cmp_max_gap"] == pytest.approx(4.0, rel=1e-12)
    assert a["cmp_sign_autocorr"] == pytest.approx(-0.25, rel=1e-12)
    assert a["cmp_idle_fraction"] == pytest.approx(0.25, rel=1e-12)
    assert b["cmp_idle_fraction"] is None  # os_idle_fraction nula
    # fit y spectrum: solo diffs.
    assert a["cmp_slope"] == pytest.approx(3.0, rel=1e-12)
    assert a["cmp_accel"] == pytest.approx(-0.5, rel=1e-12)
    assert b["cmp_jump_share"] is None
    assert a["cmp_fourier_band_1"] == pytest.approx(2.5, rel=1e-12)
    assert b["cmp_haar_level_7"] == pytest.approx(-1.5, rel=1e-12)
    assert a["cmp_energy_loc"] == pytest.approx(2.5, rel=1e-12)
    # LEFT JOIN desde summaries: C no tiene fila en ninguna de las demás familias.
    assert c["cmp_ticks"] is not None
    assert all(
        c[f"cmp_{m}"] is None for m in PATH_MEASURES + FIT_MEASURES + BAND_MEASURES
    )


def test_compare_omitted_family_has_no_columns_even_if_registered_before(inputs):
    con = duckdb.connect()
    load_compare(con, **inputs, fit=_family(FIT_SCHEMA, FIT_VALUES))
    assert "cmp_slope" in _view(con)[0]
    load_compare(
        con, **inputs
    )  # la relación `fit` sigue registrada, pero no se entrega
    assert "cmp_slope" not in _view(con)[0]


def test_compare_creates_derived_with_path(inputs):
    con = duckdb.connect()
    load_compare(con, **inputs, path=_family(PATH_SCHEMA, PATH_VALUES))
    assert con.execute("SELECT count(*) FROM derived").fetchone()[0] == 4
    assert con.execute("SELECT count(net_duration) FROM derived").fetchone()[0] == 2


def test_compare_is_idempotent(inputs):
    con = duckdb.connect()
    load_compare(con, **inputs)
    first = _view(con)
    load_compare(con, **inputs)
    assert _view(con) == first


# --- Errores -------------------------------------------------------------------------


def test_pair_without_label_fails_naming_the_measure(inputs):
    schema = _pairs_schema(FIT_MEASURES + ["curvature"])
    values = {**FIT_VALUES, "c_curvature": (1.0, 1.0), "os_curvature": (2.0, 2.0)}
    con = duckdb.connect()
    with pytest.raises(ValueError, match="curvature") as error:
        load_compare(con, **inputs, fit=_family(schema, values))
    assert "etiqueta" in str(error.value)


def test_label_without_pair_fails_naming_the_measure(inputs):
    schema = _pairs_schema(["slope", "accel"])  # falta jump_share
    con = duckdb.connect()
    with pytest.raises(ValueError, match="jump_share") as error:
        load_compare(con, **inputs, fit=_family(schema, FIT_VALUES))
    assert "pareja" in str(error.value)


def test_single_column_in_place_of_a_pair_fails(inputs):
    schema = pa.schema(
        [
            *KEYS,
            ("c_slope", F64),
            ("os_slope", F64),
            ("c_accel", F64),
            ("c_jump_share", F64),
        ]
    )
    con = duckdb.connect()
    with pytest.raises(ValueError, match="accel"):
        load_compare(con, **inputs, fit=_family(schema, FIT_VALUES))


def test_measure_repeated_across_relations_fails(inputs):
    schema = _pairs_schema(FIT_MEASURES + ["ticks"])
    values = {**FIT_VALUES, "c_ticks": (1.0, 1.0), "os_ticks": (2.0, 2.0)}
    con = duckdb.connect()
    with pytest.raises(ValueError, match="ticks") as error:
        load_compare(con, **inputs, fit=_family(schema, values))
    assert "summaries" in str(error.value) and "fit" in str(error.value)
