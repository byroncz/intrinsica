"""La familia A (`fit.parquet`, TRD-L3 §7.2.2): un fixture calculado a mano, el oráculo de Python
puro y el motor nativo sobre el lago de pruebas (ITSC-399, ADR-L3-18).

Tres cosas se sostienen mutuamente: los 9 valores de cada fila de un fixture chico, escritos a
mano en esta prueba con sus operaciones a la vista; el oráculo `fitref` (Python puro, sin el
motor), que debe dar esos mismos valores y los del lago; y el motor, que debe dar los del
oráculo con tolerancia relativa de 1e-12 y los nulos exactos. Los row groups de L1 son chicos
para que los eventos crucen row groups y tramos.
"""

import math
from decimal import Decimal
from fractions import Fraction as F

import fitref
import pyarrow.parquet as pq
import pytest
from dc_frames import family_path, l2_path
from frames_lake import theta_text, write_l2
from l3_dc_descriptors.schema import (
    FIT_SCHEMA,
    FIT_STATE_VERSION,
    SUMMARIES_SCHEMA,
)
from l3_dc_descriptors.summarize import DEFAULT_KEY
from l3_lake import M1, Tick, build_lake, fixture_ticks, month_cuts, write_ticks
from native_driver import fit_native, path_native, summarize_native
from pyutils import content_hash

ROW_GROUP = 50
RTOL = 1e-12
# Sobre el lago, donde hay eventos de 2 o 3 ticks apiñados al final de la fase, los momentos crudos
# en coordenadas normalizadas pierden dígitos al cerrar (ADR-L3-18, «Precisión»): el peor de los
# 7 785 valores del lago difiere 3,5e-9 del oráculo exacto. 1e-7 deja margen sin esconder un
# error de fórmula, que daría diferencias de orden 1.
RTOL_LAKE = 1e-7
# Un valor esperado de 0 (una aceleración nula de verdad) admite este piso absoluto: el motor
# cuantiza a 2⁻⁸⁰ y la división final deja ruido de ≈ 1e-16.
ATOL = 1e-12
NAMES = fitref.COLUMNS


# ---------------------------------------------------------------------------------------
# Lo común: leer lo que el motor escribió y compararlo columna a columna.


def fit_file(l3, text, month=M1) -> str:
    return family_path(str(l3), DEFAULT_KEY, text, month, "fit")


def rows_of(path: str) -> dict[tuple[Decimal, int], dict]:
    table = pq.read_table(path, schema=FIT_SCHEMA)
    return {(r["theta"], r["confirm_agg_trade_id"]): r for r in table.to_pylist()}


def assert_value(got, wanted, name: str, where, rtol: float = RTOL) -> None:
    """Un valor de la fila: nulidad exacta y DOUBLE a `rtol` relativo (con piso para el 0)."""
    if wanted is None:
        assert got is None, (where, name, got)
        return
    assert got is not None, (where, name)
    assert math.isfinite(got), (where, name, got)  # nunca ±inf ni NaN
    assert math.isclose(got, wanted, rel_tol=rtol, abs_tol=ATOL), (
        where,
        name,
        got,
        wanted,
    )


def assert_rows(got: dict, wanted: dict, rtol: float = RTOL) -> int:
    """`got` (filas del archivo) contra `wanted` (`(θ, id) → 9 valores`): mismas claves."""
    assert got.keys() == wanted.keys()
    for key, row in got.items():
        for name, value in zip(NAMES, wanted[key], strict=True):
            assert_value(row[name], value, name, key, rtol)
    return len(got)


def distinct_later_times(ticks, t0: int) -> int:
    """Cuántos `t` distintos mayores que `t0` tienen los ticks `(precio, t, id)`."""
    return len({t for _, t, _ in ticks if t > t0})


def check_ev_jump_share_laws(key, row, s, e, ticks) -> None:
    """Las cuatro leyes de `ev_jump_share` (TRD-L3 §7.2.2) de una fila con el valor no nulo.

    `ticks` son los `(precio, t, id)` de `(R, E]`. `p_prev` es el precio del tick de mayor
    `(t, id)` con `t < extreme_time`, o `p_R` si no hay.
    """
    ev, theta = row["ev_jump_share"], float(key[0])
    p_r, p_e = float(e["reference_price"]), float(e["extreme_price"])
    if s["os_ticks"] == 0:  # (1) la fase `ev` es la fase `c`
        assert ev == pytest.approx(row["c_jump_share"], abs=1e-9, rel=0), (key, ev)
    elif s["os_duration"] > 0:  # (2)
        assert -1e-9 <= ev <= 1 + theta + 1e-9, (key, ev)
    elif e["extreme_time"] == e["confirm_time"]:  # (3)
        assert 0 < ev <= 1 + theta + 1e-9, (key, ev)
    else:  # (4) `p_prev` es del instante de confirmación
        span = p_e - p_r
        low = (p_e - float(s["confirm_instant_price_best"])) / span
        high = (p_e - float(s["confirm_instant_price_worst"])) / span
        assert low - 1e-9 <= ev <= high + 1e-9, (key, ev, low, high)
        earlier = [(t, i, p) for p, t, i in ticks if t < e["extreme_time"]]
        p_prev = float(max(earlier)[2]) / 1e8 if earlier else p_r  # ticks en 1e-8
        worst, best = (
            float(s["confirm_instant_price_worst"]),
            float(s["confirm_instant_price_best"]),
        )
        assert min(worst, best) <= p_prev <= max(worst, best), (key, p_prev)
        assert ev == pytest.approx((p_e - p_prev) / span, rel=1e-9, abs=1e-12), key


def check_invariants(path: str, summaries: str, events: str, series) -> int:
    """Las leyes de las `*_jump_share` y la regla de nulos de TRD-L3 §7.2.2 de un
    `fit.parquet`, leyendo duraciones de `summaries.parquet` y anclas de `events.parquet`
    (L2) y los ticks de la lista: ninguna usa el motor ni el oráculo. Devuelve las filas
    revisadas. Las leyes son estrictas, sin tolerancia de frecuencia.
    """
    summary = {
        (r["theta"], r["confirm_agg_trade_id"]): r
        for r in pq.read_table(summaries, schema=SUMMARIES_SCHEMA).to_pylist()
    }
    anchors = {r["confirm_agg_trade_id"]: r for r in pq.read_table(events).to_pylist()}
    got = rows_of(path)
    assert got.keys() == summary.keys()
    for key, row in got.items():
        s, e = summary[key], anchors[key[1]]
        for name in NAMES:
            if row[name] is not None:
                assert math.isfinite(row[name]), (key, name)  # nunca ±inf ni NaN
        has_os = s["os_ticks"] > 0
        # Primer y último `t` de cada fase: C es el último tick de la confirmación y E el último
        # del overshoot (su extremo).
        c_first = e["confirm_time"] - s["c_duration"]
        ev_last = e["extreme_time"] if has_os else e["confirm_time"]
        c_slope_null = s["c_duration"] == 0
        os_slope_null = not has_os or s["os_duration"] == 0
        ev_slope_null = c_first == ev_last
        # La aceleración es nula, además, si t_fin = t_0 o p_fin = p_0, o si los ticks con
        # t > t_0 comparten el mismo t (cuenta sobre la lista de ticks, no el motor).
        c_ticks = series.between(e["reference_agg_trade_id"], e["confirm_agg_trade_id"])
        o_ticks = series.between(e["confirm_agg_trade_id"], e["extreme_agg_trade_id"])
        c_accel_null = (
            c_slope_null
            or e["confirm_time"] == e["reference_time"]
            or e["confirm_price"] == e["reference_price"]
            or distinct_later_times(c_ticks, e["reference_time"]) < 2
        )
        os_accel_null = (
            os_slope_null
            or e["extreme_time"] == e["confirm_time"]
            or e["extreme_price"] == e["confirm_price"]
            or distinct_later_times(o_ticks, e["confirm_time"]) < 2
        )
        ev_accel_null = (
            ev_slope_null
            or ev_last == e["reference_time"]
            or e["extreme_price"] == e["reference_price"]
            or distinct_later_times(c_ticks + o_ticks, e["reference_time"]) < 2
        )
        wanted_null = {
            "c_slope": c_slope_null,
            "os_slope": os_slope_null,
            "ev_slope": ev_slope_null,
            "c_accel": c_accel_null,
            "os_accel": os_accel_null,
            "ev_accel": ev_accel_null,
            "c_jump_share": e["confirm_price"] == e["reference_price"],
            "os_jump_share": not has_os or e["extreme_price"] == e["confirm_price"],
            "ev_jump_share": e["extreme_price"] == e["reference_price"],
        }
        assert {n: row[n] is None for n in NAMES} == wanted_null, key
        # Leyes de TRD-L3 §7.2.2, todas estrictas. `c`: ningún precio anterior a `confirm_time`
        # cruza el umbral, así que `0 < c_jump_share ≤ 1`. `os`: el extremo es el máximo o el
        # mínimo del overshoot, así que `os_jump_share ≥ 0`.
        assert 0 < row["c_jump_share"] <= 1, (key, row["c_jump_share"])
        if row["os_jump_share"] is not None:
            assert row["os_jump_share"] >= 0, (key, row["os_jump_share"])
        # `ev`: cuatro leyes por caso (decisión del arquitecto del 2026-11-06). Se deciden con
        # `summaries` y `events`, como la sonda `l3_fit_check.py`, y el caso 4 además se
        # comprueba con el `p_prev` que dan los ticks.
        if row["ev_jump_share"] is not None:
            check_ev_jump_share_laws(key, row, s, e, c_ticks + o_ticks)
    return len(got)


def assert_skeleton_and_footer(path: str, summaries: str) -> None:
    """Mismas filas, orden y row groups que `summaries.parquet`; unión 1:1 por clave; esquema
    del contrato y metadatos del pie (`state_version` y nada más).
    """
    fit_, summary_file = pq.ParquetFile(path), pq.ParquetFile(summaries)
    sizes = [
        summary_file.metadata.row_group(i).num_rows
        for i in range(summary_file.num_row_groups)
    ]
    assert [
        fit_.metadata.row_group(i).num_rows for i in range(fit_.num_row_groups)
    ] == sizes
    keys = ["theta", "confirm_agg_trade_id"]
    mine, theirs = fit_.read(columns=keys), summary_file.read(columns=keys)
    assert mine.equals(theirs)
    assert len(set(map(tuple, zip(*(c.to_pylist() for c in mine.columns))))) == len(
        mine
    )
    assert fit_.schema_arrow.remove_metadata().equals(FIT_SCHEMA)
    assert [f.name for f in FIT_SCHEMA][2:] == list(NAMES)
    meta = {
        k: v for k, v in pq.read_schema(path).metadata.items() if k != b"ARROW:schema"
    }
    assert meta == {b"state_version": b"1.0.0"}
    assert FIT_STATE_VERSION == "1.0.0"


# ---------------------------------------------------------------------------------------
# El fixture a mano: diez eventos de un solo θ y un mes. Precios en USD, tiempos en µs.
#
# Cada bloque son los ticks de un evento: el primero es R, `c` es la posición del tick C y
# `e` la del tick E (la misma si no hay overshoot). Los bloques van uno detrás de otro, con
# ids consecutivos y un salto de 10 s entre ellos. Los row groups de L1 son de 4 ticks y los
# de L2 de 2 eventos. `confirm` es el `confirm_price` de L2 cuando no es el del tick C.

S = 1_000_000
BASE = 1_501_545_600 * S  # 2017-08-01 00:00 UTC
THETA = 1_000_000  # 0,01 × 10⁸
TEXT = theta_text(THETA)

# (nombre, dirección, ticks (µs desde R, precio), posición de C, posición de E, confirm_price)
BLOCKS = [
    # Alza con overshoot: confirmación lineal (y = τ) y overshoot de dos ticks.
    (
        "upturn",
        +1,
        [
            (0, "1000"),
            (100, "1010"),
            (200, "1020"),
            (400, "1040"),
            (500, "1050"),
            (800, "1070"),
        ],
        3,
        5,
        None,
    ),
    # Baja con overshoot vacío: C es el último tick y E = C.
    (
        "downturn_empty",
        -1,
        [(0, "2000"), (10, "1990"), (20, "1980"), (30, "1960")],
        3,
        3,
        None,
    ),
    # Fases de un tick: el único tick de la confirmación es C y el de overshoot es E.
    ("one_tick_phases", +1, [(0, "100"), (10, "105"), (20, "108")], 1, 2, None),
    # t_fin = t_0 en la confirmación: R y sus ticks, en el mismo instante.
    (
        "confirmation_in_one_instant",
        +1,
        [(0, "100"), (0, "103"), (0, "105"), (15, "110")],
        2,
        3,
        None,
    ),
    # Todos los ticks de la confirmación en el mismo t (el de C), distinto del de R.
    ("same_time", +1, [(0, "100"), (10, "102"), (10, "104")], 2, 2, None),
    # Overshoot con ticks cuyo extremo vale lo mismo que la confirmación: p_E = p_C.
    (
        "overshoot_without_gain",
        +1,
        [(0, "1000"), (200, "1020"), (400, "1040"), (500, "1030"), (600, "1040")],
        2,
        4,
        None,
    ),
    # Confirmación parabólica y = ¾τ² + ¼τ sobre τ = k/4 con p = 1 000 + 1 600·y.
    (
        "parabola",
        +1,
        [(0, "1000"), (100, "1175"), (200, "1500"), (300, "1975"), (400, "2600")],
        4,
        4,
        None,
    ),
    # Trayectoria lineal de 14 ticks (p = 1 000 + 10·i en t = 100·i): cruza cuatro row groups.
    (
        "linear",
        +1,
        [(100 * i, str(1000 + 10 * i)) for i in range(15)],
        12,
        14,
        None,
    ),
    # Los dos primeros eventos de la sonda del puente (ITSC-398), con los mismos ticks.
    (
        "bridge_up",
        +1,
        [(0, "100"), (2, "102"), (5, "104"), (8, "109"), (10, "110")],
        4,
        4,
        None,
    ),
    # Baja con un empate en C: el tick C vale 191 y L2 dice 190.
    (
        "bridge_down_tie",
        -1,
        [(0, "200"), (10, "198"), (10, "195"), (30, "192"), (40, "190"), (40, "191")],
        5,
        5,
        "190",
    ),
]


def f(x: F | float) -> float:
    return float(x)


def hand_values() -> dict[str, tuple]:
    """Los 9 valores de cada evento, calculados a mano, por nombre de bloque:
    `(c_slope, os_slope, ev_slope, c_accel, os_accel, ev_accel, c_jump, os_jump, ev_jump)`.

    Las pendientes son de `p / p_R` contra `t` en segundos: `b · (p_fin − p_0) · 10⁶ /
    (p_R · Δt)` con `b` la pendiente de `y` sobre `τ`. Un `None` es un nulo de la regla.
    """
    return {
        # c: y = τ exacta (τ = ¼, ½, 1): b = 1, pendiente 40·10⁶ / (1 000 · 400) = 100.
        #    p_prev = 1 020: salto (1 040 − 1 020)/40. a = 0.
        # os: τ = ¼, 1 con y = ⅓, 1: b = (1 − ⅓)/(¾) = 8/9; pendiente (8/9)·30·10⁶/(1 000·400).
        #     La parábola por esos dos puntos tiene a = −4/9. p_prev = 1 050: (1 070 − 1 050)/30.
        # ev: mínimos cuadrados de p sobre t (100, 200, 400, 500, 800): Σ(t−t̄)(p−p̄) = 26 000,
        #     Σ(t−t̄)² = 300 000 → 0,08666… por µs → ·10⁶/1 000. a = −2 176 / 7 917 (momentos de
        #     τ = k/8, k = 1, 2, 4, 5, 8; y = m/70, m = 10, 20, 40, 50, 70). p_prev = 1 050.
        "upturn": (
            100.0,
            f(F(8, 9) * 30 * S / (1000 * 400)),
            f(F(26_000, 300_000) * S / 1000),
            0.0,
            f(F(-4, 9)),
            f(F(-2_176, 7_917)),
            0.5,
            f(F(20, 30)),
            f(F(20, 70)),
        ),
        # Confirmación = evento (E = C): p de 1 990, 1 980, 1 960 en t = 10, 20, 30 (R en 0):
        # Σ(t−t̄)(p−p̄) = −300, Σ(t−t̄)² = 200 → −1,5 por µs → ·10⁶/2 000 = −750.
        # y = ¼, ½, 1 en τ = ⅓, ⅔, 1 → a = 81/152. p_prev = 1 980: (1 960 − 1 980)/(−40) = ½.
        "downturn_empty": (
            -750.0,
            None,
            -750.0,
            f(F(81, 152)),
            None,
            f(F(81, 152)),
            0.5,
            None,
            0.5,
        ),
        # c y os de un tick: sin pendiente ni aceleración y sin tick anterior: salto 1.
        # ev: dos ticks (10; 105) y (20; 108) desde R = (0; 100): pendiente 0,3 por µs →
        # ·10⁶/100 = 3 000; τ = ½, 1 con y = 5/8, 1 → a = −½; p_prev = 105: (108 − 105)/8.
        "one_tick_phases": (None, None, 3000.0, None, None, -0.5, 1.0, 1.0, 0.375),
        # c: R y los ticks en t = 0 = t_C → sin coordenadas ni tick anterior: salto 1.
        # os: un tick: salto 1. ev: t = 0, 0, 15 con p = 103, 105, 110: pendiente 60/150 = 0,4
        # por µs → 4 000; un solo τ > 0 distinto → sin aceleración; p_prev = 105 (mayor id con
        # t < 15): (110 − 105)/10.
        "confirmation_in_one_instant": (
            None,
            None,
            4000.0,
            None,
            None,
            None,
            1.0,
            1.0,
            0.5,
        ),
        # Dos ticks en t = 10 = t_C: ningún tick es anterior a t_C → salto 1; sin pendiente.
        "same_time": (None, None, None, None, None, None, 1.0, None, 1.0),
        # c: τ = ½, 1 con y = ½, 1 (lineal): pendiente 100, a = 0, p_prev = 1 020 → ½.
        # os (p_E = p_C): ticks (500; 1 030) y (600; 1 040): 0,1 por µs → 100; sin a ni salto.
        # ev: p = 1 020, 1 040, 1 030, 1 040 en t = 200, 400, 500, 600: Σ(t−t̄)(p−p̄) = 3 750,
        #     Σ(t−t̄)² = 87 500 → 300/7. a = −1 791/1 766 (τ = k/6, k = 2, 4, 5, 6; y = m/4,
        #     m = 2, 4, 3, 4). p_prev = 1 030: (1 040 − 1 030)/40 = ¼.
        "overshoot_without_gain": (
            100.0,
            100.0,
            f(F(300, 7)),
            0.0,
            None,
            f(F(-1_791, 1_766)),
            0.5,
            None,
            0.25,
        ),
        # y = ¾τ² + ¼τ: a = ¾ exacto. b de mínimos cuadrados con intercepto = 95/80 →
        # pendiente (19/16) · 1 600 · 10⁶ / (1 000 · 400) = 4 750. p_prev = 1 975: 625/1 600.
        "parabola": (
            4750.0,
            None,
            4750.0,
            0.75,
            None,
            0.75,
            f(F(625, 1_600)),
            None,
            f(F(625, 1_600)),
        ),
        # Todo sobre una recta de 0,1 por µs con p_R = 1 000: pendiente 100 en las tres fases
        # y aceleración 0. Saltos: c, p_prev = 1 110 (tick 11): (1 120 − 1 110)/120 = 1/12;
        # os, p_prev = 1 130: (1 140 − 1 130)/20 = ½; ev: (1 140 − 1 130)/140 = 1/14.
        "linear": (
            100.0,
            100.0,
            100.0,
            0.0,
            0.0,
            0.0,
            f(F(1, 12)),
            0.5,
            f(F(1, 14)),
        ),
        # Evento 1 de la sonda: a = 1080/5759 y salto 1/10 (la sonda los calculó a mano). La
        # pendiente: t = 2, 5, 8, 10 con p = 102, 104, 109, 110: Σ(t−t̄)(p−p̄) = 39,75,
        # Σ(t−t̄)² = 36,75 → (53/49) por µs → ·10⁶/100 = 530 000/49.
        "bridge_up": (
            f(F(530_000, 49)),
            None,
            f(F(530_000, 49)),
            f(F(1_080, 5_759)),
            None,
            f(F(1_080, 5_759)),
            0.1,
            None,
            0.1,
        ),
        # Evento 2 de la sonda (empate en C): a = −22/39 y salto 1/5. Pendiente: t = 10, 10,
        # 30, 40, 40 con p = 198, 195, 192, 190, 191: Σ(t−t̄)(p−p̄) = −186, Σ(t−t̄)² = 920 →
        # −186/920 por µs → ·10⁶/200 = −23 250/23.
        "bridge_down_tie": (
            f(F(-23_250, 23)),
            None,
            f(F(-23_250, 23)),
            f(F(-22, 39)),
            None,
            f(F(-22, 39)),
            0.2,
            None,
            0.2,
        ),
    }


def hand_layout():
    """Los ticks `(id, tiempo, precio, lado)` y los eventos de L2 del fixture, y el id del
    tick C de cada bloque.
    """
    ticks, rows, confirm_ids = [], [], {}
    next_id, offset = 1, 0
    for name, direction, block, c, e, confirm_price in BLOCKS:
        ids = [next_id + k for k in range(len(block))]
        for k, (dt, price) in enumerate(block):
            ticks.append((ids[k], BASE + offset + dt, price, +1))
        row = {"direction": direction}
        for label, k, price in (
            ("reference", 0, block[0][1]),
            ("confirm", c, confirm_price or block[c][1]),
            ("extreme", e, block[e][1] if e != c else (confirm_price or block[c][1])),
        ):
            row[f"{label}_price"] = Decimal(price)
            row[f"{label}_time"] = BASE + offset + block[k][0]
            row[f"{label}_agg_trade_id"] = ids[k]
        rows.append(row)
        confirm_ids[name] = ids[c]
        next_id += len(block)
        offset += 10 * S
    return ticks, rows, confirm_ids


TICKS, EVENT_ROWS, CONFIRM_IDS = hand_layout()


def write_hand_lake(base):
    l1, l2, l3 = base / "l1", base / "l2", base / "l3"
    write_ticks(
        l1,
        [Tick(i, t, Decimal(p), Decimal(1), side < 0) for i, t, p, side in TICKS],
        M1,
        4,
    )
    write_l2(l2, THETA, EVENT_ROWS, 2, M1)
    return l1, l2, l3


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    base = tmp_path_factory.mktemp("hand")
    l1, l2, l3 = write_hand_lake(base)
    results = fit_native(l1, l2, l3, M1, [TEXT])
    return l1, l2, l3, results[Decimal(TEXT)]


def hand_wanted() -> dict[tuple[Decimal, int], tuple]:
    theta = Decimal(TEXT)
    values = hand_values()
    return {(theta, CONFIRM_IDS[name]): values[name] for name in values}


def test_the_hand_fixture_has_every_case(hand):
    """Los diez eventos cubren los casos que la prueba promete."""
    _, l2, l3, result = hand
    summary = pq.read_table(
        fit_file(l3, TEXT).replace("fit.parquet", "summaries.parquet"),
        schema=SUMMARIES_SCHEMA,
    ).to_pylist()
    events = pq.read_table(l2_path(str(l2), DEFAULT_KEY, TEXT, M1)).to_pylist()
    assert [r["direction"] for r in events].count(1) >= 1
    assert [r["direction"] for r in events].count(
        -1
    ) >= 2  # un downturn con y sin empate
    names = [b[0] for b in BLOCKS]
    by_name = dict(zip(names, summary, strict=True))
    assert by_name["downturn_empty"]["os_ticks"] == 0  # overshoot vacío
    assert by_name["one_tick_phases"]["os_ticks"] == 1  # overshoot de un tick
    assert by_name["one_tick_phases"]["c_ticks"] == 1
    assert (
        by_name["same_time"]["c_duration"] == 0 and by_name["same_time"]["c_ticks"] == 2
    )
    assert by_name["confirmation_in_one_instant"]["c_duration"] == 0
    # El empate en C: dos ticks en el instante de confirmación.
    assert by_name["bridge_down_tie"]["confirm_instant_ticks"] == 2
    # Un evento por varios row groups de L1 de 4 ticks: el lineal cruza al menos cuatro.
    group = {t[0]: i // 4 for i, t in enumerate(TICKS)}
    spans = [
        group[r["extreme_agg_trade_id"]] - group[r["reference_agg_trade_id"]] + 1
        for r in events
    ]
    assert max(spans) >= 4 and min(spans) >= 1
    assert result.rows == len(BLOCKS) == len(hand_values())


def test_the_engine_gives_the_hand_calculated_rows(hand):
    _, _, _, result = hand
    assert assert_rows(rows_of(result.path), hand_wanted()) == len(BLOCKS)


def test_the_oracle_gives_the_hand_calculated_rows(hand):
    """El oráculo, de tres maneras (listas de ticks y anclas, la serie filtrada por id y las
    tramas del lector), da los valores escritos a mano.
    """
    l1, l2, _, _ = hand
    wanted = hand_wanted()
    theta = Decimal(TEXT)
    series = fitref.Series(
        [Tick(i, t, Decimal(p), Decimal(1), side < 0) for i, t, p, side in TICKS]
    )
    by_lists, by_series = {}, {}
    for row in EVENT_ROWS:
        r, c, e = (
            row[f"{n}_agg_trade_id"] for n in ("reference", "confirm", "extreme")
        )
        key = (theta, c)
        by_series[key] = fitref.from_ticks(row, series)
        phase = [(fitref.scaled(Decimal(p)), t, i) for i, t, p, _ in TICKS]
        by_lists[key] = fitref.fit_row(
            [x for x in phase if r < x[2] <= c],
            [x for x in phase if c < x[2] <= e],
            reference=(row["reference_time"], fitref.scaled(row["reference_price"])),
            confirm=(row["confirm_time"], fitref.scaled(row["confirm_price"])),
            extreme=(row["extreme_time"], fitref.scaled(row["extreme_price"])),
        )
    by_reader = fitref.from_lake(l1, l2, M1, [TEXT])
    for found in (by_lists, by_series, by_reader):
        as_rows = {k: dict(zip(NAMES, v, strict=True)) for k, v in found.items()}
        assert assert_rows(as_rows, wanted) == len(BLOCKS)


def test_the_oracle_reproduces_the_three_events_of_the_bridge_probe():
    """Los tres eventos a mano de `l3_fit_bridge_probe` (ITSC-398), con sus ticks: la
    aceleración y el salto que la sonda calculó, y una pendiente a mano.
    """
    # (R = (t, p), ticks (p, t, orden), C = (t, p_L2)) del evento 1, 2 y 3 de la sonda.
    events = [
        ((0, 100), [(102, 2, 1), (104, 5, 2), (109, 8, 3), (110, 10, 4)], (10, 110)),
        (
            (100, 200),
            [(198, 110, 1), (195, 110, 2), (192, 130, 3), (190, 140, 4), (191, 140, 5)],
            (140, 190),
        ),
        ((0, 1000), [(1012, 5, 1), (1004, 10, 2), (1010, 20, 3)], (20, 1010)),
    ]
    accel = [F(1080, 5759), F(-22, 39), F(-864, 505)]
    jump = [F(1, 10), F(1, 5), F(3, 5)]
    # Pendientes en fracción de p_R por segundo: Σ(t−t̄)(p−p̄)/Σ(t−t̄)² por µs, ·10⁶/p_R.
    slopes = [F(530_000, 49), F(-23_250, 23), F(-200, 7)]
    for (reference, ticks, confirm), a, j, s in zip(
        events, accel, jump, slopes, strict=True
    ):
        c_slope, _, _, c_accel, _, _, c_jump, *_ = fitref.fit_row(
            ticks, [], reference=reference, confirm=confirm, extreme=confirm
        )
        assert_value(c_accel, float(a), "c_accel", reference)
        assert_value(c_jump, float(j), "c_jump_share", reference)
        assert_value(c_slope, float(s), "c_slope", reference)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("lot", [None, 1, 2, 5])
@pytest.mark.parametrize("order", ["forward", "reversed", "swapped", "shuffled"])
def test_the_hand_rows_do_not_depend_on_threads_lots_or_push_order(
    hand, tmp_path, threads, lot, order
):
    """Con 11 row groups de L1 de 4 ticks y tramos de fronteras de 1, 2 y 5 eventos (los
    eventos cruzan row groups y tramos), el archivo es el mismo bit a bit.
    """
    l1, l2, _, baseline = hand
    for seed in range(2):
        result = fit_native(
            l1, l2, tmp_path / f"{seed}", M1, [TEXT], threads=threads, lot=lot,
            order=order, seed=seed,
        )[Decimal(TEXT)]  # fmt: skip
        assert result.content_hash == baseline.content_hash
        assert assert_rows(rows_of(result.path), hand_wanted()) == len(BLOCKS)


def test_the_hand_file_keeps_the_skeleton_and_the_footer(hand):
    _, l2, l3, result = hand
    summaries = fit_file(l3, TEXT).replace("fit.parquet", "summaries.parquet")
    events = l2_path(str(l2), DEFAULT_KEY, TEXT, M1)
    series = fitref.Series(
        [Tick(i, t, Decimal(p), Decimal(1), side < 0) for i, t, p, side in TICKS]
    )
    assert check_invariants(result.path, summaries, events, series) == len(BLOCKS)
    assert_skeleton_and_footer(result.path, summaries)


# ---------------------------------------------------------------------------------------
# El lago de pruebas: ticks reales con huecos de ids, dos meses, cinco θ y row groups de 50
# ticks. S corre primero (la familia se escribe contra su `summaries.parquet`).


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    ticks = fixture_ticks(seed=11, drop=0.12)
    lake = build_lake(
        tmp_path_factory.mktemp("fit"),
        ticks,
        month_cuts(ticks, 2300),
        l1_row_group=ROW_GROUP,
    )
    summaries = {
        month: summarize_native(
            lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
        )
        for month in lake.months
    }
    return lake, summaries


@pytest.fixture(scope="module")
def baseline(lake):
    """La corrida de referencia de cada mes: 4 hilos, fronteras todas antes, orden de ida."""
    lake_, _ = lake
    return {
        month: fit_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
        for month in lake_.months
    }


def lake_files(lake_, month, theta: Decimal):
    text = f"{theta:.8f}"
    return (
        fit_file(lake_.l3, text, month),
        fit_file(lake_.l3, text, month).replace("fit.parquet", "summaries.parquet"),
        l2_path(str(lake_.l2), DEFAULT_KEY, text, month),
    )


def test_the_lake_has_the_edge_cases(lake, baseline):
    """El lago trae de verdad cruces de row groups, overshoot vacío y nulos de cada clase."""
    lake_, _ = lake
    crossing = empty = no_slope = no_accel = jump_one = 0
    group_of = {}
    for month in lake_.months:
        for i, tick in enumerate(lake_.ticks[month]):
            group_of[tick.id] = (month, i // ROW_GROUP)
    for month, results in baseline.items():
        for theta, result in results.items():
            rows = rows_of(result.path)
            for event in lake_.events[int(theta.scaleb(8)), month]:
                row = rows[theta, event["confirm_agg_trade_id"]]
                crossing += (
                    group_of[event["reference_agg_trade_id"]]
                    != group_of[event["extreme_agg_trade_id"]]
                )
                empty += row["os_slope"] is None
                no_slope += row["c_slope"] is None
                no_accel += row["c_accel"] is None and row["c_slope"] is not None
                jump_one += row["c_jump_share"] == 1.0
    assert min(crossing, empty, no_slope, jump_one) > 20, (
        crossing, empty, no_slope, no_accel, jump_one,
    )  # fmt: skip


def test_the_rows_hold_the_invariants(lake, baseline):
    lake_, _ = lake
    series = fitref.Series(lake_.all_ticks())
    checked = 0
    for month, results in baseline.items():
        for theta in results:
            path, summaries, events = lake_files(lake_, month, theta)
            checked += check_invariants(path, summaries, events, series)
    assert checked == sum(len(rows) for rows in lake_.events.values())


def test_the_engine_matches_the_oracle_over_the_reader_frames(lake, baseline):
    """Todos los θ y meses del lago: los 9 valores del motor son los del oráculo, que lee
    cada evento con el lector de `dc_frames` (1e-12 relativo, nulos exactos).
    """
    lake_, _ = lake
    compared = 0
    for month, results in baseline.items():
        wanted = fitref.from_lake(lake_.l1, lake_.l2, month, lake_.theta_texts)
        got = {k: v for r in results.values() for k, v in rows_of(r.path).items()}
        compared += assert_rows(got, wanted, RTOL_LAKE)
    assert compared == sum(len(rows) for rows in lake_.events.values())


def test_the_engine_matches_the_oracle_over_a_filter_of_the_tick_list(lake, baseline):
    """Lo mismo sin lector ni Arrow: la pertenencia es `R < id ≤ C` y `C < id ≤ E` sobre la
    lista de ticks y los tiempos y precios de ancla de las filas de L2.
    """
    lake_, _ = lake
    series = fitref.Series(lake_.all_ticks())
    compared = 0
    for month, results in baseline.items():
        for theta, result in results.items():
            events = lake_.events[int(theta.scaleb(8)), month]
            wanted = {
                (theta, e["confirm_agg_trade_id"]): fitref.from_ticks(e, series)
                for e in events
            }
            compared += assert_rows(rows_of(result.path), wanted, RTOL_LAKE)
    assert compared == sum(len(rows) for rows in lake_.events.values())


def test_the_skeleton_and_the_footer_are_those_of_summaries(lake, baseline):
    lake_, _ = lake
    for month, results in baseline.items():
        for theta, result in results.items():
            path, summaries, _ = lake_files(lake_, month, theta)
            assert result.path == path
            assert_skeleton_and_footer(path, summaries)
            assert result.rows == pq.ParquetFile(path).metadata.num_rows
            table = pq.read_table(path, schema=FIT_SCHEMA)
            assert content_hash(table) == result.content_hash


def test_s_and_t_do_not_change(lake, baseline):
    """Escribir la familia A no toca `summaries.parquet` ni `path.parquet`: el `content_hash`
    de S es el que S entregó, y el de T es el mismo antes y después de correr A.
    """
    lake_, summaries = lake
    for month, results in summaries.items():
        for theta, result in results.items():
            _, path, _ = lake_files(lake_, month, theta)
            table = pq.read_table(path, schema=SUMMARIES_SCHEMA)
            assert content_hash(table) == result.content_hash
    month = lake_.months[-1]
    first = path_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    fit_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    again = path_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    for theta, result in again.items():
        assert result.content_hash == first[theta].content_hash
    after = summarize_native(
        lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts, series_start=M1
    )
    for theta, result in after.items():
        assert result.content_hash == summaries[month][theta].content_hash


def test_a_repeated_run_gives_the_same_hash_and_rows(lake, baseline):
    lake_, _ = lake
    for _ in range(3):
        for month, results in baseline.items():
            again = fit_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
            for theta, result in again.items():
                assert result.content_hash == results[theta].content_hash


def assert_same_as_baseline(lake_, baseline, **kwargs) -> None:
    for month, results in baseline.items():
        got = fit_native(
            lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts, **kwargs
        )
        for theta, result in got.items():
            assert result.content_hash == results[theta].content_hash, (month, theta)
            assert result.rows == results[theta].rows
            table = pq.read_table(result.path, schema=FIT_SCHEMA)
            assert content_hash(table) == result.content_hash


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("lot", [1, 5, 10, 25, 100])
def test_boundaries_in_lots_and_the_thread_count_change_nothing(
    lake, baseline, threads, lot
):
    """Las fronteras en tramos de 1, 5, 10, 25 y 100 eventos, entregadas justo antes del row
    group que las necesita, dan el mismo archivo que todas antes del primer `push`.
    """
    assert_same_as_baseline(lake[0], baseline, threads=threads, lot=lot)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("order", ["reversed", "swapped", "shuffled"])
def test_the_push_order_does_not_change_a_bit(lake, baseline, threads, order):
    for seed in range(3):
        assert_same_as_baseline(
            lake[0], baseline, threads=threads, order=order, seed=seed
        )


def test_lots_and_a_shuffled_order_together_change_nothing(lake, baseline):
    for seed, lot in enumerate((1, 10)):
        assert_same_as_baseline(
            lake[0], baseline, threads=4, order="shuffled", seed=seed, lot=lot
        )
