"""La familia T (`path.parquet`, TRD-L3 §7.5): un fixture calculado a mano, el oráculo de
Python puro y el motor nativo sobre el lago de pruebas (ITSC-393, ADR-L3-17).

Tres cosas se sostienen mutuamente: los 13 valores de cada fila de un fixture chico, escritos
a mano en esta prueba con sus operaciones a la vista; el oráculo `pathref` (Python puro, sin
el motor), que debe dar esos mismos valores y los del lago; y el motor, que debe dar los del
oráculo con tolerancia relativa de 1e-12 en los DOUBLE y exactos en los INT64 y los nulos. Los
row groups de L1 son chicos para que los eventos crucen row groups y tramos.
"""

import itertools
import math
from decimal import Decimal
from fractions import Fraction as F

import pathref
import pyarrow.parquet as pq
import pytest
from dc_frames import l2_path
from frames_lake import theta_text, write_l2
from l3_dc_descriptors.schema import (
    PATH_SCHEMA,
    PATH_STATE_VERSION,
    SILENCE_US,
    SUMMARIES_SCHEMA,
)
from l3_dc_descriptors.summarize import DEFAULT_KEY
from l3_lake import M1, Tick, build_lake, fixture_ticks, month_cuts, write_ticks
from native_driver import path_native, summarize_native
from pyutils import content_hash

ROW_GROUP = 50
RTOL = 1e-12
NAMES = pathref.COLUMNS
INT64 = ("c_max_gap", "os_max_gap")


# ---------------------------------------------------------------------------------------
# Lo común: leer lo que el motor escribió y compararlo columna a columna.


def path_file(l3, text, month=M1) -> str:
    from dc_frames import family_path

    return family_path(str(l3), DEFAULT_KEY, text, month, "path")


def rows_of(path: str) -> dict[tuple[Decimal, int], dict]:
    table = pq.read_table(path, schema=PATH_SCHEMA)
    return {(r["theta"], r["confirm_agg_trade_id"]): r for r in table.to_pylist()}


def assert_value(got, wanted, name: str, where) -> None:
    """Un valor de la fila: nulidad exacta, INT64 exacto, DOUBLE a 1e-12 relativo."""
    if wanted is None:
        assert got is None, (where, name, got)
        return
    assert got is not None, (where, name)
    if name in INT64:
        assert isinstance(got, int) and got == wanted, (where, name, got, wanted)
    else:
        assert math.isfinite(got), (where, name, got)
        assert math.isclose(got, wanted, rel_tol=RTOL, abs_tol=0.0), (
            where,
            name,
            got,
            wanted,
        )


def assert_rows(got: dict, wanted: dict) -> int:
    """`got` (filas del archivo) contra `wanted` (`(θ, id) → 13 valores`): mismas claves."""
    assert got.keys() == wanted.keys()
    for key, row in got.items():
        for name, value in zip(NAMES, wanted[key], strict=True):
            assert_value(row[name], value, name, key)
    return len(got)


def check_invariants(path: str, summaries: str, events: str) -> int:
    """Las invariantes de TRD-L3 §7.5 de un `path.parquet`, leyendo duraciones y cantidades
    de `summaries.parquet` y los precios de ancla de `events.parquet` (L2): ninguna usa el
    motor ni el oráculo. Devuelve las filas revisadas.
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
        has_c_duration = s["c_duration"] > 0
        has_os_duration = has_os and s["os_duration"] > 0
        # Los nulos están exactamente donde la regla dice.
        wanted_null = {
            "os_realized_var": not has_os,
            "os_max_gap": not has_os,
            "os_tortuosity": not has_os or e["extreme_price"] == e["confirm_price"],
            "c_sign_autocorr": s["c_ticks"] < 2
            or 0 in (s["c_quantity_buy"], s["c_quantity_sell"]),
            "os_sign_autocorr": s["os_ticks"] < 2
            or 0 in (s["os_quantity_buy"], s["os_quantity_sell"]),
            "c_idle_fraction": not has_c_duration,
            "os_idle_fraction": not has_os_duration,
            "c_t25": not has_c_duration,
            "c_t50": not has_c_duration,
            "c_t75": not has_c_duration,
            "c_realized_var": False,
            "c_tortuosity": False,
            "c_max_gap": False,
        }
        assert {n: row[n] is None for n in NAMES} == wanted_null, key
        assert row["c_realized_var"] >= 0
        assert row["c_tortuosity"] >= 1
        assert 0 <= row["c_max_gap"] <= s["c_duration"]
        if has_os:
            assert row["os_realized_var"] >= 0
            assert 0 <= row["os_max_gap"] <= s["os_duration"]
            if row["os_tortuosity"] is not None:
                assert row["os_tortuosity"] >= 1
        for name in ("c_sign_autocorr", "os_sign_autocorr"):
            if row[name] is not None:
                assert abs(row[name]) <= 1
        for name in ("c_idle_fraction", "os_idle_fraction"):
            if row[name] is not None:
                assert 0 <= row[name] <= 1
        if SILENCE_US > s["c_duration"] and row["c_idle_fraction"] is not None:
            assert row["c_idle_fraction"] == 0
        if has_c_duration:
            assert 0 <= row["c_t25"] <= row["c_t50"] <= row["c_t75"] <= 1
    return len(got)


def assert_skeleton_and_footer(path: str, summaries: str) -> None:
    """Mismas filas, orden y row groups que `summaries.parquet`; unión 1:1 por clave;
    esquema del contrato y metadatos del pie (`state_version`, `silence_us`).
    """
    path_file_, summary_file = pq.ParquetFile(path), pq.ParquetFile(summaries)
    sizes = [
        summary_file.metadata.row_group(i).num_rows
        for i in range(summary_file.num_row_groups)
    ]
    assert [
        path_file_.metadata.row_group(i).num_rows
        for i in range(path_file_.num_row_groups)
    ] == sizes
    keys = ["theta", "confirm_agg_trade_id"]
    mine, theirs = path_file_.read(columns=keys), summary_file.read(columns=keys)
    assert mine.equals(theirs)
    assert len(set(map(tuple, zip(*(c.to_pylist() for c in mine.columns))))) == len(
        mine
    )
    assert path_file_.schema_arrow.remove_metadata().equals(PATH_SCHEMA)
    assert [f.name for f in PATH_SCHEMA][2:] == list(NAMES)
    meta = pq.read_schema(path).metadata
    assert meta[b"state_version"] == b"1.0.0" == PATH_STATE_VERSION.encode()
    assert meta[b"silence_us"] == b"10000000" == str(SILENCE_US).encode()


# ---------------------------------------------------------------------------------------
# El fixture a mano: seis eventos encadenados (E_i = R_{i+1}) de un solo θ y un mes.
#
# Precios en USD, tiempos en µs y lados +1 (compra, `is_buyer_maker` falso) o −1 (venta).
# Los row groups de L1 son de 4 ticks y los de L2 de 2 eventos. `S` es un segundo.

S = 1_000_000
BASE = 1_501_545_600 * S  # 2017-08-01 00:00 UTC
THETA = 1_000_000  # 0,01 × 10⁸
TEXT = theta_text(THETA)
# (id, tiempo desde BASE, precio, lado). Los ids 23 y 24 faltan: un hueco del proveedor.
TICKS = [
    # A, upturn, R = 1, C = 8, E = 11. Instante de confirmación: ids 7 y 8.
    (1, 0, "100.0", +1),
    (2, 15 * S, "100.2", +1),
    (3, 16 * S, "100.4", +1),
    (4, 28 * S, "100.3", -1),  # hueco de 12 s: en silencio
    (5, 38 * S, "100.7", +1),  # hueco de exactamente 10 s: no
    (6, 47 * S, "100.8", -1),
    (7, 54 * S, "100.9", +1),
    (8, 54 * S, "101.5", -1),
    (9, 74 * S, "101.7", +1),  # 20 s tras C: no es un hueco de la fase de overshoot
    (10, 79 * S, "101.4", +1),
    (11, 104 * S, "102.0", -1),  # hueco de 25 s
    # B, downturn atómico de un tick y sin overshoot. R = 11, C = E = 12.
    (12, 120 * S, "100.0", +1),
    # C, upturn con los tres lados iguales. R = 12, C = 15, E = 16.
    (13, 121 * S, "100.6", +1),
    (14, 126 * S, "101.9", +1),
    (15, 126 * S, "102.4", +1),
    (16, 140 * S, "103.0", +1),  # overshoot de un tick
    # D, downturn atómico de dos ticks. R = 16, C = 18, E = 19.
    (17, 150 * S, "101.4", -1),
    (18, 150 * S, "100.8", +1),
    (
        19,
        160 * S,
        "101.0",
        +1,
    ),  # overshoot de un tick en el mismo precio que confirm_price
    # E, upturn con huecos de 9,999999 s, 10 s y 10,000001 s. R = 19, C = 28, E = 30.
    (20, 161 * S, "101.2", +1),
    (21, 161 * S + 9_999_999, "101.4", -1),
    (22, 161 * S + 9_999_999 + 10_000_000, "101.9", -1),
    (25, 191 * S, "102.3", +1),
    (26, 191 * S + 1, "101.8", -1),
    (27, 195 * S, "102.6", +1),
    (28, 195 * S, "103.5", +1),
    (29, 196 * S, "104.0", +1),
    (30, 206 * S, "104.5", +1),
    # F, downturn con un hueco de 43 s. R = 30, C = 36, E = 39.
    (31, 207 * S, "104.3", -1),
    (32, 207 * S, "104.1", -1),
    (33, 250 * S, "104.0", +1),
    (34, 251 * S, "103.9", -1),
    (35, 260 * S, "103.7", -1),
    (36, 260 * S, "103.4", +1),
    (37, 270 * S, "103.2", -1),
    (38, 271 * S, "103.6", +1),
    (39, 272 * S, "103.0", -1),
]
# (dirección, R, C, E, reference_price, confirm_price, extreme_price)
EVENTS = [
    (+1, 1, 8, 11, "100.0", "101.0", "102.0"),
    (-1, 11, 12, 12, "102.0", "100.0", "100.0"),
    (+1, 12, 15, 16, "100.0", "102.0", "103.0"),
    (-1, 16, 18, 19, "103.0", "101.0", "101.0"),
    (+1, 19, 28, 30, "101.0", "103.0", "104.5"),
    (-1, 30, 36, 39, "104.5", "103.5", "103.0"),
]


def sq(a: str, b: str) -> float:
    """(ln(b/a))² de un paso de precio, con el cociente exacto: log1p((b − a)/a)."""
    a, b = F(a), F(b)
    return math.log1p(float((b - a) / a)) ** 2


def ratio(numerator: str, denominator: str) -> float:
    return float(F(numerator) / F(denominator))


def row(**columns) -> dict:
    """Los 13 valores de un evento por nombre de columna."""
    assert columns.keys() == set(NAMES)
    return columns


def hand_rows() -> dict[int, dict]:
    """Los 13 valores de cada evento, calculados a mano, por `confirm_agg_trade_id`."""
    return {
        # A. Anclas: R = 100, C = 101 (confirm_price; el tick C vale 101,5), E = 102.
        8: row(
            # Pasos de la confirmación: 100 → 100,2 → 100,4 → 100,3 → 100,7 → 100,8 → 100,9 → 101,5.
            c_realized_var=sum(
                sq(a, b)
                for a, b in itertools.pairwise(
                    [
                        "100",
                        "100.2",
                        "100.4",
                        "100.3",
                        "100.7",
                        "100.8",
                        "100.9",
                        "101.5",
                    ]
                )
            ),
            # Camino 0,2 + 0,2 + 0,1 + 0,4 + 0,1 + 0,1 + 0,6 = 1,7; neto |101 − 100| = 1.
            c_tortuosity=ratio("1.7", "1"),
            # Lados + + − + − + −: n = 7, media 1/7. Con a = 6/7 y b = −8/7: el numerador es
            # a² + 5ab = (36 − 240)/49 y el denominador 4a² + 3b² = 336/49: −204/336.
            c_sign_autocorr=-17 / 28,
            # Tiempos 15, 16, 28, 38, 47, 54, 54 s: huecos 1, 12, 10, 9, 7, 0 s.
            c_max_gap=12 * S,
            # Solo el de 12 s supera 10 s (el de 10 s exactos no); duración 54 − 15 = 39 s.
            c_idle_fraction=12 / 39,
            # 4(p − 100) ≥ k·1: p ≥ 100,25 (id 3, 16 s), p ≥ 100,5 (id 5, 38 s), p ≥ 100,75
            # (id 6, 47 s); (t* − 15)/39.
            c_t25=1 / 39,
            c_t50=23 / 39,
            c_t75=32 / 39,
            # Overshoot, ancla 101: 101 → 101,7 → 101,4 → 102.
            os_realized_var=sq("101", "101.7")
            + sq("101.7", "101.4")
            + sq("101.4", "102"),
            # Camino 0,7 + 0,3 + 0,6 = 1,6; neto |102 − 101| = 1.
            os_tortuosity=ratio("1.6", "1"),
            # Lados + + −: n = 3, media 1/3, a = 2/3, b = −4/3: (a² + ab)... = (4/9 − 8/9)/(24/9).
            os_sign_autocorr=-1 / 6,
            # Tiempos 74, 79, 104 s: huecos 5 y 25 s (el de 20 s hasta el tick C no cuenta).
            os_max_gap=25 * S,
            os_idle_fraction=25 / 30,  # duración 104 − 74 = 30 s
        ),
        # B. Downturn atómico: un tick (id 12) que cruza de 102 a 100. Sin overshoot.
        12: row(
            c_realized_var=sq("102", "100"),
            c_tortuosity=ratio("2", "2"),  # camino |100 − 102| = 2, neto 2
            c_sign_autocorr=None,  # un tick
            c_max_gap=0,
            c_idle_fraction=None,  # c_duration = 0
            c_t25=None,
            c_t50=None,
            c_t75=None,
            os_realized_var=None,
            os_tortuosity=None,
            os_sign_autocorr=None,
            os_max_gap=None,
            os_idle_fraction=None,
        ),
        # C. Anclas R = 100, C = 102, E = 103. Tres compras: la autocorrelación es nula.
        15: row(
            # 100 → 100,6 → 101,9 → 102,4.
            c_realized_var=sq("100", "100.6")
            + sq("100.6", "101.9")
            + sq("101.9", "102.4"),
            c_tortuosity=ratio("2.4", "2"),  # camino 0,6 + 1,3 + 0,5; neto 2
            c_sign_autocorr=None,  # todos los ticks del mismo lado
            c_max_gap=5 * S,  # tiempos 121, 126, 126 s
            c_idle_fraction=0.0,  # 5 s de duración, menos que silence_us: ningún hueco
            # 4(p − 100) ≥ 2k: p ≥ 100,5 (id 13, t_1: 0/5), p ≥ 101 y p ≥ 101,5 (id 14, 5/5).
            c_t25=0.0,
            c_t50=1.0,
            c_t75=1.0,
            os_realized_var=sq("102", "103"),  # ancla 102 → un tick de 103
            os_tortuosity=ratio("1", "1"),
            os_sign_autocorr=None,  # un tick
            os_max_gap=0,
            os_idle_fraction=None,  # os_duration = 0
        ),
        # D. Downturn atómico de dos ticks: R = 103, C = 101 (ticks 101,4 y 100,8), E = 101.
        18: row(
            c_realized_var=sq("103", "101.4") + sq("101.4", "100.8"),
            c_tortuosity=ratio("2.2", "2"),  # camino 1,6 + 0,6; neto |101 − 103| = 2
            # Lados − +: n = 2, media 0: numerador (−1)(1), denominador 1 + 1.
            c_sign_autocorr=-1 / 2,
            c_max_gap=0,  # los dos ticks tienen el mismo tiempo
            c_idle_fraction=None,
            c_t25=None,
            c_t50=None,
            c_t75=None,
            os_realized_var=0.0,  # ancla 101 → un tick de 101: ln(1) = 0
            os_tortuosity=None,  # p_E = p_C: desplazamiento neto 0
            os_sign_autocorr=None,
            os_max_gap=0,
            os_idle_fraction=None,
        ),
        # E. Anclas R = 101, C = 103 (el tick C vale 103,5), E = 104,5.
        28: row(
            # 101 → 101,2 → 101,4 → 101,9 → 102,3 → 101,8 → 102,6 → 103,5.
            c_realized_var=sum(
                sq(a, b)
                for a, b in itertools.pairwise(
                    [
                        "101",
                        "101.2",
                        "101.4",
                        "101.9",
                        "102.3",
                        "101.8",
                        "102.6",
                        "103.5",
                    ]
                )
            ),
            # Camino 0,2 + 0,2 + 0,5 + 0,4 + 0,5 + 0,8 + 0,9 = 3,5; neto 2.
            c_tortuosity=ratio("3.5", "2"),
            # Lados + − − + − + +: media 1/7, a = 6/7, b = −8/7. Pares (+−) (−−) (−+) (+−)
            # (−+) (++): 4ab + b² + a² = (−192 + 64 + 36)/49 = −92/49; denominador 336/49.
            c_sign_autocorr=-92 / 336,
            # Huecos en µs: 9 999 999, 10 000 000, 10 000 001, 1, 3 999 999, 0.
            c_max_gap=10_000_001,
            # Solo el de 10 000 001 µs supera 10 s; duración 195 − 161 = 34 s.
            c_idle_fraction=10_000_001 / 34_000_000,
            # 4(p − 101) ≥ 2k: p ≥ 101,5 (id 22, 180 999 999), p ≥ 102 (id 25, 191 s),
            # p ≥ 102,5 (id 27, 195 s); (t* − 161 s)/34 s.
            c_t25=19_999_999 / 34_000_000,
            c_t50=30_000_000 / 34_000_000,
            c_t75=1.0,
            # Overshoot, ancla 103 (no el 103,5 del tick C): 103 → 104 → 104,5.
            os_realized_var=sq("103", "104") + sq("104", "104.5"),
            os_tortuosity=ratio("1.5", "1.5"),  # camino 1 + 0,5; neto 1,5
            os_sign_autocorr=None,  # dos compras
            os_max_gap=10 * S,  # un hueco de exactamente 10 s
            os_idle_fraction=0.0,  # no supera silence_us
        ),
        # F. Anclas R = 104,5, C = 103,5 (el tick C vale 103,4), E = 103.
        36: row(
            # 104,5 → 104,3 → 104,1 → 104 → 103,9 → 103,7 → 103,4.
            c_realized_var=sum(
                sq(a, b)
                for a, b in itertools.pairwise(
                    ["104.5", "104.3", "104.1", "104.0", "103.9", "103.7", "103.4"]
                )
            ),
            # Camino 0,2 + 0,2 + 0,1 + 0,1 + 0,2 + 0,3 = 1,1; neto |103,5 − 104,5| = 1.
            c_tortuosity=ratio("1.1", "1"),
            # Lados − − + − − +: n = 6, S = −2, media −1/3, b = −2/3 (−), a = 4/3 (+).
            # Pares (−−) (−+) (+−) (−−) (−+): 2b² + 3ab = (8 − 24)/9 = −16/9; denominador
            # 4b² + 2a² = 48/9.
            c_sign_autocorr=-16 / 48,
            c_max_gap=43 * S,  # tiempos 207, 207, 250, 251, 260, 260 s
            c_idle_fraction=43 / 53,  # el hueco de 9 s no cuenta; duración 53 s
            # 4(104,5 − p) ≥ k: p ≤ 104,25 (id 32, t_1), p ≤ 104 (id 33, 250 s, justo el
            # borde), p ≤ 103,75 (id 35, 260 s).
            c_t25=0.0,
            c_t50=43 / 53,
            c_t75=1.0,
            # Overshoot, ancla 103,5: 103,5 → 103,2 → 103,6 → 103.
            os_realized_var=sq("103.5", "103.2")
            + sq("103.2", "103.6")
            + sq("103.6", "103"),
            os_tortuosity=ratio("1.3", "0.5"),  # camino 0,3 + 0,4 + 0,6; neto 0,5
            # Lados − + −: media −1/3, b = −2/3, a = 4/3: 2ab/(2b² + a²) = (−16/9)/(24/9).
            os_sign_autocorr=-2 / 3,
            os_max_gap=1 * S,  # tiempos 270, 271, 272 s
            os_idle_fraction=0.0,
        ),
    }


def write_hand_lake(base):
    """Escribe L1 (row groups de 4 ticks) y L2 (de 2 eventos) del fixture. Devuelve las raíces."""
    l1, l2, l3 = base / "l1", base / "l2", base / "l3"
    by_id = {t[0]: t for t in TICKS}
    ticks = [
        Tick(i, BASE + time, Decimal(price), Decimal(1), side < 0)
        for i, time, price, side in TICKS
    ]
    write_ticks(l1, ticks, M1, 4)
    rows = []
    for direction, r, c, e, reference, confirm, extreme in EVENTS:
        row = {"direction": direction}
        for name, tick_id, price in (
            ("reference", r, reference),
            ("confirm", c, confirm),
            ("extreme", e, extreme),
        ):
            row[f"{name}_price"] = Decimal(price)
            row[f"{name}_time"] = BASE + by_id[tick_id][1]
            row[f"{name}_agg_trade_id"] = tick_id
        rows.append(row)
    write_l2(l2, THETA, rows, 2, M1)
    return l1, l2, l3


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    base = tmp_path_factory.mktemp("hand")
    l1, l2, l3 = write_hand_lake(base)
    results = path_native(l1, l2, l3, M1, [TEXT])
    return l1, l2, l3, results[Decimal(TEXT)]


def hand_wanted() -> dict[tuple[Decimal, int], tuple]:
    theta = Decimal(TEXT)
    return {
        (theta, event): tuple(values[name] for name in NAMES)
        for event, values in hand_rows().items()
    }


def test_the_hand_fixture_has_every_case(hand):
    """Las seis filas del fixture cubren los casos que la prueba promete."""
    _, l2, l3, result = hand
    summary = pq.read_table(
        path_file(l3, TEXT).replace("path.parquet", "summaries.parquet"),
        schema=SUMMARIES_SCHEMA,
    ).to_pylist()
    events = pq.read_table(l2_path(str(l2), DEFAULT_KEY, TEXT, M1)).to_pylist()
    assert [r["direction"] for r in events] == [1, -1, 1, -1, 1, -1]
    assert [r["c_duration"] for r in summary].count(0) == 2  # cruces atómicos
    assert [r["c_ticks"] for r in summary if r["c_duration"] == 0] == [1, 2]
    assert [r["os_ticks"] for r in summary] == [3, 0, 1, 1, 2, 3]  # vacío y de un tick
    # Todos los lados iguales, con más de un tick, en confirmación y en overshoot.
    assert (summary[2]["c_quantity_sell"], summary[2]["c_ticks"]) == (0, 3)
    assert (summary[4]["os_quantity_sell"], summary[4]["os_ticks"]) == (0, 2)
    # Un evento por row group de L1 de 4 ticks: cruza al menos tres.
    group = {t[0]: i // 4 for i, t in enumerate(TICKS)}
    spans = [group[e] - group[r] + 1 for _, r, _, e, *_ in EVENTS]
    assert max(spans) >= 4 and all(s >= 1 for s in spans)
    # Huecos de 10 s menos 1 µs, 10 s exactos y 10 s más 1 µs entre ticks consecutivos.
    gaps = {b[1] - a[1] for a, b in itertools.pairwise(TICKS)}
    assert {10 * S - 1, 10 * S, 10 * S + 1} <= gaps
    assert result.rows == len(EVENTS)


def test_the_engine_gives_the_hand_calculated_rows(hand):
    _, _, _, result = hand
    assert assert_rows(rows_of(result.path), hand_wanted()) == len(EVENTS)


def test_the_oracle_gives_the_hand_calculated_rows(hand):
    """El oráculo, de tres maneras (listas de ticks y anclas, la serie filtrada por id y
    las tramas del lector), da los valores escritos a mano.
    """
    l1, l2, _, _ = hand
    wanted = hand_wanted()
    theta = Decimal(TEXT)

    def phase(low, high):
        return [
            pathref.tick(BASE + time, Decimal(price), side < 0)
            for i, time, price, side in TICKS
            if low < i <= high
        ]

    series = pathref.Series(
        [
            Tick(i, BASE + time, Decimal(price), Decimal(1), side < 0)
            for i, time, price, side in TICKS
        ]
    )
    by_lists, by_series = {}, {}
    for direction, r, c, e, reference, confirm, extreme in EVENTS:
        by_lists[theta, c] = pathref.path_row(
            phase(r, c),
            phase(c, e),
            reference=pathref.scaled(Decimal(reference)),
            confirm=pathref.scaled(Decimal(confirm)),
            extreme=pathref.scaled(Decimal(extreme)),
            direction=direction,
        )
        by_series[theta, c] = pathref.from_ticks(
            {
                "direction": direction,
                "reference_agg_trade_id": r,
                "confirm_agg_trade_id": c,
                "extreme_agg_trade_id": e,
                "reference_price": Decimal(reference),
                "confirm_price": Decimal(confirm),
                "extreme_price": Decimal(extreme),
            },
            series,
        )
    by_reader = pathref.from_lake(l1, l2, M1, [TEXT])
    for found in (by_lists, by_series, by_reader):
        as_rows = {k: dict(zip(NAMES, v, strict=True)) for k, v in found.items()}
        assert assert_rows(as_rows, wanted) == len(EVENTS)


def test_the_hand_rows_hold_the_invariants(hand):
    _, l2, l3, result = hand
    summaries = path_file(l3, TEXT).replace("path.parquet", "summaries.parquet")
    events = l2_path(str(l2), DEFAULT_KEY, TEXT, M1)
    assert check_invariants(result.path, summaries, events) == len(EVENTS)
    assert_skeleton_and_footer(result.path, summaries)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("lot", [None, 1, 2, 5])
@pytest.mark.parametrize("order", ["forward", "reversed", "swapped", "shuffled"])
def test_the_hand_rows_do_not_depend_on_threads_lots_or_push_order(
    hand, tmp_path, threads, lot, order
):
    """Con 10 row groups de L1 de 4 ticks y tramos de fronteras de 1, 2 y 5 eventos (los
    eventos cruzan row groups y tramos), el archivo es el mismo bit a bit.
    """
    l1, l2, _, baseline = hand
    for seed in range(2):
        result = path_native(
            l1, l2, tmp_path / f"{seed}", M1, [TEXT], threads=threads, lot=lot,
            order=order, seed=seed,
        )[Decimal(TEXT)]  # fmt: skip
        assert result.content_hash == baseline.content_hash
        assert assert_rows(rows_of(result.path), hand_wanted()) == len(EVENTS)


# ---------------------------------------------------------------------------------------
# El lago de pruebas: ticks reales con huecos de ids, dos meses, cinco θ y row groups de 50
# ticks. S corre primero (la familia se escribe contra su `summaries.parquet`).


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    ticks = fixture_ticks(seed=11, drop=0.12)
    lake = build_lake(
        tmp_path_factory.mktemp("path"),
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
        month: path_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
        for month in lake_.months
    }


def lake_files(lake_, month, theta: Decimal):
    text = f"{theta:.8f}"
    return (
        path_file(lake_.l3, text, month),
        path_file(lake_.l3, text, month).replace("path.parquet", "summaries.parquet"),
        l2_path(str(lake_.l2), DEFAULT_KEY, text, month),
    )


def test_the_lake_has_the_edge_cases(lake, baseline):
    """El lago trae de verdad cruces de row groups, overshoot vacío, nulos y huecos largos."""
    lake_, _ = lake
    crossing = empty = silent = idle = one_side = 0
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
                empty += row["os_max_gap"] is None
                silent += row["c_idle_fraction"] is None
                idle += bool(row["c_idle_fraction"] or row["os_idle_fraction"])
                one_side += row["c_sign_autocorr"] is None
    assert min(crossing, empty, silent, idle, one_side) > 20, (
        crossing, empty, silent, idle, one_side,
    )  # fmt: skip


def test_the_rows_hold_the_invariants(lake, baseline):
    lake_, _ = lake
    checked = 0
    for month, results in baseline.items():
        for theta in results:
            path, summaries, events = lake_files(lake_, month, theta)
            checked += check_invariants(path, summaries, events)
    assert checked == sum(len(rows) for rows in lake_.events.values())


def test_the_engine_matches_the_oracle_over_the_reader_frames(lake, baseline):
    """Todos los θ y meses del lago: los 13 valores del motor son los del oráculo, que lee
    cada evento con el lector de `dc_frames` (1e-12 relativo en DOUBLE, exacto en INT64 y nulos).
    """
    lake_, _ = lake
    compared = 0
    for month, results in baseline.items():
        wanted = pathref.from_lake(lake_.l1, lake_.l2, month, lake_.theta_texts)
        got = {k: v for r in results.values() for k, v in rows_of(r.path).items()}
        compared += assert_rows(got, wanted)
    assert compared == sum(len(rows) for rows in lake_.events.values())


def test_the_engine_matches_the_oracle_over_a_filter_of_the_tick_list(lake, baseline):
    """Lo mismo sin lector ni Arrow: la pertenencia es `R < id ≤ C` y `C < id ≤ E` sobre la
    lista de ticks y los precios de ancla de las filas de L2.
    """
    lake_, _ = lake
    series = pathref.Series(lake_.all_ticks())
    compared = 0
    for month, results in baseline.items():
        for theta, result in results.items():
            events = lake_.events[int(theta.scaleb(8)), month]
            wanted = {
                (theta, e["confirm_agg_trade_id"]): pathref.from_ticks(e, series)
                for e in events
            }
            compared += assert_rows(rows_of(result.path), wanted)
    assert compared == sum(len(rows) for rows in lake_.events.values())


def test_the_skeleton_and_the_footer_are_those_of_summaries(lake, baseline):
    lake_, _ = lake
    for month, results in baseline.items():
        for theta, result in results.items():
            path, summaries, _ = lake_files(lake_, month, theta)
            assert result.path == path
            assert_skeleton_and_footer(path, summaries)
            assert result.rows == pq.ParquetFile(path).metadata.num_rows
            table = pq.read_table(path, schema=PATH_SCHEMA)
            assert content_hash(table) == result.content_hash


def test_the_summaries_hashes_do_not_change(lake, baseline):
    """Escribir la familia T no toca `summaries.parquet`: su `content_hash` sigue siendo el
    que S entregó, y volver a correr S da el mismo.
    """
    lake_, summaries = lake
    for month, results in summaries.items():
        for theta, result in results.items():
            _, path, _ = lake_files(lake_, month, theta)
            table = pq.read_table(path, schema=SUMMARIES_SCHEMA)
            assert content_hash(table) == result.content_hash
    month = lake_.months[-1]
    again = summarize_native(
        lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts, series_start=M1
    )
    for theta, result in again.items():
        assert result.content_hash == summaries[month][theta].content_hash


def test_a_repeated_run_gives_the_same_hash_and_rows(lake, baseline):
    lake_, _ = lake
    for _ in range(3):
        for month, results in baseline.items():
            again = path_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
            for theta, result in again.items():
                assert result.content_hash == results[theta].content_hash


def assert_same_as_baseline(lake_, baseline, **kwargs) -> None:
    for month, results in baseline.items():
        got = path_native(
            lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts, **kwargs
        )
        for theta, result in got.items():
            assert result.content_hash == results[theta].content_hash, (month, theta)
            assert result.rows == results[theta].rows
            table = pq.read_table(result.path, schema=PATH_SCHEMA)
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
