"""La familia F (`spectrum.parquet`, TRD-L3 §7.2.3): un fixture calculado a mano, el oráculo con
`numpy.fft` y el motor nativo sobre el lago de pruebas (ITSC-412, ADR-L3-19).

Tres cosas se sostienen mutuamente: los 30 valores de cada fila de un fixture chico, escritos
a mano en esta prueba con sus operaciones a la vista (ocho eventos: alza y baja, overshoot vacío,
fases de 15 y 16 retornos, retornos alternados, senoide de período 8, una fase plana, un evento que
cruza row groups y tramos y un evento sobre el presupuesto); el oráculo `specref` (Python con
`numpy.fft`, sin el motor), que debe dar esos valores y los del lago; y el motor, que debe dar los
del oráculo con tolerancia absoluta de 1e-9 y los nulos exactos. Los valores a mano salen de
retornos ideales y el búfer guarda `f32`: se comparan con 1e-6 (veredicto de la Épica ITSC-410).
"""

import math
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import specref
from dc_frames import family_path, l2_path
from frames_lake import theta_text, write_l2
from l3_dc_descriptors import findings, summarize
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.schema import (
    SPECTRUM_BANDS,
    SPECTRUM_BUDGET_TICKS,
    SPECTRUM_FAMILY,
    SPECTRUM_N_MIN,
    SPECTRUM_SCHEMA,
    SPECTRUM_STATE_VERSION,
    SPECTRUM_WINDOW,
    SUMMARIES_SCHEMA,
)
from l3_dc_descriptors.summarize import DEFAULT_KEY
from l3_lake import M1, Tick, build_lake, fixture_ticks, month_cuts, write_ticks
from native_driver import (
    fit_native,
    path_native,
    spectrum_native,
    summarize_native,
)
from pathref import Series, scaled
from pyutils import content_hash

ROW_GROUP = 50
# Entre el motor (`rustfft`) y el oráculo (`numpy.fft`): las fracciones son de orden 1.
TOLERANCE = 1e-9
# La ubicación de la energía no pasa por la FFT: motor y oráculo son dos caminos de suma en `f64`.
LOCATION_TOLERANCE = 1e-12
LOCATIONS = ("c_energy_loc", "os_energy_loc")
# Contra los valores a mano: salen de retornos ideales y el motor guarda `f32`.
HAND_TOLERANCE = 1e-6
NAMES = specref.COLUMNS
SQRT2 = math.sqrt(2)


# ---------------------------------------------------------------------------------------
# El esquema es el contrato


def test_the_schema_is_the_key_and_30_nullable_doubles_in_the_inventory_order():
    names = [f.name for f in SPECTRUM_SCHEMA]
    assert names[:2] == ["theta", "confirm_agg_trade_id"]
    assert names[2:] == list(NAMES)
    assert len(names) == 32
    expected = [
        f"{phase}_{register}_{unit}_{band}"
        for register, unit in (("fourier", "band"), ("haar", "level"))
        for phase in ("c", "os")
        for band in range(1, 8)
    ]
    # Las dos ubicaciones de la energía total van al final, después de `os_haar_level_7`.
    assert names[2:] == [*expected, "c_energy_loc", "os_energy_loc"]
    assert names[-3] == "os_haar_level_7"
    assert names[2] == "c_fourier_band_1" and names[16] == "c_haar_level_1"
    assert all(
        str(f.type) == "double" and f.nullable for f in list(SPECTRUM_SCHEMA)[2:]
    )
    assert not SPECTRUM_SCHEMA.field("theta").nullable
    assert not SPECTRUM_SCHEMA.field("confirm_agg_trade_id").nullable
    assert (SPECTRUM_FAMILY, SPECTRUM_STATE_VERSION) == ("spectrum", "1.2.0")
    assert (SPECTRUM_N_MIN, SPECTRUM_BANDS, SPECTRUM_WINDOW) == (16, 7, "hann")
    assert SPECTRUM_BUDGET_TICKS == 130_000_000


# ---------------------------------------------------------------------------------------
# Lo común: leer lo que el motor escribió y compararlo columna a columna.


def spectrum_file(l3, text, month=M1) -> str:
    return family_path(str(l3), DEFAULT_KEY, text, month, SPECTRUM_FAMILY)


def rows_of(path: str) -> dict[tuple[Decimal, int], dict]:
    table = pq.read_table(path, schema=SPECTRUM_SCHEMA)
    return {(r["theta"], r["confirm_agg_trade_id"]): r for r in table.to_pylist()}


def assert_row(
    got: dict, wanted: tuple, where, tolerance: float, location_tolerance=None
) -> None:
    """Una fila: nulidad exacta y fracciones a `tolerance` absoluta (las ubicaciones, a
    `location_tolerance` si se da).
    """
    for name, value in zip(NAMES, wanted, strict=True):
        if value is None:
            assert got[name] is None, (where, name, got[name])
        else:
            assert got[name] is not None, (where, name)
            limit = (
                location_tolerance
                if name in LOCATIONS and location_tolerance
                else tolerance
            )
            assert abs(got[name] - value) <= limit, (where, name, got[name], value)


def assert_rows(
    got: dict, wanted: dict, tolerance: float, location_tolerance=None
) -> int:
    assert got.keys() == wanted.keys()
    for key, row in got.items():
        assert_row(row, wanted[key], key, tolerance, location_tolerance)
    return len(got)


def bands_present(n: int) -> int:
    """Cuántas bandas (y niveles de Haar) tienen bins con `n` retornos: la 5 pide 32, la 6 pide
    64 y la 7 pide 128."""
    return 0 if n < 16 else 4 if n < 32 else 5 if n < 64 else 6 if n < 128 else 7


# ---------------------------------------------------------------------------------------
# El fixture a mano: ocho eventos de un solo θ y un mes.
#
# Cada evento son los ticks de un bloque: el primero es R, luego la confirmación y el overshoot,
# con los retornos `ln(p_k / p_{k−1})` que dice la lista (el primero sale del ancla R). Los precios
# son enteros × 10⁻⁸ sobre 10⁷ USD, redondeados: el error del entero es de 10⁻¹⁵ relativo.
# Una deriva constante en los retornos no cambia ninguna fracción (la serie se centra), y sirve
# para que C sea el máximo (alza) o el mínimo (baja) de la confirmación, que exigen los invariantes
# de S. Los row groups de L1 son de 4 ticks y los de L2 de 2 eventos.

S = 1_000_000
BASE = 1_501_545_600 * S  # 2017-08-01 00:00 UTC
THETA = 1_000_000  # 0,01 × 10⁸
TEXT = theta_text(THETA)
START = 10**15
R = 1e-3
DRIFT = 1.1 * R  # más que la amplitud: los precios solo suben (o solo bajan)

ALTERNATING = [R, -R] * 8  # n = 16: período 2
SQUARE = [R, R, -R, -R] * 4  # n = 16: período 4
SINE = [R * math.sin(2 * math.pi * k / 8) for k in range(1, 129)]  # período 8


def shifted(values: list[float], drift: float) -> list[float]:
    return [v + drift for v in values]


# Las 7 columnas de una banda o nivel; `None` donde la banda no tiene bins.
ALTERNATING_N16 = [1, 0, 0, 0, None, None, None]
# Cuadrada de período 4, n = 16: bin 4 (período 4, banda 2) con vecinos 3 (5,33, banda 2) y 5
# (3,2, banda 1): Hann deja 2/3 en el centro y 1/6 en cada vecino → banda 1 = 1/6, banda 2 = 5/6.
# Haar: los pares (r, r) y (−r, −r) no tienen detalle; a₁ = ±√2·r y el nivel 2 lo tiene todo.
SQUARE_F16 = [1 / 6, 5 / 6, 0, 0, None, None, None]
SQUARE_H16 = [0, 1, 0, 0, None, None, None]
# Senoide de período 8: con n = 64 (bin 8) o 128 (bin 16), el centro y el vecino de período 9,1 o
# 8,5 en la banda 3 y el de 7,1 o 7,5 en la 2: banda 2 = 1/6, banda 3 = 5/6. Haar, con a = √2/2:
# nivel 1 = (2 − √2)/4, nivel 2 = 1/8, nivel 3 = (3 + 2√2)/8; los demás, 0. Con n = 128 existe la
# banda 7 (k = 1): vale 0.
SINE_F64 = [0, 1 / 6, 5 / 6, 0, 0, 0, None]
SINE_H64 = [(2 - SQRT2) / 4, 1 / 8, (3 + 2 * SQRT2) / 8, 0, 0, 0, None]
SINE_F128 = [0, 1 / 6, 5 / 6, 0, 0, 0, 0]
SINE_H128 = [(2 - SQRT2) / 4, 1 / 8, (3 + 2 * SQRT2) / 8, 0, 0, 0, 0]
NOTHING = [None] * 7

# (nombre, dirección, retornos de la confirmación, retornos del overshoot, ids que se salta antes
# del último tick, (c_fourier, os_fourier, c_haar, os_haar)).
BLOCKS = [
    # Baja: retornos alternados (toda la energía en la banda 1) y overshoot de 64 retornos senoidales.
    (
        "alternating_sine",
        -1,
        ALTERNATING,
        SINE[:64],
        0,
        (ALTERNATING_N16, SINE_F64, ALTERNATING_N16, SINE_H64),
    ),
    # Alza con overshoot vacío: senoide de período 8 de 64 retornos.
    (
        "sine_up",
        +1,
        shifted(SINE[:64], DRIFT),
        [],
        0,
        (SINE_F64, NOTHING, SINE_H64, NOTHING),
    ),
    # Baja con una confirmación cuadrada de 16 retornos y un overshoot alternado de 16.
    (
        "square_down",
        -1,
        shifted(SQUARE, -2 * R),
        ALTERNATING,
        0,
        (SQUARE_F16, ALTERNATING_N16, SQUARE_H16, ALTERNATING_N16),
    ),
    # Una fase de 15 retornos deja todo nulo y una de 16 calcula las bandas 1 a 4.
    (
        "fifteen_then_sixteen",
        +1,
        shifted(SINE[:15], DRIFT),
        SQUARE,
        0,
        (NOTHING, SQUARE_F16, NOTHING, SQUARE_H16),
    ),
    # 15 retornos y overshoot vacío: las 30 columnas nulas.
    ("fifteen_only", +1, shifted(SINE[:15], DRIFT), [], 0, (NOTHING,) * 4),
    # Una confirmación plana (todos los retornos 0) no tiene energía: nulas sus columnas.
    (
        "flat_confirmation",
        +1,
        [0.0] * 20,
        ALTERNATING,
        0,
        (NOTHING, ALTERNATING_N16, NOTHING, ALTERNATING_N16),
    ),
    # Un evento largo (192 ticks, unos 50 row groups de L1): las siete bandas de la confirmación.
    (
        "sine_128",
        +1,
        shifted(SINE[:128], DRIFT),
        SINE[:64],
        0,
        (SINE_F128, SINE_F64, SINE_H128, SINE_H64),
    ),
    # Sobre el presupuesto: 150 M de ids entre R y E con 5 ticks. No se calcula nada.
    (
        "over_budget",
        +1,
        [R, R, R],
        [-R, R],
        150_000_000,
        (NOTHING,) * 4,
    ),
]


def walk(returns: list[float]) -> list[int]:
    """Los precios (× 10⁻⁸) del tick R y de los ticks que siguen los retornos dados."""
    return [START, *(round(START * math.exp(x)) for x in np.cumsum(returns))]


def hand_layout():
    """Los ticks `(id, tiempo, precio × 10⁸)` y las filas de L2 del fixture, el id del tick C de
    cada bloque y el aviso del evento sobre el presupuesto.
    """
    ticks, rows, confirm_ids, over = [], [], {}, []
    next_id, offset = 1, 0
    for name, direction, c_returns, os_returns, jump, _ in BLOCKS:
        prices = walk([*c_returns, *os_returns])
        c, e = len(c_returns), len(c_returns) + len(os_returns)
        ids = [next_id + k for k in range(len(prices))]
        ids[-1] += jump
        for k, (id_, price) in enumerate(zip(ids, prices, strict=True)):
            ticks.append((id_, BASE + offset + 10 * k, price))
        # El tick C es el extremo de la confirmación y E el último del overshoot, cuyo extremo
        # (máximo en un alza, mínimo en una baja) es `extreme_price`.
        pick = max if direction == 1 else min
        assert prices[c] == pick(prices[1 : c + 1]), name  # invariante 3 de S
        extreme = pick(prices[c + 1 : e + 1]) if e > c else prices[c]
        row = {"direction": direction}
        for label, k, price in (
            ("reference", 0, prices[0]),
            ("confirm", c, prices[c]),
            ("extreme", e, extreme),
        ):
            row[f"{label}_price"] = Decimal(price).scaleb(-8)
            row[f"{label}_time"] = BASE + offset + 10 * k
            row[f"{label}_agg_trade_id"] = ids[k]
        rows.append(row)
        confirm_ids[name] = ids[c]
        if name == "over_budget":
            over.append((ids[c], ids[e] - ids[0]))
        next_id = ids[-1] + 1
        offset += 10 * S
    return ticks, rows, confirm_ids, over


TICKS, EVENT_ROWS, CONFIRM_IDS, OVER_BUDGET = hand_layout()


def hand_location(returns: list[float]) -> float | None:
    """La ubicación de los retornos ideales de una fase, con una suma simple de Python: sin
    `numpy`, sin el oráculo y sin el motor. Nula con menos de 16 retornos o sin energía.
    """
    n = len(returns)
    energy = sum(r * r for r in returns)
    if n < SPECTRUM_N_MIN or energy == 0:
        return None
    return sum((k - 0.5) / n * r * r for k, r in enumerate(returns, start=1)) / energy


def hand_wanted() -> dict[tuple[Decimal, int], tuple]:
    theta = Decimal(TEXT)
    return {
        (theta, CONFIRM_IDS[name]): (
            *c_f,
            *os_f,
            *c_h,
            *os_h,
            hand_location(c_returns),
            hand_location(os_returns),
        )
        for name, _, c_returns, os_returns, _, (c_f, os_f, c_h, os_h) in BLOCKS
    }


def write_hand_lake(base):
    l1, l2, l3 = base / "l1", base / "l2", base / "l3"
    write_ticks(
        l1,
        [Tick(i, t, Decimal(p).scaleb(-8), Decimal(1), False) for i, t, p in TICKS],
        M1,
        4,
    )
    write_l2(l2, THETA, EVENT_ROWS, 2, M1)
    return l1, l2, l3


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    base = tmp_path_factory.mktemp("hand")
    l1, l2, l3 = write_hand_lake(base)
    results = spectrum_native(l1, l2, l3, M1, [TEXT])
    return l1, l2, l3, results[Decimal(TEXT)]


def test_the_hand_fixture_has_every_case(hand):
    _, l2, l3, result = hand
    summary = {
        r["confirm_agg_trade_id"]: r
        for r in pq.read_table(
            spectrum_file(l3, TEXT).replace("spectrum.parquet", "summaries.parquet"),
            schema=SUMMARIES_SCHEMA,
        ).to_pylist()
    }
    by_name = {name: summary[CONFIRM_IDS[name]] for name, *_ in BLOCKS}
    events = pq.read_table(l2_path(str(l2), DEFAULT_KEY, TEXT, M1)).to_pylist()
    directions = [r["direction"] for r in events]
    assert directions.count(1) >= 1 and directions.count(-1) >= 1
    assert by_name["sine_up"]["os_ticks"] == 0  # overshoot vacío
    assert (
        by_name["fifteen_only"]["c_ticks"] == 15
        and by_name["fifteen_only"]["os_ticks"] == 0
    )
    assert by_name["fifteen_then_sixteen"]["c_ticks"] == 15
    assert by_name["fifteen_then_sixteen"]["os_ticks"] == 16
    assert by_name["alternating_sine"]["c_ticks"] == 16
    assert by_name["alternating_sine"]["os_ticks"] == 64
    assert by_name["sine_128"]["c_ticks"] == 128
    # Un evento por varios row groups de L1 de 4 ticks: el largo cruza más de cuarenta.
    group = {t[0]: i // 4 for i, t in enumerate(TICKS)}
    spans = [
        group[r["extreme_agg_trade_id"]] - group[r["reference_agg_trade_id"]] + 1
        for r in events
    ]
    assert max(spans) >= 40 and min(spans) >= 2
    assert result.rows == len(BLOCKS)


def test_the_engine_gives_the_hand_calculated_rows(hand):
    _, _, _, result = hand
    assert assert_rows(rows_of(result.path), hand_wanted(), HAND_TOLERANCE) == len(
        BLOCKS
    )


def test_the_oracle_gives_the_hand_calculated_rows(hand):
    """El oráculo, de dos maneras (listas de ticks y tramas del lector), da los valores a mano."""
    l1, l2, _, _ = hand
    wanted = hand_wanted()
    theta = Decimal(TEXT)
    series = Series(
        [Tick(i, t, Decimal(p).scaleb(-8), Decimal(1), False) for i, t, p in TICKS]
    )
    by_lists = {}
    for row in EVENT_ROWS:
        r, c, e = (
            row[f"{n}_agg_trade_id"] for n in ("reference", "confirm", "extreme")
        )
        key = (theta, c)
        if key not in wanted or (c, e - r) in OVER_BUDGET:
            continue  # el lector no conoce el presupuesto: ese evento se prueba aparte
        by_lists[key] = specref.spectrum_row(
            [p for p, _, _ in series.between(r, c)],
            [p for p, _, _ in series.between(c, e)],
            reference=scaled(row["reference_price"]),
            confirm=scaled(row["confirm_price"]),
        )
    by_reader = {
        k: v
        for k, v in specref.from_lake(l1, l2, M1, [TEXT]).items()
        if (k[1], 0) not in [(c, 0) for c, _ in OVER_BUDGET]
    }
    wanted_rest = {k: v for k, v in wanted.items() if k[1] != OVER_BUDGET[0][0]}
    for found in (by_lists, by_reader):
        assert (
            assert_rows(
                {k: dict(zip(NAMES, v, strict=True)) for k, v in found.items()},
                wanted_rest,
                HAND_TOLERANCE,
            )
            == len(BLOCKS) - 1
        )


def test_the_engine_matches_the_oracle_on_the_hand_lake(hand):
    l1, l2, _, result = hand
    got = rows_of(result.path)
    oracle = specref.from_lake(l1, l2, M1, [TEXT])
    over = OVER_BUDGET[0][0]
    compared = 0
    for key, wanted in oracle.items():
        if key[1] == over:
            continue
        assert_row(got[key], wanted, key, TOLERANCE, LOCATION_TOLERANCE)
        compared += 1
    assert compared == len(BLOCKS) - 1


def test_the_location_by_hand():
    """`LOC_X` de la sonda da 41,5/80; la energía al inicio o al final, 1/(2n) y 1 − 1/(2n); pareja,
    0,5; y sin energía, nula. El oráculo y la suma simple de Python dan lo mismo.
    """
    loc_x = [6.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, -6.0]
    assert specref.location(np.array(loc_x)) == pytest.approx(41.5 / 80, abs=1e-15)
    assert hand_location([*loc_x, *loc_x]) == pytest.approx(
        specref.location(np.array([*loc_x, *loc_x])), abs=1e-15
    )
    assert specref.location(np.array([1.0] + [0.0] * 15)) == pytest.approx(1 / 32)
    assert specref.location(np.array([0.0] * 15 + [1.0])) == pytest.approx(31 / 32)
    assert specref.location(np.full(20, -0.003)) == pytest.approx(0.5, abs=1e-15)
    assert specref.location(np.zeros(20)) is None


def test_constant_nonzero_returns_have_location_one_half_and_no_fourier(tmp_path):
    """Una fase de retornos iguales y no nulos no tiene energía tras centrar (Fourier y Haar
    nulos) pero sí ubicación: 0,5. Cada tick duplica el precio, así los retornos son `ln 2`, exactos.
    """
    prices = [1_000_000 << k for k in range(21)]  # R y 20 ticks de la confirmación
    ticks = [
        Tick(i + 1, BASE + 10 * i, Decimal(p).scaleb(-8), Decimal(1), False)
        for i, p in enumerate(prices)
    ]
    row = {"direction": 1}
    for label, k in (("reference", 0), ("confirm", 20), ("extreme", 20)):
        row[f"{label}_price"] = Decimal(prices[k]).scaleb(-8)
        row[f"{label}_time"] = BASE + 10 * k
        row[f"{label}_agg_trade_id"] = k + 1
    l1, l2, l3 = tmp_path / "l1", tmp_path / "l2", tmp_path / "l3"
    write_ticks(l1, ticks, M1, 4)
    write_l2(l2, THETA, [row], 1, M1)
    result = spectrum_native(l1, l2, l3, M1, [TEXT])[Decimal(TEXT)]
    got = rows_of(result.path)[Decimal(TEXT), 21]
    for name in NAMES[:-2]:
        assert got[name] is None, name
    assert abs(got["c_energy_loc"] - 0.5) <= 1e-12
    assert got["os_energy_loc"] is None  # overshoot vacío
    wanted = specref.from_lake(l1, l2, M1, [TEXT])[Decimal(TEXT), 21]
    assert_row(got, wanted, "retornos constantes", 1e-12)


def test_the_event_over_the_budget_is_all_null_and_warns_once(hand):
    _, _, _, result = hand
    row = rows_of(result.path)[Decimal(TEXT), OVER_BUDGET[0][0]]
    assert all(row[name] is None for name in NAMES)
    assert result.over_budget == OVER_BUDGET
    assert OVER_BUDGET[0][1] > SPECTRUM_BUDGET_TICKS
    ctx = RunContext(
        mode="monthly",
        run_id="run",
        image_version="0.4.0+test",
        series_start=M1,
        landing_root="l1",
        events_root="l2",
        summaries_root="l3",
        dq_root="dq",
        family=SPECTRUM_FAMILY,
    )
    confirm, ticks = result.over_budget[0]
    finding = findings.event_over_budget(
        ctx,
        Unit(*M1),
        theta=TEXT,
        confirm_agg_trade_id=confirm,
        ticks=ticks,
        budget=SPECTRUM_BUDGET_TICKS,
    )
    assert finding.details == {
        "theta": TEXT,
        "confirm_agg_trade_id": OVER_BUDGET[0][0],
        "ticks": OVER_BUDGET[0][1],
        "budget": 130_000_000,
        "family": "spectrum",
    }


def test_a_lower_budget_nulls_exactly_the_events_over_it(hand, tmp_path):
    l1, l2, _, baseline = hand
    base = rows_of(baseline.path)
    span = {
        r["confirm_agg_trade_id"]: r["extreme_agg_trade_id"]
        - r["reference_agg_trade_id"]
        for r in EVENT_ROWS
    }
    limit = 100
    result = spectrum_native(l1, l2, tmp_path, M1, [TEXT], budget_ticks=limit)[
        Decimal(TEXT)
    ]
    got = rows_of(result.path)
    over = {c for c, n in span.items() if n > limit}
    assert over and {c for c, _ in result.over_budget} == over
    assert dict(result.over_budget) == {c: span[c] for c in over}
    for key, row in got.items():
        if key[1] in over:
            assert all(row[n] is None for n in NAMES), key
        else:
            assert row == base[key], key
    footer = pq.read_schema(result.path).metadata
    assert footer[b"budget_ticks"] == str(limit).encode()


def test_the_same_event_is_null_below_its_ticks_and_full_above(hand, tmp_path):
    """Subir el presupuesto no cambia ningún valor: solo el evento que antes lo pasaba deja de ser
    nulo (ITSC-418). Se baja el presupuesto a los ticks de un evento y a uno menos."""
    l1, l2, _, baseline = hand
    base = rows_of(baseline.path)
    span = {
        r["confirm_agg_trade_id"]: r["extreme_agg_trade_id"]
        - r["reference_agg_trade_id"]
        for r in EVENT_ROWS
    }
    over = OVER_BUDGET[0][0]
    full = [
        c
        for c, n in span.items()
        if c != over and any(base[Decimal(TEXT), c][name] is not None for name in NAMES)
    ]
    assert full, "el fixture necesita un evento con valores"
    target = max(full, key=span.__getitem__)

    def run(budget):
        # Los dos presupuestos escriben el mismo archivo: se leen antes de la corrida siguiente.
        result = spectrum_native(l1, l2, tmp_path, M1, [TEXT], budget_ticks=budget)
        result = result[Decimal(TEXT)]
        return result, rows_of(result.path)

    (below, got_below), (at, got_at) = run(span[target] - 1), run(span[target])
    assert all(got_below[Decimal(TEXT), target][n] is None for n in NAMES)
    assert (target, span[target]) in below.over_budget
    assert (target, span[target]) not in at.over_budget
    # Con el presupuesto en el límite el evento tiene sus valores, los del archivo base, y
    # ninguna otra fila cambia: salvo las de los eventos que el presupuesto menor ya anulaba.
    assert got_at[Decimal(TEXT), target] == base[Decimal(TEXT), target]
    nulled = {c for c, _ in below.over_budget}
    for key, row in got_below.items():
        if key[1] not in nulled:
            assert row == base[key], key
    for key, row in got_at.items():
        if key[1] not in {c for c, _ in at.over_budget}:
            assert row == base[key], key


def test_the_layer_budget_is_the_engine_default():
    """`SPECTRUM_BUDGET_TICKS` (el que va al pie del archivo) y el valor por defecto del motor
    (`DEFAULT_BUDGET_TICKS` de dc_descriptors, también el del binding) son el mismo número."""
    root = Path(__file__).resolve().parents[3]
    rust = (root / "shared/dc_descriptors/src/spectrum.rs").read_text()
    match = re.search(r"pub const DEFAULT_BUDGET_TICKS: u64 = ([\d_]+);", rust)
    assert match and int(match[1].replace("_", "")) == SPECTRUM_BUDGET_TICKS
    stub = (root / "shared/dc_pyo3/dc_pyo3.pyi").read_text()
    match = re.search(r"budget_ticks: int = ([\d_]+),", stub)
    assert match and int(match[1].replace("_", "")) == SPECTRUM_BUDGET_TICKS


def permutation_ids():
    return ["forward", "reversed", "swapped", "shuffled"]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("lot", [None, 1, 2, 5])
@pytest.mark.parametrize("order", permutation_ids())
def test_the_hand_rows_do_not_depend_on_threads_lots_or_push_order(
    hand, tmp_path, threads, lot, order
):
    """Con row groups de L1 de 4 ticks y tramos de fronteras de 1, 2 y 5 eventos (los eventos
    cruzan row groups y tramos), el archivo es el mismo bit a bit.
    """
    l1, l2, _, baseline = hand
    for seed in range(2):
        result = spectrum_native(
            l1, l2, tmp_path / f"{seed}", M1, [TEXT], threads=threads, lot=lot,
            order=order, seed=seed,
        )[Decimal(TEXT)]  # fmt: skip
        assert result.content_hash == baseline.content_hash
        assert result.over_budget == OVER_BUDGET
        assert assert_rows(rows_of(result.path), hand_wanted(), HAND_TOLERANCE) == len(
            BLOCKS
        )


def assert_skeleton_and_footer(path: str, summaries: str, budget: int | None = None):
    """Mismas filas, orden y row groups que `summaries.parquet`; unión 1:1 por clave; esquema del
    contrato y metadatos del pie (`state_version` y los cuatro parámetros de F).
    """
    mine, theirs = pq.ParquetFile(path), pq.ParquetFile(summaries)
    sizes = [
        theirs.metadata.row_group(i).num_rows for i in range(theirs.num_row_groups)
    ]
    assert [
        mine.metadata.row_group(i).num_rows for i in range(mine.num_row_groups)
    ] == sizes
    keys = ["theta", "confirm_agg_trade_id"]
    assert mine.read(columns=keys).equals(theirs.read(columns=keys))
    assert mine.schema_arrow.remove_metadata().equals(SPECTRUM_SCHEMA)
    meta = {
        k: v for k, v in pq.read_schema(path).metadata.items() if k != b"ARROW:schema"
    }
    assert meta == {
        b"state_version": b"1.2.0",
        b"n_min": b"16",
        b"bands": b"7",
        b"window": b"hann",
        b"budget_ticks": str(budget or SPECTRUM_BUDGET_TICKS).encode(),
    }


def test_the_orchestrator_would_write_the_footer_of_the_hand_file(hand):
    """`summarize._OUTPUT` (lo que recibe `FamilyWriter`) es lo que el archivo lleva en el pie."""
    _, _, _, result = hand
    schema, version, metadata = summarize._OUTPUT[SPECTRUM_FAMILY]
    assert schema is SPECTRUM_SCHEMA and version == SPECTRUM_STATE_VERSION
    assert dict(metadata) == {
        "n_min": "16",
        "bands": "7",
        "window": "hann",
        "budget_ticks": "130000000",
    }
    footer = pq.read_schema(result.path).metadata
    for key, value in {"state_version": version, **metadata}.items():
        assert footer[key.encode()] == value.encode()


def test_the_hand_file_keeps_the_skeleton_and_the_footer(hand):
    _, _, l3, result = hand
    summaries = spectrum_file(l3, TEXT).replace("spectrum.parquet", "summaries.parquet")
    assert_skeleton_and_footer(result.path, summaries)


# ---------------------------------------------------------------------------------------
# El lago de pruebas: ticks reales con huecos de ids, dos meses, cinco θ y row groups de 50
# ticks. S corre primero (la familia se escribe contra su `summaries.parquet`).


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    ticks = fixture_ticks(seed=11, drop=0.12)
    lake = build_lake(
        tmp_path_factory.mktemp("spectrum"),
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
        month: spectrum_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
        for month in lake_.months
    }


def lake_files(lake_, month, theta: Decimal):
    text = f"{theta:.8f}"
    path = spectrum_file(lake_.l3, text, month)
    return path, path.replace("spectrum.parquet", "summaries.parquet")


def phase_counts(lake_, baseline_):
    """`(fase, retornos) → filas` del lago, con los retornos de `summaries.parquet`."""
    out = {}
    for month, results in baseline_.items():
        for theta in results:
            _, summaries = lake_files(lake_, month, theta)
            for r in pq.read_table(summaries, schema=SUMMARIES_SCHEMA).to_pylist():
                out[theta, r["confirm_agg_trade_id"]] = (r["c_ticks"], r["os_ticks"])
    return out


def test_the_lake_has_the_edge_cases(lake, baseline):
    """El lago trae de verdad cruces de row groups, overshoot vacío y fases de cada tamaño."""
    lake_, _ = lake
    crossing = empty = short = 0
    sizes = set()
    group_of = {}
    for month in lake_.months:
        for i, tick in enumerate(lake_.ticks[month]):
            group_of[tick.id] = (month, i // ROW_GROUP)
    counts = phase_counts(lake_, baseline)
    for month, results in baseline.items():
        for theta, result in results.items():
            rows = rows_of(result.path)
            for event in lake_.events[int(theta.scaleb(8)), month]:
                key = (theta, event["confirm_agg_trade_id"])
                row = rows[key]
                crossing += (
                    group_of[event["reference_agg_trade_id"]]
                    != group_of[event["extreme_agg_trade_id"]]
                )
                c_ticks, os_ticks = counts[key]
                empty += os_ticks == 0
                short += 0 < c_ticks < SPECTRUM_N_MIN
                present = sum(row[n] is not None for n in NAMES[:7])
                sizes.add(present)
    assert crossing > 20 and empty > 20 and short > 20, (crossing, empty, short)
    # Hay filas con 0 (fase corta), 4, 5, 6 y 7 bandas presentes en la confirmación.
    assert sizes >= {0, 4, 5, 6, 7}, sizes


def is_flat(phase: str, register: str, event: dict, series: Series, n: int) -> bool:
    """Si la serie del registro no tiene energía: todos sus retornos iguales. Se decide con los
    precios de los ticks (`R < id ≤ C` o `C < id ≤ E`) y el ancla, sin motor ni oráculo; con
    precios enteros, retornos iguales son precios iguales. Haar mira solo los últimos
    `2^⌊log₂ n⌋` retornos.
    """
    low, high = ("reference", "confirm") if phase == "c" else ("confirm", "extreme")
    prices = [scaled(event[f"{low}_price"])]
    prices += [
        p
        for p, _, _ in series.between(
            event[f"{low}_agg_trade_id"], event[f"{high}_agg_trade_id"]
        )
    ]
    assert len(prices) == n + 1
    if register == "haar":
        prices = prices[-((1 << (n.bit_length() - 1)) + 1) :]
    return len(set(prices)) == 1


def test_the_rows_hold_the_invariants(lake, baseline):
    """Cada fracción en [0, 1]; las bandas no nulas de una fase suman 1; nulos exactamente donde
    la regla lo dice (fase de menos de 16 retornos, overshoot vacío, banda o nivel sin bins, o
    serie sin energía); ni ±inf ni NaN. Sin motor ni oráculo: con `summaries.parquet` y los ticks.
    """
    lake_, _ = lake
    series = Series(lake_.all_ticks())
    counts = phase_counts(lake_, baseline)
    checked = flat = 0
    for month, results in baseline.items():
        for theta, result in results.items():
            events = {
                e["confirm_agg_trade_id"]: e
                for e in lake_.events[int(theta.scaleb(8)), month]
            }
            for key, row in rows_of(result.path).items():
                for phase, n in zip(("c", "os"), counts[key], strict=True):
                    for register, unit in (("fourier", "band"), ("haar", "level")):
                        values = [
                            row[f"{phase}_{register}_{unit}_{b}"] for b in range(1, 8)
                        ]
                        assert all(v is None or math.isfinite(v) for v in values)
                        present = [v for v in values if v is not None]
                        if not present:
                            # Fase corta, overshoot vacío o serie sin energía.
                            if n >= SPECTRUM_N_MIN:
                                flat += 1
                                assert is_flat(phase, register, events[key[1]], series, n), (
                                    key, phase, register,
                                )  # fmt: skip
                            continue
                        assert n >= SPECTRUM_N_MIN, (key, phase, register, n)
                        assert not is_flat(phase, register, events[key[1]], series, n)
                        assert len(present) == bands_present(n), (
                            key,
                            phase,
                            register,
                            n,
                        )
                        # Las bandas presentes van primero: las nulas son las más lentas.
                        assert values[: len(present)] == present
                        assert all(0.0 <= v <= 1.0 for v in present), (key, values)
                        assert abs(sum(present) - 1.0) <= 1e-9, (key, values)
                    # La ubicación: nula con menos de 16 retornos, con overshoot vacío o con todos
                    # los retornos en cero (precios iguales, ancla incluida); si no, en (0, 1).
                    loc = row[f"{phase}_energy_loc"]
                    zero_energy = n >= SPECTRUM_N_MIN and is_flat(
                        phase, "fourier", events[key[1]], series, n
                    )
                    if n < SPECTRUM_N_MIN or zero_energy:
                        assert loc is None, (key, phase, loc)
                    else:
                        assert loc is not None and 0.0 < loc < 1.0, (key, phase, loc)
                checked += 1
    assert checked == sum(len(rows) for rows in lake_.events.values())


def test_the_engine_matches_the_oracle_over_the_reader_frames(lake, baseline):
    """Todos los θ y meses del lago: los 30 valores del motor son los del oráculo, que lee cada
    evento con el lector de `dc_frames` (1e-9, nulos exactos).
    """
    lake_, _ = lake
    compared = 0
    for month, results in baseline.items():
        wanted = specref.from_lake(lake_.l1, lake_.l2, month, lake_.theta_texts)
        got = {k: v for r in results.values() for k, v in rows_of(r.path).items()}
        compared += assert_rows(
            got, {k: tuple(v) for k, v in wanted.items()}, TOLERANCE, LOCATION_TOLERANCE
        )
    assert compared == sum(len(rows) for rows in lake_.events.values())


def test_the_skeleton_and_the_footer_are_those_of_summaries(lake, baseline):
    lake_, _ = lake
    for month, results in baseline.items():
        for theta, result in results.items():
            path, summaries = lake_files(lake_, month, theta)
            assert result.path == path
            assert_skeleton_and_footer(path, summaries)
            assert result.rows == pq.ParquetFile(path).metadata.num_rows
            table = pq.read_table(path, schema=SPECTRUM_SCHEMA)
            assert content_hash(table) == result.content_hash
            assert result.over_budget == []


def test_s_t_and_a_do_not_change(lake, baseline):
    """Escribir la familia F no toca `summaries.parquet`, `path.parquet` ni `fit.parquet`: sus
    `content_hash` son los mismos antes y después de correr F.
    """
    lake_, summaries = lake
    month = lake_.months[-1]
    first_path = path_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    first_fit = fit_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    spectrum_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    again_path = path_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    again_fit = fit_native(lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts)
    for theta in first_path:
        assert again_path[theta].content_hash == first_path[theta].content_hash
        assert again_fit[theta].content_hash == first_fit[theta].content_hash
    for month_, results in summaries.items():
        for theta, result in results.items():
            _, path = lake_files(lake_, month_, theta)
            table = pq.read_table(path, schema=SUMMARIES_SCHEMA)
            assert content_hash(table) == result.content_hash


def test_a_repeated_run_gives_the_same_hash_and_rows(lake, baseline):
    lake_, _ = lake
    for _ in range(3):
        for month, results in baseline.items():
            again = spectrum_native(
                lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts
            )
            for theta, result in again.items():
                assert result.content_hash == results[theta].content_hash


def assert_same_as_baseline(lake_, baseline_, **kwargs) -> None:
    for month, results in baseline_.items():
        got = spectrum_native(
            lake_.l1, lake_.l2, lake_.l3, month, lake_.theta_texts, **kwargs
        )
        for theta, result in got.items():
            assert result.content_hash == results[theta].content_hash, (month, theta)
            assert result.rows == results[theta].rows
            table = pq.read_table(result.path, schema=SPECTRUM_SCHEMA)
            assert content_hash(table) == result.content_hash


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("lot", [1, 5, 10, 25, 100])
def test_boundaries_in_lots_and_the_thread_count_change_nothing(
    lake, baseline, threads, lot
):
    """Las fronteras en tramos de 1, 5, 10, 25 y 100 eventos, entregadas justo antes del row group
    que las necesita, dan el mismo archivo que todas antes del primer `push`, con 1 y 4 hilos.
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


def test_a_lower_budget_on_the_lake_nulls_only_the_events_over_it(
    lake, baseline, tmp_path
):
    lake_, _ = lake
    month = lake_.months[0]
    limit = 60
    got = spectrum_native(
        lake_.l1, lake_.l2, tmp_path, month, lake_.theta_texts, budget_ticks=limit
    )
    seen = 0
    for theta, result in got.items():
        base = rows_of(baseline[month][theta].path)
        spans = {
            e["confirm_agg_trade_id"]: e["extreme_agg_trade_id"]
            - e["reference_agg_trade_id"]
            for e in lake_.events[int(theta.scaleb(8)), month]
        }
        over = {c for c, n in spans.items() if n > limit}
        assert dict(result.over_budget) == {c: spans[c] for c in over}
        for key, row in rows_of(result.path).items():
            if key[1] in over:
                assert all(row[n] is None for n in NAMES)
            else:
                assert row == base[key]
        seen += len(over)
    assert seen > 10
