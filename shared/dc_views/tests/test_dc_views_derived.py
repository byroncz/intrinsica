"""Vista `derived` (familia D) sobre un fixture sintético con valores calculados a mano."""

from decimal import Decimal
from importlib.resources import files

import duckdb
import pyarrow as pa
import pytest
from dc_views import DERIVED_VERSION, load_derived

D18 = pa.decimal128(18, 8)
D9 = pa.decimal128(9, 8)

REL = 1e-12

COLUMNS_D = [
    "c_instant_bias",
    "c_flow_imbalance",
    "os_flow_imbalance",
    "c_tick_rate",
    "os_tick_rate",
    "c_quantity_rate",
    "os_quantity_rate",
    "c_mean_trade",
    "os_mean_trade",
    "c_range",
    "os_range",
    "c_vwap_distance",
    "os_vwap_distance",
    "c_speed",
    "os_speed",
    "ev_speed",
    "atomic_sweep",
    "empty_overshoot",
    "provider_gap",
]
FLAGS = ["atomic_sweep", "empty_overshoot", "provider_gap"]

SUMMARIES_SCHEMA = pa.schema(
    [
        ("theta", D9),
        ("confirm_agg_trade_id", pa.int64()),
        ("c_ticks", pa.int64()),
        ("c_quantity_buy", D18),
        ("c_quantity_sell", D18),
        ("c_price_max", D18),
        ("c_price_min", D18),
        ("c_duration", pa.int64()),
        ("c_vwap", D18),
        ("os_ticks", pa.int64()),
        ("os_quantity_buy", D18),
        ("os_quantity_sell", D18),
        ("os_price_max", D18),
        ("os_price_min", D18),
        ("os_duration", pa.int64()),
        ("os_vwap", D18),
        ("confirm_instant_ticks", pa.int64()),
        ("confirm_instant_price_best", D18),
        ("confirm_instant_price_worst", D18),
    ]
)
EVENTS_SCHEMA = pa.schema(
    [
        ("reference_price", D18),
        ("reference_time", pa.int64()),
        ("reference_agg_trade_id", pa.int64()),
        ("confirm_price", D18),
        ("confirm_time", pa.int64()),
        ("confirm_agg_trade_id", pa.int64()),
        ("extreme_price", D18),
        ("extreme_time", pa.int64()),
        ("extreme_agg_trade_id", pa.int64()),
        ("direction", pa.int8()),
        ("theta", D9),
    ]
)


def fila(**valores):
    return valores


# Cada fila: (summaries, events). Los precios y las cantidades se escriben como
# cadenas decimales; los tiempos, en µs.
ROWS = {
    # A: upturn con hueco ninguno, overshoot de 80 ticks.
    "A": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=1000,
            c_ticks=100,
            c_quantity_buy="6",
            c_quantity_sell="2",
            c_price_max="101.5",
            c_price_min="100.25",
            c_duration=2_000_000,
            c_vwap="100.5",
            os_ticks=80,
            os_quantity_buy="5",
            os_quantity_sell="3",
            os_price_max="103",
            os_price_min="101",
            os_duration=5_000_000,
            os_vwap="102",
            confirm_instant_ticks=2,
            confirm_instant_price_best="101.5",
            confirm_instant_price_worst="101",
        ),
        fila(
            reference_price="100",
            reference_time=0,
            reference_agg_trade_id=900,
            confirm_price="101",
            confirm_time=2_000_000,
            confirm_agg_trade_id=1000,
            extreme_price="103",
            extreme_time=8_000_000,
            extreme_agg_trade_id=1080,
            direction=1,
            theta="0.01",
        ),
    ),
    # B: downturn.
    "B": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=2000,
            c_ticks=100,
            c_quantity_buy="1",
            c_quantity_sell="3",
            c_price_max="199.75",
            c_price_min="197.5",
            c_duration=500_000,
            c_vwap="198.5",
            os_ticks=50,
            os_quantity_buy="3",
            os_quantity_sell="5",
            os_price_max="198",
            os_price_min="195",
            os_duration=1_500_000,
            os_vwap="196.5",
            confirm_instant_ticks=2,
            confirm_instant_price_best="197.5",
            confirm_instant_price_worst="198",
        ),
        fila(
            reference_price="200",
            reference_time=10_000_000,
            reference_agg_trade_id=1900,
            confirm_price="198",
            confirm_time=11_000_000,
            confirm_agg_trade_id=2000,
            extreme_price="195",
            extreme_time=13_000_000,
            extreme_agg_trade_id=2050,
            direction=-1,
            theta="0.01",
        ),
    ),
    # C: cruce atómico (c_duration = 0), upturn.
    "C": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=3000,
            c_ticks=4,
            c_quantity_buy="2",
            c_quantity_sell="2",
            c_price_max="51",
            c_price_min="50.25",
            c_duration=0,
            c_vwap="50.5",
            os_ticks=6,
            os_quantity_buy="3",
            os_quantity_sell="1",
            os_price_max="52",
            os_price_min="51",
            os_duration=3_000_000,
            os_vwap="51.5",
            confirm_instant_ticks=4,
            confirm_instant_price_best="51",
            confirm_instant_price_worst="50.5",
        ),
        fila(
            reference_price="50",
            reference_time=20_000_000,
            reference_agg_trade_id=2996,
            confirm_price="50.75",
            confirm_time=20_500_000,
            confirm_agg_trade_id=3000,
            extreme_price="52",
            extreme_time=24_000_000,
            extreme_agg_trade_id=3006,
            direction=1,
            theta="0.01",
        ),
    ),
    # D: overshoot vacío en un downturn (extreme_* = confirm_*).
    "D": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=4000,
            c_ticks=100,
            c_quantity_buy="5",
            c_quantity_sell="3",
            c_price_max="79.9",
            c_price_min="78.5",
            c_duration=4_000_000,
            c_vwap="79.25",
            os_ticks=0,
            os_quantity_buy="0",
            os_quantity_sell="0",
            os_price_max=None,
            os_price_min=None,
            os_duration=None,
            os_vwap=None,
            confirm_instant_ticks=2,
            confirm_instant_price_best="78.5",
            confirm_instant_price_worst="79.5",
        ),
        fila(
            reference_price="80",
            reference_time=30_000_000,
            reference_agg_trade_id=3900,
            confirm_price="79",
            confirm_time=34_000_000,
            confirm_agg_trade_id=4000,
            extreme_price="79",
            extreme_time=34_000_000,
            extreme_agg_trade_id=4000,
            direction=-1,
            theta="0.01",
        ),
    ),
    # E: hueco del proveedor en la confirmación (150 ticks de 200 ids).
    "E": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=5000,
            c_ticks=150,
            c_quantity_buy="10",
            c_quantity_sell="0",
            c_price_max="121.8",
            c_price_min="120.5",
            c_duration=5_000_000,
            c_vwap="121",
            os_ticks=100,
            os_quantity_buy="0",
            os_quantity_sell="2",
            os_price_max="123",
            os_price_min="121.2",
            os_duration=2_000_000,
            os_vwap="122",
            confirm_instant_ticks=3,
            confirm_instant_price_best="121.8",
            confirm_instant_price_worst="121.5",
        ),
        fila(
            reference_price="120",
            reference_time=40_000_000,
            reference_agg_trade_id=4800,
            confirm_price="121.5",
            confirm_time=45_000_000,
            confirm_agg_trade_id=5000,
            extreme_price="123",
            extreme_time=47_000_000,
            extreme_agg_trade_id=5100,
            direction=1,
            theta="0.01",
        ),
    ),
    # F: extreme_time = reference_time (los tres puntos en el mismo µs) y
    # overshoot vacío.
    "F": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=6000,
            c_ticks=100,
            c_quantity_buy="3",
            c_quantity_sell="1",
            c_price_max="61.5",
            c_price_min="60.5",
            c_duration=0,
            c_vwap="60.9",
            os_ticks=0,
            os_quantity_buy="0",
            os_quantity_sell="0",
            os_price_max=None,
            os_price_min=None,
            os_duration=None,
            os_vwap=None,
            confirm_instant_ticks=100,
            confirm_instant_price_best="61.5",
            confirm_instant_price_worst="60.5",
        ),
        fila(
            reference_price="60",
            reference_time=50_000_000,
            reference_agg_trade_id=5900,
            confirm_price="61",
            confirm_time=50_000_000,
            confirm_agg_trade_id=6000,
            extreme_price="61",
            extreme_time=50_000_000,
            extreme_agg_trade_id=6000,
            direction=1,
            theta="0.01",
        ),
    ),
    # G: otro θ con el mismo confirm_agg_trade_id que A (la unión usa los dos).
    "G": (
        fila(
            theta="0.02",
            confirm_agg_trade_id=1000,
            c_ticks=100,
            c_quantity_buy="2",
            c_quantity_sell="2",
            c_price_max="10.3",
            c_price_min="10.05",
            c_duration=1_000_000,
            c_vwap="10.2",
            os_ticks=10,
            os_quantity_buy="3",
            os_quantity_sell="1",
            os_price_max="10.5",
            os_price_min="10.1",
            os_duration=2_000_000,
            os_vwap="10.3",
            confirm_instant_ticks=2,
            confirm_instant_price_best="10.3",
            confirm_instant_price_worst="10.25",
        ),
        fila(
            reference_price="10",
            reference_time=60_000_000,
            reference_agg_trade_id=900,
            confirm_price="10.25",
            confirm_time=61_000_000,
            confirm_agg_trade_id=1000,
            extreme_price="10.5",
            extreme_time=63_000_000,
            extreme_agg_trade_id=1010,
            direction=1,
            theta="0.02",
        ),
    ),
    # H: overshoot de un tick (os_duration = 0 sin ser vacío): las tres razones
    # con os_duration en el denominador salen nulas y las demás os_* no.
    "H": (
        fila(
            theta="0.01",
            confirm_agg_trade_id=7000,
            c_ticks=100,
            c_quantity_buy="3",
            c_quantity_sell="1",
            c_price_max="70.8",
            c_price_min="70",
            c_duration=1_000_000,
            c_vwap="70.5",
            os_ticks=1,
            os_quantity_buy="2",
            os_quantity_sell="0",
            os_price_max="70.9",
            os_price_min="70.9",
            os_duration=0,
            os_vwap="70.9",
            confirm_instant_ticks=2,
            confirm_instant_price_best="70.8",
            confirm_instant_price_worst="70.7",
        ),
        fila(
            reference_price="70",
            reference_time=70_000_000,
            reference_agg_trade_id=6900,
            confirm_price="70.7",
            confirm_time=71_000_000,
            confirm_agg_trade_id=7000,
            extreme_price="70.9",
            extreme_time=71_500_000,
            extreme_agg_trade_id=7001,
            direction=1,
            theta="0.01",
        ),
    ),
}

# Los 19 valores de cada fila, en el orden de COLUMNS_D, calculados a mano.
# None es nulo. Entre paréntesis, la cuenta.
EXPECTED = {
    "A": [
        0.005,  # (101.5 - 101) / 100
        0.5,  # (6 - 2) / 8
        0.25,  # (5 - 3) / 8
        50.0,  # 100 / 2 s
        16.0,  # 80 / 5 s
        4.0,  # 8 / 2 s
        1.6,  # 8 / 5 s
        0.08,  # 8 / 100
        0.1,  # 8 / 80
        0.0125,  # (101.5 - 100.25) / 100
        0.02,  # (103 - 101) / 100
        -0.005,  # (100.5 - 101) / 100
        0.01,  # (102 - 101) / 100
        0.005,  # (101 - 100) / (100 * 2 s)
        0.004,  # (103 - 101) / (100 * 5 s)
        0.00375,  # (103 - 100) / (100 * 8 s)
        False,
        False,
        False,
    ],
    "B": [
        0.0025,  # -1 * (197.5 - 198) / 200
        -0.5,  # (1 - 3) / 4
        -0.25,  # (3 - 5) / 8
        200.0,  # 100 / 0.5 s
        100 / 3,  # 50 / 1.5 s
        8.0,  # 4 / 0.5 s
        16 / 3,  # 8 / 1.5 s
        0.04,  # 4 / 100
        0.16,  # 8 / 50
        0.01125,  # (199.75 - 197.5) / 200
        0.015,  # (198 - 195) / 200
        0.0025,  # (198.5 - 198) / 200
        -0.0075,  # (196.5 - 198) / 200
        -0.02,  # (198 - 200) / (200 * 0.5 s)
        -0.01,  # (195 - 198) / (200 * 1.5 s)
        -5 / 600,  # (195 - 200) / (200 * 3 s)
        False,
        False,
        False,
    ],
    "C": [
        0.005,  # (51 - 50.75) / 50
        0.0,  # (2 - 2) / 4
        0.5,  # (3 - 1) / 4
        None,  # c_duration = 0
        2.0,  # 6 / 3 s
        None,  # c_duration = 0
        4 / 3,  # 4 / 3 s
        1.0,  # 4 / 4
        2 / 3,  # 4 / 6
        0.015,  # (51 - 50.25) / 50
        0.02,  # (52 - 51) / 50
        -0.005,  # (50.5 - 50.75) / 50
        0.015,  # (51.5 - 50.75) / 50
        None,  # c_duration = 0
        1 / 120,  # (52 - 50.75) / (50 * 3 s)
        0.01,  # (52 - 50) / (50 * 4 s)
        True,
        False,
        False,
    ],
    "D": [
        0.00625,  # -1 * (78.5 - 79) / 80
        0.25,  # (5 - 3) / 8
        None,  # os_quantity_buy + os_quantity_sell = 0
        25.0,  # 100 / 4 s
        None,  # os_duration nulo
        2.0,  # 8 / 4 s
        None,  # os_duration nulo
        0.08,  # 8 / 100
        None,  # os_ticks = 0
        0.0175,  # (79.9 - 78.5) / 80
        None,  # os_price_* nulos
        0.003125,  # (79.25 - 79) / 80
        None,  # os_vwap nulo
        -0.003125,  # (79 - 80) / (80 * 4 s)
        None,  # os_duration nulo
        -0.003125,  # (79 - 80) / (80 * 4 s)
        False,
        True,
        False,
    ],
    "E": [
        0.0025,  # (121.8 - 121.5) / 120
        1.0,  # (10 - 0) / 10
        -1.0,  # (0 - 2) / 2
        30.0,  # 150 / 5 s
        50.0,  # 100 / 2 s
        2.0,  # 10 / 5 s
        1.0,  # 2 / 2 s
        1 / 15,  # 10 / 150
        0.02,  # 2 / 100
        1.3 / 120,  # (121.8 - 120.5) / 120
        0.015,  # (123 - 121.2) / 120
        -1 / 240,  # (121 - 121.5) / 120
        1 / 240,  # (122 - 121.5) / 120
        0.0025,  # (121.5 - 120) / (120 * 5 s)
        0.00625,  # (123 - 121.5) / (120 * 2 s)
        1 / 280,  # (123 - 120) / (120 * 7 s)
        False,
        False,
        True,  # 150 < 5000 - 4800
    ],
    "F": [
        1 / 120,  # (61.5 - 61) / 60
        0.5,  # (3 - 1) / 4
        None,  # os_quantity_buy + os_quantity_sell = 0
        None,  # c_duration = 0
        None,  # os_duration nulo
        None,  # c_duration = 0
        None,  # os_duration nulo
        0.04,  # 4 / 100
        None,  # os_ticks = 0
        1 / 60,  # (61.5 - 60.5) / 60
        None,  # os_price_* nulos
        -0.1 / 60,  # (60.9 - 61) / 60
        None,  # os_vwap nulo
        None,  # c_duration = 0
        None,  # os_duration nulo
        None,  # extreme_time = reference_time
        True,
        True,
        False,
    ],
    "G": [
        0.005,  # (10.3 - 10.25) / 10
        0.0,  # (2 - 2) / 4
        0.5,  # (3 - 1) / 4
        100.0,  # 100 / 1 s
        5.0,  # 10 / 2 s
        4.0,  # 4 / 1 s
        2.0,  # 4 / 2 s
        0.04,  # 4 / 100
        0.4,  # 4 / 10
        0.025,  # (10.3 - 10.05) / 10
        0.04,  # (10.5 - 10.1) / 10
        -0.005,  # (10.2 - 10.25) / 10
        0.005,  # (10.3 - 10.25) / 10
        0.025,  # (10.25 - 10) / (10 * 1 s)
        0.0125,  # (10.5 - 10.25) / (10 * 2 s)
        1 / 60,  # (10.5 - 10) / (10 * 3 s)
        False,
        False,
        False,
    ],
    "H": [
        1 / 700,  # (70.8 - 70.7) / 70
        0.5,  # (3 - 1) / 4
        1.0,  # (2 - 0) / 2
        100.0,  # 100 / 1 s
        None,  # os_duration = 0
        4.0,  # 4 / 1 s
        None,  # os_duration = 0
        0.04,  # 4 / 100
        2.0,  # 2 / 1
        2 / 175,  # (70.8 - 70) / 70
        0.0,  # (70.9 - 70.9) / 70
        -1 / 350,  # (70.5 - 70.7) / 70
        1 / 350,  # (70.9 - 70.7) / 70
        0.01,  # (70.7 - 70) / (70 * 1 s)
        None,  # os_duration = 0
        3 / 350,  # (70.9 - 70) / (70 * 1.5 s)
        False,
        False,
        False,
    ],
}


def _decimal(value):
    return None if value is None else Decimal(value)


def _table(schema, rows):
    columns = {}
    for field in schema:
        values = [row[field.name] for row in rows]
        if pa.types.is_decimal(field.type):
            values = [_decimal(v) for v in values]
        columns[field.name] = pa.array(values, type=field.type)
    return pa.table(columns, schema=schema)


@pytest.fixture
def summaries():
    return _table(SUMMARIES_SCHEMA, [s for s, _ in ROWS.values()])


@pytest.fixture
def events():
    return _table(EVENTS_SCHEMA, [e for _, e in ROWS.values()])


@pytest.fixture
def con(summaries, events):
    con = duckdb.connect()
    load_derived(con, summaries, events)
    yield con
    con.close()


def _key(summary):
    return (Decimal(summary["theta"]), summary["confirm_agg_trade_id"])


def _rows_by_key(con):
    names = [d[0] for d in con.execute("SELECT * FROM derived").description]
    return {
        (row[0], row[1]): dict(zip(names, row, strict=True))
        for row in con.execute("SELECT * FROM derived").fetchall()
    }


def test_header_matches_derived_version():
    sql = files("dc_views").joinpath("derived.sql").read_text(encoding="utf-8")
    assert sql.splitlines()[0] == f"-- derived_version: {DERIVED_VERSION}"
    assert DERIVED_VERSION == "1.0.0"


def test_load_derived_returns_version(summaries, events):
    con = duckdb.connect()
    assert load_derived(con, summaries, events) == DERIVED_VERSION


def test_names_order_and_types(con):
    described = con.execute("DESCRIBE derived").fetchall()
    got = [(name, dtype) for name, dtype, *_ in described]
    expected = (
        [("theta", "DECIMAL(9,8)"), ("confirm_agg_trade_id", "BIGINT")]
        + [(name, "DOUBLE") for name in COLUMNS_D[:16]]
        + [(name, "BOOLEAN") for name in FLAGS]
    )
    assert got == expected


def test_values_match_hand_computed(con):
    got = _rows_by_key(con)
    assert len(got) == len(ROWS)
    for name, (summary, _) in ROWS.items():
        row = got[_key(summary)]
        for column, want in zip(COLUMNS_D, EXPECTED[name], strict=True):
            have = row[column]
            if want is None:
                assert have is None, f"{name}.{column}: esperaba nulo, dio {have}"
            elif isinstance(want, bool):
                assert have is want, f"{name}.{column}: esperaba {want}, dio {have}"
            else:
                assert have == pytest.approx(want, rel=REL), f"{name}.{column}"


def test_extra_summaries_columns_pass_through(summaries, events):
    n = summaries.num_rows
    extended = summaries.append_column("year", pa.array([2026] * n, pa.int32()))
    extended = extended.append_column("month", pa.array([9] * n, pa.int32()))
    con = duckdb.connect()
    load_derived(con, extended, events)
    names = [d[0] for d in con.execute("SELECT * FROM derived").description]
    assert names == ["theta", "confirm_agg_trade_id", "year", "month"] + COLUMNS_D
    assert con.execute("SELECT DISTINCT year, month FROM derived").fetchall() == [
        (2026, 9)
    ]


def test_load_twice_gives_same_rows(summaries, events):
    con = duckdb.connect()
    load_derived(con, summaries, events)
    first = con.execute("SELECT * FROM derived ORDER BY ALL").fetchall()
    load_derived(con, summaries, events)
    second = con.execute("SELECT * FROM derived ORDER BY ALL").fetchall()
    assert first == second


def test_two_runs_on_same_input_give_same_rows(summaries, events):
    runs = []
    for _ in range(2):
        con = duckdb.connect()
        load_derived(con, summaries, events)
        runs.append(con.execute("SELECT * FROM derived ORDER BY ALL").fetchall())
    assert runs[0] == runs[1]


def test_one_row_per_summary_and_one_to_one_join(con, summaries):
    assert con.execute("SELECT count(*) FROM derived").fetchone()[0] == (
        summaries.num_rows
    )
    assert (
        con.execute(
            "SELECT count(*) FROM (SELECT DISTINCT theta, confirm_agg_trade_id "
            "FROM derived)"
        ).fetchone()[0]
        == summaries.num_rows
    )


def test_ratio_invariants(con):
    inputs = {_key(s): (s, e) for s, e in ROWS.values()}
    for key, row in _rows_by_key(con).items():
        summary, event = inputs[key]
        theta = float(row["theta"])
        assert abs(row["c_flow_imbalance"]) <= 1 + REL
        if row["os_flow_imbalance"] is not None:
            assert abs(row["os_flow_imbalance"]) <= 1 + REL
        # c_range >= theta no es invariante: el tick de referencia no pertenece a la fase
        # (TRD-L3 §6.1 (ADR-L3-01), consecuencia 3). Lo que vale es la cota de
        # confirmación, que D no puede calcular porque no trae c_price_max ni c_price_min.
        assert row["c_range"] >= 0
        assert row["c_instant_bias"] >= -REL
        reference = float(event["reference_price"])
        if event["direction"] == 1:
            bound = (float(summary["c_price_max"]) - reference) / reference
        else:
            bound = (reference - float(summary["c_price_min"])) / reference
        assert bound >= theta * (1 - REL), key


def test_flag_invariants(con):
    durations = {_key(s): s["c_duration"] for s, _ in ROWS.values()}
    os_columns = [
        "os_flow_imbalance",
        "os_tick_rate",
        "os_quantity_rate",
        "os_mean_trade",
        "os_range",
        "os_vwap_distance",
        "os_speed",
    ]
    for key, row in _rows_by_key(con).items():
        atomic = durations[key] == 0
        assert row["atomic_sweep"] is atomic
        assert (row["c_tick_rate"] is None) is atomic
        assert row["empty_overshoot"] is all(row[c] is None for c in os_columns)
        for flag in FLAGS:
            assert row[flag] is not None
