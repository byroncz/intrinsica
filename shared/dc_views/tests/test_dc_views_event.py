"""M (tmv, osv, duration) y N (net_duration) de la vista `derived`, con valores a mano."""

import re
from importlib.resources import files

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from dc_views import SILENCE_US, load_derived
from test_dc_views_derived import EVENTS_SCHEMA, SUMMARIES_SCHEMA, _table

S = 1_000_000  # un segundo, en µs
R_TIME = (
    100 * S
)  # el tiempo de R no es 0: un resultado que ignore reference_time no pasa
THETA = "0.01"

PATH_SCHEMA = pa.schema(
    [
        ("theta", pa.decimal128(9, 8)),
        ("confirm_agg_trade_id", pa.int64()),
        ("c_idle_fraction", pa.float64()),
        ("os_idle_fraction", pa.float64()),
    ]
)


def _event(
    ident,
    prices,
    direction,
    c_s,
    os_s,
    gap_r_s,
    gap_c_s,
    idle=None,
    ticks=10,
):
    """Una fila de `summaries` y `events` armada con los tiempos de cada tramo (s).

    `prices` es `(ref, conf, ext)`. Las fases son (R, C] y (C, E]: el primer tick de la
    confirmación llega `gap_r_s` después de R, y el del overshoot, `gap_c_s` después de
    C. `os_s = None` es un overshoot vacío: `extreme_*` es igual a `confirm_*`.
    `idle` es `(c_idle_fraction, os_idle_fraction)` o `None` si no hay fila de path.
    """
    ref, conf, ext = prices
    confirm_time = R_TIME + (gap_r_s + c_s) * S
    extreme_time = confirm_time if os_s is None else confirm_time + (gap_c_s + os_s) * S
    empty = os_s is None
    summary = {
        "theta": THETA,
        "confirm_agg_trade_id": ident,
        "c_ticks": ticks,
        "c_quantity_buy": "1",
        "c_quantity_sell": "1",
        "c_price_max": "1",
        "c_price_min": "1",
        "c_duration": c_s * S,
        "c_vwap": "1",
        "os_ticks": 0 if empty else ticks,
        "os_quantity_buy": "0" if empty else "1",
        "os_quantity_sell": "0" if empty else "1",
        "os_price_max": None if empty else "1",
        "os_price_min": None if empty else "1",
        "os_duration": None if empty else os_s * S,
        "os_vwap": None if empty else "1",
        "confirm_instant_ticks": 1,
        "confirm_instant_price_best": "1",
        "confirm_instant_price_worst": "1",
    }
    event = {
        "reference_price": ref,
        "reference_time": R_TIME,
        "reference_agg_trade_id": ident - ticks,
        "confirm_price": conf,
        "confirm_time": confirm_time,
        "confirm_agg_trade_id": ident,
        "extreme_price": conf if empty else ext,
        "extreme_time": extreme_time,
        "extreme_agg_trade_id": ident if empty else ident + ticks,
        "direction": direction,
        "theta": THETA,
    }
    path = None
    if idle is not None:
        path = {
            "theta": THETA,
            "confirm_agg_trade_id": ident,
            "c_idle_fraction": idle[0],
            "os_idle_fraction": idle[1],
        }
    return summary, event, path


# nombre: (fila, tmv, osv, duration en s, net_duration en s o None). Entre paréntesis,
# la cuenta. Un hueco cuenta solo si es estrictamente mayor que 10 s.
CASES = {
    # Sin silencios: huecos de 1 s y 2 s.
    "quiet": (
        _event(1, ("100", "101", "103"), 1, 5, 4, 1, 2, idle=(0.0, 0.0)),
        3.0,  # |103 - 100| / (100 * 0.01)
        2.0,  # |103 - 101| / (100 * 0.01)
        12,  # 1 + 5 + 2 + 4
        12.0,  # sin silencios que descontar
    ),
    # Un silencio dentro de cada fase: 12 s de 30 s y 12 s de 20 s.
    "inside": (
        _event(2, ("200", "198", "190"), -1, 30, 20, 1, 3, idle=(0.4, 0.6)),
        5.0,  # |190 - 200| / (200 * 0.01)
        4.0,  # |190 - 198| / (200 * 0.01)
        54,  # 1 + 30 + 3 + 20
        30.0,  # 54 - 12 - 12
    ),
    # Un silencio de 30 s entre R y el primer tick de la confirmación.
    "gap_r": (
        _event(3, ("50", "50.5", "52"), 1, 10, 6, 30, 1, idle=(0.0, 0.0)),
        4.0,  # |52 - 50| / (50 * 0.01)
        3.0,  # |52 - 50.5| / (50 * 0.01)
        47,  # 30 + 10 + 1 + 6
        17.0,  # 47 - 30
    ),
    # Un silencio de 15 s entre C y el primer tick del overshoot.
    "gap_c": (
        _event(4, ("80", "79", "76"), -1, 8, 5, 2, 15, idle=(0.0, 0.0)),
        5.0,  # |76 - 80| / (80 * 0.01)
        3.75,  # |76 - 79| / (80 * 0.01)
        30,  # 2 + 8 + 15 + 5
        15.0,  # 30 - 15
    ),
    # Los cuatro silencios a la vez: 11 s y 10 s dentro, 11 s y 12 s en las fronteras.
    "all_four": (
        _event(5, ("10", "10.25", "10.5"), 1, 20, 40, 11, 12, idle=(0.55, 0.25)),
        5.0,  # |10.5 - 10| / (10 * 0.01)
        2.5,  # |10.5 - 10.25| / (10 * 0.01)
        83,  # 11 + 20 + 12 + 40
        39.0,  # 83 - 11 - 10 - 11 - 12
    ),
    # Huecos de frontera de exactamente 10 s: no superan el umbral, no se restan.
    "at_threshold": (
        _event(6, ("20", "20.2", "20.4"), 1, 5, 5, 10, 10, idle=(0.0, 0.0)),
        2.0,  # |20.4 - 20| / (20 * 0.01)
        1.0,  # |20.4 - 20.2| / (20 * 0.01)
        30,  # 10 + 5 + 10 + 5
        30.0,  # nada que restar
    ),
    # Overshoot vacío: extreme_* = confirm_*, osv = 0, hueco de C = 0 y os_* nulos.
    "empty": (
        _event(7, ("60", "61", "61"), 1, 9, None, 11, 0, idle=(0.0, None)),
        5 / 3,  # |61 - 60| / (60 * 0.01)
        0.0,  # |61 - 61| / (60 * 0.01)
        20,  # 11 + 9 + 0 + 0
        9.0,  # 20 - 11
    ),
    # Sin fila de path para el evento: net_duration es nula y M se calcula igual.
    "no_path_row": (
        _event(8, ("100", "99", "98"), -1, 3, 2, 1, 1),
        2.0,  # |98 - 100| / (100 * 0.01)
        1.0,  # |98 - 99| / (100 * 0.01)
        7,  # 1 + 3 + 1 + 2
        None,
    ),
}


def _inputs():
    summaries = [case[0][0] for case in CASES.values()]
    events = [case[0][1] for case in CASES.values()]
    paths = [case[0][2] for case in CASES.values() if case[0][2] is not None]
    return (
        _table(SUMMARIES_SCHEMA, summaries),
        _table(EVENTS_SCHEMA, events),
        _table(PATH_SCHEMA, paths),
    )


@pytest.fixture
def con():
    summaries, events, path = _inputs()
    con = duckdb.connect()
    load_derived(con, summaries, events, path)
    yield con
    con.close()


def _by_id(con):
    cursor = con.execute("SELECT * FROM derived")
    names = [d[0] for d in cursor.description]
    return {
        row[names.index("confirm_agg_trade_id")]: dict(zip(names, row, strict=True))
        for row in cursor.fetchall()
    }


def test_m_values(con):
    rows = _by_id(con)
    for name, (parts, tmv, osv, duration, _) in CASES.items():
        row = rows[parts[0]["confirm_agg_trade_id"]]
        assert row["tmv"] == pytest.approx(tmv, rel=1e-12), name
        assert row["osv"] == pytest.approx(osv, rel=1e-12, abs=1e-15), name
        assert row["duration"] == duration * S, name


def test_n_values(con):
    rows = _by_id(con)
    for name, (parts, _, _, _, net) in CASES.items():
        have = rows[parts[0]["confirm_agg_trade_id"]]["net_duration"]
        if net is None:
            assert have is None, name
        else:
            assert have == pytest.approx(net * S, rel=1e-12), name


def test_duration_is_both_boundary_gaps_plus_both_phases(con):
    """duration = hueco de R + c_duration + hueco de C + os_duration, en todas las filas."""
    rows = _by_id(con)
    for name, (parts, *_) in CASES.items():
        summary, event, _ = parts
        gap_r = event["confirm_time"] - summary["c_duration"] - event["reference_time"]
        gap_c = (
            event["extreme_time"]
            - (summary["os_duration"] or 0)
            - event["confirm_time"]
        )
        row = rows[summary["confirm_agg_trade_id"]]
        total = gap_r + summary["c_duration"] + gap_c + (summary["os_duration"] or 0)
        assert row["duration"] == total, name
        assert gap_r >= 0 and gap_c >= 0, name
    assert rows[7]["osv"] == 0.0  # overshoot vacío


def test_without_path_net_duration_is_null_and_m_is_not():
    summaries, events, _ = _inputs()
    con = duckdb.connect()
    load_derived(con, summaries, events)
    rows = _by_id(con)
    assert len(rows) == len(CASES)
    assert all(row["net_duration"] is None for row in rows.values())
    assert all(row["tmv"] is not None for row in rows.values())
    assert all(row["osv"] is not None for row in rows.values())
    assert all(row["duration"] is not None for row in rows.values())


def test_path_does_not_change_row_count_or_other_columns(con):
    summaries, events, _ = _inputs()
    plain = duckdb.connect()
    load_derived(plain, summaries, events)
    names = [d[0] for d in con.execute("SELECT * FROM derived").description]
    keep = ", ".join(n for n in names if n != "net_duration")
    sql = f"SELECT {keep} FROM derived ORDER BY confirm_agg_trade_id"
    assert con.execute(sql).fetchall() == plain.execute(sql).fetchall()


def test_path_from_file_and_hive_root_gives_the_same_net_duration(tmp_path, con):
    *_, path = _inputs()
    summaries, events, _ = _inputs()
    partition = (
        "provider=binance/market=spot/asset=BTCUSDT/theta=0.01000000/year=2026/month=09"
    )
    directory = tmp_path / "l3" / partition
    directory.mkdir(parents=True)
    pq.write_table(path, directory / "path.parquet")
    # Una raíz de L3 trae también los demás archivos de familia: solo entra path.parquet.
    pq.write_table(path.select(["theta"]), directory / "fit.parquet")
    want = {k: v["net_duration"] for k, v in _by_id(con).items()}
    for source in (directory / "path.parquet", tmp_path / "l3"):
        other = duckdb.connect()
        load_derived(other, summaries, events, source)
        assert {k: v["net_duration"] for k, v in _by_id(other).items()} == want


def test_path_paths_register_as_datasets(tmp_path):
    summaries, events, path = _inputs()
    pq.write_table(path, tmp_path / "path.parquet")
    registered = {}
    con = duckdb.connect()
    register = con.register

    class Spy:
        def register(self, name, obj):
            registered[name] = obj
            return register(name, obj)

        def execute(self, *args):
            return con.execute(*args)

    load_derived(Spy(), summaries, events, tmp_path / "path.parquet")
    assert isinstance(registered["path"], ds.Dataset)


def test_load_again_without_path_forgets_the_previous_path():
    summaries, events, path = _inputs()
    con = duckdb.connect()
    load_derived(con, summaries, events, path)
    assert con.execute("SELECT count(net_duration) FROM derived").fetchone()[0] == 7
    load_derived(con, summaries, events)
    assert con.execute("SELECT count(net_duration) FROM derived").fetchone()[0] == 0


def test_sql_literal_is_silence_us():
    sql = files("dc_views").joinpath("derived.sql").read_text(encoding="utf-8")
    literals = re.findall(r"/\* SILENCE_US \*/ (\d+)", sql)
    assert len(literals) == 2  # el hueco de R y el de C
    assert {int(x) for x in literals} == {SILENCE_US}
    assert SILENCE_US == 10_000_000


def test_threshold_is_strict(con):
    """El caso `at_threshold` tiene dos huecos de exactamente SILENCE_US: no se restan."""
    parts, *_ = CASES["at_threshold"]
    assert parts[1]["confirm_time"] - parts[0]["c_duration"] - R_TIME == SILENCE_US
    row = _by_id(con)[parts[0]["confirm_agg_trade_id"]]
    assert row["net_duration"] == pytest.approx(row["duration"], rel=1e-12)
