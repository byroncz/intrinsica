"""`load_derived` sobre rutas: un lago en disco con la disposición de L2 y L3."""

from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from dc_views import DERIVED_VERSION, load_derived
from test_dc_views_derived import (
    COLUMNS_D,
    EVENTS_SCHEMA,
    ROWS,
    SUMMARIES_SCHEMA,
    _table,
)

PARTITION = (
    "provider=binance/market=spot/asset=BTCUSDT/theta={theta}/year={year}/month={month}"
)
# Fila -> (θ, año, mes). H va en otro mes del mismo θ y G en otro θ.
PLACEMENT = {
    **{name: ("0.01000000", 2026, 9) for name in "ABCDEF"},
    "G": ("0.02000000", 2026, 9),
    "H": ("0.01000000", 2026, 10),
}
SUMMARIES_COLUMNS = [f.name for f in SUMMARIES_SCHEMA]
KEYS = ["theta", "confirm_agg_trade_id"]


def _write(root: Path, name: str, schema, side: int, extras: dict | None = None):
    """Escribe `<name>.parquet` en la partición de cada fila; devuelve las rutas."""
    paths = {}
    for row, (theta, year, month) in PLACEMENT.items():
        directory = root / PARTITION.format(theta=theta, year=year, month=month)
        directory.mkdir(parents=True, exist_ok=True)
        paths.setdefault((theta, year, month), []).append(ROWS[row][side])
    written = {}
    for (theta, year, month), rows in paths.items():
        path = (
            root
            / PARTITION.format(theta=theta, year=year, month=month)
            / f"{name}.parquet"
        )
        pq.write_table(_table(schema, rows), path)
        written[(theta, year, month)] = path
    return written


@pytest.fixture
def lake(tmp_path):
    summaries = _write(tmp_path / "l3", "summaries", SUMMARIES_SCHEMA, 0)
    events = _write(tmp_path / "l2", "events", EVENTS_SCHEMA, 1)
    # Una raíz de L2 trae también estos dos archivos por partición.
    for path in events.values():
        (path.parent / "carry_over.parquet").write_bytes(b"no es un events.parquet")
        (path.parent / "content_hash.json").write_text("{}")
    return {
        "summaries": summaries,
        "events": events,
        "l3": tmp_path / "l3",
        "l2": tmp_path / "l2",
    }


def _rows(con):
    cursor = con.execute("SELECT * FROM derived ORDER BY theta, confirm_agg_trade_id")
    names = [d[0] for d in cursor.description]
    return names, cursor.fetchall()


def _in_memory(names=None):
    selected = [r for r in ROWS if names is None or r in names]
    con = duckdb.connect()
    load_derived(
        con,
        _table(SUMMARIES_SCHEMA, [ROWS[r][0] for r in selected]),
        _table(EVENTS_SCHEMA, [ROWS[r][1] for r in selected]),
    )
    return _rows(con)


def test_roots_match_in_memory(lake):
    con = duckdb.connect()
    assert load_derived(con, lake["l3"], str(lake["l2"])) == DERIVED_VERSION
    names, got = _rows(con)
    want_names, want = _in_memory()
    assert names == KEYS + ["year", "month"] + COLUMNS_D
    assert want_names == KEYS + COLUMNS_D
    keep = [i for i, n in enumerate(names) if n not in ("year", "month")]
    assert [tuple(r[i] for i in keep) for r in got] == want


def test_year_month_are_integers_and_theta_stays_decimal(lake):
    con = duckdb.connect()
    load_derived(con, lake["l3"], lake["l2"])
    described = {n: t for n, t, *_ in con.execute("DESCRIBE derived").fetchall()}
    assert described["theta"] == "DECIMAL(9,8)"
    assert described["year"] == described["month"] == "INTEGER"
    got = {
        (theta, ident): (year, month)
        for theta, ident, year, month in con.execute(
            "SELECT theta, confirm_agg_trade_id, year, month FROM derived"
        ).fetchall()
    }
    for row, (theta, year, month) in PLACEMENT.items():
        summary = ROWS[row][0]
        key = (Decimal(summary["theta"]), summary["confirm_agg_trade_id"])
        assert Decimal(theta) == key[0]
        assert got[key] == (year, month)


def test_partition_files_match_in_memory(lake):
    key = ("0.01000000", 2026, 9)
    con = duckdb.connect()
    load_derived(con, lake["summaries"][key], lake["events"][key])
    names, got = _rows(con)
    want_names, want = _in_memory("ABCDEF")
    assert names == want_names == KEYS + COLUMNS_D  # sin year ni month
    assert got == want


def test_file_paths_accept_str_and_path(lake):
    key = ("0.02000000", 2026, 9)
    runs = []
    for convert in (str, Path):
        con = duckdb.connect()
        load_derived(con, convert(lake["summaries"][key]), convert(lake["events"][key]))
        runs.append(_rows(con))
    assert runs[0] == runs[1] == _in_memory("G")


def test_mixed_table_and_path(lake):
    key = ("0.02000000", 2026, 9)
    con = duckdb.connect()
    load_derived(
        con,
        pq.read_table(lake["summaries"][key]),
        lake["events"][key],
    )
    assert _rows(con) == _in_memory("G")


def test_root_with_uri_scheme(lake):
    con = duckdb.connect()
    load_derived(con, lake["l3"].as_uri(), lake["l2"].as_uri())
    assert con.execute("SELECT count(*) FROM derived").fetchone()[0] == len(ROWS)


def test_paths_register_as_datasets_not_tables(lake):
    registered = {}
    con = duckdb.connect()
    register = con.register

    class Spy:
        def register(self, name, obj):
            registered[name] = obj
            return register(name, obj)

        def execute(self, *args):
            return con.execute(*args)

    load_derived(Spy(), lake["l3"], lake["l2"])
    assert all(isinstance(o, ds.Dataset) for o in registered.values())
    assert set(registered) == {"summaries", "events"}
