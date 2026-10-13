"""L3 sobre la salida real de L2: el catálogo, el linaje y la frontera.

Las capas no se importan entre sí, pero sus pruebas sí pueden correrse una sobre la otra:
aquí L2 escribe un mes de verdad (su CLI) y L3 lo resume. Comprueba tres cosas que ninguna
suite de capa ve sola: que el parser de catálogo de L3 (una copia, RNF-L3-04) no se separe
del de L2, que el `content_hash` de las entradas de L2 que L3 guarda en `summaries_summary`
sea el que L2 dejó en `events_summary`, y que la frontera de L3 acepte los archivos de L2.
"""

import csv
import json
import random
from decimal import Decimal
from pathlib import Path

import l2_dc_events.cli as l2_cli
import l2_dc_events.thetas as l2_thetas
import l3_dc_descriptors.thetas as l3_thetas
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from l2_dc_events.landing import consolidated_path
from l3_dc_descriptors import cli as l3_cli
from l3_dc_descriptors.frontier import frontiers, list_summaries

TICKS = Path(__file__).resolve().parents[1] / "shared/dc_core/tests/fixtures/ticks.csv"
THETAS = [100_000, 250_000, 500_000]
DEC = pa.decimal128(18, 8)
KEY = ("binance", "spot", "BTCUSDT")
L1_SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", DEC, nullable=False),
        pa.field("quantity", DEC, nullable=False),
        pa.field("first_trade_id", pa.int64(), nullable=False),
        pa.field("last_trade_id", pa.int64(), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
        pa.field("is_best_match", pa.bool_(), nullable=False),
    ]
)

CATALOGS = [
    "scale: 100000000\nthetas: [10000, 20000]\n",
    "scale: 100000000\nthetas: [20000, 10000, 5000000]\n",
    "scale: 1\nthetas: [20000]\n",
    "scale: 100000000\nthetas: []\n",
    "scale: 100000000\nthetas: [20000, 20000, 9999, 5000001, 1.5]\n",
    "- 1\n",
    "scale: [\n",
]


def _outcome(module, text):
    try:
        return module.parse_thetas(text, "cat"), None
    except l2_thetas.ThetasError as error:
        return None, error.problems
    except l3_thetas.ThetasError as error:
        return None, error.problems


@pytest.mark.parametrize("text", CATALOGS)
def test_the_catalog_is_validated_like_l2(text):
    assert _outcome(l3_thetas, text) == _outcome(l2_thetas, text)


def _write_l1(root: Path) -> None:
    rng = random.Random(11)
    with open(TICKS) as handle:
        rows = list(csv.DictReader(handle))
    ids = pa.array([int(r["agg_trade_id"]) for r in rows], pa.int64())
    table = pa.table(
        {
            "agg_trade_id": ids,
            "price": pa.array([Decimal(r["price"]) for r in rows], DEC),
            "quantity": pa.array(
                [Decimal(rng.randint(1, 300_000_000)).scaleb(-8) for _ in rows], DEC
            ),
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": pa.array(
                [int(r["transact_time"]) for r in rows], pa.int64()
            ),
            "is_buyer_maker": pa.array([rng.random() < 0.5 for _ in rows]),
            "is_best_match": pa.array([True] * len(rows)),
        },
        schema=L1_SCHEMA,
    )
    path = Path(consolidated_path(str(root), *KEY, 2017, 8))
    path.parent.mkdir(parents=True)
    pq.write_table(table, path, row_group_size=900)


def _findings(root: Path, check_type: str) -> list[dict]:
    rows = ds.dataset(root, partitioning="hive").to_table().to_pylist()
    rows = [r for r in rows if r["check_type"] == check_type]
    for row in rows:
        row["details"] = json.loads(row["details"])
    return rows


@pytest.fixture
def lake(tmp_path):
    _write_l1(tmp_path / "l1")
    catalog = tmp_path / "thetas.yaml"
    catalog.write_text(f"scale: 100000000\nthetas: {THETAS}\n")
    env = {
        "L2_LANDING_ROOT": str(tmp_path / "l1"),
        "L2_EVENTS_ROOT": str(tmp_path / "l2"),
        "L2_DQ_ROOT": str(tmp_path / "dq2"),
        "L2_THETAS_URI": str(catalog),
        "IMAGE_VERSION": "0.1.0+test",
    }
    args = ["--mode", "backfill", "--from", "2017-08", "--series-start", "2017-08"]
    assert l2_cli.main(args, env) == 0
    return tmp_path, str(catalog)


def test_l3_runs_on_what_l2_wrote_and_keeps_its_lineage(lake):
    base, catalog = lake
    env = {
        "L3_LANDING_ROOT": str(base / "l1"),
        "L3_EVENTS_ROOT": str(base / "l2"),
        "L3_SUMMARIES_ROOT": str(base / "l3"),
        "L3_DQ_ROOT": str(base / "dq3"),
        "L3_THETAS_URI": catalog,
        "IMAGE_VERSION": "0.4.0+test",
    }
    args = ["--mode", "backfill", "--series-start", "2017-08"]
    assert l3_cli.main(args, env) == 0

    theirs = {
        r["details"]["theta"]: r["details"]
        for r in _findings(base / "dq2", "events_summary")
    }
    ours = _findings(base / "dq3", "summaries_summary")
    assert sorted(int(Decimal(r["details"]["theta"]).scaleb(8)) for r in ours) == THETAS
    for row in ours:
        details = row["details"]
        l2 = theirs[int(Decimal(details["theta"]).scaleb(8))]
        assert details["l2_events_content_hash"] == l2["events_content_hash"]
        assert details["l2_carry_over_content_hash"] == l2["carry_over_content_hash"]
        assert details["rows"] == l2["events"]

    # La frontera de L3 acepta los archivos de L2 y llega al mes escrito.
    listed = list_summaries(str(base / "l3"), KEY)
    reached = frontiers(
        THETAS, listed, (2017, 8), str(base / "l3"), str(base / "l2"), KEY
    )
    assert set(reached.values()) == {2017 * 12 + 7}
