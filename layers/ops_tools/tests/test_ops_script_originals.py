"""Los dos scripts de `ops-script` (`viz_check_day.py`, `viz_probe_ticks.py`) contra el lago de fixtures.

Son los originales de `gs://intrinsica-dc-ops/scripts/`, versionados tal cual: no se editan para
probarlos. `viz_check_day.py` ya acepta `--local-root`; `viz_probe_ticks.py` abre
`pafs.GcsFileSystem()` fijo, así que la prueba lo sustituye por el sistema de archivos local y
pasa en `--project` una ruta absoluta (el script arma `<project>-landing/l1/...`).
"""

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest
from pyarrow import fs as pafs

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "layers/viz_tiles/tests"))

from lake_fixture import DAY, MONTH, build_lake, events_path, landing_path, read_events
from viz_tiles.lake import CARRY_OVER, EVENTS

BASE = "provider=binance/market=spot/asset=BTCUSDT"
YM = f"year={MONTH[0]:04d}/month={MONTH[1]:02d}"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ops_lake(tmp_path):
    """El lago de fixtures con la disposición de buckets que leen los scripts de `ops-script`."""
    roots, ticks, _pending = build_lake(tmp_path / "src")
    out = tmp_path / "ops"
    l1 = out / "p-landing/l1" / BASE / YM / "consolidated.parquet"
    l1.parent.mkdir(parents=True)
    shutil.copy(landing_path(roots), l1)
    theta = min(read_events())
    text = f"{theta / 1e8:.8f}"
    for name in (EVENTS, CARRY_OVER):
        dst = out / "p-dc-events/l2" / BASE / f"theta={text}" / YM / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(events_path(roots, theta, MONTH, name), dst)
    return out, text, ticks


def test_check_day_corre_y_reporta_los_invariantes(ops_lake, capsys):
    """Los eventos de los fixtures son `events_v0`, anteriores a ADR-L2-03 (b): el script debe señalarlos."""
    out, theta, ticks = ops_lake
    check = load("viz_check_day")
    sys.argv = [
        "viz_check_day.py",
        "--project",
        "p",
        "--theta",
        theta,
        "--day",
        DAY,
        "--window",
        "00:00-01:00",
        "--w",
        "128",
        "--local-root",
        str(out),
    ]
    check.main()
    text = capsys.readouterr().out
    assert f"L1 {DAY}: {len(ticks)} ticks" in text
    assert re.search(
        r"== Invariantes L2 sobre \d+ eventos: [1-9]\d* violaciones ==", text
    )
    assert "(ADR-L2-03 b)" in text and "sin alternancia entre" in text
    for section in (
        "Eventos L2 que tocan 00:00-01:00",
        "Velas de 1 min",
        "Columnas w=128",
        "Resumen",
    ):
        assert f"== {section}" in text


def test_check_day_detecta_ids_que_no_crecen(ops_lake):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out, theta, _ticks = ops_lake
    path = out / "p-landing/l1" / BASE / YM / "consolidated.parquet"
    table = pq.read_table(path)
    ids = table["agg_trade_id"].to_pylist()
    ids[1], ids[2] = ids[2], ids[1]
    table = table.set_column(0, table.schema.field(0), pa.array(ids, pa.int64()))
    pq.write_table(table, path)
    check = load("viz_check_day")
    sys.argv = [
        "viz_check_day.py",
        "--project",
        "p",
        "--theta",
        theta,
        "--day",
        DAY,
        "--local-root",
        str(out),
    ]
    with pytest.raises(
        AssertionError, match="agg_trade_id no es estrictamente creciente"
    ):
        check.main()


def test_probe_ticks_mide_el_dia(ops_lake, capsys, monkeypatch):
    out, _theta, ticks = ops_lake
    probe = load("viz_probe_ticks")
    monkeypatch.setattr(probe.pafs, "GcsFileSystem", pafs.LocalFileSystem)
    # El script arma `<project>-landing/l1/...`: con `<out>/p` queda `<out>/p-landing/l1/...`.
    monkeypatch.setattr(
        sys, "argv", ["viz_probe_ticks.py", "--project", str(out / "p"), "--day", DAY]
    )
    probe.main()
    text = capsys.readouterr().out
    assert f"L1 {DAY}: {len(ticks):,} ticks" in text
    assert "== A. columnas planas" in text and "== B. deltas varint" in text
    assert f"== Resumen: {len(ticks):,} ticks" in text


def test_probe_ticks_varint_y_zigzag():
    probe = load("viz_probe_ticks")
    buf = bytearray()
    probe.varint(buf, 300)
    assert bytes(buf) == b"\xac\x02"
    assert [probe.zigzag(v) for v in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]
