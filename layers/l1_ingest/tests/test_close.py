import calendar
import hashlib
import io
import logging
import zipfile
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest
from dq.reader import current_findings
from l1_ingest.cli import main
from l1_ingest.schema import OUTPUT_SCHEMA
from l1_ingest.write import partition_path, write_partition
from test_cli import _env

HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
    "transact_time,is_buyer_maker,is_best_match\n"
)


def _publish_month(root, year, month, n):
    """Publica un mensual con ids 1..n, una fila por id, sin huecos ni desorden."""
    directory = root / "data/spot/monthly/aggTrades/BTCUSDT"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"BTCUSDT-aggTrades-{year}-{month:02d}"
    rows = "".join(
        f"{i},100.00000000,1.00000000,{i},{i},{1709600000000 + i},True,True\n"
        for i in range(1, n + 1)
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(f"{name}.csv", HEADER + rows)
    data = buffer.getvalue()
    (directory / f"{name}.zip").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (directory / f"{name}.zip.CHECKSUM").write_text(f"{digest}  {name}.zip\n")


def _seed(tmp_path, year, month, name, ids):
    n = len(ids)
    table = pa.table(
        {
            "agg_trade_id": ids,
            "price": [Decimal(1)] * n,
            "quantity": [Decimal(1)] * n,
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": list(range(n)),
            "is_buyer_maker": [True] * n,
            "is_best_match": [True] * n,
        },
        schema=OUTPUT_SCHEMA,
    )
    write_partition(
        table,
        partition_path(
            tmp_path / "landing", "binance", "spot", "BTCUSDT", year, month, name
        ),
    )


def _seed_days(tmp_path, days, extra_rows=None):
    """Un provisional por día con el id igual al día; `extra_rows` {día: ids}."""
    for day in days:
        ids = (extra_rows or {}).get(day, [day])
        _seed(tmp_path, 2024, 3, f"provisional-day={day:02d}.parquet", ids)


def _provisionals(tmp_path):
    return sorted((tmp_path / "landing").rglob("provisional-day=*.parquet"))


def _finding(tmp_path, check_type):
    rows = current_findings(tmp_path / "dq").to_pylist()
    (row,) = [r for r in rows if r["check_type"] == check_type]
    return row


def _close(tmp_path, base, *extra):
    argv = ["--mode", "monthly-close", "--from", "2024-03", *extra]
    return main(argv, _env(tmp_path, base))


def test_close_with_matching_provisionals(tmp_path, publish_zip):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    _seed_days(tmp_path, range(1, 32))
    assert _close(tmp_path, base) == 0

    assert next((tmp_path / "landing").rglob("consolidated.parquet"))
    assert _provisionals(tmp_path) == []
    drift = _finding(tmp_path, "daily_monthly_drift")
    assert (drift["status"], drift["mode"], drift["stage"]) == (
        "pass",
        "monthly-close",
        "canonical",
    )
    rows = current_findings(tmp_path / "dq").to_pylist()
    assert {r["mode"] for r in rows} == {"monthly-close"}
    assert {r["stage"] for r in rows} == {"canonical"}


def test_close_with_extra_rows_reports_drift(tmp_path, publish_zip):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    _seed_days(tmp_path, range(1, 32), extra_rows={5: [5, 5]})
    assert _close(tmp_path, base) == 0

    drift = _finding(tmp_path, "daily_monthly_drift")
    assert (drift["status"], drift["severity"]) == ("fail", "warning")
    assert drift["metric_value"] == -1
    assert _provisionals(tmp_path) == []  # el drift no impide el cierre


def test_close_with_missing_day_reports_it(tmp_path, publish_zip):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    _seed_days(tmp_path, [d for d in range(1, 32) if d != 10])
    assert _close(tmp_path, base) == 0

    drift = _finding(tmp_path, "daily_monthly_drift")
    assert drift["status"] == "fail"
    assert drift["metric_value"] == 1
    assert '"missing_days": [10]' in drift["details"].replace("\n", "")


def test_close_without_provisionals_passes(tmp_path, publish_zip):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    assert _close(tmp_path, base) == 0

    drift = _finding(tmp_path, "daily_monthly_drift")
    assert drift["status"] == "pass"
    assert '"provisionals": 0' in drift["details"].replace("\n", "")


def test_close_seam_with_previous_month(tmp_path, publish_zip):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    _seed(tmp_path, 2024, 2, "consolidated.parquet", [-5, -4, -3, -2, -1, 0])
    assert _close(tmp_path, base) == 0

    seam = _finding(tmp_path, "seam_discontinuity")
    assert (seam["status"], seam["stage"], seam["mode"]) == (
        "pass",
        "canonical",
        "monthly-close",
    )


def test_close_repeated_does_not_rewrite(tmp_path, publish_zip, caplog):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    assert _close(tmp_path, base) == 0
    consolidated = next((tmp_path / "landing").rglob("consolidated.parquet"))
    before = consolidated.stat().st_mtime_ns
    findings = len(current_findings(tmp_path / "dq").to_pylist())

    with caplog.at_level(logging.INFO):
        assert _close(tmp_path, "http://127.0.0.1:1") == 0  # sin descargar
    assert any("mes ya cerrado" in m for m in caplog.messages)
    assert consolidated.stat().st_mtime_ns == before
    assert len(current_findings(tmp_path / "dq").to_pylist()) == findings


def test_close_repairs_leftover_provisionals(tmp_path, publish_zip, caplog):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    assert _close(tmp_path, base) == 0
    consolidated = next((tmp_path / "landing").rglob("consolidated.parquet"))
    before = consolidated.stat().st_mtime_ns
    _seed_days(tmp_path, [1])  # cierre interrumpido: quedó un provisional

    with caplog.at_level(logging.INFO):
        assert _close(tmp_path, "http://127.0.0.1:1") == 0  # sin descargar
    assert any("cierre completado" in m for m in caplog.messages)
    assert consolidated.stat().st_mtime_ns == before
    assert _provisionals(tmp_path) == []
    rows = current_findings(tmp_path / "dq").to_pylist()
    drift = [r for r in rows if r["check_type"] == "daily_monthly_drift"]
    assert [r["status"] for r in drift].count("fail") == 1


def test_close_force_reprocesses_closed_month(tmp_path, publish_zip):
    _, base = publish_zip
    _publish_month(tmp_path, 2024, 3, 31)
    assert _close(tmp_path, base) == 0
    assert _close(tmp_path, base, "--force") == 0
    assert len(list((tmp_path / "manifest").rglob("*.parquet"))) == 2


def test_close_defaults_to_previous_utc_month(tmp_path, publish_zip):
    _, base = publish_zip
    today = datetime.now(UTC).date()
    year, month = (
        (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    )
    _publish_month(tmp_path, year, month, calendar.monthrange(year, month)[1])
    assert main(["--mode", "monthly-close"], _env(tmp_path, base)) == 0
    partition = next((tmp_path / "landing").rglob("consolidated.parquet"))
    assert f"year={year:04d}/month={month:02d}" in str(partition)


@pytest.mark.parametrize("argv", [["--mode", "monthly-close", "--to", "2024-03"]])
def test_close_to_without_from_is_usage_error(tmp_path, argv):
    assert main(argv, _env(tmp_path, "http://127.0.0.1:1")) == 2
