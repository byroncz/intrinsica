import json
import logging
from datetime import UTC, date, datetime

import pytest
from dq.reader import current_findings
from l1_ingest.cli import main
from l1_ingest.pipeline import Unit
from l1_ingest.publication import missing_source_check, publication_date

URL = "https://data.binance.vision/x.zip"
SEPTEMBER = Unit(2026, 9)
DAY = Unit(2026, 10, 3)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        (Unit(2026, 9), date(2026, 10, 5)),  # 1-oct es jueves
        (Unit(2026, 2), date(2026, 3, 2)),  # 1-mar es domingo
        (Unit(2026, 6), date(2026, 7, 6)),  # 1-jul es miércoles
        (Unit(2025, 9), date(2025, 10, 6)),  # 1-oct es miércoles
        (Unit(2026, 3), date(2026, 4, 6)),  # 1-abr es miércoles
        (Unit(2024, 4), date(2024, 5, 6)),  # 1-may es miércoles
        (Unit(2023, 12), date(2024, 1, 1)),  # 1-ene es lunes: el propio día 1
        (Unit(2026, 12), date(2027, 1, 4)),  # cruza de año
        (Unit(2026, 10, 3), date(2026, 10, 4)),  # diario: D+1
        (Unit(2026, 12, 31), date(2027, 1, 1)),
    ],
)
def test_publication_date(unit, expected):
    assert publication_date(unit) == expected


def test_monthly_before_first_monday_is_info():
    check = missing_source_check(SEPTEMBER, date(2026, 10, 3), URL)
    assert (check.check_type, check.severity, check.status) == (
        "source_not_published",
        "info",
        "pass",
    )
    assert check.metric_value == 2
    assert check.details["expected_publication"] == "2026-10-05"
    assert check.details["reason"] == (
        "dentro del calendario de Binance; publicación esperada el 2026-10-05"
    )
    assert check.details["source_url"] == URL


def test_monthly_on_first_monday_is_warning():
    check = missing_source_check(SEPTEMBER, date(2026, 10, 5), URL)
    assert (check.check_type, check.severity, check.status) == (
        "source_not_published",
        "warning",
        "fail",
    )
    assert check.metric_value == 0
    assert check.details["reason"] == "día de publicación, aún sin archivo"


def test_monthly_after_first_monday_is_delayed_error():
    check = missing_source_check(SEPTEMBER, date(2026, 10, 8), URL)
    assert (check.check_type, check.severity, check.status) == (
        "source_delayed",
        "error",
        "fail",
    )
    assert check.metric_value == 3
    assert check.details["reason"] == (
        "3 días de retraso sobre el calendario publicado de Binance"
    )


@pytest.mark.parametrize(
    ("today", "check_type", "severity", "metric"),
    [
        (date(2026, 10, 3), "source_not_published", "info", 1),  # el propio día D
        (date(2026, 10, 4), "source_not_published", "warning", 0),  # D+1
        (date(2026, 10, 7), "source_delayed", "error", 3),
    ],
)
def test_daily_follows_d_plus_one(today, check_type, severity, metric):
    check = missing_source_check(DAY, today, URL)
    assert (check.check_type, check.severity, check.metric_value) == (
        check_type,
        severity,
        metric,
    )
    assert check.details["unit"] == "binance/spot/BTCUSDT/2026-10-03"


def _env(tmp_path, base):
    return {
        "L1_LANDING_ROOT": str(tmp_path / "landing"),
        "L1_DQ_ROOT": str(tmp_path / "dq"),
        "L1_MANIFEST_ROOT": str(tmp_path / "manifest"),
        "L1_SOURCE_BASE_URL": base,
        "IMAGE_VERSION": "0.1.0+test",
    }


def _clock(year, month, day, hour=12):
    return lambda: datetime(year, month, day, hour, tzinfo=UTC)


def _run(tmp_path, base, argv, clock):
    return main(argv, _env(tmp_path, base), clock)


def _assert_nothing_written(tmp_path):
    assert not (tmp_path / "landing").exists()
    assert not (tmp_path / "manifest").exists()


def _only_finding(tmp_path):
    (row,) = current_findings(tmp_path / "dq").to_pylist()
    return row


@pytest.mark.parametrize(
    ("today", "exit_code", "check_type", "severity", "status", "metric"),
    [
        ((2026, 10, 3), 0, "source_not_published", "info", "pass", 2),
        ((2026, 10, 5), 0, "source_not_published", "warning", "fail", 0),
        ((2026, 10, 8), 3, "source_delayed", "error", "fail", 3),
    ],
)
def test_main_monthly_404_by_run_date(
    tmp_path, http_server, today, exit_code, check_type, severity, status, metric
):
    _, base = http_server  # servidor vacío: todo da 404
    argv = ["--mode", "backfill", "--from", "2026-09", "--force"]
    assert _run(tmp_path, base, argv, _clock(*today, hour=23)) == exit_code

    row = _only_finding(tmp_path)
    assert (row["check_type"], row["severity"], row["status"]) == (
        check_type,
        severity,
        status,
    )
    assert row["metric_value"] == metric
    assert (row["layer"], row["mode"], row["stage"]) == ("l1", "backfill", "canonical")
    assert (row["asset"], row["year"], row["month"]) == ("BTCUSDT", 2026, 9)
    # Ninguna de las tres ramas escribe Parquet ni manifiesto.
    _assert_nothing_written(tmp_path)


@pytest.mark.parametrize(
    ("today", "exit_code", "check_type"),
    [
        ((2026, 10, 3), 0, "source_not_published"),
        ((2026, 10, 4), 0, "source_not_published"),
        ((2026, 10, 7), 3, "source_delayed"),
    ],
)
def test_main_daily_404_by_run_date(
    tmp_path, http_server, today, exit_code, check_type
):
    _, base = http_server
    argv = ["--mode", "daily", "--from", "2026-10-03"]
    assert _run(tmp_path, base, argv, _clock(*today)) == exit_code

    row = _only_finding(tmp_path)
    assert (row["check_type"], row["stage"]) == (check_type, "provisional")
    _assert_nothing_written(tmp_path)


def test_main_monthly_close_404_is_delayed(tmp_path, http_server):
    _, base = http_server
    argv = ["--mode", "monthly-close", "--from", "2026-09"]
    assert _run(tmp_path, base, argv, _clock(2026, 10, 8)) == 3
    assert _only_finding(tmp_path)["check_type"] == "source_delayed"
    _assert_nothing_written(tmp_path)


def test_main_404_does_not_retry(tmp_path, http_server, monkeypatch):
    from l1_ingest import download

    _, base = http_server
    waits = []
    monkeypatch.setattr(download.time, "sleep", waits.append)
    argv = ["--mode", "backfill", "--from", "2026-09"]
    assert _run(tmp_path, base, argv, _clock(2026, 10, 3)) == 0
    assert waits == []


def test_main_404_logs_one_json_line(tmp_path, http_server, caplog):
    _, base = http_server
    caplog.set_level(logging.INFO)
    argv = ["--mode", "backfill", "--from", "2026-09"]
    assert _run(tmp_path, base, argv, _clock(2026, 10, 8)) == 3

    lines = [json.loads(m) for m in caplog.messages if m.startswith("{")]
    (line,) = lines
    assert line["check_type"] == "source_delayed"
    assert line["severity"] == "error"
    assert line["metric_value"] == 3
    assert line["details"]["reason"].startswith("3 días de retraso")
    assert caplog.messages[-1].startswith("sonda:")  # la sonda sigue cerrando


def test_main_published_file_is_untouched_by_the_rule(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    argv = ["--mode", "backfill", "--from", "2024-03"]
    assert _run(tmp_path, base, argv, _clock(2026, 10, 8)) == 0
    assert (tmp_path / "landing").exists()
    assert "source_delayed" not in {
        r["check_type"] for r in current_findings(tmp_path / "dq").to_pylist()
    }
