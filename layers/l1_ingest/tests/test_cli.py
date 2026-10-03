import json
import logging
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dq.reader import current_findings
from l1_ingest.cli import UsageError, main, resolve_unit
from l1_ingest.download import ChecksumError
from l1_ingest.schema import OUTPUT_SCHEMA
from l1_ingest.write import partition_path, write_partition


def _env(tmp_path, base, **extra):
    return {
        "L1_LANDING_ROOT": str(tmp_path / "landing"),
        "L1_DQ_ROOT": str(tmp_path / "dq"),
        "L1_MANIFEST_ROOT": str(tmp_path / "manifest"),
        "L1_SOURCE_BASE_URL": base,
        "IMAGE_VERSION": "0.1.0+test",
        **extra,
    }


@pytest.mark.parametrize(
    ("mode", "from_", "to", "index", "expected"),
    [
        (
            "daily",
            "2024-02-28",
            "2024-03-02",
            0,
            (
                2024,
                2,
                28,
            ),
        ),
        ("daily", "2024-02-28", "2024-03-02", 2, (2024, 3, 1)),
        ("daily", "2023-12-31", "2024-01-01", 1, (2024, 1, 1)),
        ("daily", "2024-03-05", None, 0, (2024, 3, 5)),
        ("backfill", "2023-11", "2024-02", 2, (2024, 1, None)),
        ("backfill", "2023-11", "2024-02", 3, (2024, 2, None)),
        ("backfill", "2024-03", None, 0, (2024, 3, None)),
    ],
)
def test_resolve_unit(mode, from_, to, index, expected):
    assert resolve_unit(mode, from_, to, index) == expected


@pytest.mark.parametrize(
    ("mode", "from_", "to", "index"),
    [
        ("daily", "2024-03", None, 0),
        ("backfill", "2024-03-01", None, 0),
        ("backfill", "2024-13", None, 0),
        ("daily", "2024-03-05", "2024-03-04", 0),
        ("backfill", "2024-03", "2024-04", 2),
        ("backfill", "2024-03", None, -1),
    ],
)
def test_resolve_unit_invalid(mode, from_, to, index):
    with pytest.raises(UsageError):
        resolve_unit(mode, from_, to, index)


def test_main_day_end_to_end(tmp_path, publish_zip):
    publish, base = publish_zip
    publish(day=6)
    env = _env(tmp_path, base, CLOUD_RUN_TASK_INDEX="1")
    argv = ["--mode", "daily", "--from", "2024-03-05", "--to", "2024-03-07"]
    assert main(argv, env) == 0

    partition = next((tmp_path / "landing").rglob("provisional-day=06.parquet"))
    table = pq.read_table(partition)
    assert table.schema.equals(OUTPUT_SCHEMA)
    assert table.num_rows == 5
    assert table["transact_time"].to_pylist()[0] == 1_709_600_000_000_000

    findings = current_findings(tmp_path / "dq").to_pylist()
    assert {f["check_type"] for f in findings} == {
        "checksum_fail",
        "header_detected",
        "timestamp_unit_corrected",
        "reorder_applied",
        "aggid_gap",
        "aggid_duplicate",
        "provider_invalid_marker",
        "seam_skipped",  # sin día previo: 2024-03-05 no existe en el landing
    }
    assert {f["layer"] for f in findings} == {"l1"}
    assert {f["stage"] for f in findings} == {"provisional"}
    assert {f["image_version"] for f in findings} == {"0.1.0+test"}

    manifest_file = next((tmp_path / "manifest").rglob("*.parquet"))
    manifest = pq.read_table(manifest_file).to_pylist()
    assert len(manifest) == 1
    assert manifest[0]["granularity"] == "daily"
    assert manifest[0]["source_url"].endswith("BTCUSDT-aggTrades-2024-03-06.zip")


def test_main_month_backfill(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    assert main(["--mode", "backfill", "--from", "2024-03"], _env(tmp_path, base)) == 0

    assert next((tmp_path / "landing").rglob("consolidated.parquet"))
    findings = current_findings(tmp_path / "dq").to_pylist()
    assert {f["stage"] for f in findings} == {"canonical"}
    assert {f["mode"] for f in findings} == {"backfill"}


def test_main_force_flag(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    argv = ["--mode", "backfill", "--from", "2024-03"]
    assert main(argv, _env(tmp_path, base)) == 0
    assert main(argv, _env(tmp_path, base)) == 0  # salta
    assert len(list((tmp_path / "manifest").rglob("*.parquet"))) == 1
    assert main([*argv, "--force"], _env(tmp_path, base)) == 0
    assert len(list((tmp_path / "manifest").rglob("*.parquet"))) == 2


def test_main_usage_errors(tmp_path, capsys):
    env = _env(tmp_path, "http://127.0.0.1:1")
    assert main(["--mode", "daily", "--from", "2024-03"], env) == 2
    assert main(["--mode", "nope", "--from", "2024-03"], env) == 2
    assert main(["--mode", "backfill"], env) == 2
    bad_index = {**env, "CLOUD_RUN_TASK_INDEX": "1"}
    assert main(["--mode", "backfill", "--from", "2024-03"], bad_index) == 2
    del env["L1_DQ_ROOT"]
    assert main(["--mode", "backfill", "--from", "2024-03"], env) == 2
    assert "L1_DQ_ROOT" in capsys.readouterr().err


PROBE = re.compile(r"sonda: unit=(\S+) mode=(\S+) rss_peak_mib=(\d+) wall_s=(\d+\.\d)")


def _probe_lines(caplog):
    return [m for m in caplog.messages if m.startswith("sonda:")]


def test_main_probe_line_after_final_line(tmp_path, publish_zip, caplog):
    publish, base = publish_zip
    publish(day=6)
    caplog.set_level(logging.INFO)
    argv = ["--mode", "daily", "--from", "2024-03-06"]
    assert main(argv, _env(tmp_path, base)) == 0

    assert len(_probe_lines(caplog)) == 1
    assert caplog.messages[-1].startswith("sonda:")
    # La costura diaria (aquí sin día previo) corre entre "fin unidad" y la
    # sonda: WARNING del log más el INFO que deja emit_findings al persistir
    # el hallazgo seam_skipped.
    assert json.loads(caplog.messages[-2])["check_type"] == "seam_skipped"
    assert caplog.messages[-3].startswith("costura omitida")
    assert caplog.messages[-4].startswith("fin unidad=")
    unit, mode, rss, wall = PROBE.fullmatch(caplog.messages[-1]).groups()
    assert (unit, mode) == ("binance/spot/BTCUSDT/2024-03-06", "daily")
    assert int(rss) > 0
    assert float(wall) > 0


def test_main_probe_line_on_checksum_abort(tmp_path, publish_zip, caplog):
    publish, base = publish_zip
    publish(checksum="0" * 64)
    caplog.set_level(logging.INFO)
    with pytest.raises(ChecksumError):
        main(["--mode", "backfill", "--from", "2024-03"], _env(tmp_path, base))

    assert len(_probe_lines(caplog)) == 1
    assert caplog.messages[-1].startswith("sonda:")


def _seed(tmp_path, year, month, name, first, last):
    """Escribe una partición previa con ids first..last."""
    n = last - first + 1
    table = pa.table(
        {
            "agg_trade_id": list(range(first, last + 1)),
            "price": [Decimal(1)] * n,
            "quantity": [Decimal(1)] * n,
            "first_trade_id": list(range(first, last + 1)),
            "last_trade_id": list(range(first, last + 1)),
            "transact_time": list(range(n)),
            "is_buyer_maker": [True] * n,
            "is_best_match": [True] * n,
        },
        schema=OUTPUT_SCHEMA,
    )
    path = partition_path(
        tmp_path / "landing", "binance", "spot", "BTCUSDT", year, month, name
    )
    write_partition(table, path)


def _daily(tmp_path, base, day):
    date = f"2024-03-{day:02d}"
    return main(["--mode", "daily", "--from", date], _env(tmp_path, base))


def _seams(tmp_path):
    rows = current_findings(tmp_path / "dq").to_pylist()
    return [r for r in rows if r["check_type"] == "seam_discontinuity"]


def test_daily_seam_pass(tmp_path, publish_zip):
    publish, base = publish_zip
    publish(day=6)
    _seed(tmp_path, 2024, 3, "provisional-day=05.parquet", -4, 0)
    assert _daily(tmp_path, base, 6) == 0
    (seam,) = _seams(tmp_path)
    assert (seam["mode"], seam["stage"], seam["status"]) == (
        "daily",
        "provisional",
        "pass",
    )


def test_daily_seam_fail_on_id_gap(tmp_path, publish_zip):
    publish, base = publish_zip
    publish(day=6)
    _seed(tmp_path, 2024, 3, "provisional-day=05.parquet", -10, -5)
    assert _daily(tmp_path, base, 6) == 0
    (seam,) = _seams(tmp_path)
    assert seam["status"] == "fail"
    assert seam["metric_value"] == 5


def test_daily_seam_first_day_uses_last_provisional_of_previous_month(
    tmp_path, publish_zip
):
    publish, base = publish_zip
    publish(day=1)
    _seed(tmp_path, 2024, 2, "provisional-day=28.parquet", -9, -1)
    _seed(tmp_path, 2024, 2, "provisional-day=29.parquet", -4, 0)
    assert _daily(tmp_path, base, 1) == 0
    (seam,) = _seams(tmp_path)
    assert seam["status"] == "pass"


def test_daily_missing_previous_day_emits_seam_skipped_finding(
    tmp_path, publish_zip, caplog
):
    publish, base = publish_zip
    publish(day=6)
    with caplog.at_level(logging.WARNING):
        assert _daily(tmp_path, base, 6) == 0
    assert _seams(tmp_path) == []  # no hay seam_discontinuity: no se evaluó
    assert any("provisional-day=05.parquet" in m for m in caplog.messages)

    rows = current_findings(tmp_path / "dq").to_pylist()
    (skipped,) = [r for r in rows if r["check_type"] == "seam_skipped"]
    assert (skipped["severity"], skipped["status"]) == ("info", "pass")
    assert skipped["metric_value"] is None
    details = json.loads(skipped["details"])
    assert details["reason"] == "previous_missing"
    assert "provisional-day=05.parquet" in details["expected_path"]


def test_daily_closed_month_neither_downloads_nor_writes(tmp_path, publish_zip, caplog):
    publish, base = publish_zip
    publish(day=6)
    _seed(tmp_path, 2024, 3, "consolidated.parquet", 1, 5)
    argv = ["--mode", "daily", "--from", "2024-03-06", "--force"]
    with caplog.at_level(logging.WARNING):
        assert main(argv, _env(tmp_path, base)) == 0
    assert any("mes cerrado" in m for m in caplog.messages)
    assert not list((tmp_path / "landing").rglob("provisional-day=06.parquet"))
    assert not (tmp_path / "manifest").exists()
    assert not (tmp_path / "dq").exists()


def test_daily_defaults_to_previous_utc_day(tmp_path, publish_zip):
    publish, base = publish_zip
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    publish(yesterday.year, yesterday.month, yesterday.day)
    assert main(["--mode", "daily"], _env(tmp_path, base)) == 0
    name = f"provisional-day={yesterday.day:02d}.parquet"
    assert next((tmp_path / "landing").rglob(name))


def test_to_without_from_is_usage_error(tmp_path, capsys):
    env = _env(tmp_path, "http://127.0.0.1:1")
    assert main(["--mode", "daily", "--to", "2024-03-07"], env) == 2
    assert main(["--mode", "backfill"], env) == 2
    assert "--from" in capsys.readouterr().err
