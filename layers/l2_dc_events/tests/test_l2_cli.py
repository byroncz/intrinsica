import logging

import pytest
from l2_dc_events.cli import UsageError, main, resolve_unit

SERIES = ["--series-start", "2017-08"]


def _env(tmp_path, **extra):
    return {
        "L2_LANDING_ROOT": str(tmp_path / "landing"),
        "L2_EVENTS_ROOT": str(tmp_path / "events"),
        "L2_DQ_ROOT": str(tmp_path / "dq"),
        "IMAGE_VERSION": "0.1.0+test",
        **extra,
    }


@pytest.mark.parametrize(
    ("from_", "to", "index", "expected"),
    [
        ("2023-11", "2024-02", 0, (2023, 11)),
        ("2023-11", "2024-02", 2, (2024, 1)),
        ("2023-11", "2024-02", 3, (2024, 2)),
        ("2024-03", None, 0, (2024, 3)),
    ],
)
def test_resolve_unit(from_, to, index, expected):
    assert resolve_unit(from_, to, index) == expected


@pytest.mark.parametrize(
    ("from_", "to", "index"),
    [
        ("2024-03-01", None, 0),
        ("2024-13", None, 0),
        ("2024-3", None, 0),
        ("2024-04", "2024-03", 0),
        ("2024-03", "2024-04", 2),
        ("2024-03", None, -1),
    ],
)
def test_resolve_unit_invalid(from_, to, index):
    with pytest.raises(UsageError):
        resolve_unit(from_, to, index)


def test_main_processes_the_unit_of_the_task_index(
    tmp_path, fixture_ticks, write_month, caplog
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=8)
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-08", "--to", "2017-09"]
    with caplog.at_level(logging.INFO):
        # El mes 0 es el primero de la serie; el 1 lee su carry-over.
        assert main(argv, _env(tmp_path, CLOUD_RUN_TASK_INDEX="0")) == 0
        caplog.clear()
        assert main(argv, _env(tmp_path, CLOUD_RUN_TASK_INDEX="1")) == 0
    assert "binance/spot/BTCUSDT/2017-09: 4735 ticks, 5 row groups" in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-09 mode=backfill" in caplog.text
    assert "ticks=4735" in caplog.text
    assert "ticks_s_core=" in caplog.text


def test_main_fails_closed_without_the_previous_carry_over(
    tmp_path, fixture_ticks, write_month, capsys
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-08", "--to", "2017-09"]
    assert main(argv, _env(tmp_path, CLOUD_RUN_TASK_INDEX="1")) == 1
    assert "carry-over" in capsys.readouterr().err
    assert not list((tmp_path / "events").rglob("*.parquet"))


@pytest.mark.parametrize("mode", ["monthly", "backfill"])
def test_every_mode_requires_the_series_start(
    tmp_path, fixture_ticks, write_month, mode
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = ["--mode", mode, "--from", "2017-09"]
    assert main(argv, _env(tmp_path)) == 2
    assert main([*argv, "--series-start", "2017-09"], _env(tmp_path)) == 0


def test_resuming_a_backfill_mid_series_does_not_start_cold(
    tmp_path, fixture_ticks, write_month, capsys
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-09"]
    assert main(argv, _env(tmp_path)) == 1
    assert "carry-over" in capsys.readouterr().err
    assert not list((tmp_path / "events").rglob("*.parquet"))


def test_a_unit_before_the_series_start_is_rejected(tmp_path, capsys):
    argv = ["--series-start", "2017-09", "--mode", "backfill", "--from", "2017-08"]
    assert main(argv, _env(tmp_path)) == 2
    assert "anterior a --series-start" in capsys.readouterr().err


def test_main_uses_the_asset_flag(tmp_path, fixture_ticks, write_month, caplog):
    write_month(fixture_ticks, row_group_size=5_000, asset="ETHUSDT")
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-08", "--asset", "ETHUSDT"]
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    assert "binance/spot/ETHUSDT/2017-08" in caplog.text


@pytest.mark.parametrize("missing", ["L2_LANDING_ROOT", "L2_EVENTS_ROOT", "L2_DQ_ROOT"])
def test_main_requires_the_three_roots(tmp_path, capsys, missing):
    env = _env(tmp_path)
    del env[missing]
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-08"]
    assert main(argv, env) == 2
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--mode", "backfill"],
        ["--mode", "daily", "--from", "2017-08"],
        ["--mode", "backfill", "--from", "2017-08-01"],
        ["--mode", "backfill", "--from", "2017-08", "--to", "2017-07"],
    ],
)
def test_main_invalid_usage_exits_with_2(tmp_path, argv):
    assert main(argv, _env(tmp_path)) == 2


@pytest.mark.parametrize("index", ["x", "1", "-1"])
def test_main_rejects_a_bad_task_index(tmp_path, index):
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-08"]
    assert main(argv, _env(tmp_path, CLOUD_RUN_TASK_INDEX=index)) == 2


def test_main_fails_without_a_consolidated_month(tmp_path, capsys):
    argv = [*SERIES, "--mode", "backfill", "--from", "2017-08"]
    assert main(argv, _env(tmp_path)) == 1
    assert "no ha publicado el mes" in capsys.readouterr().err
