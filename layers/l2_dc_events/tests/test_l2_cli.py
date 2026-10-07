import logging
from datetime import date

import pytest
from l2_dc_events import cli, cpu
from l2_dc_events.cli import UsageError, main, previous_month, resolve_range
from l2_dc_events.cpu import CpuLimit

SERIES = ["--series-start", "2017-08"]
BACKFILL = [*SERIES, "--mode", "backfill"]


def _env(tmp_path, **extra):
    return {
        "L2_LANDING_ROOT": str(tmp_path / "landing"),
        "L2_EVENTS_ROOT": str(tmp_path / "events"),
        "L2_DQ_ROOT": str(tmp_path / "dq"),
        "IMAGE_VERSION": "0.1.0+test",
        **extra,
    }


def _carry_overs(tmp_path, year, month):
    """Los `carry_over.parquet` que dejó el mes, uno por θ."""
    return sorted(
        (tmp_path / "events").rglob(
            f"year={year:04d}/month={month:02d}/carry_over.parquet"
        )
    )


def _three_months(write_month, fixture_ticks):
    for month in (8, 9, 10):
        write_month(fixture_ticks, row_group_size=1_000, year=2017, month=month)


@pytest.mark.parametrize(
    ("from_", "to", "expected"),
    [
        ("2023-11", "2024-02", [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]),
        ("2024-03", None, [(2024, 3)]),
        ("2024-03", "2024-03", [(2024, 3)]),
    ],
)
def test_resolve_range(from_, to, expected):
    assert resolve_range(from_, to) == expected


@pytest.mark.parametrize(
    ("from_", "to"),
    [
        ("2024-03-01", None),
        ("2024-13", None),
        ("2024-3", None),
        ("2024-04", "2024-03"),
    ],
)
def test_resolve_range_invalid(from_, to):
    with pytest.raises(UsageError):
        resolve_range(from_, to)


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2024, 3, 15), "2024-02"),
        (date(2024, 3, 1), "2024-02"),
        (date(2024, 1, 31), "2023-12"),
    ],
)
def test_previous_month(today, expected):
    assert previous_month(today) == expected


def test_backfill_chains_the_range_in_one_process(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    # Solo puede leer el carry-over de septiembre si agosto ya lo escribió.
    for month in (8, 9, 10):
        assert len(_carry_overs(tmp_path, 2017, month)) == 50
    units = [
        line.split("unit=")[1].split()[0]
        for line in caplog.text.splitlines()
        if "sonda:" in line
    ]
    assert units == [f"binance/spot/BTCUSDT/2017-{m:02d}" for m in (8, 9, 10)]
    assert "ticks_s_core=" in caplog.text
    # La sonda desglosa la espera y la escritura y nombra al θ pesado y al último.
    for field in (
        "backpressure_s=",
        "drain_s=",
        "publish_s=",
        "encode_s=",
        "close_s=",
        "move_s=",
        "carry_write_s=",
        "heavy_theta=",
        "last_theta_done_s=",
    ):
        assert field in caplog.text


@pytest.mark.parametrize("index", ["1", "2"])
def test_backfill_refuses_a_task_array(tmp_path, fixture_ticks, write_month, index):
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    assert main(argv, _env(tmp_path, CLOUD_RUN_TASK_INDEX=index)) == 2
    assert not list((tmp_path / "events").rglob("*.parquet"))


def test_backfill_accepts_task_index_zero(tmp_path, fixture_ticks, write_month):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=8)
    argv = [*BACKFILL, "--from", "2017-08"]
    assert main(argv, _env(tmp_path, CLOUD_RUN_TASK_INDEX="0")) == 0


def test_backfill_resumes_from_the_first_incomplete_month(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    assert main([*argv[:-1], "2017-08"], _env(tmp_path)) == 0
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    assert "unidad binance/spot/BTCUSDT/2017-08: ningún θ la necesita" in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-08" not in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-09" in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-10" in caplog.text


def test_resuming_reprocesses_the_months_after_the_first_incomplete_one(
    tmp_path, fixture_ticks, write_month, caplog
):
    """Un θ sin carry-over en septiembre arrastra a octubre aunque tenga salida."""
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    assert main(argv, _env(tmp_path)) == 0
    _carry_overs(tmp_path, 2017, 9)[0].unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    assert "sonda: unit=binance/spot/BTCUSDT/2017-08" not in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-09" in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-10" in caplog.text
    assert len(_carry_overs(tmp_path, 2017, 9)) == 50


def test_an_invalid_carry_over_counts_as_missing(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-09"]
    assert main(argv, _env(tmp_path)) == 0
    # La frontera de ese θ retrocede a agosto: solo septiembre se reprocesa.
    _carry_overs(tmp_path, 2017, 9)[0].write_bytes(b"no es parquet")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    assert "sonda: unit=binance/spot/BTCUSDT/2017-08" not in caplog.text
    assert "sonda: unit=binance/spot/BTCUSDT/2017-09" in caplog.text
    assert len(_carry_overs(tmp_path, 2017, 9)) == 50


def test_backfill_with_everything_done_does_nothing(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    assert main(argv, _env(tmp_path)) == 0
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    assert "sonda:" not in caplog.text
    assert "los 50 θ están al día" in caplog.text


def test_force_starts_at_from(tmp_path, fixture_ticks, write_month, caplog):
    _three_months(write_month, fixture_ticks)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    assert main(argv, _env(tmp_path)) == 0
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main([*argv, "--force"], _env(tmp_path)) == 0
    for month in (8, 9, 10):
        assert f"sonda: unit=binance/spot/BTCUSDT/2017-{month:02d}" in caplog.text
    assert "se salta" not in caplog.text


def test_a_failure_stops_the_range(tmp_path, fixture_ticks, write_month, capsys):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=8)
    # Falta septiembre: octubre existe pero no debe tocarse.
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=10)
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-10"]
    assert main(argv, _env(tmp_path)) == 1
    assert "no ha publicado el mes" in capsys.readouterr().err
    assert len(_carry_overs(tmp_path, 2017, 8)) == 50
    assert not _carry_overs(tmp_path, 2017, 9)
    assert not _carry_overs(tmp_path, 2017, 10)
    assert not list((tmp_path / "events").rglob("month=10"))


def test_main_fails_closed_without_the_previous_carry_over(
    tmp_path, fixture_ticks, write_month, capsys
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = [*BACKFILL, "--from", "2017-09"]
    assert main(argv, _env(tmp_path)) == 1
    assert "carry-over" in capsys.readouterr().err
    assert not list((tmp_path / "events").rglob("*.parquet"))


def test_the_probe_line_reports_the_effective_cpu_limit_and_the_phases(
    tmp_path, fixture_ticks, write_month, monkeypatch, caplog
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=8)
    monkeypatch.setattr(cpu, "cpu_limit", lambda: CpuLimit(2.0, 6, "cgroup-v2"))
    argv = [*BACKFILL, "--from", "2017-08"]
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    (line,) = [m for m in caplog.messages if m.startswith("sonda:")]
    fields = dict(part.split("=") for part in line.split()[2:])
    assert (fields["cores"], fields["cores_visible"]) == ("2", "6")
    assert fields["cores_source"] == "cgroup-v2"
    # ticks_s_core se calcula con el límite efectivo, no con los cores visibles.
    wall, ticks = float(fields["wall_s"]), int(fields["ticks"])
    assert int(fields["ticks_s_core"]) == round(ticks / wall / 2)
    assert (fields["row_groups"], int(fields["bytes_in"]) > 0) == ("5", True)
    phases = ("read_s", "decode_s", "detect_s", "carry_s", "wait_s", "other_s")
    # Las fases del hilo principal más lo no explicado suman la pared (a 0,1 s por redondeo).
    assert abs(sum(float(fields[p]) for p in phases) - wall) <= 0.6
    assert float(fields["write_s"]) > 0


def test_monthly_defaults_to_the_previous_month(
    tmp_path, fixture_ticks, write_month, monkeypatch, caplog
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 10, 3))
    argv = ["--mode", "monthly", "--series-start", "2017-09"]
    with caplog.at_level(logging.INFO):
        assert main(argv, _env(tmp_path)) == 0
    assert "sonda: unit=binance/spot/BTCUSDT/2017-09 mode=monthly" in caplog.text


def test_monthly_across_the_year_boundary(
    tmp_path, fixture_ticks, write_month, monkeypatch, caplog
):
    for month in (11, 12):
        write_month(fixture_ticks, row_group_size=1_000, year=2017, month=month)
    series = ["--series-start", "2017-11"]
    assert main([*series, "--mode", "backfill", "--to", "2017-11"], _env(tmp_path)) == 0
    # Hoy es 2018-01-02: el mes anterior es 2017-12 y su carry-over previo, 2017-11.
    monkeypatch.setattr(cli, "_today", lambda: date(2018, 1, 2))
    with caplog.at_level(logging.INFO):
        assert main([*series, "--mode", "monthly"], _env(tmp_path)) == 0
    assert "sonda: unit=binance/spot/BTCUSDT/2017-12 mode=monthly" in caplog.text
    assert len(_carry_overs(tmp_path, 2017, 12)) == 50


def test_monthly_takes_the_series_start_from_the_environment(
    tmp_path, fixture_ticks, write_month
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = ["--mode", "monthly", "--from", "2017-09"]
    assert main(argv, _env(tmp_path, L2_SERIES_START="2017-09")) == 0


def test_the_flag_wins_over_the_environment(tmp_path, fixture_ticks, write_month):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = ["--mode", "monthly", "--from", "2017-09", "--series-start", "2017-09"]
    assert main(argv, _env(tmp_path, L2_SERIES_START="2017-08")) == 0


@pytest.mark.parametrize("mode", ["monthly", "backfill"])
@pytest.mark.parametrize("declared", [{}, {"L2_SERIES_START": ""}])
def test_every_mode_requires_the_series_start(tmp_path, capsys, mode, declared):
    argv = ["--mode", mode, "--from", "2017-09"]
    assert main(argv, _env(tmp_path, **declared)) == 2
    assert "L2_SERIES_START" in capsys.readouterr().err


def test_a_bad_series_start_in_the_environment_is_a_usage_error(tmp_path):
    argv = ["--mode", "monthly", "--from", "2017-09"]
    assert main(argv, _env(tmp_path, L2_SERIES_START="2017")) == 2


def test_force_requires_from(tmp_path, capsys):
    argv = [*BACKFILL, "--force"]
    assert main(argv, _env(tmp_path)) == 2
    assert "--force exige --from" in capsys.readouterr().err


def test_monthly_rejects_to(tmp_path):
    argv = [*SERIES, "--mode", "monthly", "--from", "2017-08", "--to", "2017-09"]
    assert main(argv, _env(tmp_path)) == 2


def test_resuming_a_backfill_mid_series_does_not_start_cold(
    tmp_path, fixture_ticks, write_month, capsys
):
    write_month(fixture_ticks, row_group_size=1_000, year=2017, month=9)
    argv = [*BACKFILL, "--from", "2017-09"]
    assert main(argv, _env(tmp_path)) == 1
    assert "carry-over" in capsys.readouterr().err
    assert not list((tmp_path / "events").rglob("*.parquet"))


def test_a_unit_before_the_series_start_is_rejected(tmp_path, capsys):
    argv = ["--series-start", "2017-09", "--mode", "backfill", "--from", "2017-08"]
    assert main(argv, _env(tmp_path)) == 2
    assert "anterior a la serie" in capsys.readouterr().err


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
    assert "L1 no ha publicado ningún mes" in capsys.readouterr().err
