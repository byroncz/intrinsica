"""Una unidad por proceso hijo en backfill (ITSC-372, TRD-L3 ADR-L3-16, «Memoria acotada por
construcción»): mismo resultado que en proceso, el padre sigue vivo si el hijo muere y
`monthly` no cambia.
"""

import pytest
import unit_process_probe
from cli_support import emitted, env_for, write_catalog
from l3_dc_descriptors import cli, unit_process
from l3_dc_descriptors.cli import main
from l3_lake import build_lake, fixture_ticks, month_cuts

TICKS = fixture_ticks()
BACKFILL = ["--mode", "backfill", "--series-start", "2017-08", "--force"]
RANGE = ["--from", "2017-08", "--to", "2017-10"]


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    return build_lake(
        tmp_path_factory.mktemp("lake"), TICKS, month_cuts(TICKS, 1500, 3000)
    )


@pytest.fixture
def catalog(tmp_path, lake):
    return write_catalog(tmp_path / "thetas.yaml", lake.thetas)


def run(lake, base, catalog, *argv, **extra) -> int:
    """Una corrida con su propia raíz de L3 y de DQ bajo `base`."""
    extra = {
        "L3_SUMMARIES_ROOT": str(base / "l3"),
        "L3_DQ_ROOT": str(base / "dq"),
        **extra,
    }
    return main(list(argv), env_for(lake, base, catalog, **extra))


def hashes(base) -> dict:
    return {
        (r["details"]["theta"], r["month"]): r["details"]["content_hash"]
        for r in emitted(base, "summaries_summary")
    }


def timings(base) -> list[dict]:
    return [r["details"] for r in emitted(base, "unit_timing")]


def test_the_child_process_gives_the_same_hashes_as_the_cli_process(
    lake, catalog, tmp_path
):
    inline, child = tmp_path / "inline", tmp_path / "child"
    assert run(lake, inline, catalog, *BACKFILL, *RANGE, L3_UNIT_PROCESS="0") == 0
    assert run(lake, child, catalog, *BACKFILL, *RANGE, L3_UNIT_PROCESS="1") == 0
    assert hashes(child) == hashes(inline)
    assert len(hashes(child)) == len(lake.thetas) * 3
    assert [t["spawn_s"] for t in timings(inline)] == [None] * 3
    spawn = [t["spawn_s"] for t in timings(child)]
    assert all(s is not None and 0 < s < 30 for s in spawn), spawn
    # El hijo es el que midió su pico: una unidad tiene su propio `VmHWM`.
    assert {t["rss_peak_scope"] for t in timings(child)} == {"unit"}


def test_the_child_process_is_the_default_in_backfill(lake, catalog, tmp_path):
    env = env_for(lake, tmp_path, catalog)
    del env["L3_UNIT_PROCESS"]
    argv = [*BACKFILL, "--from", "2017-08", "--to", "2017-08"]
    assert main(argv, env) == 0
    assert [t["spawn_s"] is not None for t in timings(tmp_path)] == [True]


def test_a_child_killed_by_a_signal_is_a_failed_unit_and_the_parent_survives(
    lake, catalog, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(unit_process, "_child", unit_process_probe.kill_in_september)
    assert run(lake, tmp_path, catalog, *BACKFILL, *RANGE, L3_UNIT_PROCESS="1") == 1
    died = emitted(tmp_path, "unit_process_died")
    assert [(r["year"], r["month"]) for r in died] == [(2017, 9)]
    assert died[0]["severity"] == "error"
    assert died[0]["details"] == {
        "month": "2017-09",
        "exit_signal": 9,
        "signal_name": "SIGKILL",
        "exit_code": -9,
        "error": None,
    }
    assert "unit_process_died" in capsys.readouterr().err
    # Agosto quedó escrito; septiembre no dejó nada y octubre no se tocó (el rango se detiene).
    written = (tmp_path / "l3").rglob("summaries.parquet")
    assert {p.parts[-2] for p in written} == {"month=08"}


def test_a_child_that_raises_is_a_failed_unit_with_its_traceback(
    lake, catalog, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        unit_process, "_child", unit_process_probe.fail_with_an_exception
    )
    assert run(lake, tmp_path, catalog, *BACKFILL, *RANGE, L3_UNIT_PROCESS="1") == 1
    (died,) = emitted(tmp_path, "unit_process_died")
    assert died["details"]["exit_signal"] is None
    assert died["details"]["exit_code"] == 1
    assert "Traceback" in died["details"]["error"]


def test_monthly_runs_its_only_unit_in_the_cli_process(
    lake, catalog, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("monthly no lanza un hijo")

    monkeypatch.setattr(cli, "run_unit", forbidden)
    argv = ["--mode", "monthly", "--series-start", "2017-08", "--from", "2017-08"]
    assert run(lake, tmp_path, catalog, *argv, L3_UNIT_PROCESS="1") == 0
    assert [t["spawn_s"] for t in timings(tmp_path)] == [None]


@pytest.mark.parametrize("value", ["2", "yes", "-1"])
def test_the_switch_is_zero_or_one(lake, catalog, tmp_path, capsys, value):
    code = run(lake, tmp_path, catalog, *BACKFILL, *RANGE, L3_UNIT_PROCESS=value)
    assert code == 2
    assert "L3_UNIT_PROCESS" in capsys.readouterr().err
