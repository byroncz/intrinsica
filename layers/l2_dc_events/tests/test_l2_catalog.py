"""El catálogo de θ como dato y la frontera por θ (ITSC-285, TRD-L2 §7.3 y §8)."""

import json
import logging
import shutil
from datetime import date

import pyarrow.parquet as pq
import pytest
from l2_dc_events import cli
from l2_dc_events.cli import main
from l2_dc_events.write import format_theta

SERIES = ["--series-start", "2017-08"]
BACKFILL = [*SERIES, "--mode", "backfill"]
OLD = [31_313, 500_000]
NEW = 1_000_000


def _env(tmp_path, **extra):
    return {
        "L2_LANDING_ROOT": str(tmp_path / "landing"),
        "L2_EVENTS_ROOT": str(tmp_path / "events"),
        "L2_DQ_ROOT": str(tmp_path / "dq"),
        "IMAGE_VERSION": "0.1.0+test",
        **extra,
    }


def _catalog(tmp_path, *thetas, name="thetas.yaml"):
    path = tmp_path / name
    path.write_text(f"scale: 100000000\nthetas: {list(thetas)}\n")
    return str(path)


def _run(tmp_path, *argv, **env):
    return main([str(a) for a in argv], _env(tmp_path, **env))


def _three_months(write_month, fixture_ticks):
    for month in (8, 9, 10):
        write_month(fixture_ticks, row_group_size=1_000, year=2017, month=month)


def _files(tmp_path, root="events", name="*.parquet"):
    """Los archivos de `root` con su mtime: la huella de lo que se escribió."""
    return {
        str(p): p.stat().st_mtime_ns
        for p in (tmp_path / root).rglob(name)
        if p.is_file()
    }


def _partition(tmp_path, theta, month, name="events.parquet"):
    return (
        tmp_path
        / "events"
        / "provider=binance"
        / "market=spot"
        / "asset=BTCUSDT"
        / f"theta={format_theta(theta)}"
        / "year=2017"
        / f"month={month:02d}"
        / name
    )


def _findings(tmp_path, check_type):
    if not (tmp_path / "dq").exists():
        return []
    rows = pq.read_table(tmp_path / "dq", partitioning="hive").to_pylist()
    return [r for r in rows if r["check_type"] == check_type]


def _hashes(tmp_path):
    """`(θ, mes) → hashes` de los `events_summary` que hay en el lago de DQ."""
    return {
        (json.loads(r["details"])["theta"], r["month"]): (
            json.loads(r["details"])["events_content_hash"],
            json.loads(r["details"])["carry_over_content_hash"],
        )
        for r in _findings(tmp_path, "events_summary")
    }


def _units(caplog):
    return [
        line.split("unit=")[1].split()[0].rsplit("/", 1)[1]
        for line in caplog.text.splitlines()
        if "sonda:" in line
    ]


@pytest.mark.parametrize(
    "body",
    [
        "scale: 100000000\nthetas: [9999, 20000]\n",
        "scale: 100000000\nthetas: [20000, 5000001]\n",
        "scale: 100000000\nthetas: [20000, 20000]\n",
        "scale: 1000\nthetas: [20000]\n",
        "scale: 100000000\nthetas: [0.0003]\n",
    ],
)
def test_an_invalid_catalog_exits_2_with_a_finding_and_writes_no_data(
    tmp_path, fixture_ticks, write_month, capsys, body
):
    write_month(fixture_ticks, row_group_size=1_000)
    path = tmp_path / "bad.yaml"
    path.write_text(body)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", path) == 2
    assert "catálogo de θ inválido" in capsys.readouterr().err
    assert not list((tmp_path / "events").rglob("*"))
    (row,) = _findings(tmp_path, "theta_catalog_invalid")
    assert (row["severity"], row["status"]) == ("error", "fail")
    details = json.loads(row["details"])
    assert details["source"] == str(path)
    assert details["problems"]


def test_a_missing_catalog_object_is_invalid_too(tmp_path, fixture_ticks, write_month):
    write_month(fixture_ticks, row_group_size=1_000)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", tmp_path / "nada.yaml") == 2
    (row,) = _findings(tmp_path, "theta_catalog_invalid")
    assert "no se pudo leer" in json.loads(row["details"])["problems"][0]
    assert not (tmp_path / "events").exists()


def test_the_catalog_comes_from_the_environment_and_the_flag_wins(
    tmp_path, fixture_ticks, write_month
):
    write_month(fixture_ticks, row_group_size=1_000)
    from_env = _catalog(tmp_path, 31_313, name="env.yaml")
    from_flag = _catalog(tmp_path, 500_000, name="flag.yaml")
    assert _run(tmp_path, *BACKFILL, L2_THETAS_URI=from_env) == 0
    assert _partition(tmp_path, 31_313, 8).exists()
    assert not _partition(tmp_path, 500_000, 8).exists()
    assert (
        _run(tmp_path, *BACKFILL, "--thetas-uri", from_flag, L2_THETAS_URI=from_env)
        == 0
    )
    assert _partition(tmp_path, 500_000, 8).exists()


def test_without_a_catalog_it_uses_the_seed(tmp_path, fixture_ticks, write_month):
    write_month(fixture_ticks, row_group_size=1_000)
    assert _run(tmp_path, *BACKFILL) == 0
    assert len(list((tmp_path / "events").rglob("carry_over.parquet"))) == 50


def test_a_new_theta_only_runs_from_the_start_and_the_old_ones_do_not_change(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    two = _catalog(tmp_path, *OLD, name="two.yaml")
    three = _catalog(tmp_path, *OLD, NEW, name="three.yaml")
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", two) == 0
    before_files = _files(tmp_path)
    before_hashes = _hashes(tmp_path)
    assert len(before_files) == 3 * 2 * 2
    assert len(before_hashes) == 3 * 2

    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert _run(tmp_path, *BACKFILL, "--thetas-uri", three) == 0

    # Se recorrió el rango entero (el nuevo empieza en --series-start), una vez por mes.
    assert _units(caplog) == ["2017-08", "2017-09", "2017-10"]
    after_files = _files(tmp_path)
    # Los archivos de los θ viejos no se tocaron: mismo mtime, no solo mismo contenido.
    assert {p: t for p, t in after_files.items() if p in before_files} == before_files
    assert len(after_files) == 3 * 3 * 2
    for month in (8, 9, 10):
        assert _partition(tmp_path, NEW, month).exists()
        assert _partition(tmp_path, NEW, month, "carry_over.parquet").exists()
    # Solo el θ nuevo dejó resumen en esta corrida, y los hashes de los viejos no cambian.
    after_hashes = _hashes(tmp_path)
    assert {
        k: v for k, v in after_hashes.items() if k in before_hashes
    } == before_hashes
    assert {theta for theta, _ in after_hashes} == {*OLD, NEW}
    assert len(_findings(tmp_path, "events_summary")) == 3 * 2 + 3


def test_the_hashes_of_a_theta_do_not_depend_on_who_runs_with_it(
    tmp_path, fixture_ticks, write_month
):
    _three_months(write_month, fixture_ticks)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", _catalog(tmp_path, *OLD, NEW)) == 0
    together = _hashes(tmp_path)
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copytree(tmp_path / "landing", alone / "landing")
    assert _run(alone, *BACKFILL, "--thetas-uri", _catalog(alone, NEW)) == 0
    assert {k: v for k, v in together.items() if k[0] == NEW} == _hashes(alone)


def test_with_everything_up_to_date_it_writes_nothing(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    catalog = _catalog(tmp_path, *OLD)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", catalog) == 0
    events, dq = _files(tmp_path), _files(tmp_path, "dq")
    # Ni siquiera hace falta L1: ningún mes se lee.
    for month in (8, 9, 10):
        (write_month.landing / "provider=binance/market=spot/asset=BTCUSDT").joinpath(
            "year=2017", f"month={month:02d}", "consolidated.parquet"
        ).rename(tmp_path / f"moved-{month}")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert (
            _run(tmp_path, *BACKFILL, "--thetas-uri", catalog, "--to", "2017-10") == 0
        )
    assert (_files(tmp_path), _files(tmp_path, "dq")) == (events, dq)
    assert "sonda:" not in caplog.text
    assert "los 2 θ están al día" in caplog.text


def test_a_month_nobody_needs_is_skipped_without_reading_l1(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    old, new = 31_313, 500_000
    assert (
        _run(
            tmp_path,
            *BACKFILL,
            "--thetas-uri",
            _catalog(tmp_path, old),
            "--to",
            "2017-08",
        )
        == 0
    )
    both = _catalog(tmp_path, old, new, name="both.yaml")
    assert (
        _run(
            tmp_path,
            *BACKFILL,
            "--thetas-uri",
            both,
            "--thetas",
            format_theta(new),
            "--to",
            "2017-10",
        )
        == 0
    )
    # `old` está en agosto y `new` en octubre: solo `old` necesita septiembre y octubre.
    (write_month.landing / "provider=binance/market=spot/asset=BTCUSDT").joinpath(
        "year=2017", "month=08", "consolidated.parquet"
    ).unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert _run(tmp_path, *BACKFILL, "--thetas-uri", both) == 0
    assert _units(caplog) == ["2017-09", "2017-10"]
    assert "unidad binance/spot/BTCUSDT/2017-08: ningún θ la necesita" in caplog.text
    assert _partition(tmp_path, old, 10).exists()


def test_a_gap_in_the_chain_sets_the_frontier_before_it(
    tmp_path, fixture_ticks, write_month, caplog
):
    _three_months(write_month, fixture_ticks)
    catalog = _catalog(tmp_path, *OLD)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", catalog) == 0
    # Falta septiembre de un θ aunque octubre exista: su frontera es agosto.
    _partition(tmp_path, OLD[0], 9, "carry_over.parquet").unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert _run(tmp_path, *BACKFILL, "--thetas-uri", catalog) == 0
    assert f"θ={OLD[0]} frontera=2017-08" in caplog.text
    assert f"θ={OLD[1]} frontera=2017-10" in caplog.text
    # Septiembre y octubre se reprocesan, pero solo para el θ rezagado.
    assert _units(caplog) == ["2017-09", "2017-10"]
    assert _partition(tmp_path, OLD[0], 9, "carry_over.parquet").exists()


def test_to_defaults_to_the_last_consolidated_month_never_a_provisional(
    tmp_path, fixture_ticks, write_month
):
    for month in (8, 9):
        write_month(fixture_ticks, row_group_size=1_000, month=month)
    write_month(
        fixture_ticks, row_group_size=1_000, month=10, name="provisional-day=01.parquet"
    )
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", _catalog(tmp_path, *OLD)) == 0
    assert _partition(tmp_path, OLD[0], 9, "carry_over.parquet").exists()
    assert not _partition(tmp_path, OLD[0], 10).parent.exists()


def test_from_is_optional_in_backfill_and_the_floor_is_the_series_start(
    tmp_path, fixture_ticks, write_month
):
    write_month(fixture_ticks, row_group_size=1_000, month=8)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", _catalog(tmp_path, *OLD)) == 0
    assert _partition(tmp_path, OLD[0], 8).exists()


def test_the_thetas_flag_narrows_the_run_to_a_subset_of_the_catalog(
    tmp_path, fixture_ticks, write_month, capsys
):
    write_month(fixture_ticks, row_group_size=1_000)
    catalog = _catalog(tmp_path, *OLD)
    assert (
        _run(tmp_path, *BACKFILL, "--thetas-uri", catalog, "--thetas", "0.00031313")
        == 0
    )
    assert _partition(tmp_path, OLD[0], 8).exists()
    assert not _partition(tmp_path, OLD[1], 8).exists()
    for bad in ("0.00099999", "0.0003131", "uno"):
        assert _run(tmp_path, *BACKFILL, "--thetas-uri", catalog, "--thetas", bad) == 2
    assert "--thetas" in capsys.readouterr().err


def test_removing_a_theta_reports_it_as_info_and_deletes_nothing(
    tmp_path, fixture_ticks, write_month
):
    _three_months(write_month, fixture_ticks)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", _catalog(tmp_path, *OLD)) == 0
    files = _files(tmp_path)
    assert _findings(tmp_path, "theta_config_drift") == []
    only_one = _catalog(tmp_path, OLD[0], name="one.yaml")
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", only_one) == 0
    assert _files(tmp_path) == files
    (row,) = _findings(tmp_path, "theta_config_drift")
    assert (row["severity"], row["status"]) == ("info", "pass")
    assert json.loads(row["details"]) == {
        "theta": OLD[1],
        "months": 3,
        "last_month": "2017-10",
    }


def test_monthly_skips_a_lagging_theta_and_leaves_a_finding(
    tmp_path, fixture_ticks, write_month, monkeypatch, caplog
):
    _three_months(write_month, fixture_ticks)
    old, behind, never = 31_313, 500_000, NEW
    assert (
        _run(
            tmp_path,
            *BACKFILL,
            "--thetas-uri",
            _catalog(tmp_path, old),
            "--to",
            "2017-09",
        )
        == 0
    )
    assert (
        _run(
            tmp_path,
            *BACKFILL,
            "--thetas-uri",
            _catalog(tmp_path, behind, name="b.yaml"),
            "--to",
            "2017-08",
        )
        == 0
    )
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 11, 2))
    catalog = _catalog(tmp_path, old, behind, never, name="all.yaml")
    with caplog.at_level(logging.INFO):
        assert (
            _run(tmp_path, "--mode", "monthly", *SERIES, "--thetas-uri", catalog) == 0
        )
    # Solo `old` (frontera = mes previo) avanza; los otros dos no se procesan.
    assert _partition(tmp_path, old, 10).exists()
    assert not _partition(tmp_path, behind, 10).exists()
    assert not _partition(tmp_path, never, 10).exists()
    rows = {
        json.loads(r["details"])["theta"]: json.loads(r["details"])["frontier"]
        for r in _findings(tmp_path, "theta_behind_frontier")
    }
    assert rows == {behind: "2017-08", never: None}
    assert all(
        (r["severity"], r["status"]) == ("warning", "fail")
        for r in _findings(tmp_path, "theta_behind_frontier")
    )
    # Con el backfill sin from, el rezagado se pone al día y monthly ya no avisa.
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", catalog) == 0
    assert _partition(tmp_path, behind, 10).exists()
    assert _partition(tmp_path, never, 10).exists()


def test_monthly_with_every_theta_behind_does_not_read_l1_and_fails(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 11, 2))
    argv = ["--mode", "monthly", *SERIES, "--thetas-uri", _catalog(tmp_path, *OLD)]
    with caplog.at_level(logging.INFO):
        assert _run(tmp_path, *argv) == 1
    assert "sonda:" not in caplog.text
    assert len(_findings(tmp_path, "theta_behind_frontier")) == 2
    assert not list((tmp_path / "events").rglob("*.parquet"))


def test_monthly_does_not_repeat_a_month_a_theta_already_has(
    tmp_path, fixture_ticks, write_month, monkeypatch, caplog
):
    _three_months(write_month, fixture_ticks)
    catalog = _catalog(tmp_path, *OLD)
    assert _run(tmp_path, *BACKFILL, "--thetas-uri", catalog) == 0
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 11, 2))
    with caplog.at_level(logging.INFO):
        assert (
            _run(tmp_path, "--mode", "monthly", *SERIES, "--thetas-uri", catalog) == 0
        )
    assert "sonda:" not in caplog.text
    assert _findings(tmp_path, "theta_behind_frontier") == []
