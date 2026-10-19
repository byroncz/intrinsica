"""La CLI de la imagen: modos, opciones, códigos de salida y hallazgos (TRD-L3 §8.2 a §8.4)."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from cli_support import emitted, env_for, l3_files, write_catalog
from frames_lake import write_carry_over
from l3_dc_descriptors import SUMMARIES_SCHEMA, cli
from l3_dc_descriptors.cli import UsageError, main, previous_month, resolve_range
from l3_lake import M2, M3, build_lake, fixture_ticks, month_cuts
from pyutils import content_hash

TICKS = fixture_ticks()
SMALL, BIG = 100_000, 2_000_000
SERIES = ["--series-start", "2017-08"]
BACKFILL = ["--mode", "backfill", *SERIES]
PARTITION = "provider=binance/market=spot/asset=BTCUSDT"


@pytest.fixture
def lake(tmp_path):
    return build_lake(tmp_path, TICKS, month_cuts(TICKS, 1500, 3000))


@pytest.fixture
def catalog(tmp_path, lake):
    return write_catalog(tmp_path / "thetas.yaml", lake.thetas)


def run(lake, tmp_path, catalog, *argv, **extra) -> int:
    return main(list(argv), env_for(lake, tmp_path, catalog, **extra))


def theta_text(theta: int) -> str:
    return f"{Decimal(theta).scaleb(-8):.8f}"


def month_dir(root, theta: int, month) -> str:
    return f"{root}/{PARTITION}/theta={theta_text(theta)}/year={month[0]}/month={month[1]:02d}"


def l2_file(lake, theta: int, month, name: str):
    return (
        lake.l2
        / PARTITION
        / (f"theta={theta_text(theta)}/year={month[0]}/month={month[1]:02d}/{name}")
    )


def summaries(lake, theta: int, month):
    return pq.ParquetFile(month_dir(lake.l3, theta, month) + "/summaries.parquet")


def by_month(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(row["month"], []).append(row)
    return out


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
    [("2024-03-01", None), ("2024-13", None), ("2024-3", None), ("2024-04", "2024-03")],
)
def test_resolve_range_invalid(from_, to):
    with pytest.raises(UsageError):
        resolve_range(from_, to)


def test_previous_month_crosses_the_year():
    assert previous_month(date(2024, 1, 15)) == "2023-12"
    assert previous_month(date(2024, 3, 1)) == "2024-02"


def test_backfill_writes_one_row_per_event_and_keeps_the_skeleton(
    lake, tmp_path, catalog
):
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    for month in lake.months:
        for theta in lake.thetas:
            out = summaries(lake, theta, month)
            events = pq.ParquetFile(l2_file(lake, theta, month, "events.parquet"))
            assert out.metadata.num_rows == len(lake.events[theta, month])
            assert [
                out.metadata.row_group(i).num_rows for i in range(out.num_row_groups)
            ] == [
                events.metadata.row_group(i).num_rows
                for i in range(events.num_row_groups)
            ]
            assert out.schema_arrow.remove_metadata().equals(SUMMARIES_SCHEMA)


def test_backfill_findings_carry_the_hashes_and_the_lineage(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    rows = emitted(tmp_path)
    assert {r["layer"] for r in rows} == {"l3"}
    assert {r["mode"] for r in rows} == {"backfill"}
    assert {r["stage"] for r in rows} == {"canonical"}
    assert {r["image_version"] for r in rows} == {"0.4.0+test"}
    assert {r["check_type"] for r in rows} == {"summaries_summary", "unit_timing"}

    summary = emitted(tmp_path, "summaries_summary")
    assert len(summary) == len(lake.thetas) * len(lake.months)
    for row in summary:
        month, theta = (row["year"], row["month"]), row["details"]["theta"]
        theta_int = int(Decimal(theta).scaleb(8))
        written = pq.read_table(
            month_dir(lake.l3, theta_int, month) + "/summaries.parquet",
            schema=SUMMARIES_SCHEMA,
        )
        events = pq.read_table(l2_file(lake, theta_int, month, "events.parquet"))
        carry = pq.read_table(l2_file(lake, theta_int, month, "carry_over.parquet"))
        assert row["details"] == {
            "theta": theta,
            "family": "summaries",
            "rows": written.num_rows,
            "content_hash": content_hash(written),
            "l2_events_content_hash": content_hash(events),
            "l2_carry_over_content_hash": content_hash(carry),
            "unassigned_ticks": row["details"]["unassigned_ticks"],
        }
        assert row["metric_value"] == written.num_rows
        assert isinstance(row["details"]["unassigned_ticks"], int)
    assert len(emitted(tmp_path, "unit_timing")) == len(lake.months)


def test_a_second_run_writes_nothing_and_does_not_need_l1(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--to", "2017-10") == 0
    before = {p: p.stat().st_mtime_ns for p in l3_files(lake)}
    findings = len(emitted(tmp_path))
    for path in lake.l1.rglob("consolidated.parquet"):
        path.unlink()
    assert run(lake, tmp_path, catalog, *BACKFILL, "--to", "2017-10") == 0
    assert {p: p.stat().st_mtime_ns for p in l3_files(lake)} == before
    assert len(emitted(tmp_path)) == findings


def test_the_default_end_is_the_last_consolidated_month(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    assert {r["month"] for r in emitted(tmp_path, "unit_timing")} == {8, 9, 10}
    # Un provisional posterior no cuenta (ADR-L3-08).
    (lake.l1 / PARTITION / "year=2017/month=11").mkdir(parents=True)
    (lake.l1 / PARTITION / "year=2017/month=11/provisional-day=01.parquet").touch()
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    assert len(emitted(tmp_path, "unit_timing")) == 3


def test_the_frontier_decides_where_each_theta_resumes(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--to", "2017-09") == 0
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    months = by_month(emitted(tmp_path, "unit_timing"))
    assert {m: len(rows) for m, rows in months.items()} == {8: 1, 9: 1, 10: 1}


def test_a_new_theta_catches_up_from_the_series_start(lake, tmp_path):
    first = write_catalog(tmp_path / "a.yaml", lake.thetas[:3])
    assert run(lake, tmp_path, first, *BACKFILL) == 0
    full = write_catalog(tmp_path / "b.yaml", lake.thetas)
    assert run(lake, tmp_path, full, *BACKFILL) == 0
    rows = emitted(tmp_path, "summaries_summary")
    new = {theta_text(t) for t in lake.thetas[3:]}
    assert {r["month"] for r in rows if r["details"]["theta"] in new} == {8, 9, 10}
    assert len(rows) == len(lake.thetas) * 3


def test_thetas_in_decimal_narrow_the_run(lake, tmp_path, catalog):
    assert (
        run(lake, tmp_path, catalog, *BACKFILL, "--thetas", "0.00100000, 0.02000000")
        == 0
    )
    done = {r["details"]["theta"] for r in emitted(tmp_path, "summaries_summary")}
    assert done == {"0.00100000", "0.02000000"}
    assert len(l3_files(lake)) == 2 * len(lake.months)


@pytest.mark.parametrize("thetas", ["0.00100001", "abc", "0.0010000012", "nan"])
def test_thetas_outside_the_catalog_are_a_usage_error(
    lake, tmp_path, catalog, thetas, capsys
):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--thetas", thetas) == 2
    assert l3_files(lake) == []
    assert "--thetas" in capsys.readouterr().err


def test_force_reprocesses_the_range_with_the_same_content_hash(
    lake, tmp_path, catalog
):
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    first = {
        (r["month"], r["details"]["theta"]): r["details"]["content_hash"]
        for r in emitted(tmp_path, "summaries_summary")
    }
    before = {p: p.stat().st_mtime_ns for p in l3_files(lake)}
    assert run(lake, tmp_path, catalog, *BACKFILL, "--force", "--from", "2017-09") == 0
    rows = emitted(tmp_path, "summaries_summary")
    again = rows[len(first) :]
    assert {r["month"] for r in again} == {9, 10}
    assert len(again) == 2 * len(lake.thetas)
    for row in again:
        assert (
            first[row["month"], row["details"]["theta"]]
            == row["details"]["content_hash"]
        )
    after = {p: p.stat().st_mtime_ns for p in l3_files(lake)}
    changed = {p for p in after if after[p] != before[p]}
    assert all("month=08" not in str(p) for p in changed)
    assert len(changed) == 2 * len(lake.thetas)


def test_force_without_from_is_a_usage_error(lake, tmp_path, catalog, capsys):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--force") == 2
    assert "--force exige --from" in capsys.readouterr().err
    assert l3_files(lake) == []


@pytest.mark.parametrize("family", ["path", "fit", ""])
def test_another_family_is_a_usage_error(lake, tmp_path, catalog, family, capsys):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--family", family) == 2
    assert "--family" in capsys.readouterr().err
    assert l3_files(lake) == []


def test_the_summaries_family_is_the_default_and_can_be_named(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--family", "summaries") == 0
    assert len(l3_files(lake)) == len(lake.thetas) * len(lake.months)


@pytest.mark.parametrize("index", ["1", "7", "x"])
def test_only_task_zero_runs(lake, tmp_path, catalog, index):
    code = run(lake, tmp_path, catalog, *BACKFILL, CLOUD_RUN_TASK_INDEX=index)
    assert code == 2
    assert l3_files(lake) == []


def test_task_zero_runs(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, CLOUD_RUN_TASK_INDEX="0") == 0


def test_the_series_start_can_come_from_the_environment(lake, tmp_path, catalog):
    code = run(
        lake,
        tmp_path,
        catalog,
        "--mode",
        "backfill",
        L3_SERIES_START="2017-08",
    )
    assert code == 0
    assert len(l3_files(lake)) == len(lake.thetas) * len(lake.months)


def test_the_series_start_is_declared_not_inferred(lake, tmp_path, catalog, capsys):
    assert run(lake, tmp_path, catalog, "--mode", "backfill") == 2
    assert "--series-start" in capsys.readouterr().err


def test_a_range_before_the_series_is_a_usage_error(lake, tmp_path, catalog, capsys):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--from", "2017-07") == 2
    assert "anterior" in capsys.readouterr().err


@pytest.mark.parametrize("name", cli.ROOT_VARS)
def test_every_root_is_required(lake, tmp_path, catalog, name, capsys):
    env = env_for(lake, tmp_path, catalog)
    del env[name]
    assert main(BACKFILL, env) == 2
    assert name in capsys.readouterr().err


def test_a_missing_catalog_uri_is_a_usage_error_with_a_finding(lake, tmp_path, capsys):
    assert run(lake, tmp_path, None, *BACKFILL) == 2
    (row,) = emitted(tmp_path)
    assert row["check_type"] == "theta_catalog_invalid"
    assert row["severity"] == "error" and row["status"] == "fail"
    assert row["details"]["source"] is None
    assert "L3_THETAS_URI" in row["details"]["problems"][0]
    assert l3_files(lake) == []


def test_the_catalog_uri_can_come_from_the_option(lake, tmp_path, catalog):
    code = main([*BACKFILL, "--thetas-uri", catalog], env_for(lake, tmp_path, None))
    assert code == 0


def test_an_invalid_catalog_is_a_usage_error_with_a_finding(lake, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("scale: 100\nthetas: [1, 1]\n")
    assert run(lake, tmp_path, str(bad), *BACKFILL) == 2
    (row,) = emitted(tmp_path)
    assert row["check_type"] == "theta_catalog_invalid"
    assert row["details"]["source"] == str(bad)
    assert len(row["details"]["problems"]) >= 2
    assert l3_files(lake) == []


def test_an_unreadable_catalog_is_invalid(lake, tmp_path):
    code = run(lake, tmp_path, str(tmp_path / "no-existe.yaml"), *BACKFILL)
    assert code == 2
    (row,) = emitted(tmp_path)
    assert row["check_type"] == "theta_catalog_invalid"
    assert "no se pudo leer" in row["details"]["problems"][0]


def test_dropping_a_theta_from_the_catalog_is_only_reported(lake, tmp_path):
    full = write_catalog(tmp_path / "a.yaml", lake.thetas)
    assert run(lake, tmp_path, full, *BACKFILL) == 0
    smaller = write_catalog(tmp_path / "b.yaml", lake.thetas[:-1])
    assert run(lake, tmp_path, smaller, *BACKFILL) == 0
    (row,) = emitted(tmp_path, "theta_config_drift")
    assert row["severity"] == "info" and row["status"] == "pass"
    assert row["details"] == {
        "theta": theta_text(lake.thetas[-1]),
        "months": 3,
        "last_month": "2017-10",
    }
    assert len(l3_files(lake)) == len(lake.thetas) * 3


def test_a_missing_l1_month_stops_the_range(lake, tmp_path, catalog):
    (lake.l1 / PARTITION / "year=2017/month=09/consolidated.parquet").unlink()
    assert run(lake, tmp_path, catalog, *BACKFILL) == 1
    (row,) = emitted(tmp_path, "input_missing")
    assert row["severity"] == "error" and row["month"] == 9
    assert row["details"]["expected_path"].endswith(
        "year=2017/month=09/consolidated.parquet"
    )
    assert {r["month"] for r in emitted(tmp_path, "unit_timing")} == {8}


def test_a_provisional_only_month_is_reported(lake, tmp_path, catalog):
    month = lake.l1 / PARTITION / "year=2017/month=09"
    (month / "consolidated.parquet").rename(month / "provisional-day=01.parquet")
    assert run(lake, tmp_path, catalog, *BACKFILL) == 1
    (row,) = emitted(tmp_path, "input_provisional_only")
    assert row["details"]["provisionals"] == 1
    assert emitted(tmp_path, "input_missing") == []


def test_l2_behind_stops_only_the_theta_and_fails_the_run(lake, tmp_path, catalog):
    for name in ("events.parquet", "carry_over.parquet"):
        l2_file(lake, BIG, M3, name).unlink()
    assert run(lake, tmp_path, catalog, *BACKFILL) == 1
    rows = emitted(tmp_path, "l2_input_missing")
    assert [r["details"] for r in rows] == [
        {"theta": theta_text(BIG), "month": "2017-10", "file": "events.parquet"},
        {"theta": theta_text(BIG), "month": "2017-10", "file": "carry_over.parquet"},
    ]
    assert all(r["severity"] == "error" and r["month"] == 10 for r in rows)
    # Los demás θ llegaron a octubre; este se quedó en septiembre.
    for theta in lake.thetas:
        published = Path(month_dir(lake.l3, theta, M3), "summaries.parquet").exists()
        assert published == (theta != BIG)
    assert summaries(lake, BIG, M2).metadata.num_rows == len(lake.events[BIG, M2])


def test_the_range_of_a_theta_without_l2_is_not_retried_in_later_months(
    lake, tmp_path, catalog
):
    for month in (M2, M3):
        for name in ("events.parquet", "carry_over.parquet"):
            l2_file(lake, SMALL, month, name).unlink()
    assert run(lake, tmp_path, catalog, *BACKFILL) == 1
    # Un solo mes con hallazgos, el primero que falta; octubre no se vuelve a reportar.
    assert {r["month"] for r in emitted(tmp_path, "l2_input_missing")} == {9}


def test_a_unit_error_becomes_its_finding_and_stops_the_range(lake, tmp_path, catalog):
    write_carry_over(lake.l2, SMALL, M2, 1, has_pending=False)
    assert run(lake, tmp_path, catalog, *BACKFILL) == 1
    (row,) = emitted(tmp_path, "l2_state_inconsistent")
    assert row["severity"] == "error" and row["status"] == "fail"
    assert row["month"] == 9
    assert row["details"] == {
        "theta": theta_text(SMALL),
        "month": "2017-09",
        "direction": 1,
        "has_pending_event": False,
    }
    # El mes 08 quedó escrito; 09 no dejó nada y 10 no se tocó.
    assert {r["month"] for r in emitted(tmp_path, "unit_timing")} == {8}
    assert len(l3_files(lake)) == len(lake.thetas)


def test_unit_timing_has_what_the_probe_needs(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL) == 0
    for row in emitted(tmp_path, "unit_timing"):
        details = row["details"]
        for name in (
            "wall_s",
            "read_s",
            "aggregate_s",
            "write_s",
            "rss_peak_mib",
            "row_groups_month",
            "row_groups_previous",
            "events",
            "ticks",
            "theta_stats",
            "cores",
            "cores_source",
        ):
            assert name in details
        assert row["metric_value"] == details["wall_s"]
        assert len(details["theta_stats"]) == len(lake.thetas)
        assert details["events"] == sum(t["events"] for t in details["theta_stats"])


# monthly


def monthly(lake, tmp_path, catalog, *argv, **extra) -> int:
    return run(lake, tmp_path, catalog, "--mode", "monthly", *SERIES, *argv, **extra)


def test_monthly_defaults_to_the_previous_month(lake, tmp_path, catalog, monkeypatch):
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 8, 20))
    assert monthly(lake, tmp_path, catalog) == 2  # 2017-07 es anterior a la serie
    monkeypatch.setattr(cli, "_today", lambda: date(2017, 9, 3))
    assert monthly(lake, tmp_path, catalog) == 0
    (row, *_) = emitted(tmp_path, "unit_timing")
    assert (row["mode"], row["year"], row["month"]) == ("monthly", 2017, 8)


def test_monthly_advances_the_thetas_that_reached_the_previous_month(
    lake, tmp_path, catalog
):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--to", "2017-08") == 0
    assert monthly(lake, tmp_path, catalog, "--from", "2017-09") == 0
    rows = emitted(tmp_path, "summaries_summary")
    assert {(r["mode"], r["month"]) for r in rows} == {("backfill", 8), ("monthly", 9)}
    assert len(l3_files(lake)) == 2 * len(lake.thetas)


def test_monthly_leaves_a_theta_that_already_has_the_month(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--to", "2017-09") == 0
    before = len(emitted(tmp_path))
    assert monthly(lake, tmp_path, catalog, "--from", "2017-09") == 0
    assert len(emitted(tmp_path)) == before


def test_monthly_reports_the_stragglers_and_goes_on(lake, tmp_path, catalog):
    ahead = write_catalog(tmp_path / "ahead.yaml", lake.thetas[:-1])
    assert run(lake, tmp_path, ahead, *BACKFILL, "--to", "2017-08") == 0
    assert monthly(lake, tmp_path, catalog, "--from", "2017-09") == 0
    (row,) = emitted(tmp_path, "theta_behind_frontier")
    assert row["severity"] == "warning" and row["status"] == "fail"
    assert row["details"] == {
        "theta": theta_text(lake.thetas[-1]),
        "frontier": None,
        "reason": "l3_frontier",
    }
    assert len(l3_files(lake)) == len(lake.thetas[:-1]) * 2


def test_monthly_names_l2_as_the_reason_when_it_has_no_month(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, "--to", "2017-08") == 0
    for name in ("events.parquet", "carry_over.parquet"):
        l2_file(lake, BIG, M2, name).unlink()
    assert monthly(lake, tmp_path, catalog, "--from", "2017-09") == 0
    (row,) = emitted(tmp_path, "theta_behind_frontier")
    assert row["details"] == {
        "theta": theta_text(BIG),
        "frontier": "2017-08",
        "reason": "l2_missing",
    }
    assert len(l3_files(lake)) == 2 * len(lake.thetas) - 1


def test_monthly_without_any_ready_theta_fails_without_reading_l1(
    lake, tmp_path, catalog
):
    (lake.l1 / PARTITION / "year=2017/month=09/consolidated.parquet").unlink()
    assert monthly(lake, tmp_path, catalog, "--from", "2017-09") == 1
    rows = emitted(tmp_path, "theta_behind_frontier")
    assert len(rows) == len(lake.thetas)
    assert {r["details"]["reason"] for r in rows} == {"l3_frontier"}
    assert emitted(tmp_path, "input_missing") == []
    assert l3_files(lake) == []


def test_monthly_starts_the_series_without_a_frontier(lake, tmp_path, catalog):
    assert monthly(lake, tmp_path, catalog, "--from", "2017-08") == 0
    assert len(l3_files(lake)) == len(lake.thetas)
    assert emitted(tmp_path, "theta_behind_frontier") == []


@pytest.mark.parametrize("option", [["--to", "2017-09"], ["--force"]])
def test_monthly_takes_one_month_and_does_not_force(lake, tmp_path, catalog, option):
    assert monthly(lake, tmp_path, catalog, "--from", "2017-08", *option) == 2
    assert l3_files(lake) == []


def test_l3_profile_makes_the_unit_emit_unit_profile(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, L3_PROFILE="1") == 0
    profiles = emitted(tmp_path, "unit_profile")
    assert {r["month"] for r in profiles} == {
        r["month"] for r in emitted(tmp_path, "unit_timing")
    }
    assert len(profiles) == len(lake.months)


def test_without_l3_profile_the_cli_emits_no_unit_profile(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, L3_PROFILE="0") == 0
    assert emitted(tmp_path, "unit_profile") == []


# L3_THREADS y L3_PREFETCH (ITSC-357)


def test_l3_threads_and_prefetch_reach_the_unit(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, L3_THREADS="2", L3_PREFETCH="0") == 0
    for row in emitted(tmp_path, "unit_timing"):
        assert (row["details"]["threads"], row["details"]["prefetch"]) == (2, 0)


def test_without_l3_threads_the_pool_follows_the_vcpus(lake, tmp_path, catalog):
    assert run(lake, tmp_path, catalog, *BACKFILL, L3_THREADS="", L3_PREFETCH="") == 0
    for row in emitted(tmp_path, "unit_timing"):
        details = row["details"]
        assert 1 <= details["threads"] <= len(lake.thetas)
        assert details["threads"] <= max(1, int(details["cores"]))
        assert details["prefetch"] == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("L3_THREADS", "0"),
        ("L3_THREADS", "cuatro"),
        ("L3_PREFETCH", "-1"),
        ("L3_PREFETCH", "1.5"),
    ],
)
def test_a_bad_l3_threads_or_prefetch_is_a_usage_error(
    lake, tmp_path, catalog, name, value
):
    assert run(lake, tmp_path, catalog, *BACKFILL, **{name: value}) == 2
    assert l3_files(lake) == []
