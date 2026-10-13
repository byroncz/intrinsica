"""`unit_timing`: lo que ITSC-335 necesita para el umbral 3 de TRD-L3 §10.3."""

from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import pytest
from dc_frames import lake as frames_lake
from l3_dc_descriptors import timing
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.pipeline import process_unit
from l3_lake import KEY, M1, M2, M3, build_lake, fixture_ticks, month_cuts

TICKS = fixture_ticks()


@pytest.fixture
def lake(tmp_path):
    return build_lake(tmp_path, TICKS, month_cuts(TICKS, 1500, 3000))


def context(lake, tmp_path) -> RunContext:
    return RunContext(
        mode="backfill",
        run_id="run",
        image_version="0.4.0+test",
        series_start=M1,
        landing_root=str(lake.l1),
        events_root=str(lake.l2),
        summaries_root=str(lake.l3),
        dq_root=str(tmp_path / "dq"),
    )


def unit_of(month) -> Unit:
    return Unit(*month)


def l1_row_groups(lake, month) -> int:
    path = lake.l1 / "provider=binance/market=spot/asset=BTCUSDT"
    path = path / f"year={month[0]}/month={month[1]:02d}/consolidated.parquet"
    return pq.ParquetFile(path).num_row_groups


class Spy:
    """Cuenta los `read_row_group` de L1 por debajo del gancho, como la sonda de ops_tools."""

    def __init__(self, monkeypatch) -> None:
        self.reads: list[tuple[str, int]] = []
        real = timing._OPEN

        def spy(path, what):
            parquet = real(path, what)
            read = parquet.read_row_group

            def read_row_group(i, columns=None, **kwargs):
                if path.endswith("consolidated.parquet"):
                    self.reads.append((path, i))
                return read(i, columns=columns, **kwargs)

            parquet.read_row_group = read_row_group
            return parquet

        monkeypatch.setattr(timing, "_OPEN", spy)

    def of(self, month) -> int:
        marker = f"year={month[0]}/month={month[1]:02d}/"
        return sum(marker in path for path, _ in self.reads)


def test_it_counts_the_row_groups_the_reader_decodes(lake, tmp_path, monkeypatch):
    spy = Spy(monkeypatch)
    ctx = context(lake, tmp_path)
    for month in lake.months:
        before = len(spy.reads)
        result = process_unit(unit_of(month), ctx, lake.thetas)
        assert result.ok
        counts = result.timing.row_groups
        decoded = spy.reads[before:]
        assert counts.month == sum(f"month={month[1]:02d}/" in p for p, _ in decoded)
        assert counts.previous == len(decoded) - counts.month
        assert counts.month_total == l1_row_groups(lake, month)
        # Cada row group se decodifica una sola vez (ADR-L3-14), también los de meses
        # anteriores que necesitan los eventos que cruzan el borde.
        assert counts.repeated == 0
        assert len(set(decoded)) == len(decoded)
        assert counts.bytes_in > 0
    assert spy.of(M1) >= l1_row_groups(lake, M1)


def test_events_that_cross_the_month_edge_read_previous_months(lake, tmp_path):
    ctx = context(lake, tmp_path)
    first = process_unit(unit_of(M1), ctx, lake.thetas).timing.row_groups
    later = process_unit(unit_of(M3), ctx, lake.thetas).timing.row_groups
    assert first.previous == 0
    assert later.previous > 0


def test_the_hook_does_not_touch_the_reader_outside_a_unit(lake):
    path = str(
        lake.l1 / "provider=binance/market=spot/asset=BTCUSDT/year=2017/month=08"
    )
    with pytest.raises(Exception):  # noqa: B017 - solo importa que no sea un conteo
        frames_lake.open_parquet(path + "/no-existe.parquet", "L1")
    parquet = frames_lake.open_parquet(path + "/consolidated.parquet", "L1")
    try:
        assert "read_row_group" not in vars(parquet)
    finally:
        parquet.close()


def test_the_phases_explain_the_wall(lake, tmp_path):
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    details = result.timing.details()
    phases = ("gate_s", "read_s", "aggregate_s", "write_s", "reconcile_s", "lineage_s")
    assert all(details[name] >= 0 for name in (*phases, "other_s"))
    assert sum(details[name] for name in phases) <= details["wall_s"] + 0.01
    assert details["read_s"] > 0 and details["write_s"] > 0
    assert details["lineage_s"] > 0


def test_it_reports_events_and_ticks_per_theta_and_in_total(lake, tmp_path):
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    stats = {s.theta: s for s in result.timing.thetas}
    assert set(stats) == set(lake.theta_texts)
    for theta, text in zip(lake.thetas, lake.theta_texts, strict=True):
        rows = lake.events[theta, M2]
        assert stats[text].events == len(rows)
    totals = result.timing.details()
    assert totals["events"] == sum(s.events for s in stats.values())
    assert totals["ticks"] == sum(s.ticks for s in stats.values())
    assert totals["thetas"] == len(lake.thetas)
    # Σ_cerrados cuenta los ticks de los meses anteriores: son más que los del mes.
    assert any(s.ticks_before_month > 0 for s in stats.values())


def test_it_reports_memory_and_the_cpu_limit(lake, tmp_path):
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    details = result.timing.details()
    assert details["rss_peak_mib"] > 0
    assert details["rss_peak_scope"] in {"unit", "process"}
    assert details["cores"] > 0 and details["cores_visible"] >= 1
    assert details["cores_source"] in {"cgroup-v2", "cgroup-v1", "affinity"}


def test_units_in_parallel_threads_do_not_mix_their_counts(lake, tmp_path):
    ctx = context(lake, tmp_path)
    serial = {
        month: process_unit(unit_of(month), ctx, lake.thetas).timing.row_groups
        for month in lake.months
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda m: process_unit(unit_of(m), ctx, lake.thetas), lake.months)
        )
    for month, result in zip(lake.months, results, strict=True):
        assert result.ok
        counts, expected = result.timing.row_groups, serial[month]
        assert (counts.month, counts.previous, counts.repeated, counts.bytes_in) == (
            expected.month,
            expected.previous,
            expected.repeated,
            expected.bytes_in,
        )


def test_a_failed_unit_returns_its_finding_and_no_timing(lake, tmp_path):
    path = lake.l2 / "provider=binance/market=spot/asset=BTCUSDT"
    (path / "theta=0.02000000/year=2017/month=09/events.parquet").unlink()
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert not result.ok and result.timing is None
    (finding,) = result.findings
    assert finding.check_type == "l2_input_missing"
    assert finding.layer == "l3" and finding.mode == "backfill"
    assert (finding.year, finding.month) == (2017, 9)
    assert KEY == ("binance", "spot", "BTCUSDT")
