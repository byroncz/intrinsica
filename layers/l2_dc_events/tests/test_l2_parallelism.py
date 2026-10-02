"""ITSC-290: hilos del fan-out, escritores y lectura anticipada no cambian la salida."""

import dataclasses

import pytest
from l2_dc_events import pipeline
from l2_dc_events.cpu import CpuLimit
from l2_dc_events.pipeline import RunContext, Unit, process_unit

AUG = Unit(2017, 8)

# (cores, visibles): un core, cuatro cores y cuotas fraccionarias como las que
# entrega Cloud Run con 2 vCPU (1,61, 1,86 y 1,97).
LIMITS = [(1.0, 1), (4.0, 4), (1.97, 4), (1.61, 6), (0.5, 4)]


@pytest.fixture
def ctx(write_month, tmp_path):
    return RunContext(
        mode="backfill",
        run_id="test",
        image_version="0.1.0+test",
        series_start=(2017, 8),
        landing_root=write_month.landing,
        events_root=tmp_path / "events",
        dq_root=tmp_path / "dq",
    )


def _hashes(ctx, monkeypatch, limit, read_ahead):
    monkeypatch.setattr(
        pipeline.cpu, "cpu_limit", lambda: CpuLimit(limit[0], limit[1], "cgroup-v1")
    )
    result = process_unit(AUG, dataclasses.replace(ctx, read_ahead=read_ahead))
    return result.events_hashes, result.carry_over_hashes, result.timing


def test_hashes_do_not_depend_on_the_cpu_limit_nor_the_read_ahead(
    fixture_ticks, write_month, ctx, monkeypatch
):
    write_month(fixture_ticks, row_group_size=500)
    base_events, base_carry, _ = _hashes(ctx, monkeypatch, (1.0, 1), 0)
    assert len(set(base_events)) > 1
    for limit in LIMITS:
        for read_ahead in (0, 1, 2, 4):
            events, carry, _ = _hashes(ctx, monkeypatch, limit, read_ahead)
            assert (events, carry) == (base_events, base_carry), (limit, read_ahead)


def test_the_unit_uses_the_threads_that_the_cpu_module_decides(
    fixture_ticks, write_month, ctx, monkeypatch
):
    write_month(fixture_ticks, row_group_size=1_000)
    _, _, timing = _hashes(ctx, monkeypatch, (1.97, 4), 2)
    assert (timing.fanout_threads, timing.write_workers) == (2, 2)
    _, _, timing = _hashes(ctx, monkeypatch, (0.5, 4), 2)
    assert (timing.fanout_threads, timing.write_workers) == (1, 1)


def test_detect_cpu_is_the_thread_cpu_during_the_fan_out(
    fixture_ticks, write_month, ctx, monkeypatch
):
    write_month(fixture_ticks, row_group_size=1_000)
    _, _, timing = _hashes(ctx, monkeypatch, (2.0, 4), 2)
    assert 0 < timing.detect_cpu_s <= timing.detect_s + 0.05
