"""`unit_timing`: lo que ITSC-335 necesita para el umbral 3 de TRD-L3 §10.3."""

import cProfile
import pstats
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pyarrow.parquet as pq
import pytest
from dc_frames import lake as frames_lake
from dq import Severity, Status
from l3_dc_descriptors import memory, pipeline, timing
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

        def spy(path, what, metadata=None, pre_buffer=False):
            parquet = real(path, what, metadata, pre_buffer)
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


SERIAL = {"threads": 1, "readers": 1, "inflight": 1, "writers": 1, "engine_threads": 1}


def test_the_phases_explain_the_wall(lake, tmp_path):
    # Las fases del hilo principal son tramos seguidos de la pared y suman a lo más
    # `wall_s`; `read_s`, `write_s` y `read_wall_s` corren dentro de `aggregate_s`.
    ctx = replace(context(lake, tmp_path), **SERIAL)
    result = process_unit(unit_of(M2), ctx, lake.thetas)
    details = result.timing.details()
    phases = (
        "gate_s",
        "setup_s",
        "aggregate_s",
        "publish_s",
        "reconcile_s",
    )
    assert all(details[name] >= 0 for name in (*phases, "other_s"))
    assert sum(details[name] for name in phases) <= details["wall_s"] + 0.01
    assert details["read_s"] > 0 and details["write_s"] > 0
    assert details["read_wall_s"] > 0 and "lineage_s" not in details
    assert details["read_s"] <= details["aggregate_s"] + 0.01
    assert details["write_s"] <= details["aggregate_s"] + 0.05
    assert details["read_wall_s"] <= details["wall_s"]


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


# CPU y perfil (ITSC-356)


def test_it_reports_the_cpu_of_each_unit_within_what_the_cores_allow(lake, tmp_path):
    ctx = context(lake, tmp_path)
    # Varias unidades en el mismo proceso: getrusage es acumulado, cada una usa su diferencia.
    for month in lake.months:
        result = process_unit(unit_of(month), ctx, lake.thetas)
        details = result.timing.details()
        assert details["cpu_user_s"] >= 0 and details["cpu_system_s"] >= 0
        # `details()` redondea a 3 decimales. Comparar un valor redondeado con la suma o el
        # cociente de otros redondeados acumula errores y falló a veces con 0,002 fijo
        # (ITSC-368). Cada valor publicado se compara con el sin redondear del `Timing`: la
        # diferencia es el redondeo de ese único valor, a lo más 0,0005 (más margen de coma flotante).
        timing = result.timing
        # Las definiciones se afirman sobre los valores sin redondear, que no tienen error.
        assert timing.cpu_s == pytest.approx(timing.cpu_user_s + timing.cpu_system_s)
        assert timing.cpu_utilization == pytest.approx(
            timing.cpu_s / (timing.wall_s * timing.cpu.cores)
        )
        rounding = 0.0005 + 1e-9
        assert details["cpu_user_s"] == pytest.approx(timing.cpu_user_s, abs=rounding)
        assert details["cpu_system_s"] == pytest.approx(
            timing.cpu_system_s, abs=rounding
        )
        assert details["cpu_s"] == pytest.approx(timing.cpu_s, abs=rounding)
        assert 0 < details["cpu_s"] <= details["wall_s"] * details["cores"] * 1.05
        assert details["cpu_utilization"] == pytest.approx(
            timing.cpu_utilization, abs=rounding
        )


def test_it_does_not_profile_without_the_flag(lake, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("se instaló el perfilador sin L3_PROFILE")

    monkeypatch.setattr(pipeline.cProfile, "Profile", boom)
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert result.ok
    assert "unit_profile" not in {f.check_type for f in result.findings}


def test_it_emits_unit_profile_with_the_top_functions(lake, tmp_path):
    ctx = replace(context(lake, tmp_path), profile=True)
    result = process_unit(unit_of(M2), ctx, lake.thetas)
    assert result.ok
    (profile,) = (f for f in result.findings if f.check_type == "unit_profile")
    (timing,) = (f for f in result.findings if f.check_type == "unit_timing")
    assert (profile.severity, profile.status) == (Severity.INFO, Status.PASS)
    assert (profile.year, profile.month) == M2
    details = profile.details
    assert details["wall_s"] == pytest.approx(result.timing.wall_s, abs=0.001)
    entry = {"function", "file", "line", "ncalls", "tottime_s", "cumtime_s"}
    for key, column in (("top_tottime", "tottime_s"), ("top_cumtime", "cumtime_s")):
        rows = details[key]
        assert len(rows) == 40 and all(set(r) == entry for r in rows)
        values = [r[column] for r in rows]
        assert values == sorted(values, reverse=True)
        assert all(r["ncalls"] >= 1 for r in rows)
    # El perfil ve la unidad entera: la función más alta por tiempo acumulado la contiene.
    assert (
        details["top_cumtime"][0]["cumtime_s"] >= details["top_tottime"][0]["tottime_s"]
    )
    assert any(r["function"] == "summarize_month" for r in details["top_cumtime"])
    # unit_timing conserva los mismos campos con y sin perfil.
    plain = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    (plain_timing,) = (f for f in plain.findings if f.check_type == "unit_timing")
    assert timing.details.keys() == plain_timing.details.keys()


def test_a_failed_unit_has_no_profile(lake, tmp_path):
    path = lake.l2 / "provider=binance/market=spot/asset=BTCUSDT"
    (path / "theta=0.02000000/year=2017/month=09/events.parquet").unlink()
    ctx = replace(context(lake, tmp_path), profile=True)
    result = process_unit(unit_of(M2), ctx, lake.thetas)
    assert not result.ok
    assert {f.check_type for f in result.findings} == {"l2_input_missing"}


# Hilos (ITSC-357)


@pytest.mark.parametrize(
    "knobs",
    [
        SERIAL,
        {"threads": 2, "readers": 2, "inflight": 2, "writers": 2, "engine_threads": 2},
        {"threads": 4, "readers": 1, "inflight": 4, "writers": 4, "engine_threads": 4},
        {"threads": 50, "readers": 2, "inflight": 1, "writers": 1, "engine_threads": 3},
    ],
)
def test_the_counts_do_not_depend_on_the_knobs(lake, tmp_path, knobs):
    # El gancho de row groups vive en una ContextVar: con el lector en otro hilo, un
    # contexto sin propagar dejaría `row_groups_month` en cero.
    base = process_unit(
        unit_of(M3), replace(context(lake, tmp_path), **SERIAL), lake.thetas
    ).timing
    ctx = replace(context(lake, tmp_path), **knobs)
    timing = process_unit(unit_of(M3), ctx, lake.thetas).timing
    assert timing.row_groups.month > 0 and timing.row_groups.previous > 0
    counts, expected = timing.row_groups, base.row_groups
    assert (
        counts.month,
        counts.previous,
        counts.repeated,
        counts.bytes_in,
        counts.month_total,
    ) == (
        expected.month,
        expected.previous,
        expected.repeated,
        expected.bytes_in,
        expected.month_total,
    )


def test_it_reports_the_knobs_it_used(lake, tmp_path):
    knobs = {
        "threads": 2,
        "readers": 1,
        "inflight": 3,
        "writers": 2,
        "engine_threads": 3,
    }
    ctx = replace(context(lake, tmp_path), **knobs)
    details = process_unit(unit_of(M2), ctx, lake.thetas).timing.details()
    assert {name: details[name] for name in knobs} == knobs
    assert 1 <= details["inflight_max"] <= 3
    # Nunca más hilos de E/S que θ.
    ctx = replace(context(lake, tmp_path), threads=50)
    details = process_unit(unit_of(M2), ctx, lake.thetas).timing.details()
    assert details["threads"] == len(lake.thetas)
    assert (details["readers"], details["inflight"], details["writers"]) == (2, 4, 12)
    default = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert 1 <= default.timing.details()["threads"] <= len(lake.thetas)


def test_the_engine_phase_and_the_read_volume_are_measured(lake, tmp_path):
    knobs = {
        "threads": 4,
        "readers": 2,
        "inflight": 2,
        "writers": 2,
        "engine_threads": 2,
    }
    ctx = replace(context(lake, tmp_path), **knobs)
    timing = process_unit(unit_of(M2), ctx, lake.thetas).timing
    details = timing.details()
    assert details["aggregate_s"] > 0 and details["write_s"] > 0
    assert details["other_s"] >= 0
    assert 0 < details["engine_busy"] <= 1
    # Un ítem por (row group, θ con eventos): los row groups que el motor recibió.
    assert details["items"] > 0
    assert details["items"] % len(lake.thetas) == 0
    assert details["items"] // len(lake.thetas) == timing.row_groups.month + (
        timing.row_groups.previous
    )
    # Los bytes leídos suman los de L1 que cuenta el gancho y los de L2.
    assert details["read_bytes"] > details["bytes_in"] > 0
    assert 0 < details["read_wall_s"] <= details["wall_s"]
    # Condición de merge de ITSC-357: el RSS del fixture con hilos queda dentro del límite.
    assert 0 < details["rss_peak_mib"] <= 2048


def test_the_profile_sees_the_calls_of_the_pool_threads():
    # profiling.py lo afirma: desde Python 3.12 `cProfile` (sys.monitoring) cuenta las
    # llamadas de todos los hilos, no solo las del que lo activa.
    def busy(n):
        return sum(range(n))

    def run():
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(busy, [1000] * 12))

    profiler = cProfile.Profile()
    profiler.runcall(run)
    calls = {f[2]: v[1] for f, v in pstats.Stats(profiler).stats.items()}
    assert calls["busy"] == 12


def test_it_reports_the_kernel_counters_that_explain_the_system_cpu(lake, tmp_path):
    # Fallos de página y cambios de contexto de la unidad (diferencia de `getrusage`): con
    # ellos se separa la CPU de sistema en memoria, candados y red (ADR-L3-16, ITSC-365).
    ctx = context(lake, tmp_path)
    for month in lake.months:
        details = process_unit(unit_of(month), ctx, lake.thetas).timing.details()
        assert details["minor_faults"] > 0
        assert details["major_faults"] >= 0
        assert details["voluntary_switches"] > 0
        assert details["involuntary_switches"] >= 0


def test_unit_timing_records_the_mmap_threshold(lake, tmp_path, monkeypatch):
    # El umbral que `tune_allocators` dejó fijado viaja en el hallazgo, para que cada
    # corrida diga con cuál corrió (None si quedó dinámico).
    monkeypatch.setenv("MALLOC_MMAP_THRESHOLD_", "65536")
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert result.timing.details()["mmap_threshold"] == 65536
    (row,) = (f for f in result.findings if f.check_type == "unit_timing")
    assert row.details["mmap_threshold"] == 65536
    monkeypatch.delenv("MALLOC_MMAP_THRESHOLD_")
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert result.timing.details()["mmap_threshold"] is None


def test_unit_timing_records_the_trim_threshold(lake, tmp_path, monkeypatch):
    # Lo que la imagen fijó en `MALLOC_TRIM_THRESHOLD_` viaja en el hallazgo (None si no).
    monkeypatch.setenv("MALLOC_TRIM_THRESHOLD_", "1073741824")
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert result.timing.details()["trim_threshold"] == 1073741824
    (row,) = (f for f in result.findings if f.check_type == "unit_timing")
    assert row.details["trim_threshold"] == 1073741824
    monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_")
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    assert result.timing.details()["trim_threshold"] is None


def test_unit_timing_records_the_rss_before_and_after_the_trim(lake, tmp_path):
    # ITSC-371: la unidad devuelve la memoria al kernel al cerrar y deja el RSS de antes y de
    # después (VmRSS, no el pico) en el hallazgo.
    result = process_unit(unit_of(M2), context(lake, tmp_path), lake.thetas)
    details = result.timing.details()
    assert details["rss_before_trim_mib"] > 0
    assert details["rss_after_trim_mib"] > 0
    (row,) = (f for f in result.findings if f.check_type == "unit_timing")
    assert row.details["rss_before_trim_mib"] == details["rss_before_trim_mib"]
    assert row.details["rss_after_trim_mib"] == details["rss_after_trim_mib"]


def test_the_unit_trims_also_when_it_fails(lake, tmp_path, monkeypatch):
    # Con éxito o con error: un fallo a medio mes también deja arenas llenas.
    calls = []
    monkeypatch.setattr(
        pipeline, "release_to_kernel", lambda: calls.append(1) or (None, None)
    )
    ctx = context(lake, tmp_path)
    assert process_unit(unit_of(M2), ctx, lake.thetas).ok
    assert calls == [1]
    # Sin el mes anterior en L2 la unidad falla dentro de `summarize_month`.
    broken = replace(ctx, events_root=str(tmp_path / "nothing"))
    assert not process_unit(unit_of(M2), broken, lake.thetas).ok
    assert calls == [1, 1]


def test_unit_timing_records_the_memory_samples(lake, tmp_path):
    # ITSC-375: el pico y el cierre (antes y después del trim) con la contabilidad de glibc.
    ctx = replace(context(lake, tmp_path), memory_sample_s=0.005)
    result = process_unit(unit_of(M2), ctx, lake.thetas)
    details = result.timing.details()
    assert details["mem_samples"] >= 1
    keys = {
        "rss_mib",
        "arena_mib",
        "hblkhd_mib",
        "uordblks_mib",
        "fordblks_mib",
        "keepcost_mib",
        "arrow_mib",
        "engine_inflight",
        "elapsed_s",
    }
    for name in ("mem_peak", "mem_close_before_trim", "mem_close_after_trim"):
        assert set(details[name]) == keys, name
    # El cierre ya no tiene motor: sin tramos anclados.
    assert details["mem_close_before_trim"]["engine_inflight"] is None
    assert (
        details["mem_peak"]["elapsed_s"]
        <= details["mem_close_before_trim"]["elapsed_s"]
    )
    assert details["mem_close_before_trim"]["elapsed_s"] <= details["wall_s"]
    assert 0 < details["mem_peak_coverage"] <= 1.05
    (row,) = (f for f in result.findings if f.check_type == "unit_timing")
    assert row.details["mem_peak"] == details["mem_peak"]
    assert row.details["mem_samples"] == details["mem_samples"]


def test_the_memory_sampler_stops_with_the_unit_also_when_it_fails(lake, tmp_path):
    ctx = replace(context(lake, tmp_path), events_root=str(tmp_path / "nothing"))
    assert not process_unit(unit_of(M2), ctx, lake.thetas).ok
    assert not [t for t in threading.enumerate() if t.name == "l3-memory-sampler"]


def test_unit_timing_records_the_periodic_trim(lake, tmp_path):
    # ITSC-376: cuántos trims hizo el muestreador, cuánto tardaron y con qué marca.
    base = replace(context(lake, tmp_path), memory_sample_s=0.005)
    on = process_unit(
        unit_of(M2), replace(base, trim_rss_mib=1, trim_min_s=0.0), lake.thetas
    )
    details = on.timing.details()
    assert details["trim_rss_mib"] == 1
    assert details["trims"] >= 1 and details["trim_s"] > 0
    (row,) = (f for f in on.findings if f.check_type == "unit_timing")
    for key in ("trims", "trim_s", "trim_rss_mib"):
        assert row.details[key] == details[key]
    # Sin la variable rige la marca por defecto, que un mes chico no alcanza.
    default = process_unit(unit_of(M2), base, lake.thetas).timing.details()
    assert default["trim_rss_mib"] == memory.DEFAULT_TRIM_RSS_MIB == 1024
    assert (default["trims"], default["trim_s"]) == (0, 0.0)
    off = process_unit(unit_of(M2), replace(base, trim_rss_mib=0), lake.thetas)
    assert off.timing.details()["trim_rss_mib"] == 0 and off.timing.trims == 0
