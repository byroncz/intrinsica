"""El orquestador sobre el motor nativo (ITSC-363, ADR-L3-16): lo que escribe no depende de
los hilos ni de los topes.

Estas pruebas no miden velocidad. Corren `summarize_month` con 1, 2 y 4 hilos del motor, 1 y
2 lectores de L1, 1 y 4 row groups en vuelo y 1 y 4 escritores, con las fronteras de L2 en un
solo lote o en lotes de pocos eventos, y exigen exactamente la misma tabla, el mismo
`content_hash` y la misma conciliación que la referencia de Python (`pyref`, en serie). Un
lago con row groups de L1 repartidos a mano fuerza un evento que cruza tres row groups, uno
cuya referencia es el último tick de un row group y uno que cubre varios row groups enteros.
"""

import ast
import importlib
import inspect
import itertools
import logging
import math
import pkgutil
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import l3_dc_descriptors
import numpy as np
import pathref
import pyarrow as pa
import pyarrow.parquet as pq
import pyref
import pytest
from dc_frames import FamilySkeletonMismatch, FamilyWriter, FramesInputError
from dc_frames.lake import EventFile, l1_path
from l3_dc_descriptors import (
    SUMMARIES_SCHEMA,
    L2InputMissing,
    SkeletonMismatch,
    boundaries,
    native,
    outputs,
    summarize,
    summarize_month,
)
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.cpu import CpuLimit
from l3_dc_descriptors.pipeline import process_unit
from l3_dc_descriptors.schema import PATH_SCHEMA
from l3_dc_descriptors.timing import BusyClock, Phases, ReadMeter
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts
from native_driver import path_native

ENGINE_THREADS = (1, 2, 4)
READERS = (1, 2)
INFLIGHT = (1, 4)
WRITERS = (1, 4)
KNOBS = list(itertools.product(ENGINE_THREADS, READERS, INFLIGHT, WRITERS))
THREAD_PREFIXES = (
    "l3-theta",
    "l3-meta",
    "l3-write",
    "l3-collector",
    "dc-frames-read",
)


@pytest.fixture(scope="module")
def ticks():
    return fixture_ticks(seed=5, drop=0.05)


@pytest.fixture(scope="module")
def lake(tmp_path_factory, ticks):
    # Row groups chicos: cada uno lo tocan los 5 θ, y hay eventos que cruzan row groups.
    return build_lake(
        tmp_path_factory.mktemp("lake"),
        ticks,
        month_cuts(ticks, 1500, 3000),
        l1_row_group=150,
    )


def l3_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith(THREAD_PREFIXES)]


def run_all(lake, runner=summarize_month, **options) -> dict:
    """Los resultados y las tablas de todos los meses, leídas antes de la siguiente corrida
    (que reescribe los mismos archivos).
    """
    results, tables = {}, {}
    for month in lake.months:
        results[month] = runner(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            **options,
        )
        for theta, result in results[month].items():
            tables[month, theta] = pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
    return {"results": results, "tables": tables}


@pytest.fixture(scope="module")
def reference(lake):
    return run_all(lake, pyref.summarize_month)


def assert_same(got: dict, expected: dict) -> None:
    assert got["results"].keys() == expected["results"].keys()
    for month, per_theta in expected["results"].items():
        for theta, result in per_theta.items():
            other = got["results"][month][theta]
            assert other.content_hash == result.content_hash
            assert other.rows == result.rows
            assert other.reconciliation == result.reconciliation
    left, right = expected["tables"], got["tables"]
    assert left.keys() == right.keys()
    assert all(left[key].equals(right[key]) for key in left)
    assert sum(t.num_rows for t in left.values()) > 0


@pytest.fixture
def tiny_l2(monkeypatch):
    """Tramos de L2 de un row group y lotes de 3 eventos: muchos lotes por θ, que el motor
    suelta a medida que cierran sus eventos.
    """
    monkeypatch.setattr(boundaries, "L2_LOT_BYTES", 1)
    monkeypatch.setattr(boundaries, "L2_BATCH_ROWS", 3)


@pytest.mark.parametrize(("engine_threads", "readers", "inflight", "writers"), KNOBS)
def test_the_knobs_give_the_same_table_and_hash(
    lake, reference, engine_threads, readers, inflight, writers
):
    got = run_all(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
    )
    assert_same(got, reference)
    assert l3_threads() == []


@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight", "writers"),
    [(1, 1, 1, 1), (2, 2, 4, 1), (4, 1, 4, 4), (4, 2, 1, 4)],
)
def test_the_boundaries_in_small_lots_change_nothing(
    lake, reference, tiny_l2, engine_threads, readers, inflight, writers
):
    got = run_all(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
    )
    assert_same(got, reference)


@pytest.mark.parametrize("threads", [1, 2, 50])
def test_the_io_pool_changes_nothing(lake, reference, threads):
    assert_same(run_all(lake, threads=threads), reference)
    assert l3_threads() == []


@pytest.mark.parametrize("meta_workers", [1, 3, summarize.MAX_META_WORKERS])
def test_the_metadata_pool_changes_nothing(lake, reference, meta_workers):
    assert_same(run_all(lake, threads=4, meta_workers=meta_workers), reference)
    assert l3_threads() == []


def test_the_defaults_give_the_same_result_too(lake, reference):
    assert_same(run_all(lake), reference)


@pytest.mark.parametrize(
    "option",
    [
        "threads",
        "readers",
        "inflight",
        "writers",
        "engine_threads",
        "inflight_bytes",
        "meta_workers",
    ],
)
def test_it_rejects_a_knob_out_of_range(lake, option):
    with pytest.raises(ValueError, match=option):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            **{option: 0},
        )


def test_the_metadata_pool_has_a_cap(lake):
    with pytest.raises(ValueError, match="meta_workers"):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            meta_workers=summarize.MAX_META_WORKERS + 1,
        )


def test_the_metadata_pool_does_not_depend_on_the_cores(monkeypatch):
    # Son esperas de red, no CPU: el ancho es el mismo con 1 vCPU que con 8, y nunca pasa de θ.
    for cores in (0.5, 1.0, 8.0):
        monkeypatch.setattr(
            summarize, "cpu_limit", lambda cores=cores: CpuLimit(cores, 8, "cgroup-v2")
        )
        assert summarize._meta_workers(None, 50) == summarize.DEFAULT_META_WORKERS
    assert summarize._meta_workers(None, 5) == 5
    assert summarize._meta_workers(3, 50) == 3


def test_it_never_uses_more_io_threads_than_thetas():
    assert summarize._workers(8, 5) == 5
    assert summarize._workers(3, 50) == 3
    assert 1 <= summarize._workers(None, 50) <= 50


@pytest.mark.parametrize(
    ("cores", "thetas", "expected"),
    [
        (3.72, 50, 12),
        (4.0, 50, 12),
        (8.0, 50, 24),
        (8.0, 5, 5),
        (0.2, 50, 1),
        (0.5, 50, 2),
    ],
)
def test_the_default_pool_is_three_threads_per_vcpu_rounded_up(
    monkeypatch, cores, thetas, expected
):
    monkeypatch.setattr(summarize, "cpu_limit", lambda: CpuLimit(cores, 8, "cgroup-v2"))
    assert summarize._workers(None, thetas) == expected


def test_ordered_returns_in_item_order_and_raises_the_first_failure_in_item_order():
    def work(n: int) -> int:
        time.sleep(0.02 * (5 - n))  # los últimos terminan antes que los primeros
        if n in (2, 4):
            raise RuntimeError(f"falla {n}")
        return n * 10

    with ThreadPoolExecutor(5) as pool:
        assert summarize.ordered(pool, lambda n: n * 10, range(5)) == [
            0,
            10,
            20,
            30,
            40,
        ]
        with pytest.raises(RuntimeError, match="falla 2"):
            summarize.ordered(pool, work, range(5))
    with pytest.raises(RuntimeError, match="falla 2"):
        summarize.ordered(None, work, range(5))


def test_a_pool_of_one_thread_is_no_pool():
    with summarize.theta_pool(1) as pool:
        assert pool is None
    with summarize.theta_pool(3) as pool:
        assert isinstance(pool, ThreadPoolExecutor)


def _modules():
    for info in pkgutil.iter_modules(l3_dc_descriptors.__path__):
        if info.name == "__main__":  # importarlo ejecuta la CLI
            continue
        yield importlib.import_module(f"l3_dc_descriptors.{info.name}")


def test_no_module_declares_global_state():
    # Ningún `global` y ningún contenedor mutable a nivel de módulo.
    for module in _modules():
        tree = ast.parse(inspect.getsource(module))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)], module
        mutable = [
            name
            for name, value in vars(module).items()
            if isinstance(value, (list, dict, set, bytearray))
            and not name.startswith("__")
        ]
        assert mutable == [], (module.__name__, mutable)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_the_lineage_is_the_same_with_any_pool(lake, tmp_path, threads):
    def lineage(**options):
        ctx = RunContext(
            mode="backfill",
            run_id="run",
            image_version="0.8.0+test",
            series_start=M1,
            landing_root=str(lake.l1),
            events_root=str(lake.l2),
            summaries_root=str(lake.l3),
            dq_root=str(tmp_path / "dq"),
            **options,
        )
        result = process_unit(Unit(*M2), ctx, lake.thetas)
        assert result.ok
        rows = [f for f in result.findings if f.check_type == "summaries_summary"]
        return {
            f.details["theta"]: (
                f.details["l2_events_content_hash"],
                f.details["l2_carry_over_content_hash"],
                f.details["content_hash"],
            )
            for f in rows
        }, result.timing

    expected, _ = lineage(threads=1)
    got, timing = lineage(threads=threads)
    assert got == expected and len(got) == len(lake.thetas)
    assert timing.phases.gate_s > 0 and not hasattr(timing.phases, "lineage_s")


def test_the_content_hash_is_the_same_with_and_without_the_periodic_trim(
    lake, tmp_path
):
    """ITSC-376: devolver memoria al kernel durante la unidad no cambia un bit de la salida."""

    def hashes(**options):
        ctx = RunContext(
            mode="backfill",
            run_id="run",
            image_version="0.8.0+test",
            series_start=M1,
            landing_root=str(lake.l1),
            events_root=str(lake.l2),
            summaries_root=str(lake.l3),
            dq_root=str(tmp_path / "dq"),
            memory_sample_s=0.001,
            **options,
        )
        result = process_unit(Unit(*M2), ctx, lake.thetas)
        assert result.ok, result.error
        rows = [f for f in result.findings if f.check_type == "summaries_summary"]
        return {f.details["theta"]: f.details["content_hash"] for f in rows}, result

    without, off = hashes(trim_rss_mib=0)
    trimmed, on = hashes(trim_rss_mib=1, trim_min_s=0.0)
    assert trimmed == without and len(trimmed) == len(lake.thetas)
    assert off.timing.trims == 0 and off.timing.trim_rss_mib == 0
    assert on.timing.trims >= 1 and on.timing.trim_rss_mib == 1


def test_the_phases_report_how_it_ran(lake):
    phases = Phases()
    summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M2,
        lake.theta_texts,
        series_start=M1,
        phases=phases,
        threads=3,
        engine_threads=2,
        readers=1,
        inflight=3,
        writers=2,
    )
    assert (phases.threads, phases.engine_threads, phases.readers) == (3, 2, 1)
    assert phases.meta_workers == min(summarize.DEFAULT_META_WORKERS, len(lake.thetas))
    assert (phases.inflight, phases.writers) == (3, 2)
    assert 1 <= phases.inflight_max <= 3
    assert phases.items > 0 and phases.items % len(lake.thetas) == 0
    assert 0 < phases.engine_busy <= 1
    assert phases.aggregate_s > 0 and phases.setup_s > 0 and phases.publish_s > 0
    assert phases.read.bytes > 0 and phases.read.clock.seconds > 0
    assert phases.write.seconds > 0


# Row groups de L1 repartidos a mano


def relayout(lake, month, sizes: list[int]) -> None:
    """Reescribe `consolidated.parquet` del mes con row groups de esos tamaños (el último,
    lo que quede). L2 solo habla de ids, así que no cambia.
    """
    path = l1_path(str(lake.l1), ("binance", "spot", "BTCUSDT"), month)
    table = pq.read_table(path)
    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    start = 0
    for size in [*sizes, table.num_rows]:
        stop = min(start + size, table.num_rows)
        if stop > start:
            writer.write_table(table.slice(start, stop - start), row_group_size=size)
        start = stop
    writer.close()


def position(ticks, tick_id: int) -> int:
    return next(i for i, t in enumerate(ticks) if t.id == tick_id)


def hand_made_cuts(lake, month) -> list[int]:
    """Cortes de row group tales que, en algún θ, haya un evento que cruza tres row groups
    o más y otro cuya referencia es el último tick de un row group.
    """
    ticks = lake.ticks[month]
    present = {t.id for t in ticks}
    events = [
        e
        for theta in lake.thetas
        for e in lake.events[theta, month]
        if {e["reference_agg_trade_id"], e["extreme_agg_trade_id"]} <= present
    ]
    long = max(
        events, key=lambda e: e["extreme_agg_trade_id"] - e["reference_agg_trade_id"]
    )
    start = position(ticks, long["reference_agg_trade_id"])
    stop = position(ticks, long["extreme_agg_trade_id"])
    assert stop - start >= 30, "el lago no trae un evento largo"
    # La referencia del evento largo es el último tick de su row group; el evento abarca
    # varios row groups de 10 ticks, y los demás row groups son de 25.
    cuts = []
    done = 0
    while done < start + 1:
        step = min(25, start + 1 - done)
        cuts.append(step)
        done += step
    cuts += [10] * ((stop - start) // 10 + 1)
    return cuts


def crossing(ticks, cuts, events) -> tuple[int, int, int]:
    """Cuántos eventos cruzan tres row groups o más, cuántos tienen su referencia como
    último tick de un row group y cuántos cruzan un row group entero sin ninguna de sus
    tres fronteras (referencia, confirmación, extremo) dentro.
    """
    edges = list(itertools.accumulate(cuts))
    group = {}
    for g, (a, b) in enumerate(itertools.pairwise([0, *edges, len(ticks)])):
        for i in range(a, b):
            group[ticks[i].id] = g
    inside = [
        e
        for e in events
        if {e["reference_agg_trade_id"], e["extreme_agg_trade_id"]} <= group.keys()
    ]
    wide = sum(
        group[e["extreme_agg_trade_id"]] - group[e["reference_agg_trade_id"]] >= 2
        for e in inside
    )
    bare = 0
    for e in inside:
        first, stop = (
            group[e["reference_agg_trade_id"]],
            group[e["extreme_agg_trade_id"]],
        )
        between = set(range(first + 1, stop)) - {group.get(e["confirm_agg_trade_id"])}
        bare += bool(between)
    last = {ticks[edge - 1].id for edge in edges}
    on_edge = sum(e["reference_agg_trade_id"] in last for e in events)
    return wide, on_edge, bare


@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight"), [(1, 1, 1), (4, 2, 4), (2, 2, 2)]
)
def test_events_over_three_row_groups_and_on_a_row_group_edge(
    tmp_path, ticks, engine_threads, readers, inflight
):
    lake = build_lake(tmp_path / "lake", ticks, month_cuts(ticks, 1500, 3000))
    month = lake.months[1]
    cuts = hand_made_cuts(lake, month)
    relayout(lake, month, cuts)
    events = [e for theta in lake.thetas for e in lake.events[theta, month]]
    wide, on_edge, bare = crossing(lake.ticks[month], cuts, events)
    assert wide > 0 and on_edge > 0 and bare > 0
    wanted = pyref.summarize_month(
        lake.l1, lake.l2, tmp_path / "py", month, lake.theta_texts, series_start=M1
    )
    got = summarize_month(
        lake.l1,
        lake.l2,
        tmp_path / "rs",
        month,
        lake.theta_texts,
        series_start=M1,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
    )
    for theta, result in wanted.items():
        assert got[theta].content_hash == result.content_hash
        assert got[theta].rows == result.rows
        assert got[theta].reconciliation == result.reconciliation
        assert pq.read_table(got[theta].path, schema=SUMMARIES_SCHEMA).equals(
            pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
        )


# Errores


def test_a_failing_writer_aborts_the_unit_without_publishing_and_without_threads(
    lake, tmp_path, monkeypatch
):
    real = FamilyWriter.write_row_group
    fail = set(lake.theta_texts[1::2])

    def failing(self, batch):
        theta = f"{batch.column('theta')[0].as_py():.8f}"
        if theta in fail:
            raise FamilySkeletonMismatch("mensaje", f"falla {theta}")
        real(self, batch)

    monkeypatch.setattr(FamilyWriter, "write_row_group", failing)
    for kwargs in ({"threads": 1, "writers": 1}, {"threads": 4, "writers": 4}):
        with pytest.raises(SkeletonMismatch) as caught:
            summarize_month(
                lake.l1,
                lake.l2,
                tmp_path,
                M1,
                lake.theta_texts,
                series_start=M1,
                **kwargs,
            )
        assert caught.value.details["theta"] in fail
        assert l3_threads() == []
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_a_failing_commit_is_the_skeleton_error_of_its_theta(
    lake, tmp_path, monkeypatch
):
    def refuse(self):
        raise FamilySkeletonMismatch("mensaje", "faltan row groups")

    monkeypatch.setattr(FamilyWriter, "commit", refuse)
    with pytest.raises(SkeletonMismatch) as caught:
        summarize_month(
            lake.l1, lake.l2, tmp_path, M1, lake.theta_texts, series_start=M1, threads=4
        )
    assert caught.value.details["theta"] == lake.theta_texts[0]
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]
    assert l3_threads() == []


def test_a_missing_events_file_is_the_gate_error_of_the_first_theta(
    lake, tmp_path, ticks
):
    base = build_lake(tmp_path / "gate", ticks, month_cuts(ticks, 1500, 3000))
    thetas = base.theta_texts
    for text in thetas[:2]:
        events = (
            base.l2
            / f"provider=binance/market=spot/asset=BTCUSDT/theta={text}/year=2017/month=09"
            / "events.parquet"
        )
        events.unlink()
    messages = set()
    for threads in (1, 4):
        with pytest.raises(L2InputMissing) as caught:
            summarize_month(
                base.l1, base.l2, base.l3, M2, thetas, series_start=M1, threads=threads
            )
        messages.add(str(caught.value))
        assert caught.value.details["theta"] == thetas[0]
        assert [p for p in base.l3.rglob("*") if p.is_file()] == []
        assert l3_threads() == []
    assert len(messages) == 1


# Contrapresión: a lo más K row groups en el motor y pocos sellados sin escribir


def test_inflight_never_passes_the_cap(lake):
    for inflight in (1, 2, 3):
        phases = Phases()
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            phases=phases,
            inflight=inflight,
            readers=2,
        )
        assert 1 <= phases.inflight_max <= inflight


def test_slow_writers_hold_back_the_pushes(lake, monkeypatch, tmp_path):
    """Con escritores lentos, lo sellado sin escribir se queda cerca del tope y la unidad
    igual termina con el mismo resultado; sin tope se acumula todo lo que el motor sella.
    """
    seen: list[int] = []
    real = outputs.SealedWriters.put

    def spying(self, sealed):
        real(self, sealed)
        seen.append(self.pending)

    slow = FamilyWriter.write_row_group

    def sleepy(self, batch):
        time.sleep(0.004)
        slow(self, batch)

    monkeypatch.setattr(outputs.SealedWriters, "put", spying)
    monkeypatch.setattr(FamilyWriter, "write_row_group", sleepy)
    peaks, hashes = {}, {}
    for cap in (1, 10**6):
        seen.clear()
        monkeypatch.setattr(summarize, "PENDING_PER_WRITER", cap)
        results = summarize_month(
            lake.l1,
            lake.l2,
            tmp_path / str(cap),
            M2,
            lake.theta_texts,
            series_start=M1,
            writers=1,
            inflight=2,
            readers=1,
        )
        peaks[cap] = max(seen)
        hashes[cap] = {t: r.content_hash for t, r in results.items()}
    wanted = pyref.summarize_month(
        lake.l1, lake.l2, tmp_path / "py", M2, lake.theta_texts, series_start=M1
    )
    assert hashes[1] == hashes[10**6] == {t: r.content_hash for t, r in wanted.items()}
    assert peaks[1] < peaks[10**6]


@pytest.mark.parametrize("cap", [1, 40_000, summarize.DEFAULT_INFLIGHT_BYTES])
def test_the_byte_cap_gives_the_same_table_and_hash(lake, reference, cap):
    """ITSC-372: el tope de bytes de eventos abiertos solo decide cuándo se empuja, no qué se
    escribe: con 1 B cada row group avanza solo y el resultado es el de la referencia.
    """
    got = run_all(
        lake,
        threads=4,
        engine_threads=4,
        readers=2,
        inflight=4,
        writers=4,
        inflight_bytes=cap,
    )
    assert_same(got, reference)


def test_the_unit_reports_the_open_events_it_held(lake):
    phases = Phases()
    summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M1,
        lake.theta_texts,
        series_start=M1,
        phases=phases,
        inflight_bytes=10**9,
    )
    assert phases.inflight_bytes == 10**9
    assert phases.inflight_events_max > 0
    # Las fronteras que `cover` trae por delante cuestan 81 B y no 321 (ITSC-376).
    assert (
        phases.inflight_events_max * boundaries.BOUNDARY_ROW_BYTES
        <= phases.inflight_bytes_max
        <= phases.inflight_events_max * summarize.EVENT_BYTES
    )
    # El primer row group siempre entra a un motor vacío.
    assert 0 < phases.inflight_alone_bytes_max <= phases.inflight_bytes_max


def test_an_event_costs_its_output_row_and_its_boundaries():
    """240 B de las 18 columnas de salida más 81 B de las ocho de fronteras de L2."""
    assert summarize.EVENT_BYTES == 240 + 81


class ClosingEngine:
    """Un motor que cierra cada row group, y los eventos que entregó, `work` s después de
    empujarlo. Anota cuántos row groups tuvo abiertos a la vez.
    """

    def __init__(self, work: float) -> None:
        self.work = work
        self.open = 0
        self.most = 0
        self.closed = 0
        self.lock = threading.Lock()

    def start(self, events: int) -> None:
        with self.lock:
            self.open += 1
            self.most = max(self.most, self.open)
        threading.Timer(self.work, self.close, (events,)).start()

    def close(self, events: int) -> None:
        with self.lock:
            self.open -= 1
            self.closed += events

    def inflight(self) -> int:
        with self.lock:
            return self.open

    def needed(self, position: int) -> int:
        with self.lock:
            return self.closed


class FakeFeed:
    """Las fronteras de un θ: cada row group trae `per_group` eventos nuevos."""

    position = 0

    def __init__(self, per_group: int, ahead: int = 0) -> None:
        self.per_group = per_group
        self.ahead = ahead  # de los entregados, los que `cover` trajo por delante
        self.delivered = 0

    def cover(self, high: int) -> None:
        self.delivered += self.per_group


def push_fake_row_groups(
    monkeypatch, cap: int, per_group: int = 1000, groups: int = 6, ahead: int = 0
):
    """`_push_all` con 6 row groups de `per_group` eventos (`ahead` de ellos por delante) y un
    motor que tarda 30 ms en cerrar cada uno. Devuelve el motor y las `Phases`.
    """
    engine, feed, phases = ClosingEngine(0.03), FakeFeed(per_group, ahead), Phases()
    phases.inflight, phases.writers, phases.inflight_bytes = 4, 1, cap

    def push(engine, row_group_id, ids, columns, before_month):
        engine.start(feed.per_group)

    monkeypatch.setattr(summarize.native, "push", push)
    row_groups = [
        types.SimpleNamespace(high=i, month=M1, ids=None, columns=None)
        for i in range(groups)
    ]
    idle = types.SimpleNamespace(error=None, pending=0)
    observed = types.SimpleNamespace(observe=lambda ids: None)
    pushed = summarize._push_all(
        engine, row_groups, [feed], idle, idle, [], M1, phases, observed
    )
    assert pushed == groups
    while engine.inflight():
        time.sleep(0.01)
    return engine, phases


def test_the_byte_cap_holds_back_a_row_group_while_the_open_events_pass_it(
    monkeypatch, caplog
):
    each = 1000 * summarize.EVENT_BYTES
    # Sin tope de bytes, K = 4 deja entrar cuatro row groups a la vez.
    engine, phases = push_fake_row_groups(monkeypatch, 10**12)
    assert engine.most == 4
    # Con 1,5 row groups el segundo entra (el tope se mira antes de empujar) y el tercero espera.
    engine, phases = push_fake_row_groups(monkeypatch, 3 * each // 2)
    assert engine.most == 2
    assert phases.inflight_events_max == 2000
    assert phases.inflight_bytes_max == 2 * each
    # Pasar del tope en hasta un row group es lo normal: ninguno lo pasó solo, así que
    # `inflight_alone_bytes_max` (el límite conocido) queda bajo el tope y no hay aviso.
    assert phases.inflight_bytes_max > phases.inflight_bytes
    assert phases.inflight_alone_bytes_max == each < phases.inflight_bytes
    assert caplog.records == []
    # Con medio row group, cada uno avanza solo: pasa del tope, avisa una vez y sigue.
    caplog.set_level(logging.WARNING, logger="l3_dc_descriptors.summarize")
    engine, phases = push_fake_row_groups(monkeypatch, each // 2)
    assert engine.most == 1
    assert phases.inflight_bytes_max == each > phases.inflight_bytes
    assert phases.inflight_alone_bytes_max == each > phases.inflight_bytes
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert "L3_INFLIGHT_BYTES" in message
    # ITSC-376: dice qué tramo trajo cuántos eventos y que la unidad avanza; no nombra
    # variables: las salidas (lote más chico o más memoria) las dicen las docs.
    assert "el tramo 0 trajo 1000 eventos" in message
    assert "la unidad avanza" in message
    assert "L3_LOT_EVENTS" not in message


def test_the_boundaries_cover_brought_ahead_cost_81_bytes_not_321(monkeypatch):
    """Los eventos que `cover` entregó por delante del tramo no tienen casilla: cuestan solo
    sus ocho columnas de fronteras (`BOUNDARY_ROW_BYTES`).
    """
    each = 1000 * summarize.EVENT_BYTES
    ahead = 400
    _, phases = push_fake_row_groups(monkeypatch, each // 2, ahead=ahead)
    expected = (1000 - ahead) * summarize.EVENT_BYTES + (
        ahead * boundaries.BOUNDARY_ROW_BYTES
    )
    assert phases.inflight_events_max == 1000
    assert phases.inflight_alone_bytes_max == expected < each
    # 600 × 321 + 400 × 81 = 225 000 B, y no los 321 000 de contar todo a 321 B: con el tope
    # justo sobre eso el tramo cabe solo.
    _, phases = push_fake_row_groups(monkeypatch, expected + 1, ahead=ahead)
    assert phases.inflight_alone_bytes_max == expected <= phases.inflight_bytes


def test_a_boundary_feed_counts_the_events_it_brought_ahead_of_the_covered_ids():
    pushed: list[int] = []
    engine = types.SimpleNamespace(
        push_boundaries=lambda position, entry: pushed.append(entry.rows)
    )
    references = np.arange(0, 100, 10, dtype=np.int64)  # 10 eventos: 0, 10, ..., 90
    columns_of_l2 = (
        "reference_agg_trade_id",
        "confirm_agg_trade_id",
        "confirm_time",
        "extreme_agg_trade_id",
        "direction",
        "confirm_price",
        "extreme_price",
        "reference_price",
    )

    def batches():
        for start in (0, 5):
            part = references[start : start + 5]
            columns = {name: pa.array(part) for name in columns_of_l2}
            yield pa.RecordBatch.from_pydict(columns)

    feed = native.BoundaryFeed(engine, 0, batches())
    assert feed.ahead == 0
    feed.cover(25)  # entrega el primer lote (0..40): su último, 40, ya pasa de 25
    assert (feed.delivered, feed.ahead) == (5, 2)  # 30 y 40 quedan por delante
    feed.cover(35)  # nada nuevo: 40 sigue después de 35
    assert (feed.delivered, feed.ahead) == (5, 1)
    feed.cover(40)  # 40 ya no pasa de 40: entrega el segundo lote (50..90)
    assert (feed.delivered, feed.ahead) == (10, 5)
    assert pushed == [5, 5]


# Lote acotado en eventos (ITSC-373): cada row group de L1 se parte en tramos al empujarlo


def an_l1_row_group(ticks: int, first: int = 100, month=M1) -> types.SimpleNamespace:
    """Un row group de L1 de `ticks` ticks con ids `first`, `first + 1`, ... y cuatro
    columnas de Arrow como las del lector (la última, booleana: su offset es en bits).
    """
    ids = np.arange(first, first + ticks, dtype=np.int64)
    columns = (
        pa.array(ids * 10),
        pa.array(ids * 100),
        pa.array(ids * 1000),
        pa.array(ids % 2 == 0),
    )
    return types.SimpleNamespace(
        ids=ids, columns=columns, month=month, high=int(ids[-1])
    )


def test_the_lot_ticks_come_from_the_density_of_the_month():
    # 1 000 000 de eventos de lote y 6,1 eventos por tick (2017-12): ≈ 164 000 ticks.
    assert summarize._lot_ticks(1_000_000, 6_100_000, 1_000_000) == 163_934
    # 0,66 eventos por tick (2021-01): 1,5 M de ticks, más que un row group de L1.
    assert summarize._lot_ticks(1_000_000, 660_000, 1_000_000) == 1_515_151
    assert summarize._lot_ticks(1, 10**9, 10) == 1  # nunca menos de un tick
    # Desactivado (0) o sin eventos que midan la densidad: no hay corte.
    assert summarize._lot_ticks(0, 100, 100) is None
    assert summarize._lot_ticks(1_000, 0, 100) is None


def test_a_row_group_is_cut_in_contiguous_lots_without_copying():
    rg = an_l1_row_group(100)
    lots = summarize._cut(rg, M1, 33)
    # ceil(100 / 33) = 4 tramos de tamaño casi igual, ninguno pasa de 33.
    assert [len(ids) for ids, *_ in lots] == [25, 25, 25, 25]
    assert np.concatenate([ids for ids, *_ in lots]).tolist() == rg.ids.tolist()
    start = 0
    for ids, columns, high, before_month in lots:
        assert high == int(ids[-1]) and before_month is False
        assert np.shares_memory(ids, rg.ids)
        for column, whole in zip(columns, rg.columns, strict=True):
            assert column.to_pylist() == whole.slice(start, len(ids)).to_pylist()
            # El mismo buffer de Arrow, con el desfase del tramo: `Engine.push` lo honra.
            assert column.buffers()[1].address == whole.buffers()[1].address
            assert native._bytes(column)[1] == start
        start += len(ids)


def test_a_row_group_that_fits_in_a_lot_is_pushed_whole():
    rg = an_l1_row_group(100)
    for lot_ticks in (None, 100, 1_000):
        ((ids, columns, high, before_month),) = summarize._cut(rg, M1, lot_ticks)
        assert ids is rg.ids and columns is rg.columns
        assert high == rg.high and before_month is False
    # El tramo de un mes anterior lleva la bandera de su row group.
    assert all(lot[3] for lot in summarize._cut(rg, (2017, 9), 10))


def push_lots(monkeypatch, row_groups, lot_ticks: int | None):
    """`_push_all` con un motor simulado que anota lo que recibe: los `cover(high)` de cada
    θ, cada `push` y cada `l1.observe`.
    """
    pushes, observed = [], []

    class Feed:
        position = 0
        delivered = 0
        ahead = 0

        def cover(self, high):
            pushes.append(("cover", high))

    class Engine:
        def inflight(self):
            return 0

        def needed(self, position):
            return 0

    monkeypatch.setattr(
        summarize.native,
        "push",
        lambda engine, ordinal, ids, columns, before: pushes.append(
            ("push", ordinal, len(ids), before)
        ),
    )
    phases = Phases()
    phases.inflight, phases.writers, phases.inflight_bytes = 4, 1, 10**12
    phases.lot_ticks = lot_ticks
    idle = types.SimpleNamespace(error=None, pending=0)
    watcher = types.SimpleNamespace(observe=lambda ids: observed.append(len(ids)))
    pushed = summarize._push_all(
        Engine(), row_groups, [Feed()], idle, idle, [], M1, phases, watcher
    )
    assert phases.lots == pushed
    return pushed, pushes, observed


def test_each_lot_is_a_row_group_for_the_engine_with_its_own_ordinal_and_cover(
    monkeypatch,
):
    groups = [an_l1_row_group(100, 100), an_l1_row_group(60, 200, month=(2017, 7))]
    pushed, calls, observed = push_lots(monkeypatch, groups, 30)
    assert pushed == 4 + 2
    # `cover` va antes de cada `push` y con el último id del tramo; los ordinales siguen.
    assert calls == [
        ("cover", 124), ("push", 0, 25, False),
        ("cover", 149), ("push", 1, 25, False),
        ("cover", 174), ("push", 2, 25, False),
        ("cover", 199), ("push", 3, 25, False),
        ("cover", 229), ("push", 4, 30, True),
        ("cover", 259), ("push", 5, 30, True),
    ]  # fmt: skip
    # `l1.observe` ve el row group entero, una vez, y solo los del mes.
    assert observed == [100]


def test_without_a_cut_the_lots_are_the_row_groups(monkeypatch):
    groups = [an_l1_row_group(100, 100), an_l1_row_group(60, 200)]
    pushed, calls, observed = push_lots(monkeypatch, groups, None)
    assert pushed == 2 and observed == [100, 60]
    assert [c[2] for c in calls if c[0] == "push"] == [100, 60]


def test_the_engine_receives_the_offset_of_each_lot(lake, reference, monkeypatch):
    """Lo que `native.push` entrega al motor: el mismo buffer con el desfase del tramo, en
    elementos y en bits (la columna booleana). El resultado es el de la referencia, así que
    el motor lo honra.
    """
    real = summarize.native.push
    seen = []

    def spying(engine, ordinal, ids, columns, before_month):
        seen.append((ordinal, [native._bytes(column)[1] for column in columns]))
        real(engine, ordinal, ids, columns, before_month)

    monkeypatch.setattr(summarize.native, "push", spying)
    # Tramos de ≈ 33 ticks sobre row groups de 150: los desfases no son múltiplos de 8.
    got = run_all(lake, engine_threads=2, readers=2, lot_events=10)
    assert_same(got, reference)
    offsets = {column[3] for _, column in seen}
    assert any(offset % 8 for offset in offsets)
    assert all(
        len(set(column)) == 1 for _, column in seen
    )  # los 4 con el mismo desfase


@pytest.mark.parametrize("lot_events", [1, 5, 10, 25, 100])
@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight", "writers"), [(1, 1, 1, 1), (4, 2, 4, 4)]
)
def test_the_lots_give_the_same_table_and_hash(
    lake, reference, lot_events, engine_threads, readers, inflight, writers
):
    """Con tramos de pocos ticks, hay eventos que cruzan tramos y row groups (en el lago de
    pruebas son ≈ 0,3 eventos por tick): las tablas y los `content_hash` son los de sin corte,
    porque la frontera que un evento cruza entre tramos se fusiona como entre row groups
    (ADR-L3-14/16).
    """
    got = run_all(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
        lot_events=lot_events,
    )
    assert_same(got, reference)


def test_the_lots_and_the_small_boundary_batches_give_the_same_result(
    lake, reference, tiny_l2
):
    assert_same(run_all(lake, engine_threads=4, inflight=4, lot_events=3), reference)


def test_a_normal_month_is_not_cut(lake):
    """Con la densidad del lago de pruebas el lote por defecto (1 M de eventos) es mucho mayor
    que un row group: los tramos son los row groups.
    """
    uncut, cut = Phases(), Phases()
    for phases, lot_events in ((uncut, 0), (cut, None)):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            phases=phases,
            lot_events=lot_events,
        )
    assert uncut.lot_events == 0 and uncut.lot_ticks is None
    assert cut.lot_events == summarize.DEFAULT_LOT_EVENTS == 1_000_000
    assert cut.lot_ticks is not None and cut.lot_ticks > 150  # un row group de L1
    assert cut.lots == uncut.lots > 1
    assert cut.items == uncut.items


def test_a_dense_month_is_cut_and_the_unit_says_so(lake):
    uncut, cut = Phases(), Phases()
    for phases, lot_events in ((uncut, 0), (cut, 10)):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            phases=phases,
            lot_events=lot_events,
        )
    assert cut.lot_events == 10
    assert 0 < cut.lot_ticks < 150
    assert cut.lots > 3 * uncut.lots
    # Los ítems son (tramo, θ).
    assert cut.items == cut.lots * (uncut.items // uncut.lots)


def test_the_lot_is_in_the_unit_timing(lake, tmp_path):
    ctx = RunContext(
        mode="monthly",
        run_id="run",
        image_version="test",
        series_start=M1,
        landing_root=str(lake.l1),
        events_root=str(lake.l2),
        summaries_root=str(lake.l3),
        dq_root=str(tmp_path / "dq"),
        lot_events=10,
    )
    result = process_unit(Unit(*M1), ctx, lake.thetas)
    assert result.ok, result.error
    details = result.timing.details()
    assert details["lot_events"] == 10
    assert 0 < details["lot_ticks"] < 150
    assert details["lots"] > details["row_groups_month"]


def test_it_rejects_a_negative_lot(lake):
    with pytest.raises(ValueError, match="lot_events"):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            lot_events=-1,
        )


# Escritores: cola por θ con «siguiente esperado»


class FakeWriter:
    """Un escritor que anota qué row groups recibe, de qué hilo y si hubo dos a la vez."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.order: list[int] = []
        self.active = 0
        self.overlap = False
        self.lock = threading.Lock()

    def write_row_group(self, batch: pa.RecordBatch) -> None:
        with self.lock:
            self.active += 1
            self.overlap = self.overlap or self.active > 1
        time.sleep(self.delay)
        with self.lock:
            self.order.append(batch.column("confirm_agg_trade_id")[0].as_py())
            self.active -= 1


class Wrapped:
    """Bytes con protocolo de buffer, como `SealedColumn`."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __buffer__(self, flags):
        return memoryview(self.data).__buffer__(flags)

    def __len__(self) -> int:
        return len(self.data)


def fake_sealed(position: int, index: int):
    """Un row group sellado de 1 fila en el formato de `Engine.pop_sealed`: `(θ, índice,
    filas, columnas, validez, nulos)`, con el índice como `confirm_agg_trade_id`.
    """
    columns = []
    for field in list(SUMMARIES_SCHEMA)[1:]:
        width = 16 if pa.types.is_decimal(field.type) else 8
        value = index if field.name == "confirm_agg_trade_id" else 0
        columns.append(Wrapped(value.to_bytes(width, "little", signed=True)))
    return (position, index, 1, columns, None, 0)


def test_each_theta_writes_its_row_groups_in_order_and_never_two_at_a_time():
    from concurrent.futures import ThreadPoolExecutor

    fakes = [FakeWriter(0.002) for _ in range(3)]
    clock = BusyClock()
    thetas = [Decimal("0.001"), Decimal("0.002"), Decimal("0.003")]
    with ThreadPoolExecutor(4) as pool:
        sink = outputs.SealedWriters(pool, fakes, thetas, ["a", "b", "c"], M1, clock)
        # Llegan al revés y revueltos entre los θ: 4 antes que 3, 2 antes que 1…
        order = [(0, 3), (1, 1), (0, 2), (2, 0), (1, 0), (0, 1), (2, 1), (0, 0), (1, 2)]
        for position, index in order:
            sink.put(fake_sealed(position, index))
        sink.join()
    assert sink.pending == 0 and sink.error is None
    assert [f.order for f in fakes] == [[0, 1, 2, 3], [0, 1, 2], [0, 1]]
    assert not any(f.overlap for f in fakes)
    assert clock.seconds > 0


def test_a_row_group_that_waits_for_its_predecessor_is_not_written_yet():
    from concurrent.futures import ThreadPoolExecutor

    fake = FakeWriter()
    with ThreadPoolExecutor(2) as pool:
        sink = outputs.SealedWriters(
            pool, [fake], [Decimal("0.001")], ["a"], M1, BusyClock()
        )
        sink.put(fake_sealed(0, 1))
        time.sleep(0.05)
        assert fake.order == [] and sink.pending == 1
        sink.put(fake_sealed(0, 0))
        sink.join()
    assert fake.order == [0, 1]


def test_a_writer_error_is_kept_and_stops_the_rest():
    from concurrent.futures import ThreadPoolExecutor

    class Failing(FakeWriter):
        def write_row_group(self, batch):
            raise FamilySkeletonMismatch("mensaje", "falla")

    with ThreadPoolExecutor(2) as pool:
        sink = outputs.SealedWriters(
            pool,
            [Failing(), FakeWriter()],
            [Decimal("0.001"), Decimal("0.002")],
            ["a", "b"],
            M1,
            BusyClock(),
        )
        sink.put(fake_sealed(0, 0))
        sink.put(fake_sealed(1, 0))
        with pytest.raises(SkeletonMismatch):
            sink.join()
        assert isinstance(sink.error, SkeletonMismatch)
        sink.put(fake_sealed(1, 1))  # tras el error no se encola nada
        assert sink.pending == 0


def test_the_same_row_group_twice_is_an_error():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as pool:
        sink = outputs.SealedWriters(
            pool, [FakeWriter()], [Decimal("0.001")], ["a"], M1, BusyClock()
        )
        sink.put(fake_sealed(0, 1))
        sink.put(fake_sealed(0, 1))
        with pytest.raises(SkeletonMismatch, match="dos veces"):
            sink.join()


# Fronteras de L2 por delante


@pytest.fixture(scope="module")
def events_file(lake):
    text = lake.theta_texts[0]
    path = (
        lake.l2
        / f"provider=binance/market=spot/asset=BTCUSDT/theta={text}/year=2017/month=09"
        / "events.parquet"
    )
    return str(path)


def read_all(path: str, **options):
    with EventFile(path, "x", ("direction", "confirm_price", "extreme_price")) as file:
        stream = boundaries.BatchStream(file, **options)
        try:
            return list(stream)
        finally:
            stream.close()


def test_the_stream_gives_the_events_in_order_with_and_without_a_pool(events_file):
    wanted = pq.read_table(events_file).column("confirm_agg_trade_id").to_pylist()
    from concurrent.futures import ThreadPoolExecutor

    for pool_size in (0, 1, 3):
        meter = ReadMeter()
        pool = ThreadPoolExecutor(pool_size) if pool_size else None
        try:
            batches = read_all(
                events_file, pool=pool, meter=meter, lot_bytes=1, batch_rows=4
            )
        finally:
            if pool:
                pool.shutdown()
        got = [v for b in batches for v in b.column("confirm_agg_trade_id").to_pylist()]
        assert got == wanted
        assert all(b.num_rows <= 4 for b in batches)
        assert meter.bytes > 0 and meter.clock.seconds > 0


def test_the_stream_stays_a_few_batches_ahead(events_file):
    from concurrent.futures import ThreadPoolExecutor

    with (
        ThreadPoolExecutor(2) as pool,
        EventFile(events_file, "x") as file,
    ):
        stream = boundaries.BatchStream(
            file, pool, ReadMeter(), lot_bytes=1, batch_rows=2, ahead=2
        )
        first = next(stream)
        time.sleep(0.1)
        assert first.num_rows == 2
        with stream._cond:
            assert len(stream._ready) <= 2
        stream.close()
        assert l3_threads() == []
        assert not stream._ready


def test_a_read_error_comes_out_at_the_batch_that_would_have_brought_it(events_file):
    from concurrent.futures import ThreadPoolExecutor

    class Broken:
        def __init__(self, file):
            self.file = file

        def lots(self, lot_bytes):
            yield from self.file.lots(lot_bytes)

        def compressed(self, lot):
            return self.file.compressed(lot)

        def batches(self, lot, rows, request):
            yield from itertools.islice(self.file.batches(lot, rows, request), 1)
            raise OSError("se cortó la lectura")

    with ThreadPoolExecutor(2) as pool, EventFile(events_file, "x") as file:
        stream = boundaries.BatchStream(
            Broken(file), pool, ReadMeter(), lot_bytes=1, batch_rows=2
        )
        got = 0
        with pytest.raises(OSError, match="se cortó"):
            while True:
                next(stream)
                got += 1
        assert got >= 1
        stream.close()


def test_a_busy_clock_unites_the_intervals_of_concurrent_operations():
    clock = BusyClock()
    done = threading.Event()

    def work():
        with clock.busy():
            done.wait(1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    done.set()
    for t in threads:
        t.join()
    assert 0.09 < clock.seconds < 0.4  # no 4 × 0,1 s: la pared, no la suma


def lake_events(lake, text: str, month):
    base = "provider=binance/market=spot/asset=BTCUSDT"
    return (
        lake.l2
        / f"{base}/theta={text}/year={month[0]}/month={month[1]:02d}/events.parquet"
    )


@pytest.mark.parametrize(
    "knobs",
    [
        {"threads": 1, "readers": 1, "inflight": 1, "writers": 1, "engine_threads": 1},
        {"threads": 4, "readers": 2, "inflight": 4, "writers": 4, "engine_threads": 4},
    ],
)
def test_an_engine_error_aborts_the_unit_the_same_way_with_any_knobs(
    tmp_path, ticks, knobs
):
    from l3_dc_descriptors import L2StateInconsistent

    lake = build_lake(tmp_path / "lake", ticks, month_cuts(ticks, 1500, 3000))
    month = lake.months[1]
    # Un id de confirmación que no es un tick de L1: el motor lo ubica y lo rechaza.
    missing = lake.events[lake.thetas[0], month][4]["confirm_agg_trade_id"]
    path = l1_path(str(lake.l1), ("binance", "spot", "BTCUSDT"), month)
    table = pq.read_table(path)
    keep = [i for i, v in enumerate(table["agg_trade_id"].to_pylist()) if v != missing]
    pq.write_table(table.take(keep), path, row_group_size=150)
    messages = set()
    with pytest.raises(L2StateInconsistent) as caught:
        summarize_month(
            lake.l1,
            lake.l2,
            tmp_path / "out",
            month,
            lake.theta_texts,
            series_start=M1,
            **knobs,
        )
    messages.add(str(caught.value))
    assert caught.value.details["field"] == "confirm_agg_trade_id"
    assert caught.value.details["got"] == str(missing)
    assert caught.value.details["theta"] in lake.theta_texts
    assert l3_threads() == []
    assert not [p for p in (tmp_path / "out").rglob("*") if p.is_file()]


def test_an_event_that_never_closes_is_the_same_error_as_the_reference(tmp_path, ticks):
    from l3_dc_descriptors import L2StateInconsistent

    lake = build_lake(tmp_path / "lake", ticks, month_cuts(ticks, 1500, 3000))
    month = lake.months[1]
    # L1 termina antes del extremo del último evento de un θ: el evento queda sin cerrar.
    last = lake.events[lake.thetas[0], month][-1]["extreme_agg_trade_id"]
    path = l1_path(str(lake.l1), ("binance", "spot", "BTCUSDT"), month)
    table = pq.read_table(path)
    keep = [i for i, v in enumerate(table["agg_trade_id"].to_pylist()) if v < last]
    pq.write_table(table.take(keep), path, row_group_size=150)
    errors = []
    for name, runner in (("py", pyref.summarize_month), ("rs", summarize_month)):
        with pytest.raises(L2StateInconsistent) as caught:
            runner(
                lake.l1,
                lake.l2,
                tmp_path / name,
                month,
                lake.theta_texts,
                series_start=M1,
            )
        errors.append(caught.value)
    wanted, got = errors
    assert got.check_type == wanted.check_type
    for key in ("theta", "field", "got", "wanted"):
        assert got.details[key] == wanted.details[key], key
    assert got.details["field"] == "extreme_agg_trade_id"
    assert got.details["got"] == "ausente"
    assert l3_threads() == []


def test_a_price_in_another_width_is_refused_before_the_engine_reads_it(
    tmp_path, ticks
):
    from dc_frames import FramesInputError

    lake = build_lake(tmp_path / "lake", ticks, month_cuts(ticks, 1500, 3000))
    path = lake_events(lake, lake.theta_texts[0], M2)
    table = pq.read_table(path)
    index = table.schema.get_field_index("confirm_price")
    wide = table["confirm_price"].cast(pa.decimal128(20, 8))
    pq.write_table(table.set_column(index, "confirm_price", wide), path)
    with pytest.raises(FramesInputError, match="confirm_price"):
        summarize_month(
            lake.l1, lake.l2, tmp_path / "out", M2, lake.theta_texts, series_start=M1
        )
    assert l3_threads() == []
    assert not [p for p in (tmp_path / "out").rglob("*") if p.is_file()]


def test_the_unit_reports_the_boundaries_the_engine_held(lake, tiny_l2):
    phases = Phases()
    summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M2,
        lake.theta_texts,
        series_start=M1,
        phases=phases,
        inflight=1,
    )
    # Con lotes de 3 eventos, el motor suelta cada uno al cerrar sus eventos: lo que tuvo
    # en mano es una fracción de las fronteras del mes.
    month_bytes = sum(
        pq.ParquetFile(lake_events(lake, text, M2)).metadata.num_rows
        for text in lake.theta_texts
    )
    assert 0 < phases.retained.peak
    assert phases.retained.peak < month_bytes * 65
    assert phases.retained.current == 0


# La pasada de la familia T (`--family path`): mismas tablas y mismo `content_hash` que la
# referencia con cualquier hilo, lector, escritor, lote y tamaño de lote de fronteras.


def run_path(lake, **options) -> dict:
    """Los resultados y las tablas de `path.parquet` de todos los meses (con `summaries.parquet`
    ya escrito por la referencia de S).
    """
    results, tables = {}, {}
    for month in lake.months:
        results[month] = summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            family="path",
            **options,
        )
        for theta, result in results[month].items():
            tables[month, theta] = pq.read_table(result.path, schema=PATH_SCHEMA)
    return {"results": results, "tables": tables}


@pytest.fixture(scope="module")
def path_reference(lake, reference):
    """La referencia de T: `native_driver.path_native` sobre el `summaries.parquet` de la
    referencia de S, que queda escrito en el lago.
    """
    results, tables = {}, {}
    for month in lake.months:
        results[month] = path_native(lake.l1, lake.l2, lake.l3, month, lake.theta_texts)
        for theta, result in results[month].items():
            tables[month, theta] = pq.read_table(result.path, schema=PATH_SCHEMA)
    return {"results": results, "tables": tables}


def assert_same_path(got: dict, expected: dict) -> None:
    assert got["results"].keys() == expected["results"].keys()
    for month, per_theta in expected["results"].items():
        assert got["results"][month].keys() == per_theta.keys()
        for theta, result in per_theta.items():
            other = got["results"][month][theta]
            assert other.content_hash == result.content_hash
            assert other.rows == result.rows
            assert other.reconciliation is None  # T no concilia: ya lo hizo S
    left, right = expected["tables"], got["tables"]
    assert left.keys() == right.keys()
    assert all(left[key].equals(right[key]) for key in left)
    assert sum(t.num_rows for t in left.values()) > 0


def test_the_path_reference_is_the_oracle(lake, path_reference):
    """La referencia de T es el oráculo `pathref` sobre el lector de `dc_frames`: INT64 y nulos
    exactos, DOUBLE a 1e-12 relativo.
    """
    compared = 0
    for month in lake.months:
        wanted = pathref.from_lake(lake.l1, lake.l2, month, lake.theta_texts)
        got = {
            (row["theta"], row["confirm_agg_trade_id"]): row
            for (m, _), table in path_reference["tables"].items()
            if m == month
            for row in table.to_pylist()
        }
        assert got.keys() == wanted.keys()
        for key, values in wanted.items():
            for name, value in zip(pathref.COLUMNS, values, strict=True):
                if value is None or isinstance(value, int):
                    assert got[key][name] == value, (key, name)
                else:
                    assert math.isclose(got[key][name], value, rel_tol=1e-12), (
                        key,
                        name,
                    )
        compared += len(got)
    assert compared > 0


@pytest.mark.parametrize(("engine_threads", "readers", "inflight", "writers"), KNOBS)
def test_the_path_pass_gives_the_same_table_and_hash_with_any_knobs(
    lake, path_reference, engine_threads, readers, inflight, writers
):
    got = run_path(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
    )
    assert_same_path(got, path_reference)
    assert l3_threads() == []


@pytest.mark.parametrize("threads", [1, 2, 50])
def test_the_path_pass_does_not_depend_on_the_io_pool(lake, path_reference, threads):
    assert_same_path(run_path(lake, threads=threads), path_reference)


@pytest.mark.parametrize("lot_events", [1, 5, 10, 25, 100])
@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight", "writers"), [(1, 1, 1, 1), (4, 2, 4, 4)]
)
def test_the_path_pass_gives_the_same_result_in_lots(
    lake, path_reference, lot_events, engine_threads, readers, inflight, writers
):
    """Con tramos de pocos ticks hay eventos que cruzan tramos y row groups: la fusión
    ordenada de T (ADR-L3-17) da la misma tabla y el mismo `content_hash` que sin corte.
    """
    got = run_path(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
        lot_events=lot_events,
    )
    assert_same_path(got, path_reference)


@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight", "writers"),
    [(1, 1, 1, 1), (2, 2, 4, 1), (4, 1, 4, 4), (4, 2, 1, 4)],
)
def test_the_path_pass_gives_the_same_result_with_boundaries_in_lots_of_three(
    lake, path_reference, tiny_l2, engine_threads, readers, inflight, writers
):
    got = run_path(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
    )
    assert_same_path(got, path_reference)
    assert_same_path(
        run_path(lake, engine_threads=4, inflight=4, lot_events=3), path_reference
    )


def test_a_path_run_does_not_change_a_byte_of_summaries(lake, path_reference):
    files = sorted(lake.l3.rglob("summaries.parquet"))
    assert len(files) == len(lake.thetas) * len(lake.months)
    before = {f: f.read_bytes() for f in files}
    run_path(lake, engine_threads=2, lot_events=10)
    assert {f: f.read_bytes() for f in files} == before


def test_the_path_pass_reports_the_lineage_of_l2_like_summaries(
    lake, reference, path_reference
):
    for month, per_theta in path_reference["results"].items():
        got = summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            family="path",
        )
        for theta, result in got.items():
            assert result.l2_events_content_hash == (
                reference["results"][month][theta].l2_events_content_hash
            )
            assert result.l2_carry_over_content_hash == (
                reference["results"][month][theta].l2_carry_over_content_hash
            )
            assert result.content_hash == per_theta[theta].content_hash


def test_the_path_pass_without_summaries_fails_without_publishing(lake, path_reference):
    theta = lake.theta_texts[0]
    target = lake.l3 / (
        "provider=binance/market=spot/asset=BTCUSDT/"
        f"theta={theta}/year={M1[0]}/month={M1[1]:02d}"
    )
    hidden = target / "summaries.hidden"
    (target / "summaries.parquet").rename(hidden)
    before = (target / "path.parquet").read_bytes()
    try:
        with pytest.raises(FramesInputError, match="summaries"):
            summarize_month(
                lake.l1,
                lake.l2,
                lake.l3,
                M1,
                lake.theta_texts,
                series_start=M1,
                family="path",
            )
    finally:
        hidden.rename(target / "summaries.parquet")
    assert (target / "path.parquet").read_bytes() == before
    assert l3_threads() == []


def test_an_unknown_family_is_refused(lake):
    with pytest.raises(ValueError, match="familia"):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            family="fit",
        )
