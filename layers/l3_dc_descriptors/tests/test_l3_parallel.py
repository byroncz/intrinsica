"""Lector anticipado y pool de θ por row group (ADR-L3-15): la estructura es paralelizable
sin estado compartido (ITSC-333) y la corrida con hilos no cambia nada de lo que escribe.

Estas pruebas no miden velocidad. Corren `summarize_month` con hilos 1, 2 y 4 y lector
anticipado de 0, 1 y 2 row groups y exigen exactamente la misma tabla y el mismo
`content_hash` que la referencia secuencial (`threads=1`, `prefetch=0`). Lo que sí queda
en Python puro (recomposición y VWAP por segmento) es el límite conocido de la
paralelización por hilos que documenta el README.
"""

import ast
import importlib
import inspect
import pkgutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import l3_dc_descriptors
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors import (
    SUMMARIES_SCHEMA,
    L2InputMissing,
    summarize,
    summarize_month,
)
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.cpu import CpuLimit
from l3_dc_descriptors.pipeline import process_unit
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts

THREADS = (1, 2, 4)
PREFETCH = (0, 1, 2)


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


def run_all(lake, **options) -> dict:
    """Los resultados y las tablas de todos los meses, leídas antes de la siguiente corrida
    (que reescribe los mismos archivos).
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
            **options,
        )
        for theta, result in results[month].items():
            tables[month, theta] = pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
    return {"results": results, "tables": tables}


@pytest.fixture(scope="module")
def reference(lake):
    return run_all(lake, threads=1, prefetch=0)


@pytest.mark.parametrize("prefetch", PREFETCH)
@pytest.mark.parametrize("threads", THREADS)
def test_threads_and_prefetch_give_the_same_table_and_hash(
    lake, reference, threads, prefetch
):
    got = run_all(lake, threads=threads, prefetch=prefetch)

    expected = reference["results"]
    assert got["results"].keys() == expected.keys()
    for month in expected:
        for theta, result in expected[month].items():
            other = got["results"][month][theta]
            assert other.content_hash == result.content_hash
            assert other.rows == result.rows
            assert other.reconciliation == result.reconciliation
    left, right = reference["tables"], got["tables"]
    assert left.keys() == right.keys()
    assert all(left[key].equals(right[key]) for key in left)
    assert sum(t.num_rows for t in left.values()) > 0


def test_the_defaults_give_the_same_result_too(lake, reference):
    got = run_all(lake)
    for month, per_theta in reference["results"].items():
        for theta, result in per_theta.items():
            assert got["results"][month][theta].content_hash == result.content_hash


@pytest.mark.parametrize(
    ("options", "message"),
    [({"threads": 0}, "threads"), ({"prefetch": -1}, "prefetch")],
)
def test_it_rejects_threads_or_prefetch_out_of_range(lake, options, message):
    with pytest.raises(ValueError, match=message):
        summarize_month(
            lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1, **options
        )


def test_it_never_uses_more_threads_than_thetas():
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


def l3_threads() -> list[str]:
    return [
        t.name
        for t in threading.enumerate()
        if t.name.startswith(("l3-read-ahead", "l3-theta"))
    ]


def test_a_failing_theta_raises_the_same_error_as_the_sequential_flow(
    lake, monkeypatch
):
    add = summarize._Lane.add

    def failing(self, agg, events):
        if self.text in lake.theta_texts[1::2]:  # el 2.º y el 4.º θ
            raise RuntimeError(f"falla {self.text}")
        add(self, agg, events)

    monkeypatch.setattr(summarize._Lane, "add", failing)
    messages = set()
    for threads, prefetch in ((1, 0), (4, 1), (4, 2)):
        with pytest.raises(RuntimeError) as caught:
            summarize_month(
                lake.l1,
                lake.l2,
                lake.l3,
                M1,
                lake.theta_texts,
                series_start=M1,
                threads=threads,
                prefetch=prefetch,
            )
        messages.add(str(caught.value))
        assert l3_threads() == []  # ningún hilo sobrevive al error
    assert messages == {f"falla {lake.theta_texts[1]}"}


class Source:
    """Un lector de mentira: entrega `count` row groups y anota cuántos pidió."""

    def __init__(self, count: int, fail_at: int | None = None) -> None:
        self.count, self.fail_at = count, fail_at
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pulled == self.count:
            raise StopIteration
        if self.pulled == self.fail_at:
            raise KeyError("el lector falló")
        self.pulled += 1
        return object()


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_read_ahead_never_holds_more_than_depth_plus_the_one_in_process(depth):
    source = Source(8)
    with summarize._ReadAhead(source, depth) as ahead:
        for index, item in enumerate(ahead):
            time.sleep(0.03)  # el lector tiene tiempo de sobra para adelantarse
            # `item` en proceso y a lo más `depth` más decodificados o listos.
            assert source.pulled <= index + 1 + depth
        assert source.pulled == 8
    assert l3_threads() == []


def test_the_read_ahead_relays_a_reader_error_after_the_row_groups_before_it():
    source = Source(5, fail_at=2)
    got = []
    with summarize._ReadAhead(source, 1) as ahead:
        with pytest.raises(KeyError, match="el lector falló"):
            while True:
                got.append(next(ahead))
        with pytest.raises(StopIteration):
            next(ahead)
    assert len(got) == 2 and l3_threads() == []


def test_leaving_early_stops_the_reader_and_drops_what_it_brought():
    source = Source(100)
    with summarize._ReadAhead(source, 2) as ahead:
        next(ahead)
        time.sleep(0.05)
    assert l3_threads() == []
    assert source.pulled <= 1 + 2 + 1  # lo que ya traía al salir; no sigue leyendo
    settled = source.pulled
    time.sleep(0.05)
    assert source.pulled == settled


def test_the_read_ahead_runs_in_a_copy_of_the_callers_context():
    import contextvars

    var = contextvars.ContextVar("probe", default="sin valor")
    seen = []

    class Reading(Source):
        def __next__(self):
            seen.append(var.get())
            return super().__next__()

    var.set("del hilo principal")
    with summarize._ReadAhead(Reading(2), 1) as ahead:
        list(ahead)
    assert set(seen) == {"del hilo principal"}
    assert var.get() == "del hilo principal"


def _modules():
    for info in pkgutil.iter_modules(l3_dc_descriptors.__path__):
        if info.name == "__main__":  # importarlo ejecuta la CLI
            continue
        yield importlib.import_module(f"l3_dc_descriptors.{info.name}")


def test_no_module_declares_global_state():
    # (a): ningún `global` y ningún contenedor mutable a nivel de módulo.
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


# Compuerta y linaje en el pool (ITSC-360)


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


def test_the_gate_raises_the_first_failing_theta_in_theta_order(tmp_path, ticks):
    # Los dos primeros θ no tienen events.parquet: con hilos, el error es el del primero,
    # como en serie, y no queda ningún archivo ni hilo. Lago propio: se borran archivos.
    lake = build_lake(tmp_path, ticks, month_cuts(ticks, 1500, 3000), l1_row_group=150)
    thetas = lake.theta_texts
    for text in thetas[:2]:
        lake_file(lake, text, M2, "events.parquet").unlink()
    messages = set()
    for threads in (1, 4):
        with pytest.raises(L2InputMissing) as caught:
            summarize_month(
                lake.l1,
                lake.l2,
                lake.l3,
                M2,
                thetas,
                series_start=M1,
                threads=threads,
            )
        messages.add(str(caught.value))
        assert caught.value.details["theta"] == thetas[0]
        assert [p for p in lake.l3.rglob("*") if p.is_file()] == []
        assert l3_threads() == []
    assert len(messages) == 1


def lake_file(lake, text: str, month, name: str):
    base = "provider=binance/market=spot/asset=BTCUSDT"
    return lake.l2 / f"{base}/theta={text}/year={month[0]}/month={month[1]:02d}/{name}"


@pytest.mark.parametrize("threads", THREADS)
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

    expected, _ = lineage(threads=1, prefetch=0)
    got, timing = lineage(threads=threads, prefetch=1)
    assert got == expected and len(got) == len(lake.thetas)
    assert timing.phases.lineage_s > 0 and timing.phases.gate_s > 0
