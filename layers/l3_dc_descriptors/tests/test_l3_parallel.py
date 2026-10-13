"""La estructura es paralelizable sin estado compartido (ITSC-333; decide ITSC-335).

Esta prueba no mide velocidad. Agrega los θ de cada row group en un `ThreadPoolExecutor`
en vez del bucle secuencial y exige exactamente la misma tabla y el mismo `content_hash`:
demuestra que no hay estado compartido que el GIL estuviera tapando. Lo que sí queda en
Python puro (recomposición y VWAP por segmento) es el límite conocido de la paralelización
por hilos que documenta el README.
"""

import ast
import importlib
import inspect
import pkgutil
from concurrent.futures import ThreadPoolExecutor

import l3_dc_descriptors
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors import SUMMARIES_SCHEMA, summarize, summarize_month
from l3_dc_descriptors.rowgroup import Prefixes
from l3_lake import M1, build_lake, fixture_ticks, month_cuts


@pytest.fixture(scope="module")
def ticks():
    return fixture_ticks(seed=5, drop=0.05)


def run_all(lake) -> dict:
    return {
        month: summarize_month(
            lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
        )
        for month in lake.months
    }


def tables(lake, results) -> dict:
    return {
        (month, theta): pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
        for month, per_theta in results.items()
        for theta, result in per_theta.items()
    }


@pytest.mark.parametrize("workers", [2, 8])
def test_threads_per_theta_give_the_same_table_and_hash(
    tmp_path, ticks, monkeypatch, workers
):
    # Row groups chicos: cada uno lo tocan los 5 θ, y hay eventos que cruzan row groups.
    sequential = build_lake(
        tmp_path / "seq", ticks, month_cuts(ticks, 1500, 3000), l1_row_group=150
    )
    expected = run_all(sequential)

    threaded = build_lake(
        tmp_path / "par", ticks, month_cuts(ticks, 1500, 3000), l1_row_group=150
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def consume(rg, lanes):
            agg = Prefixes(rg.frame)  # una vez por row group, solo de lectura después
            work = [pool.submit(lanes[e.theta].add, agg, e) for e in rg.thetas]
            for future in work:
                future.result()

        monkeypatch.setattr(summarize, "_consume", consume)
        got = run_all(threaded)

    assert got.keys() == expected.keys()
    for month in expected:
        for theta, result in expected[month].items():
            other = got[month][theta]
            assert other.content_hash == result.content_hash
            assert other.rows == result.rows
            assert other.reconciliation == result.reconciliation
    left, right = tables(sequential, expected), tables(threaded, got)
    assert left.keys() == right.keys()
    assert all(left[key].equals(right[key]) for key in left)
    assert sum(t.num_rows for t in left.values()) > 0


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
