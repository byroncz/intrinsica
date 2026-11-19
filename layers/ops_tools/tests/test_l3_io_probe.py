"""`l3_io_probe.py` contra un lago local mínimo y con cifras simuladas para la regla.

El lago de `tmp_path` tiene el `consolidated.parquet` de un mes con varios row groups de bytes
aleatorios (no se comprimen: los rangos pesan lo que las columnas). Con rutas locales las tres
formas de lectura y las dos de escritura corren sin red; la regla del veredicto se prueba aparte, con
cifras simuladas. La sonda no importa ninguna capa.
"""

import csv
import importlib.util
import os
import re
import sys
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
MONTH = "2023-03"
GROUPS = 8
ROWS = 70_000
COLUMNS = 3
LAST_LINE = re.compile(
    r"io probe: mes=(?P<mes>\S+) lecturas_mib=(?P<lectura_mib>\S+) a_2=(?P<a2>\S+) b_2=(?P<b2>\S+) "
    r"escrituras_mib=(?P<escritura_mib>\S+) a_12=(?P<a12>\S+) b_12=(?P<b12>\S+) "
    r"lecturas=(?P<lecturas>limita|no_limita|fallo) "
    r"escrituras=(?P<escrituras>limita|no_limita|fallo) "
    r"candado=(?P<candado>proceso|sistema_de_archivos|ninguno|fallo)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_io_probe", SCRIPTS / "l3_io_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["l3_io_probe"] = module
    spec.loader.exec_module(module)
    return module


probe = load()


def random_column(rows: int) -> pa.Array:
    buffer = pa.py_buffer(os.urandom(rows * 8))
    return pa.Array.from_buffers(pa.int64(), rows, [None, buffer])


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    root = tmp_path_factory.mktemp("l1")
    folder = root / SERIES / "year=2023" / "month=03"
    folder.mkdir(parents=True)
    table = pa.table({f"c{i}": random_column(ROWS) for i in range(COLUMNS)})
    with pq.ParquetWriter(
        folder / "consolidated.parquet", table.schema, compression="none"
    ) as writer:
        for _ in range(GROUPS):
            writer.write_table(table)
    return root


def uri(lake):
    return probe.l1_file(str(lake), ("binance", "spot", "BTCUSDT"), MONTH)


@pytest.fixture(autouse=True)
def local_env(monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)


def test_does_not_import_any_layer():
    text = (SCRIPTS / "l3_io_probe.py").read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+(\S+)", text, re.MULTILINE)
    assert not [
        name
        for name in imports
        if name.split(".")[0] in {"l3_dc_descriptors", "dc_frames", "pyutils"}
    ]


def test_ranges_are_the_row_groups_of_the_footer(lake):
    path = uri(lake)
    ranges, total = probe.sample_ranges(path, GROUPS, seed=1)
    assert total == GROUPS == len(ranges)
    size = Path(path).stat().st_size
    starts = [start for start, _ in ranges]
    assert starts == sorted(starts)
    for (start, length), (next_start, _) in pairwise(ranges):
        assert start + length <= next_start
    assert ranges[-1][0] + ranges[-1][1] <= size
    assert all(length >= ROWS * 8 * COLUMNS for _, length in ranges)


def test_sample_is_fixed_by_seed_and_capped_by_the_row_groups(lake):
    path = uri(lake)
    first, _ = probe.sample_ranges(path, 3, seed=7)
    again, _ = probe.sample_ranges(path, 3, seed=7)
    other, _ = probe.sample_ranges(path, 3, seed=8)
    assert first == again and len(first) == 3
    assert first != other
    assert len(probe.sample_ranges(path, 99, seed=7)[0]) == GROUPS


def test_missing_footer_is_an_input_error(tmp_path):
    with pytest.raises(probe.InputError):
        probe.sample_ranges(str(tmp_path / "no.parquet"), 4, seed=1)


@pytest.mark.parametrize("form", probe.READ_FORMS)
@pytest.mark.parametrize("threads", [1, 3])
def test_every_read_form_moves_exactly_the_sampled_bytes(lake, form, threads):
    ranges, _ = probe.sample_ranges(uri(lake), 5, seed=3)
    run = probe.read_run(form, threads, uri(lake), ranges)
    assert run.errors == 0 and run.reliable
    assert run.moved == sum(size for _, size in ranges)
    assert run.tasks == 5 and run.workers == min(threads, 5)
    assert run.mib_per_s > 0 and run.cpu_ratio >= 0 and 0 < run.occupancy <= 1.0001


def test_more_threads_than_ranges_use_only_as_many_workers_as_ranges(lake):
    ranges, _ = probe.sample_ranges(uri(lake), 2, seed=3)
    assert probe.read_run("a2", 32, uri(lake), ranges).workers == 2


def test_a_read_past_the_end_is_a_counted_error_not_a_crash(lake):
    size = Path(uri(lake)).stat().st_size
    run = probe.read_run("a", 2, uri(lake), [(size - 10, 100), (0, 10)])
    assert run.errors == 1 and run.moved == 10 and not run.reliable


@pytest.mark.parametrize("form", probe.WRITE_FORMS)
def test_every_write_form_creates_distinct_objects_with_the_payload(
    tmp_path, lake, form
):
    payload = os.urandom(3 * probe.MIB + 17)
    prefix = f"{tmp_path}/io_probe/x"
    run = probe.write_run(form, 4, prefix, payload, objects=6, warm_uri=uri(lake))
    files = sorted(Path(prefix).glob("*.bin"))
    assert run.errors == 0 and len(files) == 6 == len({f.name for f in files})
    assert run.moved == 6 * len(payload)
    assert all(f.read_bytes() == payload for f in files)


def test_the_two_write_forms_never_share_an_object_name(tmp_path, lake):
    payload = os.urandom(1024)
    prefix = f"{tmp_path}/x"
    for form in probe.WRITE_FORMS:
        for threads in (1, 4):
            probe.write_run(
                form, threads, prefix, payload, objects=3, warm_uri=uri(lake)
            )
    assert len(list(Path(prefix).glob("*.bin"))) == 2 * 2 * 3


def test_a_write_form_is_warmed_with_the_l1_object_before_the_clock_starts(
    tmp_path, lake, monkeypatch
):
    events = []

    class Recorder:
        def __init__(self, threads, root):
            pass

        def warm(self, warm_uri):
            events.append(("warm", warm_uri))

        def put(self, name, payload):
            events.append(("put", name))
            return len(payload)

    monkeypatch.setitem(probe.WRITERS, "z", Recorder)
    run = probe.write_run("z", 1, str(tmp_path), b"x", objects=2, warm_uri=uri(lake))
    assert run.errors == 0
    assert [kind for kind, _ in events] == ["warm", "put", "put"]
    assert events[0][1] == uri(lake)


# --- la regla, con cifras simuladas -----------------------------------------------------------


def rates(**values):
    """`a_2=10, a2_8=5` -> {("a", 2): 10, ("a2", 8): 5}."""
    return {
        (name.rsplit("_", 1)[0], int(name.rsplit("_", 1)[1])): value
        for name, value in values.items()
    }


def reads(a1=10, a8=10, a2_1=10, a2_8=10, b1=10, b8=10, a2=10, b2=10):
    return rates(a_1=a1, a_2=a2, a_8=a8, a2_1=a2_1, a2_8=a2_8, b_1=b1, b_2=b2, b_8=b8)


def test_limits_needs_twice_the_rate_and_the_boundary_counts():
    assert probe.limits(10.0, 20.0) == "limita"
    assert probe.limits(10.0, 19.9) == "no_limita"
    assert probe.limits(10.0, 5.0) == "no_limita"


def test_limits_is_fallo_when_a_figure_is_missing_or_zero():
    assert probe.limits(None, 20.0) == "fallo"
    assert probe.limits(10.0, None) == "fallo"
    assert probe.limits(0.0, 20.0) == "fallo"


def test_verdict_uses_the_production_concurrency_only():
    # Con 8 hilos (b) rinde 10 veces más, pero con 2 lectores no: las lecturas no limitan.
    got = probe.verdicts(reads(a2=50, b2=60, a8=5, b8=50), rates(a_12=100, b_12=250))
    assert got["lecturas"] == "no_limita" and got["escrituras"] == "limita"
    got = probe.verdicts(reads(a2=50, b2=100), rates(a_12=100, b_12=199))
    assert got["lecturas"] == "limita" and got["escrituras"] == "no_limita"


def test_lock_is_the_file_system_if_a2_doubles_a_at_8_threads():
    assert probe.lock(reads(a8=10, a2_8=20)) == "sistema_de_archivos"
    # Gana aunque (b) también escale.
    assert probe.lock(reads(a8=10, a2_8=25, b1=10, b8=80)) == "sistema_de_archivos"


def test_lock_is_the_process_if_neither_pyarrow_form_scales_and_google_does():
    got = reads(a1=10, a8=15, a2_1=10, a2_8=18, b1=10, b8=60)
    assert probe.lock(got) == "proceso"


def test_lock_is_none_in_every_other_case():
    # (a) escala: no hay tope.
    assert probe.lock(reads(a1=10, a8=40, a2_1=10, a2_8=15)) == "ninguno"
    # Ni pyarrow ni Google escalan: el tope no es de pyarrow.
    assert probe.lock(reads(a1=10, a8=12, a2_1=10, a2_8=12, b1=10, b8=12)) == "ninguno"
    # (a2) escala pero no dobla a (a): tampoco es «proceso».
    assert probe.lock(reads(a1=10, a8=15, a2_1=10, a2_8=25, b1=10, b8=60)) == "ninguno"


def test_lock_is_fallo_if_a_run_it_needs_is_missing():
    partial = reads()
    del partial[("a2", 8)]
    assert probe.lock(partial) == "fallo"
    assert probe.lock({}) == "fallo"


def test_a_run_with_more_than_5_percent_failed_tasks_does_not_count():
    def run(errors, tasks=24):
        return probe.Run("lecturas", "a", 2, 2, 1.0, 0.5, 1.5, tasks, MIB_TEST, errors)

    assert run(1).reliable and not run(2).reliable
    assert probe.rates([run(2)], "lecturas") == {("a", 2): None}
    assert not run(0, tasks=0).reliable


MIB_TEST = probe.MIB


def test_threads_always_include_the_ones_the_rule_needs():
    assert probe.parse_threads("4", "--read-threads", probe.REQUIRED_READ_THREADS) == (
        1,
        2,
        4,
        8,
    )
    assert probe.parse_threads(
        "1", "--write-threads", probe.REQUIRED_WRITE_THREADS
    ) == (
        1,
        12,
    )
    with pytest.raises(ValueError):
        probe.parse_threads("0", "--read-threads", ())
    with pytest.raises(ValueError):
        probe.parse_threads("x", "--read-threads", ())


def test_the_runs_that_decide_the_rule_go_first():
    plan = probe.plan_runs((1, 2, 8, 32), (1, 4, 12))
    decisive = {
        r
        for r in plan
        if (r[0] == "lecturas" and r[2] in (1, 2, 8))
        or (r[0] == "escrituras" and r[2] == 12)
    }
    assert set(plan[: len(decisive)]) == decisive
    assert len(plan) == 4 * 3 + 3 * 2 == len(set(plan))
    assert plan[-1][2] in (32, 4)


# --- la sonda de punta a punta -----------------------------------------------------------------


def run_main(lake, tmp_path, capsys, *extra):
    out = tmp_path / "salida.csv"
    code = probe.main(
        [
            "--l1-root",
            str(lake),
            "--write-root",
            str(tmp_path / "results"),
            "--out",
            str(out),
            "--groups",
            "6",
            "--object-mib",
            "1",
            *extra,
        ]
    )
    return code, capsys.readouterr().out, out


def test_probe_runs_end_to_end_and_ends_with_the_line(lake, tmp_path, capsys):
    code, text, out = run_main(lake, tmp_path, capsys)
    last = text.strip().splitlines()[-1]
    found = LAST_LINE.match(last)
    assert found, last
    assert code == 0 and found["mes"] == MONTH
    assert found["escritura_mib"] == "12.0"
    assert float(found["lectura_mib"]) > 0
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == probe.CSV_HEADER
    runs = {
        (r["direccion"], r["forma"], r["hilos"])
        for r in rows
        if r["seccion"] == "corrida"
    }
    assert len(runs) == 4 * 3 + 3 * 2
    measures = {r["medida"] for r in rows if r["seccion"] == "corrida"}
    assert {"pared_s", "mib_por_s", "cpu_ratio", "ocupacion", "errores"} <= measures
    env = {r["medida"]: r["valor"] for r in rows if r["seccion"] == "entorno"}
    assert {"cores", "pyarrow", "muestra_mib"} <= set(env)
    assert env["pyarrow"] == pa.__version__
    assert len(list((tmp_path / "results/io_probe").rglob("*.bin"))) == 6 * 12


def test_probe_leaves_one_object_per_write_task_under_its_own_prefix(
    lake, tmp_path, capsys
):
    run_main(
        lake, tmp_path, capsys, "--write-threads", "1,12", "--read-threads", "1,2,8"
    )
    written = list((tmp_path / "results/io_probe").rglob("*.bin"))
    assert len(written) == 12 * 4  # 2 formas × 2 cantidades de escritores
    assert len({f.name for f in written}) == len(written)
    assert all(f.stat().st_size == probe.MIB for f in written)


def test_budget_spent_runs_nothing_and_ends_in_fallo(lake, tmp_path, capsys):
    code, text, _ = run_main(lake, tmp_path, capsys, "--budget-s", "-1")
    assert code == 1 and "omitida, pasó el presupuesto" in text
    found = LAST_LINE.match(text.strip().splitlines()[-1])
    assert (
        found
        and found["lecturas"] == found["escrituras"] == found["candado"] == "fallo"
    )
    assert found["a2"] == found["b12"] == "-"


def test_a_run_over_the_memory_cap_is_skipped(lake, tmp_path, capsys):
    # Cada rango pesa ≈ 1,6 MiB: con 2 hilos pasa de 3 MiB, con 1 no.
    code, text, _ = run_main(lake, tmp_path, capsys, "--max-inflight-mib", "3")
    assert "lecturas:a_2: omitida, su peor caso en vuelo" in text
    assert "lecturas:a_1: omitida" not in text
    found = LAST_LINE.match(text.strip().splitlines()[-1])
    assert code == 1 and found["lecturas"] == "fallo" and found["a2"] == "-"
    assert found["escrituras"] in ("limita", "no_limita")


def test_a_missing_month_ends_in_fallo_with_code_1(tmp_path, capsys):
    code, text, _ = run_main(tmp_path / "vacio", tmp_path, capsys)
    assert code == 1 and "FALLO" in text
    found = LAST_LINE.match(text.strip().splitlines()[-1])
    assert found and found["lecturas"] == "fallo" and found["lectura_mib"] == "-"


def test_usage_errors_exit_with_2(lake, tmp_path):
    base = ["--l1-root", str(lake), "--write-root", str(tmp_path)]
    for bad in (
        ["--read-threads", "a,b"],
        ["--write-threads", "0"],
        ["--month", "2023-13"],
        ["--groups", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            probe.main(base + bad)
        assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:  # sin --write-root ni OPS_RESULTS_URI
        probe.main(["--l1-root", str(lake)])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:  # sin --l1-root ni OPS_RESULTS_URI
        probe.main(["--write-root", str(tmp_path)])
    assert exc.value.code == 2


def test_out_and_write_root_must_be_under_the_results_of_the_job(
    lake, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://p-ops/results/r1/")
    for bad in (
        ["--out", "gs://otro/x.csv", "--write-root", str(tmp_path)],
        ["--write-root", "gs://p-landing/x"],
    ):
        with pytest.raises(SystemExit) as exc:
            probe.main(["--l1-root", str(lake), *bad])
        assert exc.value.code == 2
