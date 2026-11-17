"""`l3_frontier_probe.py` contra un lago local mínimo y con tiempos simulados para la regla.

El lago de `tmp_path` tiene, por θ y mes, un `path.parquet` (columna `theta` decimal, varios row
groups y `state_version` en el pie) y el `events.parquet` con las mismas filas por row group. La
sonda no importa ninguna capa. La variante c usa procesos (`spawn`), que importan el módulo por su
nombre: por eso la prueba lo registra en `sys.modules` y pone `scripts/` en `sys.path`.
"""

import csv
import importlib.util
import re
import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
THETAS = (100_000, 250_000, 500_000, 980_520)
MONTHS = [(2017, m) for m in range(1, 7)] + [(2023, m) for m in range(1, 7)]
LAST_LINE = re.compile(
    r"frontier probe: muestra=(?P<muestra>\S+) base_1=(?P<b1>\S+)s base_2=(?P<b2>\S+)s "
    r"base_4=(?P<b4>\S+)s cpu_ratio_4=(?P<ratio>\S+) mejor_variante=(?P<mejor>\S+) "
    r"pies_por_s_mejor=(?P<pies>\S+) veredicto=(?P<veredicto>cpu|gil|espera|mixto|fallo)$"
)


def load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "l3_frontier_probe", SCRIPTS / "l3_frontier_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["l3_frontier_probe"] = module
    spec.loader.exec_module(module)
    return module


probe = load()


def write_partition(
    root: Path,
    theta: int,
    year: int,
    month: int,
    groups: list[int],
    family: bool = True,
):
    folder = (
        root / SERIES / f"theta=0.{theta:08d}" / f"year={year}" / f"month={month:02d}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    rows = sum(groups)
    if family:
        value = Decimal(theta).scaleb(-8)
        table = pa.table(
            {
                "theta": pa.array([value] * rows, pa.decimal128(18, 8)),
                "x": pa.array(range(rows), pa.int64()),
            }
        ).replace_schema_metadata({"state_version": "1.0.0"})
        name = "path.parquet"
    else:
        table = pa.table({"confirm": pa.array(range(rows), pa.int64())})
        name = "events.parquet"
    with pq.ParquetWriter(folder / name, table.schema) as writer:
        start = 0
        for size in groups:
            writer.write_table(table.slice(start, size))
            start += size


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    base = tmp_path_factory.mktemp("lake")
    l3, l2 = base / "l3", base / "l2"
    for i, theta in enumerate(THETAS):
        for j, (year, month) in enumerate(MONTHS):
            groups = [5 + (i + j) % 3] * (2 + (i * j) % 4)
            write_partition(l3, theta, year, month, groups)
            write_partition(l2, theta, year, month, groups, family=False)
    return l3, l2


def config(lake):
    return probe.Config(
        str(lake[0]), str(lake[1]), ("binance", "spot", "BTCUSDT"), "path"
    )


def all_tasks(lake):
    return sorted(
        probe.list_months(str(lake[0]), ("binance", "spot", "BTCUSDT"), "path.parquet")
    )


def test_does_not_import_any_layer():
    text = (SCRIPTS / "l3_frontier_probe.py").read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+(\S+)", text, re.MULTILINE)
    assert not [
        name
        for name in imports
        if name.split(".")[0] in {"l3_dc_descriptors", "dc_frames", "pyutils"}
    ]


def test_listing_finds_every_partition(lake):
    assert len(all_tasks(lake)) == len(THETAS) * len(MONTHS)


def test_sample_is_stratified_fixed_by_seed_and_cedes_quota(lake):
    available = set(all_tasks(lake))
    sample, strata = probe.stratified_sample(available, 24, seed=7)
    assert len(sample) == 24 == sum(strata.values())
    assert sorted(strata) == [
        "theta_chico_reciente",
        "theta_chico_viejo",
        "theta_grande_reciente",
        "theta_grande_viejo",
    ]
    assert set(strata.values()) == {6}
    assert probe.stratified_sample(available, 24, seed=7)[0] == sample
    assert probe.stratified_sample(available, 24, seed=8)[0] != sample
    # Un estrato corto cede su cuota: solo 2 tareas viejas con θ chico, el resto sale de los demás.
    poor = {t for t in available if not (t.theta < 500_000 and t.year == 2017)} | {
        t for t in available if t.theta == 100_000 and t.month <= 2 and t.year == 2017
    }
    sample, strata = probe.stratified_sample(poor, 36, seed=7)
    assert len(sample) == 36 == sum(strata.values())
    assert strata["theta_chico_viejo"] == 2
    assert probe.stratified_sample(available, 10_000, seed=1)[0] == sorted(available)


def test_base_stages_add_up_to_the_task_wall(lake):
    reader = probe.Reader(config(lake))
    for task in all_tasks(lake)[:5]:
        result = probe.check_base(reader, task)
        assert result.error is None
        assert sum(result.stage_wall) == pytest.approx(result.wall, rel=1e-6, abs=1e-6)
        assert all(w >= 0 for w in result.stage_wall + result.stage_cpu)
        assert len(result.files) == 2


def test_run_wall_is_the_sum_of_the_stages_with_one_reader(lake):
    tasks = all_tasks(lake)
    done = probe.execute("base", 1, config(lake), tasks)
    busy = sum(sum(r.stage_wall) for r in done.results)
    assert done.wall >= busy * 0.99
    assert done.wall <= busy * 1.5 + 0.05
    assert done.errors == 0
    assert 0 < done.occupancy <= 1.01


def test_variants_read_the_same_footers(lake):
    reader = probe.Reader(config(lake))
    task = all_tasks(lake)[3]
    base = probe.check_base(reader, task)
    footer_only = probe.check_base(reader, task, deep=False)
    tail = probe.check_tail(reader, task)
    assert base.files == footer_only.files == tail.files
    assert (
        footer_only.stage_wall[probe.STATS] == footer_only.stage_wall[probe.ROWS] == 0
    )
    assert tail.error is None and tail.stage_wall[probe.FOOTER] > 0
    assert sum(tail.stage_wall) == pytest.approx(tail.wall, rel=1e-6, abs=1e-6)


def test_tail_asks_again_when_the_footer_is_bigger_than_the_first_read(
    lake, monkeypatch
):
    monkeypatch.setattr(probe, "TAIL_BYTES", 16)
    result = probe.check_tail(probe.Reader(config(lake)), all_tasks(lake)[0])
    assert result.error is None
    assert result.files[0][0] > 0


def test_a_missing_file_is_a_task_error_not_a_crash(lake, tmp_path):
    cfg = config(lake)._replace(l2_root=str(tmp_path / "vacio"))
    result = probe.check_base(probe.Reader(cfg), all_tasks(lake)[0])
    assert result.error and "events" in result.error
    assert sum(result.stage_wall) == pytest.approx(result.wall, rel=1e-6, abs=1e-6)


def test_a_wrong_theta_in_the_stats_costs_the_same_and_does_not_fail(tmp_path):
    root = tmp_path / "l3"
    write_partition(root, 100_000, 2020, 1, [4, 4])
    write_partition(tmp_path / "l2", 100_000, 2020, 1, [4, 4], family=False)
    cfg = probe.Config(
        str(root), str(tmp_path / "l2"), ("binance", "spot", "BTCUSDT"), "path"
    )
    result = probe.check_base(probe.Reader(cfg), probe.Task(999_999, 2020, 1))
    assert result.error is not None  # no existe la partición de ese θ
    ok = probe.check_base(probe.Reader(cfg), probe.Task(100_000, 2020, 1))
    assert ok.error is None


def test_full_run_writes_the_csv_and_the_last_line(lake, tmp_path, capsys):
    out = tmp_path / "salida" / "probe.csv"
    code = probe.main(
        [
            "--l3-root", str(lake[0]),
            "--l2-root", str(lake[1]),
            "--sample", "16",
            "--readers", "1,2,4",
            "--processes", "2",
            "--out", str(out),
        ]
    )  # fmt: skip
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    last = LAST_LINE.fullmatch(lines[-1])
    assert last, lines[-1]
    assert last["muestra"] == "16"
    assert last["mejor"] in {f"{v}_{n}" for v in "ab" for n in (1, 2, 4)} | {
        f"base_{n}" for n in (1, 2, 4)
    } | {"c_2"}
    rows = list(csv.DictReader(out.open()))
    assert set(rows[0]) == set(probe.CSV_HEADER)
    sections = {r["seccion"] for r in rows}
    assert sections == {"entorno", "muestra", "archivo", "corrida", "etapa"}
    runs = {(r["variante"], r["lectores"]) for r in rows if r["seccion"] == "corrida"}
    assert runs == {(v, str(n)) for v in ("base", "a", "b") for n in (1, 2, 4)} | {
        ("c", "2")
    }
    stages = {r["etapa"] for r in rows if r["seccion"] == "etapa"}
    assert stages == set(probe.STAGES)
    environment = {r["medida"]: r["valor"] for r in rows if r["seccion"] == "entorno"}
    assert int(environment["cores"]) >= 1 and environment["pyarrow"] == pa.__version__
    for measure in (
        "row_groups_mediana",
        "row_groups_p95",
        "row_groups_max",
        "pie_bytes_max",
    ):
        assert any(r["medida"] == measure and r["etapa"] == "familia" for r in rows)


def test_default_out_goes_to_the_directory_given_with_a_slash(lake, tmp_path):
    code = probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(lake[1]), "--sample", "4", "--readers", "1",
         "--variants", "base", "--out", f"{tmp_path}/res/"]
    )  # fmt: skip
    assert code == 0
    assert (tmp_path / "res" / "l3_frontier_probe.csv").exists()


def test_empty_listing_fails_with_the_last_line(lake, tmp_path, capsys):
    code = probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(tmp_path / "vacio"), "--sample", "4",
         "--readers", "1", "--variants", "base", "--out", str(tmp_path / "x.csv")]
    )  # fmt: skip
    assert code == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_more_than_five_percent_of_errors_fails(lake, tmp_path, capsys):
    # El listado ve los archivos de L2, pero no son Parquet: cada tarea falla al leer el pie.
    broken = tmp_path / "l2"
    for task in all_tasks(lake):
        target = broken / probe.partition_rel(
            ("binance", "spot", "BTCUSDT"), task, "events.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"no es parquet")
    code = probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(broken), "--sample", "8",
         "--readers", "1", "--variants", "base", "--out", str(tmp_path / "x.csv")]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 1
    assert "primer error de base_1" in out
    assert out.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert (tmp_path / "x.csv").exists()


def test_a_broken_b_variant_warns_and_does_not_decide_or_win(
    lake, tmp_path, capsys, monkeypatch
):
    # `b` es la única que usa el cliente de GCS: si falla, la medición de `base` sigue valiendo.
    def broken(self, root, rel, size):
        raise ConnectionError("sin red")

    monkeypatch.setattr(probe.Reader, "tail", broken)
    code = probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(lake[1]), "--sample", "8",
         "--readers", "1,2", "--variants", "base,b", "--out", str(tmp_path / "x.csv")]
    )  # fmt: skip
    out = capsys.readouterr().out
    last = out.strip().splitlines()[-1]
    assert code == 0
    assert "AVISO b_1" in out and "AVISO b_2" in out
    assert "primer error de b_1" in out
    assert "veredicto=fallo" not in last and "mejor_variante=base_" in last


def test_a_broken_base_still_fails_even_with_a_healthy_b(lake, tmp_path, capsys):
    broken = tmp_path / "l2"
    for task in all_tasks(lake):
        target = broken / probe.partition_rel(
            ("binance", "spot", "BTCUSDT"), task, "events.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"no es parquet")
    code = probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(broken), "--sample", "8",
         "--readers", "1", "--variants", "base,a", "--out", str(tmp_path / "x.csv")]
    )  # fmt: skip
    assert code == 1
    assert "de base o c falla" in capsys.readouterr().out


def test_best_variant_skips_runs_over_the_error_limit():
    broken = probe.Result(0.0, (0.0,) * 5, (0.0,) * 5, (), "ConnectionError: x")
    fast_and_broken = probe.Run("b", 64, 1.0, 1.0, [broken] * 600)
    runs = [
        simulated("base", 1, 100.0, 90.0),
        simulated("base", 8, 98.0, 100.0),
        fast_and_broken,
    ]
    assert fast_and_broken.over_limit and not fast_and_broken.decides
    assert probe.summarize(runs, cores=8)["best"].label == "base_8"


def test_usage_errors_exit_with_two(lake, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://p-ops/results/x/")
    roots = ["--l3-root", str(lake[0]), "--l2-root", str(lake[1])]
    for argv in (
        [*roots, "--readers", "0"],
        [*roots, "--readers", "uno"],
        [*roots, "--variants", "a,b"],
        [*roots, "--variants", "base,z"],
        [*roots, "--out", "gs://otro/x.csv"],
    ):
        with pytest.raises(SystemExit) as raised:
            probe.main(argv)
        assert raised.value.code == 2, argv


@pytest.mark.parametrize(
    ("cores", "ratio", "scale", "base_pps", "process_pps", "expected"),
    [
        # CPU saturada: la razón llega al 70 % de los cores.
        (8, 5.6, 1.0, 12.0, 12.0, "cpu"),
        (8, 5.5, 1.0, 12.0, 12.0, "mixto"),
        # Un hilo ocupado y los procesos duplican: el límite está dentro del proceso.
        (8, 1.0, 1.1, 12.0, 24.0, "gil"),
        (8, 0.4, 1.0, 12.0, 30.0, "gil"),
        # Nada ocupado y los procesos no ayudan: espera de red o servidor.
        (8, 0.3, 1.0, 12.0, 13.0, "espera"),
        (8, 0.69, 1.0, 12.0, 23.9, "espera"),
        # Un núcleo ocupado, procesos sin ayuda y sin saturar: ni una cosa ni otra.
        (8, 1.0, 1.0, 12.0, 13.0, "mixto"),
        # La base sí escala: no se reproduce el problema.
        (8, 2.0, 4.0, 12.0, 13.0, "mixto"),
        # Sin la variante c no hay con qué separar GIL de espera.
        (8, 0.3, 1.0, 12.0, None, "mixto"),
        # Con 1 vCPU, un hilo ocupado ya es CPU saturada.
        (1, 0.8, 1.0, 12.0, 12.0, "cpu"),
    ],
)
def test_verdict_follows_the_rule(cores, ratio, scale, base_pps, process_pps, expected):
    assert probe.verdict(cores, ratio, scale, base_pps, process_pps) == expected


def simulated(variant, workers, wall, cpu, tasks=600):
    result = probe.Result(0.0, (0.0,) * 5, (0.0,) * 5, ((10, 100), (10, 100)), None)
    return probe.Run(variant, workers, wall, cpu, [result] * tasks)


def test_summary_and_last_line_with_simulated_times():
    runs = [
        simulated("base", 1, 100.0, 90.0),
        simulated("base", 8, 98.0, 100.0),
        simulated("base", 64, 97.0, 97.0),
        simulated("a", 64, 60.0, 60.0),
        simulated("c", 4, 25.0, 100.0),
    ]
    summary = probe.summarize(runs, cores=8)
    assert summary["verdict"] == "gil"
    assert summary["best"].label == "c_4"
    line = probe.final_line(600, summary, (1, 8, 64))
    assert line == (
        "frontier probe: muestra=600 base_1=100.0s base_8=98.0s base_64=97.0s cpu_ratio_64=1.00 "
        "mejor_variante=c_4 pies_por_s_mejor=48.0 veredicto=gil"
    )
    waiting = [*runs[:4], simulated("c", 4, 95.0, 20.0)]
    waiting[2] = simulated("base", 64, 97.0, 20.0)
    assert probe.summarize(waiting, cores=8)["verdict"] == "espera"
    saturated = [*runs[:2], simulated("base", 64, 97.0, 600.0), *runs[3:]]
    assert probe.summarize(saturated, cores=8)["verdict"] == "cpu"


def test_failure_line_keeps_the_format():
    assert probe.final_line("-", None, (1, 8, 64)) == (
        "frontier probe: muestra=- base_1=- base_8=- base_64=- cpu_ratio_64=- mejor_variante=- "
        "pies_por_s_mejor=- veredicto=fallo"
    )


def test_container_cores_is_at_least_one():
    assert probe.container_cores() >= 1


def test_quantile_nearest_rank():
    assert probe.quantile([1, 2, 3, 4], 50) == 2
    assert probe.quantile([1, 2, 3, 4], 95) == 4
    assert probe.quantile([], 95) != probe.quantile([], 95)  # nan


def test_without_roots_and_without_the_project_it_is_a_usage_error(lake, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        probe.main(["--l3-root", str(lake[0])])
    assert raised.value.code == 2


def test_an_exhausted_budget_skips_the_runs_and_fails_without_a_verdict(
    lake, tmp_path, capsys
):
    code = probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(lake[1]), "--sample", "4", "--readers", "1",
         "--budget-s", "-1", "--out", str(tmp_path / "x.csv")]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 1
    assert "omitida, pasó el presupuesto" in out
    assert out.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_runs_go_base_then_c_then_a_and_b(lake, tmp_path, capsys):
    probe.main(
        ["--l3-root", str(lake[0]), "--l2-root", str(lake[1]), "--sample", "4", "--readers", "1,2",
         "--processes", "2", "--out", str(tmp_path / "x.csv")]
    )  # fmt: skip
    order = re.findall(r"^(\w+_\d+): pared=", capsys.readouterr().out, re.MULTILINE)
    assert order == ["base_1", "base_2", "c_2", "a_1", "a_2", "b_1", "b_2"]
