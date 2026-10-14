import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / ".github/scripts/run-job-tasks.sh"
URI = "gs://proj-ops/scripts/seam.py"


def run(*entrada: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [SCRIPT, *entrada], capture_output=True, text=True, check=False
    )


def units(*entrada: str) -> list[str]:
    out = run(*entrada)
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def test_ops_script_es_una_sola_unidad():
    assert units("ops-script", "", "", URI) == ["1"]


def test_ops_script_args_no_cambian_las_unidades():
    # Los args libres no llegan a este script: no hay nada que contar.
    assert units("ops-script", "", "", "gs://proj-ops/scripts/sub/dir/otro.py") == ["1"]


@pytest.mark.parametrize(
    "script",
    [
        "",
        "seam.py",
        "gs://proj-ops",
        "gs://proj-ops/",
        "gs://proj-ops/scripts/",
        "gs://proj-ops/scripts/dos palabras.py",
        "https://proj-ops/scripts/seam.py",
        "gs:///scripts/seam.py",
    ],
)
def test_ops_script_exige_una_uri_valida(script):
    out = run("ops-script", "", "", script)
    assert out.returncode == 2
    assert "script" in out.stderr


def test_ops_script_sin_cuarto_argumento_exige_script():
    out = run("ops-script", "")
    assert out.returncode == 2
    assert "exige script" in out.stderr


@pytest.mark.parametrize(("desde", "hasta"), [("2023-03", ""), ("", "2023-03")])
def test_ops_script_no_admite_rango(desde, hasta):
    out = run("ops-script", desde, hasta, URI)
    assert out.returncode == 2
    assert "from ni to" in out.stderr


def test_los_demas_jobs_ignoran_script():
    assert units("l2-monthly", "2023-03", "", URI) == ["1"]
    assert units("l1-backfill", "2023-03", "2023-05", URI) == [
        "2023-03",
        "2023-04",
        "2023-05",
    ]


def test_l1_daily_y_l2_backfill_siguen_igual():
    assert units("l1-daily", "2025-02-27", "2025-03-01") == [
        "2025-02-27",
        "2025-02-28",
        "2025-03-01",
    ]
    assert units("l2-backfill", "", "") == ["1"]


def test_job_invalido_nombra_ops_script():
    out = run("l9-x", "2023-03")
    assert out.returncode == 2
    assert "ops-script" in out.stderr


@pytest.mark.parametrize(
    ("desde", "hasta"),
    [("", ""), ("2023-03", ""), ("2023-03", "2023-03"), ("2023-03", "2024-01")],
)
def test_viz_tiles_es_una_sola_unidad_para_todo_el_rango(desde, hasta):
    # Una tarea recorre los meses en orden; vacíos, la CLI toma el mes anterior.
    assert units("viz-tiles", desde, hasta) == ["1"]


def test_viz_tiles_sin_argumentos_de_rango():
    assert units("viz-tiles", "") == ["1"]


@pytest.mark.parametrize(
    ("desde", "hasta", "mensaje"),
    [
        ("2023-13", "", "from inválido"),
        ("2023-03-01", "", "from inválido"),
        ("2023-03", "2023-3", "to inválido"),
        ("2023-05", "2023-03", "menor que from"),
        ("", "2023-03", "to exige from"),
    ],
)
def test_viz_tiles_rechaza_rangos_invalidos(desde, hasta, mensaje):
    out = run("viz-tiles", desde, hasta)
    assert out.returncode == 2
    assert mensaje in out.stderr


def test_viz_tiles_rechaza_script():
    out = run("viz-tiles", "", "", URI)
    assert out.returncode == 2
    assert "no usa script" in out.stderr


@pytest.mark.parametrize(
    ("desde", "hasta"),
    [("", ""), ("2023-03", ""), ("2023-03", "2023-03"), ("2023-03", "2024-01")],
)
def test_viz_render_es_una_sola_unidad_para_todo_el_rango(desde, hasta):
    # Como viz-tiles: una tarea recorre los meses; vacíos, el mes anterior.
    assert units("viz-render", desde, hasta) == ["1"]


@pytest.mark.parametrize(
    ("desde", "hasta", "mensaje"),
    [
        ("2023-13", "", "from inválido"),
        ("2023-03-01", "", "from inválido"),
        ("2023-03", "2023-3", "to inválido"),
        ("2023-05", "2023-03", "menor que from"),
        ("", "2023-03", "to exige from"),
    ],
)
def test_viz_render_rechaza_rangos_invalidos(desde, hasta, mensaje):
    out = run("viz-render", desde, hasta)
    assert out.returncode == 2
    assert mensaje in out.stderr


def test_viz_render_rechaza_script():
    out = run("viz-render", "", "", URI)
    assert out.returncode == 2
    assert "viz-render no usa script" in out.stderr


@pytest.mark.parametrize(
    ("desde", "hasta"),
    [
        ("", ""),
        ("2023-03", ""),
        ("", "2024-01"),
        ("2023-03", "2023-03"),
        ("2023-03", "2024-01"),
    ],
)
def test_l3_backfill_es_una_sola_unidad_para_todo_el_rango(desde, hasta):
    # Igual que l2-backfill: from vacío lo decide la frontera de cada θ.
    assert units("l3-backfill", desde, hasta) == ["1"]


@pytest.mark.parametrize(
    ("desde", "hasta", "mensaje"),
    [
        ("2023-13", "", "from inválido"),
        ("2023-03", "2023-1", "to inválido"),
        ("2023-05", "2023-03", "menor que from"),
        ("", "2023-1", "to inválido"),
    ],
)
def test_l3_backfill_rechaza_rangos_invalidos(desde, hasta, mensaje):
    out = run("l3-backfill", desde, hasta)
    assert out.returncode == 2
    assert mensaje in out.stderr


@pytest.mark.parametrize(
    ("desde", "hasta"), [("", ""), ("2023-03", ""), ("2023-03", "2023-03")]
)
def test_l3_monthly_es_una_sola_unidad(desde, hasta):
    # Vacío, la CLI toma el mes anterior.
    assert units("l3-monthly", desde, hasta) == ["1"]


@pytest.mark.parametrize(
    ("desde", "hasta", "mensaje"),
    [
        ("2023-03", "2023-04", "un solo mes"),
        ("", "2023-04", "to exige from"),
        ("2023-3", "", "from inválido"),
    ],
)
def test_l3_monthly_rechaza_to_distinto_de_from_y_from_invalido(desde, hasta, mensaje):
    out = run("l3-monthly", desde, hasta)
    assert out.returncode == 2
    assert mensaje in out.stderr


def test_job_invalido_nombra_los_jobs_l3():
    assert "l3-backfill, l3-monthly" in run("l9-x", "2023-03").stderr
