import json
import subprocess
import sys
from pathlib import Path

from ops_tools import cli

SCRIPT = Path(__file__).parents[1] / ".github/scripts/ops-summary.sh"
URI = "gs://proj-ops/scripts/seam.py"


def summary(lines: list[str], **env) -> str:
    out = subprocess.run(
        [SCRIPT],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", **env},
    )
    return out.stdout


def job_log(source: str, *args: str, capfd) -> list[str]:
    """Las líneas que deja el entrypoint de verdad, como las lee `gcloud logging read`."""
    fetched = cli.Script(uri=URI, generation=1700000000000001, data=source.encode())
    env = {"OPS_RESULTS_ROOT": "gs://proj-ops/results", "CLOUD_RUN_EXECUTION": "e1"}
    cli.main(["--script", URI, *args], fetch=lambda _: fetched, env=env)
    # Un log de texto sale con un tabulador final.
    return [f"{line}\t" for line in capfd.readouterr().out.splitlines()]


def test_muestra_identidad_contenido_y_ultima_linea(capfd):
    source = "print('a')\nprint('θ: 50; bordes: 5400 (OK)')\n"
    out = summary(job_log(source, capfd=capfd))
    assert "| URI | `gs://proj-ops/scripts/seam.py` |" in out
    assert "| Generation | `1700000000000001` |" in out
    assert "| SHA-256 | `" in out
    assert "| Resultados | `gs://proj-ops/results/e1/` |" in out
    assert "| Código de salida | 0 |" in out
    assert "Contenido del script (2 líneas)" in out
    assert "print('a')\nprint('θ: 50; bordes: 5400 (OK)')" in out
    assert out.rstrip().endswith("θ: 50; bordes: 5400 (OK)\n```")


def test_script_que_falla_muestra_el_codigo_y_el_traceback(capfd):
    log = job_log("import sys\nprint('antes')\nsys.exit(1)", capfd=capfd)
    out = summary(log)
    assert "| Código de salida | 1 |" in out
    assert "antes" in out


def test_solo_las_ultimas_lineas_de_la_salida(capfd):
    log = job_log("for i in range(100): print(f'fila {i}')", capfd=capfd)
    out = summary(log, MAX_OUTPUT_LINES="3")
    assert "Últimas 3 de 100 líneas" in out
    assert "fila 97\nfila 98\nfila 99" in out
    assert "fila 96" not in out


def test_contenido_con_comillas_invertidas_no_rompe_el_bloque(capfd):
    source = 'print("""\n```\n""")\n'
    out = summary(job_log(source, capfd=capfd))
    assert "````python" in out


def test_contenido_truncado(capfd):
    log = job_log("x = 1\n" * 10, capfd=capfd)
    out = summary(log, MAX_CONTENT_LINES="4")
    assert "Se muestran 4 de 10 líneas." in out


def test_sin_codigo_de_salida_el_job_murio(capfd):
    log = [x for x in job_log("print(1)", capfd=capfd) if "OPS_SCRIPT_EXIT" not in x]
    assert "sin código (el job murió" in summary(log)


def test_sin_encabezado_muestra_el_log():
    out = summary(
        ["ops_tools: no se pudo leer gs://proj-ops/scripts/x.py: 403 Forbidden\t"]
    )
    assert "No se encontró el encabezado" in out
    assert "403 Forbidden" in out


def test_salida_sin_lineas(capfd):
    assert "sin salida" in summary(job_log("pass", capfd=capfd))


def test_la_meta_es_json_valido(capfd):
    line = next(x for x in job_log("pass", capfd=capfd) if x.startswith(cli.META))
    json.loads(line.removeprefix(f"{cli.META} ").rstrip("\t"))
    assert sys.executable
