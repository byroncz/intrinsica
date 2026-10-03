import hashlib
import json

import pytest
from ops_tools import cli

URI = "gs://proj-ops/scripts/seam.py"


def fake_fetch(source: str, generation: int = 1700000000000001):
    def fetch(uri: str) -> cli.Script:
        assert uri == URI
        return cli.Script(uri=uri, generation=generation, data=source.encode())

    return fetch


def run(capfd, source: str, *argv: str, env: dict | None = None):
    code = cli.main(["--script", URI, *argv], fetch=fake_fetch(source), env=env or {})
    return code, capfd.readouterr().out.splitlines()


def test_imprime_uri_generation_hash_y_contenido_antes_de_ejecutar(capfd):
    source = "print('hola')\n"
    code, lines = run(capfd, source)
    assert code == 0
    meta = json.loads(lines[0].removeprefix(f"{cli.META} "))
    assert meta == {
        "uri": URI,
        "generation": 1700000000000001,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "results_uri": "",
    }
    assert lines[1:] == [
        cli.BEGIN,
        "print('hola')",
        cli.END,
        cli.OUTPUT,
        "hola",
        f"{cli.EXIT} code=0",
    ]


@pytest.mark.parametrize("code", [1, 3, 42])
def test_propaga_el_codigo_de_salida(capfd, code):
    got, lines = run(capfd, f"import sys; sys.exit({code})")
    assert got == code
    assert lines[-1] == f"{cli.EXIT} code={code}"


def test_excepcion_sin_capturar_es_codigo_1(capfd):
    got, _ = run(capfd, "raise RuntimeError('boom')")
    assert got == 1


def test_muerte_por_senal_es_128_mas_la_senal(capfd):
    got, lines = run(capfd, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    assert got == 128 + 9
    assert lines[-1] == f"{cli.EXIT} code=137"


def test_args_tras_doble_guion_y_en_script_args(capfd):
    source = "import sys; print(sys.argv[1:])"
    _, lines = run(
        capfd, source, "--script-args", "--a 'dos palabras' 5", "--", "--b", "5"
    )
    assert (
        lines[lines.index(cli.OUTPUT) + 1] == "['--a', 'dos palabras', '5', '--b', '5']"
    )


def test_expone_la_carpeta_de_resultados_de_la_ejecucion(capfd):
    env = {
        "OPS_RESULTS_ROOT": "gs://proj-ops/results/",
        "CLOUD_RUN_EXECUTION": "ops-script-abc12",
    }
    source = "import os; print(os.environ['OPS_RESULTS_URI'])"
    _, lines = run(capfd, source, env=env)
    expected = "gs://proj-ops/results/ops-script-abc12/"
    assert lines[lines.index(cli.OUTPUT) + 1] == expected
    assert json.loads(lines[0].removeprefix(f"{cli.META} "))["results_uri"] == expected


def test_el_script_no_hereda_nada_que_no_este_en_el_entorno(capfd):
    _, lines = run(capfd, "import os; print(os.environ.get('SECRETO'))", env={})
    assert lines[lines.index(cli.OUTPUT) + 1] == "None"


def test_script_inexistente_es_codigo_1(capfd):
    def fetch(uri):
        raise cli.FetchError(f"no existe {uri}")

    assert cli.main(["--script", URI], fetch=fetch, env={}) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert f"no existe {URI}" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--script", "seam.py"],
        ["--script", "gs://proj-ops/"],
        ["--script", "gs://proj-ops/scripts/"],
        ["--script", URI, "--script-args", "'sin cerrar"],
        ["--mode", "script"],
    ],
)
def test_uso_invalido_es_codigo_2(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv, fetch=fake_fetch(""), env={})
    assert exc.value.code == 2


def test_acepta_el_mode_de_los_args_por_defecto_del_modulo(capfd):
    code, _ = run(capfd, "", "--mode", "script")
    assert code == 0


def test_split_uri():
    assert cli.split_uri(URI) == ("proj-ops", "scripts/seam.py")


class FakeBlob:
    generation = 1700000000000007
    size = 5

    def __init__(self):
        self.asked = None

    def download_as_bytes(self, if_generation_match):
        self.asked = if_generation_match
        return b"x = 1"


def fake_client(blob):
    class Client:
        def bucket(self, name):
            assert name == "proj-ops"
            return type("Bucket", (), {"get_blob": lambda _, obj: blob})()

    return Client


def test_fetch_gcs_ata_la_descarga_a_la_generation_reportada(monkeypatch):
    from google.cloud import storage

    blob = FakeBlob()
    monkeypatch.setattr(storage, "Client", fake_client(blob))
    script = cli.fetch_gcs(URI)
    assert (script.generation, script.data) == (1700000000000007, b"x = 1")
    assert blob.asked == 1700000000000007


def test_fetch_gcs_objeto_inexistente(monkeypatch):
    from google.cloud import storage

    monkeypatch.setattr(storage, "Client", fake_client(None))
    with pytest.raises(cli.FetchError, match="no existe"):
        cli.fetch_gcs(URI)


def test_fetch_gcs_traduce_el_403(monkeypatch):
    from google.api_core import exceptions
    from google.cloud import storage

    class Blob(FakeBlob):
        def download_as_bytes(self, if_generation_match):
            raise exceptions.Forbidden("sin permiso")

    monkeypatch.setattr(storage, "Client", fake_client(Blob()))
    with pytest.raises(cli.FetchError, match="403"):
        cli.fetch_gcs(URI)
