"""Entrypoint del job `ops-script`: descarga un script de GCS y lo ejecuta.

Antes de correrlo imprime su URI, su generation, su SHA-256 y su contenido
completo: sin git de por medio, esa es la trazabilidad de lo que se ejecutó.
El script corre con el mismo intérprete y entorno de la imagen y su código de
salida es el del job: un script que falla deja la ejecución en rojo.

`run-job.yml` lee estas líneas de Cloud Logging para armar el resumen de
Actions (`.github/scripts/ops-summary.sh`): los marcadores de abajo son su
contrato y cambiarlos exige cambiar ese script.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Marcadores del log. Cada uno va solo en su línea.
META = "OPS_SCRIPT_META"  # seguido de un JSON: uri, generation, sha256, results_uri
BEGIN = "----- ops-script: contenido -----"
END = "----- ops-script: fin del contenido -----"
OUTPUT = "----- ops-script: salida -----"
EXIT = "OPS_SCRIPT_EXIT"  # seguido de code=<n>

# Un script ad hoc cabe de sobra en 1 MiB; más es un archivo equivocado.
MAX_SCRIPT_BYTES = 1 << 20


@dataclass(frozen=True)
class Script:
    """Un objeto de GCS tal como se descargó: su URI, su generation y su contenido."""

    uri: str
    generation: int
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class FetchError(Exception):
    """El script no se pudo descargar."""


Fetcher = Callable[[str], Script]


def split_uri(uri: str) -> tuple[str, str]:
    """`(bucket, objeto)` de un `gs://bucket/objeto`; `ValueError` si no lo es."""
    parsed = urlparse(uri)
    name = parsed.path.lstrip("/")
    if parsed.scheme != "gs" or not parsed.netloc or not name or name.endswith("/"):
        raise ValueError(f"se esperaba gs://<bucket>/<objeto>, llegó '{uri}'")
    return parsed.netloc, name


def fetch_gcs(uri: str) -> Script:
    """Descarga el objeto y lo ata a la generation que se reporta.

    Se pide con `if_generation_match`: si alguien sobrescribe el script entre
    leer los metadatos y descargarlo, falla en vez de reportar una generation y
    ejecutar otro contenido.
    """
    from google.api_core import exceptions
    from google.cloud import storage

    bucket, name = split_uri(uri)
    try:
        blob = storage.Client().bucket(bucket).get_blob(name)
        if blob is None:
            raise FetchError(f"no existe {uri}")
        if blob.size is not None and blob.size > MAX_SCRIPT_BYTES:
            raise FetchError(f"{uri} pesa {blob.size} bytes: el máximo es 1 MiB")
        data = blob.download_as_bytes(if_generation_match=blob.generation)
    except exceptions.GoogleAPICallError as err:
        raise FetchError(f"no se pudo leer {uri}: {err}") from err
    return Script(uri=uri, generation=blob.generation, data=data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ops_tools",
        description="Corre un script Python de GCS. Los argumentos del script "
        "van tras `--` o, como un solo texto, en --script-args.",
    )
    parser.add_argument("--script", required=True, help="URI gs://bucket/objeto.py")
    parser.add_argument(
        "--script-args",
        default="",
        help="Argumentos del script como un texto, separados como en un shell. "
        "Es la vía de run-job.yml: gcloud rechaza un --args con valores repetidos.",
    )
    # Los args por defecto del módulo `layer` son `--mode <modo>`: se aceptan y
    # se ignoran para que una ejecución sin overrides falle por falta de
    # --script y no por un argumento desconocido.
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    return parser


def parse(argv: Sequence[str]) -> tuple[str, list[str]]:
    """`(uri del script, argumentos del script)`."""
    argv = list(argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1 :]
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        split_uri(ns.script)
        words = shlex.split(ns.script_args)
    except ValueError as err:
        parser.error(str(err))
    return ns.script, words + rest


def emit(line: str = "") -> None:
    print(line, flush=True)


def results_uri(env: Mapping[str, str]) -> str:
    """Dónde dejar salidas grandes: `OPS_RESULTS_ROOT/<ejecución>/`."""
    root = env.get("OPS_RESULTS_ROOT", "").rstrip("/")
    if not root:
        return ""
    return f"{root}/{env.get('CLOUD_RUN_EXECUTION') or 'local'}/"


def report(script: Script, results: str) -> None:
    """Imprime lo que se va a ejecutar, antes de ejecutarlo."""
    meta = {
        "uri": script.uri,
        "generation": script.generation,
        "sha256": script.sha256,
        "results_uri": results,
    }
    emit(f"{META} {json.dumps(meta, ensure_ascii=False)}")
    emit(BEGIN)
    text = script.data.decode("utf-8", errors="replace")
    emit(text.removesuffix("\n"))
    emit(END)


def run_script(script: Script, args: Sequence[str], env: Mapping[str, str]) -> int:
    """Ejecuta `python script.py <args>` y devuelve su código de salida."""
    with tempfile.TemporaryDirectory(prefix="ops-script-") as workdir:
        path = Path(workdir) / "script.py"
        path.write_bytes(script.data)
        emit(OUTPUT)
        proc = subprocess.run(
            [sys.executable, str(path), *args], cwd=workdir, env=dict(env), check=False
        )
    # Una señal (SIGKILL por OOM, SIGTERM por timeout) llega como código negativo.
    return proc.returncode if proc.returncode >= 0 else 128 - proc.returncode


def main(
    argv: Sequence[str] | None = None,
    *,
    fetch: Fetcher = fetch_gcs,
    env: Mapping[str, str] | None = None,
) -> int:
    """Códigos de salida: 2 por uso inválido, 1 si no se pudo descargar el
    script y, si se ejecutó, el del script."""
    env = dict(os.environ if env is None else env)
    uri, args = parse(sys.argv[1:] if argv is None else argv)
    try:
        script = fetch(uri)
    except FetchError as err:
        print(f"ops_tools: {err}", file=sys.stderr, flush=True)
        return 1
    results = results_uri(env)
    report(script, results)
    code = run_script(script, args, {**env, "OPS_RESULTS_URI": results})
    emit(f"{EXIT} code={code}")
    return code
