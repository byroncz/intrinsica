"""Prueba de permisos de la cuenta ops-script (la prueba negativa del runbook).

Comprueba que la cuenta hace lo que debe y falla con 403 en lo demás:

  - leer el lago: listar un objeto de landing y de dc-events  -> debe poder
  - escribir bajo OPS_RESULTS_URI                              -> debe poder
  - escribir en scripts/ del bucket ops                        -> 403
  - escribir en landing, dc-events, dq-findings y manifest     -> 403
  - pisar un objeto ya escrito en results/ (sin borrar)        -> 403

Una escritura que "debe fallar" y no falla deja un objeto `ops-permisos-*` en el
bucket: la corrida lo avisa para que el humano lo borre.

Última línea: `permisos: N de N como se esperaba`. Sale con 1 si alguno no.
"""

import os
import re
import sys
import uuid

from google.api_core import exceptions
from google.cloud import storage


def split(uri: str) -> tuple[str, str]:
    match = re.match(r"gs://([^/]+)/(.*)", uri)
    return match[1], match[2]


def attempt(label: str, expected: str, action) -> bool:
    """Ejecuta `action`; `expected` es `ok` o `403`. Imprime y devuelve si coincidió."""
    try:
        action()
        got = "ok"
    except exceptions.Forbidden:
        got = "403"
    except exceptions.GoogleAPICallError as err:
        got = f"error {err.code}"
    verdict = "OK" if got == expected else "FALLO"
    print(f"{verdict} {label}: esperado {expected}, obtenido {got}")
    return got == expected


def main() -> int:
    results = os.environ.get("OPS_RESULTS_URI", "")
    match = re.match(r"gs://(.+)-ops/", results)
    if match is None:
        print("sin OPS_RESULTS_URI: corre esto dentro del job ops-script")
        return 1
    project = match[1]
    client = storage.Client()
    name = f"ops-permisos-{uuid.uuid4().hex[:8]}"

    def list_one(bucket: str, prefix: str):
        return lambda: next(
            iter(client.list_blobs(bucket, prefix=prefix, max_results=1)), None
        )

    def write(bucket: str, path: str):
        return lambda: client.bucket(bucket).blob(path).upload_from_string(b"prueba")

    results_bucket, results_prefix = split(results)
    checks = [
        ("leer landing", "ok", list_one(f"{project}-landing", "l1/")),
        ("leer dc-events", "ok", list_one(f"{project}-dc-events", "l2/")),
        (
            "escribir en results/",
            "ok",
            write(results_bucket, f"{results_prefix}{name}"),
        ),
        ("pisar en results/", "403", write(results_bucket, f"{results_prefix}{name}")),
        ("escribir en scripts/", "403", write(results_bucket, f"scripts/{name}")),
        ("escribir en landing", "403", write(f"{project}-landing", f"l1/{name}")),
        ("escribir en dc-events", "403", write(f"{project}-dc-events", f"l2/{name}")),
        (
            "escribir en dq-findings",
            "403",
            write(f"{project}-dq-findings", f"l2/{name}"),
        ),
        ("escribir en manifest", "403", write(f"{project}-manifest", f"l2/{name}")),
    ]
    passed = sum(attempt(*check) for check in checks)
    print(f"permisos: {passed} de {len(checks)} como se esperaba")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
