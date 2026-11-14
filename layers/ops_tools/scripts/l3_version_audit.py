"""Auditoría de versiones de L3: ¿todos los archivos de cada familia tienen la misma versión? (ITSC-419).

Para cada familia (`summaries`, `path`, `fit`, `spectrum`), cada θ del catálogo y cada mes con
`events.parquet` en L2, lee **solo el pie** del archivo de la familia (los metadatos clave-valor de
Parquet: ningún dato) y anota su `state_version` (y `budget_ticks` en `spectrum`). Sirve para
comprobar, tras un backfill, que no quedaron meses con la versión vieja: la frontera de una
familia ya no los salta en silencio (TRD-L3 §8.2), pero esta sonda lo mide sin tocar nada.

Args:
  --catalog gs://<proyecto>-manifest/l2/thetas.yaml   (el catálogo único de θ; también local)
  --l2-root gs://<proyecto>-dc-events/l2              (también una ruta local, disposición hive)
  --l3-root gs://<proyecto>-dc-descriptors/l3         (ídem)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_version_audit.csv` bajo OPS_RESULTS_URI
                                       o, en local, en el directorio actual; una ruta que termina
                                       en `/` es un directorio)
  Sin argumentos, con OPS_RESULTS_URI, deduce el proyecto y usa los tres de arriba.

No importa ninguna capa: una importación cambiaría `uv.lock`. Lee cada pie por su ruta completa
(un listado por raíz y un `read_metadata` por archivo), nunca un dataset sobre un prefijo, que
pide un objeto sin barra y la condición IAM por prefijo de `ops-script` lo niega con 403
(runbook de operación de L3, «La trampa del 403 por prefijo»).

Memoria: solo pies y los nombres de los archivos listados (cientos de miles de bytes), ningún dato.

CSV: una fila `version` por familia y versión (con cuántos archivos la tienen) y una fila `excepcion`
por cada (familia, θ, mes) que falte, no se pueda leer o no tenga la versión mayoritaria de su
familia. La versión de `spectrum` incluye `budget_ticks`: cambiar el presupuesto es versión nueva.
Un archivo que no se lee, o sin `state_version`, cuenta como la versión `ilegible`.

Salida: la tabla por familia y versión, la ruta del CSV y, como última línea, `version audit:
familias=4 archivos=… faltantes=… versiones_mezcladas=… veredicto=ok|mezcla|faltantes`. `archivos`
son los que existen y se leyeron; `faltantes`, los (familia, θ, mes) con `events.parquet` en L2 y sin
archivo de la familia; `versiones_mezcladas`, cuántas familias tienen más de una versión.
El veredicto es `faltantes` si falta algún archivo, si no `mezcla` si alguna familia mezcla
versiones, si no `ok`.

Códigos: 0 solo con `ok` (cada familia con una única versión y ningún archivo faltante); 1 con
`mezcla` o `faltantes`, o si no se puede listar o leer el catálogo; 2 por uso. Solo lee; lo único
que escribe es el CSV de `--out`.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pyarrow.fs as pafs
import pyarrow.parquet as pq

FAMILIES = ("summaries", "path", "fit", "spectrum")
READ_WORKERS = 16
CSV_HEADER = [
    "fila",
    "familia",
    "theta",
    "mes",
    "state_version",
    "budget_ticks",
    "archivos",
    "motivo",
]
UNREADABLE = "ilegible"
FILE = re.compile(r"/theta=(0\.\d{8})/year=(\d{4})/month=(\d{2})/([^/]+)$")


class InputError(Exception):
    """No se pudo listar o leer una raíz o el catálogo: sale con código 1."""


def log(message: str) -> None:
    print(message, flush=True)


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        return pafs.FileSystem.from_uri(uri)
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def series_root(root: str, key: tuple[str, str, str]) -> str:
    provider, market, asset = key
    return f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"


def gcs_names(bucket: str, prefix: str) -> list[str]:
    """Los nombres de los objetos de `gs://bucket/prefix`, listados por la API de GCS."""
    from google.cloud import storage

    return [b.name for b in storage.Client().list_blobs(bucket, prefix=prefix)]


def list_files(root: str) -> list[str]:
    """Todos los archivos bajo `root`, de forma recursiva: objetos por prefijo en GCS."""
    if not root.startswith("gs://"):
        return [str(p) for p in Path(root).rglob("*") if p.is_file()]
    bucket, _, prefix = root.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    try:
        return [f"gs://{bucket}/{name}" for name in gcs_names(bucket, prefix)]
    except Exception as exc:
        raise InputError(
            f"no se pudo listar {root}: {type(exc).__name__}: {exc}"
        ) from exc


def listing(files: list[str], name: str) -> dict[tuple[str, str], str]:
    """`{(θ, YYYY-MM): ruta}` de los archivos `name`, entre los `files` listados."""
    found = {}
    for path in files:
        match = FILE.search(path)
        if match and match[4] == name:
            found[match[1], f"{match[2]}-{match[3]}"] = path
    return found


def read_catalog(source: str) -> list[str]:
    """Los θ del catálogo de L2 (`scale: 100000000` y `- <round(θ × 10⁸)>`) como texto."""
    try:
        fs, resolved = open_fs(source)
        with fs.open_input_stream(resolved) as stream:
            text = stream.read().decode()
    except OSError as exc:
        raise InputError(f"no se pudo leer el catálogo {source}: {exc}") from exc
    if not re.search(r"^scale:\s*100000000\s*(?:#.*)?$", text, re.MULTILINE):
        raise InputError(f"{source}: `scale` debe ser 100000000")
    integers = [
        int(m[1]) for m in re.finditer(r"^\s*-\s*(\d+)\s*(?:#.*)?$", text, re.MULTILINE)
    ]
    if not integers:
        raise InputError(f"{source}: sin θ")
    return [f"{Decimal(i).scaleb(-8):.8f}" for i in sorted(integers)]


def footer(path: str, family: str) -> tuple[str, str]:
    """`(state_version, budget_ticks)` del pie del archivo; `ilegible` si no se puede leer.

    `budget_ticks` es de `spectrum`: en las demás familias va vacío. Lee solo el pie.
    """
    try:
        fs, resolved = open_fs(path)
        metadata = pq.read_metadata(resolved, filesystem=fs).metadata or {}
    except Exception:  # noqa: BLE001 - un archivo que no se lee es un hallazgo, no un fallo
        return UNREADABLE, ""
    version = metadata.get(b"state_version")
    if version is None:
        return UNREADABLE, ""
    budget = metadata.get(b"budget_ticks") if family == "spectrum" else b""
    return version.decode(), (budget or b"").decode()


def label(version: tuple[str, str]) -> str:
    """La versión como texto: `1.2.0` o, con presupuesto, `1.2.0 (budget_ticks=130000000)`."""
    state, budget = version
    return f"{state} (budget_ticks={budget})" if budget else state


def audit(
    catalog: list[str],
    l2_files: list[str],
    l3_files: list[str],
) -> tuple[dict, list[tuple[str, str, str]], int]:
    """Lee los pies y devuelve `(versiones, faltantes, archivos)`.

    `versiones` es `{familia: {(θ, mes): versión}}` de los archivos que existen y `faltantes`, los
    `(familia, θ, mes)` con `events.parquet` de L2 y sin archivo de la familia.
    """
    thetas = set(catalog)
    wanted = {
        key: path
        for key, path in listing(l2_files, "events.parquet").items()
        if key[0] in thetas
    }
    versions: dict[str, dict[tuple[str, str], tuple[str, str]]] = {}
    missing = []
    tasks = []
    for family in FAMILIES:
        present = listing(l3_files, f"{family}.parquet")
        versions[family] = {}
        for key in sorted(wanted, key=lambda k: (k[0], k[1])):
            if key in present:
                tasks.append((family, key, present[key]))
            else:
                missing.append((family, *key))
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        read = pool.map(lambda t: footer(t[2], t[0]), tasks)
        for (family, key, _), version in zip(tasks, read, strict=True):
            versions[family][key] = version
    return versions, missing, len(tasks)


def csv_rows(versions: dict, missing: list) -> tuple[list[dict], dict[str, Counter]]:
    """Las filas del CSV y, por familia, cuántos archivos tiene cada versión."""
    counts = {family: Counter(versions[family].values()) for family in FAMILIES}
    rows = []
    for family in FAMILIES:
        for version, files in counts[family].most_common():
            rows.append(
                {
                    "fila": "version",
                    "familia": family,
                    "state_version": version[0],
                    "budget_ticks": version[1],
                    "archivos": files,
                }
            )
    for family in FAMILIES:
        majority = counts[family].most_common(1)[0][0] if counts[family] else None
        for (theta, month), version in sorted(versions[family].items()):
            if version != majority:
                rows.append(
                    {
                        "fila": "excepcion",
                        "familia": family,
                        "theta": theta,
                        "mes": month,
                        "state_version": version[0],
                        "budget_ticks": version[1],
                        "motivo": f"no es la versión mayoritaria ({label(majority)})",
                    }
                )
    rows += [
        {
            "fila": "excepcion",
            "familia": family,
            "theta": theta,
            "mes": month,
            "motivo": "falta el archivo",
        }
        for family, theta, month in sorted(missing)
    ]
    return rows, counts


def final_line(files: int, missing: int, mixed: int) -> str:
    verdict = "faltantes" if missing else "mezcla" if mixed else "ok"
    return (
        f"version audit: familias={len(FAMILIES)} archivos={files} faltantes={missing} "
        f"versiones_mezcladas={mixed} veredicto={verdict}"
    )


def write_csv(rows: list[dict], out: str) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if out.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(out)
        with fs.open_output_stream(path) as stream:
            stream.write(data)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    key = (ns.provider, ns.market, ns.asset)
    name = "l3_version_audit.csv"
    name_project = project()
    if not (ns.catalog and ns.l2_root and ns.l3_root) and name_project is None:
        parser.error(
            "sin --catalog, --l2-root y --l3-root y sin OPS_RESULTS_URI: no sé de qué "
            "proyecto es"
        )
    catalog_uri = ns.catalog or f"gs://{name_project}-manifest/l2/thetas.yaml"
    l2_root = ns.l2_root or f"gs://{name_project}-dc-events/l2"
    l3_root = ns.l3_root or f"gs://{name_project}-dc-descriptors/l3"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))

    started = time.perf_counter()
    log(f"version audit: catálogo={catalog_uri} l2={l2_root} l3={l3_root}")
    try:
        catalog = read_catalog(catalog_uri)
        l2_files = list_files(series_root(l2_root, key))
        l3_files = list_files(series_root(l3_root, key))
        versions, missing, files = audit(catalog, l2_files, l3_files)
    except InputError as exc:
        log(f"FALLO {exc}")
        return 1
    rows, counts = csv_rows(versions, missing)

    log(f"{len(catalog)} θ en el catálogo")
    log(f"{'familia':<10} {'archivos':>9}  versión")
    for family in FAMILIES:
        for version, n in counts[family].most_common():
            log(f"{family:<10} {n:>9}  {label(version)}")
        if not counts[family]:
            log(f"{family:<10} {0:>9}  (sin archivos)")
    mixed = sum(len(counts[family]) > 1 for family in FAMILIES)
    exceptions = [r for r in rows if r["fila"] == "excepcion"]
    log(f"{len(exceptions)} excepciones; faltan {len(missing)} archivos")
    try:
        write_csv(rows, out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    log(f"pared_s={time.perf_counter() - started:.1f}")
    line = final_line(files, len(missing), mixed)
    log(line)
    return 0 if line.endswith("veredicto=ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog")
    parser.add_argument("--l2-root")
    parser.add_argument("--l3-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
