"""Compara los hashes de `events_summary` entre corridas de L2 sobre el mismo mes.

L2 es determinista: dos corridas del mismo mes con la misma entrada deben dejar,
para cada θ, el mismo número de eventos y los mismos `content_hash` de
`events.parquet` y `carry_over.parquet` (sonda de L2: el resultado no depende de
la CPU). Cada corrida emite un hallazgo `events_summary` por θ y mes; aquí se
agrupan por (θ, mes) y se compara lo que dejó cada `run_id`.

Args (todos opcionales):
  --dq-root gs://<proyecto>-dq-findings/l2   (por defecto, el del proyecto)
  --month AAAA-MM    solo ese mes (por defecto, todos)
  --asset BTCUSDT

Última línea: `θ: 50; unidades: 5450 (con >= 2 corridas: 50; discrepancias: 0)`.
Una unidad es un (θ, mes). Sale con 1 si hay discrepancias, y con 1 si ninguna
unidad tiene dos corridas que comparar (no se comprobó nada).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

FIELDS = (
    "events",
    "has_pending_event",
    "events_content_hash",
    "carry_over_content_hash",
)
COLUMNS = ["check_type", "asset", "run_id", "year", "month", "details"]
MAX_PRINTED = 50


def default_root(suffix: str) -> str:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit("sin --dq-root y sin OPS_RESULTS_URI: no sé de qué proyecto es")
    return f"gs://{match[1]}-{suffix}"


def list_parquets(root: str) -> tuple[pafs.FileSystem, list[str]]:
    """El sistema de archivos y los `.parquet` bajo `root`, de forma recursiva.

    En GCS se listan los objetos por prefijo con `google-cloud-storage` (`l2/`, con barra) y
    cada archivo se abre luego por su ruta completa. Nunca se abre el prefijo como dataset:
    pyarrow pide primero el objeto `l2` sin barra y la condición IAM por prefijo de ops-script
    lo niega con 403 (ITSC-370, ITSC-354).
    """
    if not root.startswith("gs://"):
        found = (str(p) for p in Path(root).resolve().rglob("*.parquet") if p.is_file())
        return pafs.LocalFileSystem(), sorted(found)
    from google.cloud import storage

    bucket, _, prefix = root.removeprefix("gs://").partition("/")
    blobs = storage.Client().list_blobs(bucket, prefix=prefix.rstrip("/") + "/")
    names = (blob.name for blob in blobs if blob.name.endswith(".parquet"))
    return pafs.GcsFileSystem(), sorted(f"{bucket}/{name}" for name in names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dq-root")
    parser.add_argument("--month")
    parser.add_argument("--asset", default="BTCUSDT")
    ns = parser.parse_args(argv)

    root = (ns.dq_root or f"{default_root('dq-findings')}/l2").rstrip("/")
    fs, paths = list_parquets(root)
    if not paths:  # sin hallazgos no hay nada que comparar
        print("θ: 0; unidades: 0 (con >= 2 corridas: 0; discrepancias: 0)")
        return 1
    keep = (pc.field("check_type") == "events_summary") & (
        pc.field("asset") == ns.asset
    )
    if ns.month:
        year, month = (int(part) for part in ns.month.split("-"))
        keep &= (pc.field("year") == year) & (pc.field("month") == month)

    # (θ, año, mes) -> run_id -> conjunto de resúmenes que esa corrida dejó
    units: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    # Un archivo a la vez: la tabla filtrada se suelta antes de abrir el siguiente.
    for path in paths:
        with pq.ParquetFile(path, filesystem=fs) as parquet:
            table = parquet.read(columns=COLUMNS).filter(keep)
        for row in table.to_pylist():
            details = json.loads(row["details"])
            summary = tuple(details[name] for name in FIELDS)
            key = details["theta"], row["year"], row["month"]
            units[key][row["run_id"]].add(summary)
        del table

    compared = bad = 0
    thetas = {theta for theta, _, _ in units}
    for (theta, year, month), runs in sorted(units.items()):
        distinct = {summary for seen in runs.values() for summary in seen}
        # Una corrida con dos resúmenes distintos del mismo mes ya es una discrepancia.
        if len(runs) >= 2:
            compared += 1
        if len(distinct) > 1:
            bad += 1
            if bad <= MAX_PRINTED:
                print(
                    f"DISCREPANCIA θ={theta} {year}-{month:02d}: {len(runs)} corridas, {len(distinct)} resultados distintos"
                )
                for run_id, seen in sorted(runs.items()):
                    for summary in sorted(seen, key=str):
                        print(f"  {run_id}: {dict(zip(FIELDS, summary, strict=True))}")
    if bad > MAX_PRINTED:
        print(f"... y {bad - MAX_PRINTED} discrepancias más")
    print(
        f"θ: {len(thetas)}; unidades: {len(units)} "
        f"(con >= 2 corridas: {compared}; discrepancias: {bad})"
    )
    return 1 if bad or compared == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
