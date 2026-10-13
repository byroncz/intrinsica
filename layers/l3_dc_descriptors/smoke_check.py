"""Comprobaciones del humo de la imagen (`smoke.sh`), con el Python y el pyarrow de la imagen.

Se monta en el contenedor y lee `/data`. Uso:
  python smoke_check.py rows <archivos esperados>
  python smoke_check.py hashes <meses esperados dos veces>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA = Path("/data")


def fail(message: str) -> None:
    print(f"::error::{message}")
    sys.exit(1)


def row_groups(path: Path) -> list[int]:
    meta = pq.ParquetFile(path).metadata
    return [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]


def rows(expected: int) -> None:
    """Una fila por evento: cada summaries.parquet repite filas y row groups de L2."""
    files = sorted((DATA / "l3").rglob("summaries.parquet"))
    if len(files) != expected:
        fail(f"se esperaban {expected} summaries.parquet y hay {len(files)}")
    events = 0
    for path in files:
        twin = DATA / "l2" / path.relative_to(DATA / "l3").with_name("events.parquet")
        if row_groups(path) != row_groups(twin):
            fail(f"{path}: filas por row group distintas de {twin}")
        events += sum(row_groups(twin))
    if not events:
        fail("el lago de prueba no tiene eventos")
    print(f"filas OK: {events} eventos en {len(files)} archivos")


def hashes(twice: int) -> None:
    """Tras `--force`, los meses reprocesados dan el mismo content_hash que la primera vez."""
    found = ds.dataset(DATA / "dq", partitioning="hive").to_table().to_pylist()
    seen = defaultdict(list)
    for row in found:
        if row["check_type"] == "summaries_summary":
            details = json.loads(row["details"])
            seen[row["year"], row["month"], details["theta"]].append(
                details["content_hash"]
            )
    repeated = {key: values for key, values in seen.items() if len(values) > 1}
    months = {(year, month) for year, month, _ in repeated}
    if len(months) != twice:
        fail(f"se esperaban {twice} meses reprocesados y hay {len(months)}")
    different = {key for key, values in repeated.items() if len(set(values)) != 1}
    if different:
        fail(f"el content_hash cambió al reprocesar: {sorted(different)}")
    if any(len(values) > 2 for values in seen.values()):
        fail("algún (θ, mes) se escribió más de dos veces")
    print(f"content_hash OK: {len(repeated)} (θ, mes) iguales tras --force")


if __name__ == "__main__":
    command, number = sys.argv[1], int(sys.argv[2])
    {"rows": rows, "hashes": hashes}[command](number)
