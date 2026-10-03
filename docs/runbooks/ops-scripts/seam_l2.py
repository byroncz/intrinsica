"""Seam-check de L2: la costura entre cada par de meses consecutivos de cada θ.

L2 encadena los meses de un θ por carry-over (TRD-L2 §7.4, ADR-L2-06): un evento
se escribe en el mes que confirma el evento siguiente, y el que queda pendiente
al cierre viaja en el `carry_over.parquet`. Para cada θ y cada borde (M, M+1) se
comprueba:

  1. Existen el carry-over de M y el de M+1 (`falta_carry_over`).
  2. Si el carry-over de M trae un evento pendiente:
     a. su referencia coincide con el extremo del último evento de M
        (`cadena_rota`): el evento pendiente sigue a ese;
     b. el primer evento de M+1 es ese evento pendiente, referencia y
        confirmación idénticas (`pendiente_no_coincide`); y si M+1 no tiene
        eventos, el carry-over de M+1 lo conserva igual (`pendiente_perdido`).

Lee de cada `events.parquet` solo la primera y la última fila, y el carry-over
entero (una fila): son archivos chicos, así que los ~5.400 bordes salen en
minutos desde la región del bucket.

Args (todos opcionales):
  --events-root gs://<proyecto>-dc-events/l2   (por defecto, el del proyecto)
  --provider binance --market spot --asset BTCUSDT

Última línea: `θ: 50; bordes: 5400 (ok: 5400; fallos: 0)`. Sale con 1 si hay
algún fallo.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pyarrow.fs as pafs
import pyarrow.parquet as pq

PATH = re.compile(
    r"/theta=(0\.\d+)/year=(\d{4})/month=(\d{2})/(events|carry_over)\.parquet$"
)
POINTS = ("reference", "confirm", "extreme")
EVENT_COLUMNS = [f"{p}_{c}" for p in POINTS for c in ("price", "time", "agg_trade_id")]
CARRY_COLUMNS = [
    "has_pending_event",
    *[
        f"pending_{p}_{c}"
        for p in ("reference", "confirm")
        for c in ("price", "time", "agg_trade_id")
    ],
]
MAX_PRINTED = 50


def default_root(suffix: str) -> str:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit("sin --events-root y sin OPS_RESULTS_URI: no sé de qué proyecto es")
    return f"gs://{match[1]}-{suffix}"


def point(row: dict, prefix: str) -> tuple:
    """(precio, tiempo, agg_trade_id) de un punto, con el prefijo de sus columnas."""
    return tuple(row[f"{prefix}_{c}"] for c in ("price", "time", "agg_trade_id"))


def edge_events(fs: pafs.FileSystem, path: str) -> tuple[dict | None, dict | None]:
    """Primer y último evento de `events.parquet`, leyendo un row group por punta."""
    with fs.open_input_file(path) as handle:
        parquet = pq.ParquetFile(handle)
        groups = range(parquet.metadata.num_row_groups)

        def row(group: int, last: bool) -> dict | None:
            table = parquet.read_row_group(group, columns=EVENT_COLUMNS)
            if table.num_rows == 0:
                return None
            return table.slice(table.num_rows - 1 if last else 0, 1).to_pylist()[0]

        first = next((r for g in groups if (r := row(g, False))), None)
        last = next((r for g in reversed(groups) if (r := row(g, True))), None)
    return first, last


def read_carry(fs: pafs.FileSystem, path: str) -> dict:
    with fs.open_input_file(path) as handle:
        return pq.read_table(handle, columns=CARRY_COLUMNS).to_pylist()[0]


def load_month(fs: pafs.FileSystem, paths: dict[str, str]) -> dict:
    """Lo que la costura necesita de un (θ, mes): primer y último evento, carry-over."""
    first = last = None
    if "events" in paths:
        first, last = edge_events(fs, paths["events"])
    carry = read_carry(fs, paths["carry_over"]) if "carry_over" in paths else None
    return {"first": first, "last": last, "carry": carry}


def pending(carry: dict) -> tuple | None:
    if not carry["has_pending_event"]:
        return None
    return point(carry, "pending_reference"), point(carry, "pending_confirm")


def check_border(a: dict | None, b: dict | None) -> str | None:
    """El motivo del fallo del borde (a = mes M, b = mes M+1), o `None` si pasa."""
    if a is None or b is None or a["carry"] is None or b["carry"] is None:
        return "falta_carry_over"
    held = pending(a["carry"])
    if held is None:
        return None
    reference = held[0]
    if a["last"] is not None and point(a["last"], "extreme") != reference:
        return "cadena_rota"
    if b["first"] is not None:
        starts = (point(b["first"], "reference"), point(b["first"], "confirm"))
        return None if starts == held else "pendiente_no_coincide"
    return None if pending(b["carry"]) == held else "pendiente_perdido"


def ordinal(year: int, month: int) -> int:
    return year * 12 + month - 1


def label(n: int) -> str:
    year, month = divmod(n, 12)
    return f"{year:04d}-{month + 1:02d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    ns = parser.parse_args(argv)

    root = (ns.events_root or f"{default_root('dc-events')}/l2").rstrip("/")
    prefix = f"{root}/provider={ns.provider}/market={ns.market}/asset={ns.asset}"
    fs, base = pafs.FileSystem.from_uri(prefix)
    infos = fs.get_file_info(
        pafs.FileSelector(base, recursive=True, allow_not_found=True)
    )

    # (θ, ordinal del mes) -> {"events": ruta, "carry_over": ruta}
    found: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for info in infos:
        if match := PATH.search(info.path):
            theta, year, month, kind = match.groups()
            found[theta, ordinal(int(year), int(month))][kind] = info.path
    if not found:
        print(f"sin eventos bajo {prefix}")
        return 1

    keys = sorted(found)
    with ThreadPoolExecutor(16) as pool:
        loaded = dict(
            zip(keys, pool.map(lambda k: load_month(fs, found[k]), keys), strict=True)
        )

    months_by_theta: dict[str, list[int]] = defaultdict(list)
    for theta, n in keys:
        months_by_theta[theta].append(n)

    borders = failures = 0
    for theta, months in sorted(months_by_theta.items()):
        # Todos los meses entre el primero y el último del θ: un hueco es un borde roto.
        for n in range(min(months), max(months)):
            borders += 1
            reason = check_border(loaded.get((theta, n)), loaded.get((theta, n + 1)))
            if reason is not None:
                failures += 1
                if failures <= MAX_PRINTED:
                    print(f"FALLO θ={theta} {label(n)} -> {label(n + 1)}: {reason}")
    if failures > MAX_PRINTED:
        print(f"... y {failures - MAX_PRINTED} fallos más")
    print(
        f"θ: {len(months_by_theta)}; bordes: {borders} "
        f"(ok: {borders - failures}; fallos: {failures})"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
