"""Escaneo de pies Parquet de la landing de L1: el mínimo de `price` de cada mes.

Lee solo el pie (footer) de cada `consolidated.parquet`, nunca los datos: son
unos KiB por archivo, así que los 109 meses salen en segundos. El mínimo sale de
las estadísticas de los row groups. Sirve para cazar marcas inválidas
(`price = 0`, ITSC-294) sin leer 190 millones de ticks.

Args (todos opcionales):
  --landing-root gs://<proyecto>-landing/l1   (por defecto, el del proyecto)
  --provider binance --market spot --asset BTCUSDT
  --umbral 0     un mes con min(price) <= umbral es un fallo

Última línea: `meses: N; min(price): X (AAAA-MM); con min <= umbral: K`. Sale con
1 si algún mes está en fallo o no trae estadísticas.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pyarrow.fs as pafs
import pyarrow.parquet as pq

CONSOLIDATED = re.compile(r"/year=(\d{4})/month=(\d{2})/consolidated\.parquet$")


def default_root(suffix: str) -> str:
    """`gs://<proyecto>-<suffix>`: el proyecto sale del bucket de resultados."""
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit("sin --landing-root y sin OPS_RESULTS_URI: no sé de qué proyecto es")
    return f"gs://{match[1]}-{suffix}"


def footer_min(fs: pafs.FileSystem, path: str) -> tuple[int, Decimal | None]:
    """`(filas, min(price))` según el pie; `None` si algún row group no trae estadísticas."""
    with fs.open_input_file(path) as handle:
        meta = pq.ParquetFile(handle).metadata
    column = meta.schema.names.index("price")
    mins = []
    for group in range(meta.num_row_groups):
        stats = meta.row_group(group).column(column).statistics
        if stats is None or not stats.has_min_max:
            return meta.num_rows, None
        mins.append(stats.min)
    return meta.num_rows, min(mins, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--landing-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--umbral", type=Decimal, default=Decimal(0))
    ns = parser.parse_args(argv)

    root = (ns.landing_root or f"{default_root('landing')}/l1").rstrip("/")
    prefix = f"{root}/provider={ns.provider}/market={ns.market}/asset={ns.asset}"
    fs, base = pafs.FileSystem.from_uri(prefix)
    infos = fs.get_file_info(
        pafs.FileSelector(base, recursive=True, allow_not_found=True)
    )
    files = sorted(
        (f"{m[1]}-{m[2]}", info.path)
        for info in infos
        if (m := CONSOLIDATED.search(info.path))
    )
    if not files:
        print(f"sin consolidated.parquet bajo {prefix}")
        return 1

    with ThreadPoolExecutor(16) as pool:
        found = list(pool.map(lambda item: footer_min(fs, item[1]), files))

    bad = []
    for (month, _), (rows, low) in zip(files, found, strict=True):
        if low is None or low <= ns.umbral:
            bad.append(month)
            print(f"FALLO {month}: filas={rows} min(price)={low}")
    known = [
        (low, month)
        for (month, _), (_, low) in zip(files, found, strict=True)
        if low is not None
    ]
    low, month = min(known, default=(None, "-"))
    print(
        f"meses: {len(files)}; min(price): {low} ({month}); con min <= {ns.umbral}: {len(bad)}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
