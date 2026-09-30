"""Genera el fixture de equivalencia contra la v0 (ITSC-240).

Lee un día de un `consolidated.parquet` de la landing (L1), lo escribe como
`ticks.csv` y corre `segment_events_kernel` de la v0 (tag `v0.2.0-legacy`)
para varios θ. Deja en esta carpeta:

- `ticks.csv`: `agg_trade_id,price,transact_time` (price con 8 decimales,
  transact_time en µs UTC).
- `events_v0.csv`: un evento por fila y θ, con referencia, confirmación y
  extremo (el del último evento de cada θ no se conoce: columnas vacías).
- `final_state_v0.csv`: el estado final del kernel por θ.

Corre aparte del entorno del repo (Numba no soporta Python 3.14). Ver
`README.md` de esta carpeta para el comando exacto.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

V0_TAG = "v0.2.0-legacy"
V0_KERNEL = "src/intrinseca/core/kernel.py"
SCALE = 10**8
US_PER_DAY = 86_400_000_000
# θ en enteros de escala 10⁸, como los recibe `Detector::new`: 0.1 %, 0.25 %,
# 0.5 %, 1 % y 2 %.
THETAS = (100_000, 250_000, 500_000, 1_000_000, 2_000_000)
OUT = Path(__file__).resolve().parent


def load_v0_kernel(repo: Path):
    """Baja `kernel.py` del tag de la v0 a un temporal y lo importa."""
    source = subprocess.run(
        ["git", "-C", str(repo), "show", f"{V0_TAG}:{V0_KERNEL}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "kernel_v0.py"
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location("kernel_v0", tmp)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kernel_v0"] = module
    spec.loader.exec_module(module)
    return module


def read_day(parquet: Path, day: dt.date):
    """Ticks del día UTC, ordenados por (transact_time, agg_trade_id)."""
    table = pq.read_table(parquet, columns=["agg_trade_id", "price", "transact_time"])
    start = (
        int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())
        * 1_000_000
    )
    mask = pc.and_(
        pc.greater_equal(table["transact_time"], start),
        pc.less(table["transact_time"], start + US_PER_DAY),
    )
    table = table.filter(mask).sort_by(
        [("transact_time", "ascending"), ("agg_trade_id", "ascending")]
    )
    ids = table["agg_trade_id"].to_pylist()
    prices = [str(p) for p in table["price"].to_pylist()]  # Decimal exacto, 8 decimales
    times = table["transact_time"].to_pylist()
    return ids, prices, times


def fmt(price: float) -> str:
    return f"{Decimal(repr(float(price))):.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet", type=Path, help="consolidated.parquet del mes")
    parser.add_argument("day", type=dt.date.fromisoformat, help="día UTC, YYYY-MM-DD")
    parser.add_argument(
        "--repo",
        type=Path,
        default=OUT.parents[3],
        help="raíz del repo (para git show)",
    )
    args = parser.parse_args()

    kernel = load_v0_kernel(args.repo)
    ids, prices, times = read_day(args.parquet, args.day)
    n = len(ids)

    with (OUT / "ticks.csv").open("w") as f:
        f.write("agg_trade_id,price,transact_time\n")
        for i, p, t in zip(ids, prices, times):
            f.write(f"{i},{Decimal(p):.8f},{t}\n")

    # La v0 recibe precios float64, como los alimentaba. Como `quantities`
    # pasa el agg_trade_id (exacto en float64 por debajo de 2⁵³): el kernel
    # lo copia junto al precio a los búferes DC/OS y así se recuperan los
    # `agg_trade_id` de cada punto, que el kernel solo conoce como índice.
    p64 = np.array([float(Decimal(p)) for p in prices], dtype=np.float64)
    t64 = np.array(times, dtype=np.int64)
    id64 = np.array(ids, dtype=np.float64)
    dirs = np.zeros(n, dtype=np.int8)
    pos_of_id = {i: k for k, i in enumerate(ids)}

    with (
        (OUT / "events_v0.csv").open("w") as ev,
        (OUT / "final_state_v0.csv").open("w") as fs,
    ):
        ev.write(
            "theta,seq,direction,reference_price,reference_time,reference_agg_trade_id,"
            "confirm_price,confirm_time,confirm_agg_trade_id,"
            "extreme_price,extreme_time,extreme_agg_trade_id\n"
        )
        fs.write(
            "theta,n_events,trend,ext_high_price,ext_low_price,last_os_ref,orphan_start_idx\n"
        )
        for theta_int in THETAS:
            r = kernel.segment_events_kernel(
                p64,
                t64,
                id64,
                dirs,
                theta_int / SCALE,
                np.int8(0),
                np.float64(0),
                np.float64(0),
                np.float64(0),
            )
            (_, _, dc_q, _, _, _, _, _, types, dc_off, _) = r[:11]
            ref_p, ref_t, ext_p, ext_t, conf_p, conf_t = r[11:17]
            n_events, trend, ext_high, ext_low, os_ref, orphan = r[17:23]
            # Referencia: el tick previo al primer tick del DC; confirmación:
            # el último tick del DC.
            ref_id = [ids[pos_of_id[int(dc_q[dc_off[k]])] - 1] for k in range(n_events)]
            conf_id = [int(dc_q[dc_off[k + 1] - 1]) for k in range(n_events)]
            for k in range(n_events):
                last = k == n_events - 1
                # Extremo del evento k = referencia del k+1 (la v0 lo rellena igual).
                ext = (
                    ["", "", ""]
                    if last
                    else [fmt(ext_p[k]), int(ext_t[k]), ref_id[k + 1]]
                )
                row = [
                    theta_int,
                    k,
                    int(types[k]),
                    fmt(ref_p[k]),
                    int(ref_t[k]),
                    ref_id[k],
                    fmt(conf_p[k]),
                    int(conf_t[k]),
                    conf_id[k],
                    *ext,
                ]
                ev.write(",".join(str(c) for c in row) + "\n")
            fs.write(
                f"{theta_int},{int(n_events)},{int(trend)},{fmt(ext_high)},{fmt(ext_low)},"
                f"{fmt(os_ref)},{int(orphan)}\n"
            )
            print(
                f"theta={theta_int}: {int(n_events)} eventos, tendencia final {int(trend)}"
            )
    print(f"{n} ticks de {args.day}")


if __name__ == "__main__":
    main()
