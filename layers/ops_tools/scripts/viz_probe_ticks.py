
"""Sonda de tamaño: cuánto pesa un día de ticks si la página lo lleva entero.

Lee los ticks de un día de L1 (tiempo, precio, cantidad) y los codifica de dos
formas; reporta bytes crudos, en gzip (lo que serviría el bucket), en base64 +
gzip (como hoy va embebido en el HTML) y, si la librería existe, en zstd.
Solo librería estándar y pyarrow.

  A. Columnas planas: tiempo ms desde el inicio del día (int32), precio en
     centavos (int32), cantidad en 1e-8 (int64). 16 bytes por tick.
  B. Deltas varint: Δtiempo ms (varint), Δprecio en centavos (zigzag varint),
     cantidad en 1e-8 (varint). Variable, del orden de 4 a 7 bytes por tick.

Uso (ops-script): --day 2026-09-30
"""
import argparse
from collections import Counter
import base64
import datetime as dt
import gzip
import struct
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import fs as pafs

BASE = "provider=binance/market=spot/asset=BTCUSDT"


def varint(out: bytearray, v: int) -> None:
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


def zigzag(v: int) -> int:
    return (v << 1) ^ (v >> 63)


def sizes(name: str, raw: bytes) -> None:
    gz = len(gzip.compress(raw, 9))
    b64gz = len(gzip.compress(base64.b64encode(raw), 9))
    line = f"  {name:<28} crudo {len(raw):>10,} B | gzip {gz:>10,} B | base64+gzip {b64gz:>10,} B"
    try:
        from compression import zstd  # Python 3.14
        line += f" | zstd {len(zstd.compress(raw, 19)):>10,} B"
    except Exception:
        pass
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="intrinsica-dc")
    ap.add_argument("--day", default="2026-09-30")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.day)
    day0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp() * 1e6)
    day1 = day0 + 86_400_000_000
    path = f"{a.project}-landing/l1/{BASE}/year={day.year:04d}/month={day.month:02d}/consolidated.parquet"

    t0 = time.perf_counter()
    t = ds.dataset(path, filesystem=pafs.GcsFileSystem()).to_table(
        columns=["price", "quantity", "transact_time"],
        filter=(ds.field("transact_time") >= day0) & (ds.field("transact_time") < day1),
    )
    tm = t["transact_time"].to_pylist()
    px = [int(p * 100) for p in t["price"].to_pylist()]            # centavos, exacto
    qt = [int(q * 100_000_000) for q in t["quantity"].to_pylist()]  # 1e-8, exacto
    del t
    n = len(tm)
    print(f"L1 {a.day}: {n:,} ticks, lectura {time.perf_counter() - t0:.1f} s")
    ms = [(x - day0) // 1000 for x in tm]
    print(f"  precio {min(px) / 100:.2f} a {max(px) / 100:.2f} | cantidad máx {max(qt) / 1e8:.8f} | "
          f"ticks por ms máx {max(Counter(ms).values())}")

    # A. columnas planas
    t0 = time.perf_counter()
    col_t = struct.pack(f"<{n}i", *ms)
    col_p = struct.pack(f"<{n}i", *px)
    col_q = struct.pack(f"<{n}q", *qt)
    print(f"\n== A. columnas planas (int32 ms, int32 centavos, int64 1e-8), {time.perf_counter() - t0:.1f} s ==")
    sizes("tiempo", col_t)
    sizes("precio", col_p)
    sizes("cantidad", col_q)
    sizes("A total", col_t + col_p + col_q)

    # B. deltas varint
    t0 = time.perf_counter()
    bt, bp, bq = bytearray(), bytearray(), bytearray()
    pt, pp = 0, 0
    for i in range(n):
        varint(bt, ms[i] - pt)
        pt = ms[i]
        varint(bp, zigzag(px[i] - pp))
        pp = px[i]
        varint(bq, qt[i])
    print(f"\n== B. deltas varint, {time.perf_counter() - t0:.1f} s en Python ==")
    sizes("Δtiempo", bytes(bt))
    sizes("Δprecio", bytes(bp))
    sizes("cantidad", bytes(bq))
    total = bytes(bt + bp + bq)
    sizes("B total", total)
    gz = len(gzip.compress(total, 9))
    print(f"\n== Resumen: {n:,} ticks | B en gzip {gz / 1e6:.2f} MB = {gz / n:.2f} B/tick | "
          f"histórico ~3 330 días ≈ {gz * 3330 / 1e9:.1f} GB ==")


if __name__ == "__main__":
    main()
