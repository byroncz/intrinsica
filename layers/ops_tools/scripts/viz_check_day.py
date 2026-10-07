"""Contrasta L2 con lo que dibuja la vista para un día, un θ y un nivel.

Reproduce la regla de la vista (estado de la columna = estado de su último
tick, direction.py) sobre los ticks de L1 y los eventos de L2, sin leer tiles.
Imprime: (1) los eventos de L2 que tocan la ventana, (2) velas de 1 min con la
caída desde el máximo del día, (3) las columnas de la ventana con su forma y
estado, (4) resumen del día: columnas cuya forma contradice el color y por qué,
y (0) invariantes de L2 sobre todos los eventos del mes.
Solo usa la librería estándar y pyarrow (la imagen ops-script no trae numpy).

Uso (ops-script): --theta 0.01 --day 2026-09-30 --window 12:30-13:00 --w 1024
"""
import argparse
import datetime as dt
import os
from bisect import bisect_left, bisect_right
from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs as pafs

BASE = "provider=binance/market=spot/asset=BTCUSDT"
STATE = {0: "ninguno", 1: "conf↑", 2: "over↑", 3: "conf↓", 4: "over↓"}
INT64_MAX = 2**63 - 1


def hms(us, day0):
    s = (int(us) - day0) / 1e6
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="intrinsica-dc")
    ap.add_argument("--theta", required=True, help="p. ej. 0.01")
    ap.add_argument("--day", default="2026-09-30")
    ap.add_argument("--window", default="12:30-13:00")
    ap.add_argument("--w", type=int, default=1024, help="columnas del nivel (128..4096)")
    ap.add_argument("--local-root", default="", help="pruebas: raíz local en vez de GCS")
    a = ap.parse_args()

    theta = Decimal(a.theta)
    day = dt.date.fromisoformat(a.day)
    day0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp() * 1e6)
    day1 = day0 + 86_400_000_000
    h0, h1 = a.window.split("-")
    w0 = day0 + (int(h0[:2]) * 3600 + int(h0[3:5]) * 60) * 1_000_000
    w1 = day0 + (int(h1[:2]) * 3600 + int(h1[3:5]) * 60) * 1_000_000
    ym = f"year={day.year:04d}/month={day.month:02d}"
    gcs = pafs.LocalFileSystem() if a.local_root else pafs.GcsFileSystem()
    root = a.local_root.rstrip("/") + "/" if a.local_root else ""
    l1 = f"{root}{a.project}-landing/l1/{BASE}/{ym}/consolidated.parquet"
    l2dir = f"{root}{a.project}-dc-events/l2/{BASE}/theta={theta:.8f}/{ym}"

    # --- L1: ticks del día (tres columnas; los row groups se filtran por estadísticas)
    t = ds.dataset(l1, filesystem=gcs).to_table(
        columns=["agg_trade_id", "price", "transact_time"],
        filter=(ds.field("transact_time") >= day0) & (ds.field("transact_time") < day1),
    )
    ids = t["agg_trade_id"].to_pylist()
    tm = t["transact_time"].to_pylist()
    px = pc.cast(t["price"], pa.float64()).to_pylist()
    del t
    n_ticks = len(ids)
    assert all(ids[i] < ids[i + 1] for i in range(n_ticks - 1)), "agg_trade_id no es estrictamente creciente"
    assert all(tm[i] <= tm[i + 1] for i in range(n_ticks - 1)), "transact_time no es no decreciente"
    print(f"L1 {a.day}: {n_ticks} ticks, ids [{ids[0]}, {ids[-1]}]")

    # --- L2: eventos del mes y cola (carry_over)
    ev = pq.read_table(f"{l2dir}/events.parquet", filesystem=gcs).sort_by("reference_agg_trade_id")
    col = lambda n: ev[n].to_pylist()
    fl = lambda n: pc.cast(ev[n], pa.float64()).to_pylist()
    ref, con, ext = col("reference_agg_trade_id"), col("confirm_agg_trade_id"), col("extreme_agg_trade_id")
    ref_t, con_t, ext_t = col("reference_time"), col("confirm_time"), col("extreme_time")
    ref_p, con_p, ext_p = fl("reference_price"), fl("confirm_price"), fl("extreme_price")
    dirs = [int(d) for d in col("direction")]
    n_real = len(ref)
    co = pq.read_table(f"{l2dir}/carry_over.parquet", filesystem=gcs).to_pylist()[0]
    pend = None
    if co["has_pending_event"]:
        d = int(co["direction"])
        e = "ext_high" if d == 1 else "ext_low"
        pend = dict(
            ref=int(co["pending_reference_agg_trade_id"]), con=int(co["pending_confirm_agg_trade_id"]),
            ext=int(co[f"{e}_agg_trade_id"]), ext_t=int(co[f"{e}_time"]), ext_p=float(co[f"{e}_price"]), d=d,
        )
        # Igual que direction.py: el pendiente hasta su candidato, y después confirmación contraria.
        ref += [pend["ref"], pend["ext"]]
        con += [pend["con"], INT64_MAX]
        ext += [pend["ext"], INT64_MAX]
        dirs += [d, -d]
        ref_p += [float("nan"), pend["ext_p"]]
    assert all(ref[i] < ref[i + 1] for i in range(len(ref) - 1)), "referencias no crecientes"
    print(f"L2 θ={theta}: {n_real} eventos en el mes; pendiente: "
          + (f"dir={pend['d']} extremo candidato {hms(pend['ext_t'], day0)} @ {pend['ext_p']:.2f}" if pend else "no"))

    # (0) invariantes de L2 sobre todos los eventos del mes (contrato §7.2 y ADR-L2-03)
    th = float(theta)
    bad = []
    for k in range(n_real):
        mv = con_p[k] / ref_p[k] - 1
        if dirs[k] == 1 and mv < th or dirs[k] == -1 and -mv < th:
            bad.append(f"|Δ|<θ en evento {k}: Δ={mv:+.6%}")
        if k + 1 < n_real:
            if dirs[k + 1] != -dirs[k]:
                bad.append(f"sin alternancia entre {k} y {k + 1}")
            if ext[k] != ref[k + 1] or ext_p[k] != ref_p[k + 1]:
                bad.append(f"extremo de {k} ≠ referencia de {k + 1}")
        if ext_t[k] < con_t[k] or ext[k] < con[k]:
            bad.append(f"extremo anterior a la confirmación en {k}")
        if ext_t[k] == con_t[k] and ext_p[k] != con_p[k]:
            bad.append(f"evento {k}: extreme_time == confirm_time ({con_t[k]} µs) pero extreme_price "
                       f"{ext_p[k]:.2f} ≠ confirm_price {con_p[k]:.2f} (ADR-L2-03 b)")
    print(f"\n== Invariantes L2 sobre {n_real} eventos: {len(bad)} violaciones ==")
    for b in bad[:20]:
        print("  " + b)

    # (1) eventos que tocan la ventana
    print(f"\n== Eventos L2 que tocan {h0}-{h1} (ref → conf → ext; Δ = conf/ref - 1) ==")
    shown = 0
    for k in range(n_real):
        if ref_t[k] < w1 and ext_t[k] >= w0:
            shown += 1
            if shown > 40:
                print("  ... (más de 40 eventos en la ventana; acota --window)")
                break
            mv = con_p[k] / ref_p[k] - 1
            print(f"  {'↑' if dirs[k] == 1 else '↓'} ref {hms(ref_t[k], day0)} @ {ref_p[k]:.2f} | "
                  f"conf {hms(con_t[k], day0)} @ {con_p[k]:.2f} (Δ {mv:+.4%}) | ext {hms(ext_t[k], day0)} @ {ext_p[k]:.2f}"
                  f" | µs conf={con_t[k]} ext={ext_t[k]}")
    if pend and pend["ext_t"] >= w0:
        print(f"  pendiente dir={pend['d']}: extremo candidato {hms(pend['ext_t'], day0)} @ {pend['ext_p']:.2f} "
              f"(después de él la vista pinta confirmación contraria provisional)")

    # (2) velas de 1 min en la ventana, con caída desde el máximo del día hasta ese minuto
    print(f"\n== Velas de 1 min {h0}-{h1}: O H L C | caída desde el máx. del día (θ = {theta:.2%}) ==")
    cummax = []
    m = float("-inf")
    for p in px:
        if p > m:
            m = p
        cummax.append(m)
    for m0 in range(w0, w1, 60_000_000):
        i0, i1 = bisect_left(tm, m0), bisect_left(tm, m0 + 60_000_000)
        if i1 <= i0:
            continue
        seg = px[i0:i1]
        dd = min(seg) / cummax[i1 - 1] - 1
        print(f"  {hms(m0, day0)[:5]}  {seg[0]:.2f} {max(seg):.2f} {min(seg):.2f} {seg[-1]:.2f} | {dd:+.4%}"
              + ("  <-- supera θ" if -dd >= float(theta) else ""))

    # (3) columnas del nivel: regla de direction.py sobre el último tick de cada columna
    col_us = 86_400_000_000 // a.w
    rows = []  # (c, present, p_first, p_last, state, contra, causa)
    counts = {"giro": 0, "empate": 0, "provisional": 0, "menor": 0, "otro": 0}
    n_present = 0
    for c in range(a.w):
        first = bisect_left(tm, day0 + c * col_us)
        last = bisect_left(tm, day0 + (c + 1) * col_us) - 1
        if last < first:
            rows.append((c, False, None, None, 0, False, ""))
            continue
        n_present += 1
        last_id, first_id = ids[last], ids[first]
        k = bisect_left(ref, last_id) - 1  # evento de mayor referencia estrictamente menor que el id
        state = 0
        if k >= 0 and last_id <= ext[k]:
            over = last_id > con[k]
            state = (2 if over else 1) if dirs[k] == 1 else (4 if over else 3)
        p_first, p_last = px[first], px[last]
        shape = (p_last > p_first) - (p_last < p_first)
        sign = 0 if state == 0 else (1 if state <= 2 else -1)
        contra = shape != 0 and sign != 0 and shape != sign
        causa = ""
        if contra:
            if first_id <= ref[k] <= last_id:
                causa = "giro"           # el extremo (referencia del evento) cae dentro de la columna
            elif p_last == ref_p[k]:
                causa = "empate"         # mismo precio que el extremo, tick posterior
            elif k >= n_real:
                causa = "provisional"    # cola del carry-over
            elif abs(p_last / p_first - 1) < th:
                causa = "menor"          # contra-movimiento < θ dentro del evento: normal en DC
            else:
                causa = "otro"           # la barra sola recorre ≥ θ contra el estado: revisar
            counts[causa] += 1
        rows.append((c, True, p_first, p_last, state, contra, causa))

    txt = {"giro": "giro dentro de la columna", "empate": "empate con el extremo",
           "provisional": "cola provisional", "menor": "contra-movimiento < θ (normal en DC)",
           "otro": "barra ≥ θ contra el estado: REVISAR"}
    print(f"\n== Columnas w={a.w} ({col_us / 1e6:.1f} s) en {h0}-{h1}: forma(primero→último) y estado ==")
    for c in range((w0 - day0) // col_us, (w1 - day0) // col_us):
        _, present, p_first, p_last, state, contra, causa = rows[c]
        t0 = hms(day0 + c * col_us, day0)[:8]
        if not present:
            print(f"  col {c} {t0} vacía")
            continue
        arrow = "▲" if p_last > p_first else "▼" if p_last < p_first else "="
        flag = f" <-- CONTRADICE: {txt[causa]}" if contra else ""
        print(f"  col {c} {t0} {p_first:.2f}→{p_last:.2f} {arrow} {STATE[state]}{flag}")

    # (4) resumen del día
    n_contra = sum(counts.values())
    print(f"\n== Resumen {a.day} θ={theta} w={a.w}: {n_present} columnas con ticks, "
          f"{n_contra} con forma contraria al color ==")
    print(f"  giro dentro de la columna: {counts['giro']}")
    print(f"  empate con el extremo:     {counts['empate']}")
    print(f"  cola provisional:          {counts['provisional']}")
    print(f"  contra-movimiento < θ:     {counts['menor']}  (normal en DC: el evento sigue hasta revertir θ)")
    print(f"  barra ≥ θ contra el estado: {counts['otro']}  (si > 0 hay un defecto real)")
    for c, present, p_first, p_last, state, contra, causa in rows:
        if causa == "otro":
            print(f"    col {c} {hms(day0 + c * col_us, day0)[:8]} {p_first:.2f}→{p_last:.2f} {STATE[state]}")

    uri = os.environ.get("OPS_RESULTS_URI", "")
    if uri:
        from google.cloud import storage
        lines = ["col,inicio,primero,ultimo,estado,contradice,causa"]
        for c, present, p_first, p_last, state, contra, causa in rows:
            if present:
                lines.append(f"{c},{hms(day0 + c * col_us, day0)},{p_first:.2f},{p_last:.2f},"
                             f"{STATE[state]},{int(contra)},{causa}")
        b, path = uri[5:].split("/", 1)
        name = f"{path}columnas_{a.day}_theta{theta}_w{a.w}.csv"
        storage.Client().bucket(b).blob(name).upload_from_string("\n".join(lines), content_type="text/csv")
        print(f"  detalle por columna: gs://{b}/{name}")


if __name__ == "__main__":
    main()
