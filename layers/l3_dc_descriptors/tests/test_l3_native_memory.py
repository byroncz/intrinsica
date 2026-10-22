"""El pico de memoria del motor nativo es O(row groups en vuelo): no crece con los ticks
del mes (ITSC-358, RNF-L3-01, ADR-L3-16).

Mismo patrón que `test_l3_memory.py`: el pico se mide en un proceso aparte. Aquí se mide
la RSS (`VmHWM`), que sí ve la memoria de Rust y de los hilos nativos, y no el pool de
Arrow. El probe empuja a lo más `K` row groups a la vez (observa `inflight()`), recoge lo
que el motor sella y lo suelta, como hará el orquestador.
"""

import json
import os
import subprocess
import sys

T = 100_000  # ticks por row group
K = 3
# El asignador de la imagen de L3 (ITSC-367): umbral de mmap de 32 MiB y sin recorte del heap.
IMAGE_MALLOC = {
    "MALLOC_MMAP_THRESHOLD_": "33554432",
    "MALLOC_TRIM_THRESHOLD_": "1073741824",
}
# Bytes de un row group decodificado: tiempo, precio y cantidad (8 + 16 + 16 B) y el bit
# de lado.
ROW_GROUP_BYTES = T * 40 + T // 8

PROBE = """
import json, sys, time
from decimal import Decimal
import numpy as np
import dc_pyo3
from l3_dc_descriptors import native

T, K, RGS = @T@, @K@, int(sys.argv[1])
SMALL, OVER, S = 60, 20, 5000  # ticks de confirmación y de overshoot, filas de salida
total = RGS * T

# Eventos encadenados (E_i = R_{i+1}) sobre ticks de precio y cantidad 1 y tiempo = id:
# todos cumplen los invariantes. Uno de cada 4 000 cruza dos row groups y medio.
r, c, e, pos = [], [], [], 1
while True:
    long = len(r) % 4000 == 3999
    conf = (5 * T) // 2 if long else SMALL
    if pos + conf + OVER >= total:
        break
    r.append(pos); c.append(pos + conf); e.append(pos + conf + OVER)
    pos += conf + OVER
r, c, e = (np.array(x, np.int64) for x in (r, c, e))
n = len(r)
sizes = [S] * (n // S) + ([n % S] if n % S else [])
engine = native.new_engine(@THETAS@, [sizes, sizes], 2)  # dos θ, mismo esqueleto

def decimals(count, value):
    words = np.zeros((count, 2), "<i8")
    words[:, 0] = value
    return memoryview(words).cast("B"), 0

# Las fronteras de L2 de cada θ, como lotes de S filas de `events.parquet`, sin copiar. Las
# arrays las retiene el probe (el motor suelta cada lote al cerrarse sus eventos; eso lo
# prueba `test_l3_native.py`) y se arman antes de la línea base: lo que se mide abajo es lo
# que el motor añade por row group.
# Las referencias, confirmaciones y extremos son ids, y el tiempo de cada tick es su id.
direction = np.ones(n, np.int8)
price = decimals(n, 10**8)
def column(array, start):
    return memoryview(array[start:]).cast("B"), 0
batches = [
    native.EventBatch(
        min(S, n - start), column(r, start), column(c, start), column(c, start),
        column(e, start), column(direction, start), (price[0], start), (price[0], start),
    )
    for start in range(0, n, S)
]
for theta in range(2):
    for batch in batches:
        engine.push_boundaries(theta, batch)

def rss():
    with open("/proc/self/status") as f:
        fields = dict(line.split(":") for line in f)
    return {k: int(fields[k].split()[0]) * 1024 for k in ("VmRSS", "VmHWM")}

popped = 0
def drain():
    global popped
    while (sealed := engine.pop_sealed()) is not None:
        popped += sealed[2]

# El pico se mide desde aquí: el de armar la cadena de eventos (listas de Python) no cuenta.
with open("/proc/self/clear_refs", "w") as f:
    f.write("5")
baseline = rss()["VmRSS"]
peak_inflight = 0
for k in range(RGS):
    while engine.inflight() >= K:
        drain(); time.sleep(0.0005)
    lo, hi = k * T, (k + 1) * T
    ids = np.arange(lo + 1, hi + 1, dtype=np.int64)
    sold = np.packbits(ids % 2 == 0, bitorder="little")
    engine.push(
        k, T, (memoryview(ids).cast("B"), 0), (memoryview(ids).cast("B"), 0),
        decimals(T, 10**8), decimals(T, 10**8), (memoryview(sold).cast("B"), 0), False,
    )
    del ids, sold
    peak_inflight = max(peak_inflight, engine.inflight())
    drain()
stats = engine.finish()
drain()
engine.close()
print(json.dumps({
    "rows": [s[0] for s in stats], "popped": popped, "events": n,
    "baseline": baseline, "peak_inflight": peak_inflight, **rss(),
}))
"""


def probe(row_groups: int) -> dict:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE.replace("@T@", str(T))
            .replace("@K@", str(K))
            .replace("@THETAS@", '[Decimal("0.001"), Decimal("0.002")]'),
            str(row_groups),
        ],
        capture_output=True,
        text=True,
        check=False,
        # Como en la imagen de L3 (ver `memory.py`): sin esto las fronteras de la memoria
        # del proceso dependen de la heurística del asignador.
        env={**os.environ, **IMAGE_MALLOC},
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_rss_does_not_grow_with_the_ticks_of_the_month():
    short, long = probe(24), probe(192)
    for result in (short, long):
        assert result["rows"] == [result["events"]] * 2
        assert result["popped"] == 2 * result["events"]
        assert result["peak_inflight"] <= K
    assert long["events"] > 3 * short["events"]
    # Ocho veces más ticks no mueven el pico sobre la línea base: es del orden de los
    # row groups en vuelo y de los buffers de salida abiertos, no de los ticks del mes.
    over = {
        name: r["VmHWM"] - r["baseline"]
        for name, r in (("corto", short), ("largo", long))
    }
    assert over["largo"] < over["corto"] + 2 * ROW_GROUP_BYTES, over
    assert over["largo"] < (K + 4) * ROW_GROUP_BYTES + 8 * 2**20, over
