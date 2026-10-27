"""El pico de memoria de una pasada es O(row group): no crece con los eventos ni con los
ticks del evento más grande (TRD-L3 §13.9, RNF-L3-01).

Mismo patrón que `test_dc_frames_memory.py`: el pico se mide en un proceso aparte, porque
`max_memory()` del pool por defecto es el máximo desde que arrancó. El pool de Arrow no ve
los arreglos de NumPy (los prefijos de la agregación), que `tracemalloc` sí cuenta: el pico
es la suma de los dos.
"""

import json
import os
import subprocess
import sys
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from frames_lake import (
    DEC,
    EVENTS_SCHEMA,
    L1_SCHEMA,
    MONTH,
    THETA_TYPE,
    partition,
    theta_text,
    write_carry_over,
    write_l2,
)

ROW_GROUP = 100_000
# Bytes de un row group decodificado: id, tiempo, precio y cantidad (8 + 8 + 16 + 16 B).
ROW_GROUP_BYTES = ROW_GROUP * 48
GROUPS = 14
LONG = 10  # row groups que abarca el evento largo
THETAS = (100_000, 200_000)
# El asignador de la imagen de L3 (ITSC-367): umbral de mmap de 32 MiB y sin recorte del heap.
IMAGE_MALLOC = {
    "MALLOC_MMAP_THRESHOLD_": "33554432",
    "MALLOC_TRIM_THRESHOLD_": "1073741824",
}

# Lo que fija la imagen y glibc lee al arrancar: el asignador de arriba y el pool del sistema.
IMAGE_ENV = {**IMAGE_MALLOC, "ARROW_DEFAULT_MEMORY_POOL": "system"}

# ITSC-377: un lector, un tramo en el motor y un hilo del motor (R = K = 1). Con los valores por
# defecto el pico dependía de cuántos row groups alcanzaban a estar en vuelo a la vez, que es una
# carrera entre el lector y el motor (100 corridas: el evento largo pasaba al corto por -5,6 a
# 11,9 MB, casi siempre con la misma `inflight_max`; con R = K = 1, 50 corridas, por -1,6 a 3,6 MB).
# La carrera la cubre
# `test_l3_parallel.py`; aquí lo que se afirma es que el pico no depende de los ticks del evento.
PROBE = """
import json, sys, tracemalloc
import pyarrow as pa
from l3_dc_descriptors import summarize_month
from l3_dc_descriptors.timing import Phases

tracemalloc.start()

l1, l2, l3 = sys.argv[1:4]
phases = Phases()
results = summarize_month(
    l1, l2, l3, (2017, 8), ["0.001", "0.002"], series_start=(2017, 8), phases=phases,
    readers=1, inflight=1, engine_threads=1,
)
print(json.dumps({
    "rows": sum(r.rows for r in results.values()),
    "ticks": sum(r.reconciliation.closed for r in results.values()),
    "inflight_max": phases.inflight_max,
    "peak": pa.default_memory_pool().max_memory() + tracemalloc.get_traced_memory()[1],
}))
"""


# Varias unidades en un mismo proceso, como el backfill (ITSC-371). Cada una escribe en su
# propia raíz de L3 y `process_unit` hace lo que hace en producción, trim incluido; con
# `no-trim` el trim se reemplaza por una función que solo lee el RSS, para ver qué pasa sin él.
UNITS = """
import json, sys
from l3_dc_descriptors import memory, pipeline
from l3_dc_descriptors.context import RunContext, Unit

l1, l2, l3, count, trim = sys.argv[1:6]
if trim == "no-trim":
    pipeline.release_to_kernel = lambda: (memory.rss_bytes(),) * 2
thetas = [100_000 * k for k in range(1, 9)]
out = []
for i in range(int(count)):
    ctx = RunContext(
        mode="backfill", run_id="run", image_version="test", series_start=(2017, 8),
        landing_root=l1, events_root=l2, summaries_root=f"{l3}/{i}", dq_root=f"{l3}/dq",
        threads=16, readers=6, inflight=6, writers=6, engine_threads=8,
    )
    result = pipeline.process_unit(Unit(2017, 8), ctx, thetas)
    assert result.ok, result.error
    details = result.timing.details()
    keys = ("rss_after_trim_mib", "rss_peak_mib", "inflight_max")
    out.append({k: details[k] for k in keys})
print(json.dumps(out))
"""

UNIT_COUNT = 10
# Row groups de 400 000 ticks (19,2 MB decodificados): con los de 100 000 de arriba el RSS de
# la unidad es ≈ 30 MiB sobre los ≈ 120 MiB del intérprete y su ruido (± 15 MiB) tapa la deriva.
UNITS_ROW_GROUP = 400_000
UNITS_ROW_GROUP_BYTES = UNITS_ROW_GROUP * 48
# glibc crea hasta 8 arenas por vCPU y el resultado depende de cuántas: se fijan las de los
# 4 vCPU de Cloud Run (4 × 8) para que la prueba mida lo mismo en cualquier máquina.
UNITS_ENV = {**IMAGE_ENV, "MALLOC_ARENA_MAX": "32"}
# Techo de lo que queda retenido al cerrar una unidad: lo que tuvo en vuelo a la vez, R lectores
# + K row groups en el motor (`readers=6, inflight=6` en `UNITS`), decodificados (≈ 220 MiB).
# Es un techo y no lo que detecta la falta del trim: sin trim la deriva (158 a 214 MiB en 8
# corridas) también cabe, y con él fue de 61 a 198. Solo vale para estas 10 unidades; con 60 el
# RSS tras el trim sigue subiendo ≈ 2 MiB por unidad (ADR-L3-16) y aquí no se afirma lo contrario.
UNITS_IN_FLIGHT_BYTES = (6 + 6) * UNITS_ROW_GROUP_BYTES
# ITSC-377. Lo que detecta la falta del trim es cuánto sube el RSS al cerrar las últimas unidades
# sobre el de la primera (`rise`): sin trim sube lo que glibc va reteniendo, con él sube menos. La
# comparación es contra lo que la propia corrida midió y no contra un número de la máquina (antes
# 3 row groups = 57,6 MB fijos, que en CI dio 57,06): la primera unidad ya trae el nivel del
# intérprete y de las librerías de esa máquina, y restarla lo cancela. En 68 series (10 vCPU sin
# carga, 4 procesos en paralelo, 2 y 8 vCPU) con trim recuperó de 0,31 a 0,83 de lo que subía la
# serie sin él; 0,15 deja más de dos veces de margen y sigue fallando si el trim no hace nada (0).
UNITS_TRIM_MIN_SHARE = 0.15
# Sin trim la serie sube al menos lo que cada unidad tuvo en vuelo (K row groups del motor, de
# media en las 10 unidades): en las 68 series subió al menos 1,45 veces esa cuenta (mediana 2,1); el piso es 1.
UNITS_NO_TRIM_MIN_RISE = 1.0
MIB = 1 << 20


def write_flat_l1(root, ticks: int, row_group: int = ROW_GROUP) -> None:
    """Precio y cantidad 1 en todos los ticks, tiempo = id: una fase suma lo que mide."""
    ids = pa.array(np.arange(ticks, dtype=np.int64))
    # El decimal 1 sin escalar es 10⁸ en la palabra baja y 0 en la alta.
    words = np.zeros((ticks, 2), "<i8")
    words[:, 0] = 10**8
    one = pa.Array.from_buffers(DEC, ticks, [None, pa.py_buffer(words)])
    table = pa.table(
        {
            "agg_trade_id": ids,
            "price": one,
            "quantity": one,
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": ids,
            "is_buyer_maker": pa.array(np.arange(ticks) % 2 == 0),
            "is_best_match": pa.array(np.ones(ticks, bool)),
        },
        schema=L1_SCHEMA,
    )
    path = partition(root, MONTH) / "consolidated.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group)


def event(reference: int, confirm: int, extreme: int) -> dict:
    row = {"direction": 1}
    for name, tick in (
        ("reference", reference),
        ("confirm", confirm),
        ("extreme", extreme),
    ):
        row[f"{name}_price"] = Decimal(1)
        row[f"{name}_time"] = tick
        row[f"{name}_agg_trade_id"] = tick
    return row


def chain(
    start: int, small: int, long: int = LONG, row_group: int = ROW_GROUP
) -> list[dict]:
    """`small` eventos dentro de un row group y uno de `long` row groups, encadenados:
    la referencia de cada uno es el extremo del anterior (sin ticks sin dueño).
    """
    rows, position = [], start
    for _ in range(small):
        rows.append(event(position, position + 50, position + 80))
        position += 80
    rows.append(
        event(position, position + long * row_group, position + long * row_group + 5)
    )
    return rows


def write_theta(root, theta: int, rows: list[dict]) -> None:
    write_l2(root, theta, rows, 100)
    write_carry_over(
        root,
        theta,
        MONTH,
        1,
        pending_reference=rows[-1]["extreme_agg_trade_id"],
    )


def peak(root, name: str) -> dict:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
            str(root / "l1"),
            str(root / name),
            str(root / "l3" / name),
        ],
        capture_output=True,
        text=True,
        check=False,
        # Con el asignador de la imagen de L3 (ITSC-367): glibc retiene lo liberado y el pico
        # no puede crecer con los eventos ni con los ticks aun así.
        env={**os.environ, **IMAGE_MALLOC},
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_peak_is_a_row_group_and_does_not_grow_with_events(tmp_path):
    write_flat_l1(tmp_path / "l1", GROUPS * ROW_GROUP)
    for name, small in (("few", 10), ("many", 200)):
        # Dos θ en fan-out; cada uno con un evento que cruza 10 row groups.
        write_theta(tmp_path / name, THETAS[0], chain(100, small))
        write_theta(tmp_path / name, THETAS[1], chain(107, small))

    few, many = peak(tmp_path, "few"), peak(tmp_path, "many")
    assert many["rows"] == 2 * 201 and few["rows"] == 2 * 11
    assert many["ticks"] > few["ticks"] > 2 * LONG * ROW_GROUP
    # O(row group): el del pase, los prefijos y los temporales de la agregación, muy por
    # debajo de los 10 row groups del evento largo (48 MB con 100 000 ticks por row group).
    assert few["peak"] < 10 * ROW_GROUP_BYTES
    assert many["peak"] < 10 * ROW_GROUP_BYTES
    # Casi veinte veces más eventos no suben el pico. El pico de cada corrida varía entre
    # ≈ 18 y 25 MB según la carrera entre el lector y el motor (6 corridas seguidas:
    # many/few entre 0,94 y 1,02), así que se compara con holgura absoluta, como en la
    # prueba siguiente, y no con una razón.
    assert many["peak"] < few["peak"] + 2 * ROW_GROUP_BYTES


def test_peak_does_not_grow_with_the_ticks_of_the_largest_event(tmp_path):
    """El evento más grande pasa de 10 a 40 row groups y el pico no se mueve."""
    groups = 46
    write_flat_l1(tmp_path / "l1", groups * ROW_GROUP)
    for name, long in (("long", LONG), ("longest", 4 * LONG)):
        write_theta(tmp_path / name, THETAS[0], chain(100, 10, long))
        write_theta(tmp_path / name, THETAS[1], chain(107, 10, long))

    long, longest = peak(tmp_path, "long"), peak(tmp_path, "longest")
    assert longest["ticks"] > 3 * long["ticks"] > 0
    # Si el pico creciera con los ticks del evento, 30 row groups más sumarían 30 × 4,8 MB =
    # 144 MB. Lo que puede pasar de más es lo que la corrida midió: los row groups que quedaron
    # anclados de más (`inflight_max`, por el tamaño de un row group decodificado) y los
    # temporales de la agregación, que con R = K = 1 sumaron a lo más 3,6 MB en 50 corridas
    # (bajo carga y con 2 vCPU): dos row groups (9,6 MB) los cubren con margen de 2,7 veces.
    anchored = (longest["inflight_max"] - long["inflight_max"]) * ROW_GROUP_BYTES
    assert longest["peak"] < long["peak"] + max(anchored, 0) + 2 * ROW_GROUP_BYTES, (
        long,
        longest,
    )
    assert longest["peak"] < 10 * ROW_GROUP_BYTES


def units(root, trim: str) -> list[dict]:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            UNITS,
            str(root / "l1"),
            str(root / "events"),
            str(root / "l3"),
            str(UNIT_COUNT),
            trim,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **UNITS_ENV},
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def write_units_lake(root) -> None:
    write_flat_l1(root / "l1", 8 * UNITS_ROW_GROUP, UNITS_ROW_GROUP)
    for k in range(1, 9):
        write_theta(
            root / "events", 100_000 * k, chain(100 + 7 * k, 200, 5, UNITS_ROW_GROUP)
        )


@pytest.fixture(scope="module")
def unit_runs(tmp_path_factory) -> dict[str, list[dict]]:
    """Las mismas 10 unidades con el trim de producción y sin él, para compararlas."""
    root = tmp_path_factory.mktemp("units")
    write_units_lake(root)
    return {mode: units(root, mode) for mode in ("trim", "no-trim")}


def drift(runs: list[dict]) -> float:
    """Bytes que el RSS al cerrar una unidad llegó a pasar del de la primera."""
    after = [r["rss_after_trim_mib"] * MIB for r in runs]
    return max(after) - after[0]


def tail_level(runs: list[dict]) -> float:
    """Promedio, en bytes, del RSS al cerrar la mitad final de las unidades."""
    tail = [r["rss_after_trim_mib"] * MIB for r in runs[len(runs) // 2 :]]
    return sum(tail) / len(tail)


def rise(runs: list[dict], first: dict) -> float:
    """Bytes que el nivel final de la serie `runs` pasa del RSS al cerrar la primera unidad de
    `first` (la de la serie con trim, que arranca sin nada retenido de antes).
    """
    return tail_level(runs) - first["rss_after_trim_mib"] * MIB


def in_flight(runs: list[dict]) -> float:
    """Bytes que cada unidad tuvo en el motor a la vez, de media: `inflight_max` por row group."""
    return sum(r["inflight_max"] for r in runs) / len(runs) * UNITS_ROW_GROUP_BYTES


def test_rss_after_closing_each_unit_does_not_grow_with_the_units(unit_runs):
    """ITSC-371: con el asignador de la imagen glibc no devuelve al kernel lo liberado y, sin
    hacerlo al cerrar la unidad, el RSS crece unidad a unidad y un backfill largo muere por
    memoria (la causa exacta es una hipótesis, ver TRD-L3 ADR-L3-16). Las pruebas de arriba
    miden una unidad y no lo ven.
    """
    runs = unit_runs["trim"]
    # Lo que queda retenido no pasa de lo que la unidad tuvo en vuelo; ver UNITS_IN_FLIGHT_BYTES.
    assert drift(runs) <= UNITS_IN_FLIGHT_BYTES, runs
    # Y el trim recupera de verdad: las últimas unidades quedan más abajo que sin él, y lo que
    # recuperan es una parte de lo que la serie sin trim sube; ver UNITS_TRIM_MIN_SHARE. Esta
    # comparación es la que falla cuando falta el trim.
    gain = tail_level(unit_runs["no-trim"]) - tail_level(runs)
    grown = rise(unit_runs["no-trim"], runs[0])
    assert gain >= UNITS_TRIM_MIN_SHARE * grown, (gain, grown, unit_runs)
    # El pico de la última unidad lo acota la unidad más grande, no la suma de las anteriores.
    peaks = [r["rss_peak_mib"] for r in runs]
    assert peaks[-1] <= 1.3 * max(peaks[:-1]), peaks


def test_without_the_trim_the_rss_does_grow(unit_runs):
    """Contraprueba de la anterior: sin el trim el RSS al cerrar sí sube, así que la
    comparación de arriba no pasa en vacío.
    """
    runs = unit_runs["no-trim"]
    assert drift(runs) > UNITS_NO_TRIM_MIN_RISE * in_flight(runs), unit_runs


# ITSC-372. Problema 1: el RSS que una unidad deja en el proceso. El padre del backfill no corre
# unidades: cada una vive en un hijo y muere con su memoria. Se miden 6 en un proceso padre y
# se lee el `VmRSS` del padre entre una y otra.
CHILD_UNITS = """
import json, sys
from l3_dc_descriptors import memory, unit_process
from l3_dc_descriptors.context import RunContext, Unit

l1, l2, l3, count = sys.argv[1:5]
thetas = [100_000 * k for k in range(1, 9)]
out = []
for i in range(int(count)):
    ctx = RunContext(
        mode="backfill", run_id="run", image_version="test", series_start=(2017, 8),
        landing_root=l1, events_root=l2, summaries_root=f"{l3}/{i}", dq_root=f"{l3}/dq",
        threads=16, readers=6, inflight=6, writers=6, engine_threads=8,
    )
    result = unit_process.run_unit(Unit(2017, 8), ctx, thetas)
    assert result.ok, result.error
    details = next(f for f in result.findings if f.check_type == "unit_timing").details
    out.append({
        "parent_rss": memory.rss_bytes(),
        "child_peak_mib": details["rss_peak_mib"],
        "spawn_s": details["spawn_s"],
    })
print(json.dumps(out))
"""
CHILD_UNIT_COUNT = 6
# Lo que el padre puede moverse entre unidades: sus objetos de Python y los hallazgos que
# recibe. En proceso, la misma lago deriva 60 a 140 MiB en 10 unidades (ADR-L3-16).
PARENT_FLAT_BYTES = 20 * (1 << 20)


def test_the_parent_rss_is_flat_between_units_run_in_child_processes(tmp_path):
    """El padre no importa lo que una unidad toca: su VmRSS no sube de unidad en unidad y el
    hijo de cada una arranca de cero (su pico es el de una unidad, no el de la suma).
    """
    write_units_lake(tmp_path)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD_UNITS,
            str(tmp_path / "l1"),
            str(tmp_path / "events"),
            str(tmp_path / "l3"),
            str(CHILD_UNIT_COUNT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **UNITS_ENV},
    )
    assert out.returncode == 0, out.stderr
    runs = json.loads(out.stdout)
    assert len(runs) == CHILD_UNIT_COUNT
    parent = [r["parent_rss"] for r in runs]
    assert max(parent) - min(parent) <= PARENT_FLAT_BYTES, runs
    peaks = [r["child_peak_mib"] for r in runs]
    assert max(peaks) <= 1.3 * min(peaks), peaks
    assert all(0 < r["spawn_s"] < 30 for r in runs), runs


# ITSC-372. Problema 2: el pico de una unidad es O(eventos en vuelo), no O(row groups en vuelo).
# Un lago denso reproduce 2017-12 en pequeño: 3 row groups de L1 (con K = 4 caben todos a la vez) y
# un evento cada 3 ticks en 16 θ, 2,4 M de eventos por row group. Las casillas de salida abiertas
# de todo el mes caben en el motor, y lo que sellan espera a un solo escritor.
DENSE_THETAS = 16
DENSE_GROUPS = 3
DENSE_GROUP_TICKS = 450_000
DENSE_TICKS_PER_EVENT = 3
# Lo que cuesta una casilla de salida abierta (las 18 columnas de `ENGINE_SCHEMA`).
SLOT_BYTES = 240
# Menos que un row group (2,4 M de eventos × 305 B): cada uno avanza solo.
DENSE_CAP = 120 * (1 << 20)
# Un solo hilo del motor, más lento que el lector: con K = 4 solo, el lector empuja los row groups
# uno tras otro y sus casillas se acumulan (4,35 M de eventos abiertos medidos, 1,8 row groups);
# con el tope cada uno espera a que cierre el anterior (2,40 M, un row group). Con 4 hilos el motor
# cierra tan rápido como se empuja y la unidad sin tope también llega a un row group: no prueba nada.
DENSE_ENGINE_THREADS = 1
# ITSC-377. El pico baja al menos esta fracción de lo que cuestan las casillas de más (SLOT_BYTES por
# evento abierto de más, medido en la corrida). No baja la cuenta entera: el pico del RSS es la marca
# más alta del heap y glibc retiene lo que ya liberó, así que parte del ahorro no llega al RSS. Medido
# en 100 pares (lote contra row group entero) la razón fue de 0,28 a 0,70; el piso es 0,15, casi la
# mitad del mínimo. El 0,3 de ITSC-372 salía de 3 corridas y en CI dio 0,285.
DENSE_MIN_GAIN = 0.15
# El pico de dos corridas iguales se mueve (desviación 24 MiB, rango de 140 MiB en 100 corridas de la
# unidad con el tope) según cuántos row groups decodificados alcanzó a tener el lector a la vez. Es la
# holgura de la comparación con el motor sin tope, donde la cuenta de casillas puede ser 0: dos row
# groups de L1 decodificados (2 × 21,6 MB), contra una diferencia mínima medida de 25 MB a favor del tope.
DENSE_PEAK_NOISE = 2 * DENSE_GROUP_TICKS * 48
# ITSC-373: el lote en eventos de las corridas con corte (un tramo de 100 000 eventos son ≈ 18 750
# ticks y 24 tramos por row group; ≈ 30 MiB de casillas y fronteras, bajo los 120 MiB del tope) y el
# lote de las fronteras de L2: `BoundaryFeed` entrega lotes enteros por delante del tramo, hasta uno
# por θ, y esos eventos cuentan como abiertos. Con los 32 768 de producción taparían el lote.
LOT_EVENTS = 100_000
LOT_BATCH_ROWS = 4096
# ITSC-376: el trim por marca de agua de las corridas con trim. 64 MiB está muy bajo el pico de la
# unidad densa (cientos de MiB) y 0,05 s deja varios trims en su pared.
TRIM = {"trim_rss_mib": 64, "trim_min_s": 0.05}

DENSE_UNIT = """
import json, sys
from l3_dc_descriptors import boundaries, pipeline
from l3_dc_descriptors.context import RunContext, Unit

l1, l2, l3, cap, count, engine_threads, options = sys.argv[1:8]
options = json.loads(options)
if "batch_rows" in options:
    boundaries.L2_BATCH_ROWS = options["batch_rows"]
ctx = RunContext(
    mode="backfill", run_id="run", image_version="test", series_start=(2017, 8),
    landing_root=l1, events_root=l2, summaries_root=l3, dq_root=f"{l3}/dq",
    threads=8, readers=2, inflight=options.get("inflight", 4), writers=1,
    engine_threads=int(engine_threads), inflight_bytes=int(cap) or None,
    lot_events=options.get("lot_events", 0), memory_sample_s=0.02,
    trim_rss_mib=options.get("trim_rss_mib", 0), trim_min_s=options.get("trim_min_s"),
)
result = pipeline.process_unit(Unit(2017, 8), ctx, [100_000 * k for k in range(1, int(count) + 1)])
assert result.ok, result.error
details = result.timing.details()
keys = (
    "rss_peak_mib", "inflight_events_max", "inflight_bytes_max", "inflight_alone_bytes_max",
    "inflight_bytes", "events", "lot_events", "lot_ticks", "lots", "inflight_max",
    "mem_peak", "mem_close_before_trim", "mem_close_after_trim", "mem_samples",
    "mem_peak_coverage", "trims", "trim_s", "trim_rss_mib", "gate_s", "setup_s", "aggregate_s",
)
print(json.dumps({k: details[k] for k in keys}))
"""


def dense_decimal(values: np.ndarray) -> pa.Array:
    words = np.zeros((len(values), 2), "<i8")
    words[:, 0] = values
    return pa.Array.from_buffers(DEC, len(values), [None, pa.py_buffer(words)])


def write_dense_theta(root, theta: int, ticks: int, start: int) -> None:
    """Eventos de 3 ticks (referencia, confirmación a +1, extremo a +3) uno tras otro: el
    extremo de cada uno es la referencia del siguiente. Sin un objeto de Python por evento.
    """
    count = (ticks - start - 10) // DENSE_TICKS_PER_EVENT
    reference = start + DENSE_TICKS_PER_EVENT * np.arange(count, dtype=np.int64)
    one = dense_decimal(np.full(count, 10**8, "<i8"))
    columns = {"direction": pa.array(np.ones(count, "i1"))}
    for name, offset in (("reference", 0), ("confirm", 1), ("extreme", 3)):
        columns[f"{name}_price"] = one
        columns[f"{name}_time"] = pa.array(reference + offset)
        columns[f"{name}_agg_trade_id"] = pa.array(reference + offset)
    constant = np.zeros((count, 2), "<i8")
    constant[:, 0] = theta
    columns["theta"] = pa.Array.from_buffers(
        THETA_TYPE, count, [None, pa.py_buffer(constant)]
    )
    path = (
        partition(root, MONTH).parent.parent
        / f"theta={theta_text(theta)}"
        / "year=2017/month=08/events.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table([columns[f.name] for f in EVENTS_SCHEMA], schema=EVENTS_SCHEMA)
    pq.write_table(table, path, row_group_size=25_000)
    write_carry_over(root, theta, MONTH, 1, pending_reference=int(reference[-1]) + 3)


@pytest.fixture(scope="module")
def dense_runs(tmp_path_factory) -> dict[str, dict]:
    """La misma unidad densa con K = 4 solo (sin tope de bytes), con el tope de bytes y, con el
    tope, partiendo cada row group en tramos (ITSC-373) con K = 4 y con K = 1.
    """
    root = tmp_path_factory.mktemp("dense")
    ticks = DENSE_GROUPS * DENSE_GROUP_TICKS
    write_flat_l1(root / "l1", ticks, DENSE_GROUP_TICKS)
    for k in range(1, DENSE_THETAS + 1):
        write_dense_theta(root / "events", 100_000 * k, ticks, 100 + 7 * k)
    cut = {"lot_events": LOT_EVENTS, "batch_rows": LOT_BATCH_ROWS}
    runs = {}
    for name, cap, options in (
        ("k_only", 1 << 40, {}),
        ("bytes", DENSE_CAP, {}),
        ("lots", DENSE_CAP, cut),
        ("lots_one", DENSE_CAP, {**cut, "inflight": 1}),
        ("lots_trim", DENSE_CAP, {**cut, **TRIM}),
        # Los valores por defecto de producción: tope de 768 MiB y lote de 1 M de eventos.
        ("defaults", 0, {"lot_events": None, "batch_rows": LOT_BATCH_ROWS}),
    ):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                DENSE_UNIT,
                str(root / "l1"),
                str(root / "events"),
                str(root / "l3" / name),
                str(cap),
                str(DENSE_THETAS),
                str(DENSE_ENGINE_THREADS),
                json.dumps(options),
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **UNITS_ENV},
        )
        assert out.returncode == 0, out.stderr
        runs[name] = {
            **json.loads(out.stdout),
            "warned": "L3_INFLIGHT_BYTES" in out.stderr,
        }
    return runs


def test_the_byte_cap_bounds_the_open_events_and_the_peak_rss(dense_runs):
    k_only, capped = dense_runs["k_only"], dense_runs["bytes"]
    assert k_only["events"] == capped["events"] > 7_000_000
    one_group = k_only["events"] // DENSE_GROUPS
    # Con el tope, lo abierto es un row group (más lo que `cover` trae por delante), no el mes.
    assert capped["inflight_events_max"] <= 1.2 * one_group, dense_runs
    # Sin él, con un motor más lento que el lector, se acumulan los row groups: es el problema 2
    # tal como lo describe la ADR (O(row groups en vuelo)). Cuántos se acumulan es una carrera
    # entre el lector y el motor (ITSC-377: en 100 corridas, de 1,0 a 1,8 row groups, y en 38 el
    # motor alcanzó al lector y no acumuló nada), así que no se exige que haya acumulación: lo que se afirma es que el tope nunca abre más que el motor sin él.
    extra = k_only["inflight_events_max"] - capped["inflight_events_max"]
    assert extra >= 0, dense_runs
    # El RSS del pico baja lo que cuestan las casillas de más que la corrida midió (`extra`, a
    # SLOT_BYTES cada una); sin acumulación `extra` es 0 y la cuenta dice que el tope no lo sube.
    gain = (k_only["rss_peak_mib"] - capped["rss_peak_mib"]) * MIB
    floor = DENSE_MIN_GAIN * extra * SLOT_BYTES - DENSE_PEAK_NOISE
    assert gain >= floor, (extra, gain, floor, dense_runs)


def test_a_row_group_over_the_cap_is_recorded_and_the_unit_goes_on(dense_runs):
    """Con 120 MiB un solo row group denso pasa del tope: `inflight_alone_bytes_max` lo dice y
    la unidad terminó (límite conocido, TRD-L3 ADR-L3-16).
    """
    capped = dense_runs["bytes"]
    assert capped["inflight_alone_bytes_max"] > capped["inflight_bytes"] == DENSE_CAP


# ITSC-373. El lote es un tramo de a lo más LOT_EVENTS eventos, no un row group de L1.
def test_a_lot_bounds_the_open_events_to_the_lot_not_to_the_row_group(dense_runs):
    one_group = dense_runs["k_only"]["events"] // DENSE_GROUPS
    ahead = DENSE_THETAS * LOT_BATCH_ROWS  # fronteras que `cover` entrega por delante
    for name in ("lots", "lots_one"):
        run = dense_runs[name]
        assert run["events"] == dense_runs["bytes"]["events"], run
        assert run["lot_events"] == LOT_EVENTS and run["lot_ticks"] > 0, run
        # Cada row group se parte en ≈ 24 tramos.
        assert run["lots"] > 10 * DENSE_GROUPS, run
        # Y la unidad nunca empuja a un motor vacío más de lo que el tope admite.
        assert run["inflight_alone_bytes_max"] <= run["inflight_bytes"], run
    # Un tramo a la vez: los eventos abiertos son el lote, más los que `cover` trajo por delante
    # y los que cruzan el borde del tramo (a lo más uno por θ y por borde).
    lot = dense_runs["lots_one"]["inflight_events_max"]
    assert lot <= LOT_EVENTS + ahead + 10 * DENSE_THETAS, dense_runs["lots_one"]
    assert lot < one_group / 3, dense_runs
    # Con K = 4 y el tope de bytes caben varios tramos, pero no más que el tope y un tramo.
    cap_events = DENSE_CAP // (SLOT_BYTES + 65)
    assert dense_runs["lots"]["inflight_events_max"] <= (
        cap_events + LOT_EVENTS + ahead + 10 * DENSE_THETAS
    ), dense_runs["lots"]


def test_the_lot_lowers_the_peak_rss_by_what_the_open_slots_cost(dense_runs):
    uncut, cut = dense_runs["bytes"], dense_runs["lots"]
    extra = uncut["inflight_events_max"] - cut["inflight_events_max"]
    assert extra > 0, dense_runs
    gain = (uncut["rss_peak_mib"] - cut["rss_peak_mib"]) * MIB
    floor = DENSE_MIN_GAIN * extra * SLOT_BYTES
    assert gain >= floor, (extra, gain, floor, dense_runs)


# ITSC-375. La composición del RSS en el pico y al cierre. Ninguna aserción es un MiB absoluto
# (la lección de ITSC-374): son relaciones que valen en cualquier máquina.
MIN_COVERAGE = 0.9
# Tolerancia (MiB) de la invariante del trim: los MiB van redondeados a un decimal.
TRIM_TOLERANCE_MIB = 0.2
# ITSC-377: lo residente sin uso en el pico con trim periódico no pasa de esta fracción del de la
# corrida sin él. En más de 100 pares quedó entre -0,26 y 0,0 veces; con un trim que no hace nada es 1.
TRIM_MAX_RESIDUAL = 0.25


def test_the_memory_sampler_sees_the_peak_of_the_dense_unit(dense_runs):
    """El motor solo es visible para el muestreador entre su creación y su cierre. La muestra
    del pico lleva `engine_inflight` si cayó dentro de esa ventana; si cayó después (la
    publicación y el cierre de los pools de la unidad, donde el RSS todavía puede marcar un
    máximo nuevo), el campo es `None` y la muestra lo dice así: eso es el host (ITSC-377), no un
    fallo del muestreador. En 350 corridas con las fases registradas, las 8 sin `engine_inflight`
    cayeron entre 0,04 y 1,0 s después de que terminó la etapa de agregación.
    """
    for name, run in dense_runs.items():
        peak = run["mem_peak"]
        assert peak is not None and run["mem_samples"] > 1, name
        assert run["mem_peak_coverage"] >= MIN_COVERAGE, (name, run)
        assert peak["arrow_mib"] is not None, (name, peak)
        # La etapa de agregación (empujar los row groups y cerrar el motor con `finish`) termina,
        # contada desde el inicio de la unidad, a más tardar en `gate_s + setup_s + aggregate_s`
        # (la unidad arranca un poco antes que `summarize_month`, así que la cuenta se queda corta
        # y nunca marca de más). Antes de eso el motor existe y el muestreador lo ve.
        engine_until = run["gate_s"] + run["setup_s"] + run["aggregate_s"]
        if peak["elapsed_s"] <= engine_until:
            assert peak["engine_inflight"] is not None, (name, engine_until, peak)
        if peak["engine_inflight"] is not None:
            # Y lo que ve es coherente con lo que el orquestador contó: tramos anclados a lo más
            # el máximo que empujó.
            assert 0 <= peak["engine_inflight"] <= run["inflight_max"], (name, run)


def test_the_glibc_accounting_adds_up_in_the_three_samples(dense_runs):
    for name, run in dense_runs.items():
        for key in ("mem_peak", "mem_close_before_trim", "mem_close_after_trim"):
            sample = run[key]
            # Las arenas piden al sistema lo que está en uso más lo libre que retienen. Los
            # MiB van redondeados a un decimal: de ahí la tolerancia de 0,1 por término.
            assert sample["arena_mib"] == pytest.approx(
                sample["uordblks_mib"] + sample["fordblks_mib"], abs=0.2
            ), (name, key, sample)
            assert sample["hblkhd_mib"] >= 0 and sample["keepcost_mib"] >= 0


def test_the_trim_returns_pages_to_the_kernel_and_the_free_list_drops_at_most_the_top(
    dense_runs,
):
    """`malloc_trim(0)` llama a `mtrim` por arena, que hace dos cosas: `madvise(MADV_DONTNEED)`
    sobre las páginas interiores de los chunks libres, en todas las arenas (baja el RSS y no toca
    `fordblks`), y, solo en la arena principal, `systrim` sobre su top (baja `arena` y `fordblks`
    en ≈ `keepcost`, que `mallinfo2` reporta solo para ella). Entre las dos muestras del cierre
    corre además `gc.collect()`, que puede subir `fordblks`: la subida se permite y la caída no
    puede pasar de la cima. Lo retenido al cierre queda acotado: Δ`rss_mib` es cota inferior y
    `before.fordblks_mib` cota superior.

    ITSC-377: el trim de cierre nunca empeora el RSS, y devuelve algo solo si había algo que
    devolver. Eso depende de cuándo cayó el último trim periódico: con trims por marca de agua
    (`trims > 0`) el último pudo correr instantes antes del cierre y dejar el RSS igual (en CI
    quedó en 215,2 antes y después), así que ahí se afirma `after ≤ before`. Sin trims
    periódicos (`trims == 0`) nadie devolvió nada durante la unidad: el RSS al cierre incluye lo
    libre que glibc retiene (el cierre lo bajó de 363 MiB a más de 1 GiB en 650 corridas) y el
    trim de cierre tiene que bajarlo.
    """
    for name, run in dense_runs.items():
        before, after = run["mem_close_before_trim"], run["mem_close_after_trim"]
        assert after["rss_mib"] <= before["rss_mib"] + TRIM_TOLERANCE_MIB, (
            name,
            before,
            after,
        )
        if run["trims"] == 0:
            assert after["rss_mib"] < before["rss_mib"], (name, before, after)
        dropped = before["fordblks_mib"] - after["fordblks_mib"]
        assert dropped <= before["keepcost_mib"] + TRIM_TOLERANCE_MIB, (
            name,
            before,
            after,
        )


# ITSC-376. El trim por marca de agua devuelve al kernel, durante la unidad, la memoria libre que
# glibc retiene. Relaciones y no MiB absolutos (la lección de ITSC-374).
def test_the_periodic_trim_runs_and_costs_what_it_says(dense_runs):
    trimmed, untrimmed = dense_runs["lots_trim"], dense_runs["lots"]
    assert trimmed["trim_rss_mib"] == 64 and untrimmed["trim_rss_mib"] == 0
    assert trimmed["trims"] >= 1 and trimmed["trim_s"] > 0, trimmed
    assert (untrimmed["trims"], untrimmed["trim_s"]) == (0, 0.0)
    assert trimmed["events"] == untrimmed["events"]


def unused_resident(run: dict) -> float:
    """MiB residentes en el pico de `run` que no son memoria en uso del heap ni lo que el proceso
    deja residente tras un trim completo (`mem_close_after_trim`: intérprete, librerías, pilas
    de hilos y lo libre que el trim no pudo devolver). Es lo que el trim periódico puede quitar.
    """
    peak, closed = run["mem_peak"], run["mem_close_after_trim"]
    floor = closed["rss_mib"] - closed["uordblks_mib"] - closed["hblkhd_mib"]
    return peak["rss_mib"] - peak["uordblks_mib"] - peak["hblkhd_mib"] - floor


def test_the_periodic_trim_lowers_the_peak_by_part_of_the_free_memory_glibc_held(
    dense_runs,
):
    """ITSC-377. El pico del lago denso es memoria viva (≈ 450 MiB de salida sellada de Arrow
    que espera a un solo escritor) más lo libre que glibc retiene. El trim puede quitar solo lo
    segundo, y cuánto era lo libre en el pico varía de corrida a corrida; por eso no se afirma
    una fracción de la baja del pico (ITSC-376 puso 0,2 sobre lo medido en 11 corridas y en CI
    dio 0,195; con 2 vCPU llegó a 0,19 en 8 corridas) sino qué queda residente sin uso en cada
    corrida, contra la otra: sin trim, 250 a 670 MiB; con él, de -124 a +3 (el piso de la cuenta
    es el RSS que deja un trim completo, y a veces el trim periódico deja menos). Con el trim
    caído (que no haga nada) las dos cifras serían iguales.
    """
    trimmed, untrimmed = dense_runs["lots_trim"], dense_runs["lots"]
    held = unused_resident(untrimmed)
    assert held > 0, untrimmed
    assert unused_resident(trimmed) <= TRIM_MAX_RESIDUAL * held, (
        unused_resident(trimmed),
        held,
        untrimmed,
        trimmed,
    )
    # Y el pico del RSS no sube por hacerlo: el trim solo devuelve páginas libres.
    assert trimmed["rss_peak_mib"] < untrimmed["rss_peak_mib"], (untrimmed, trimmed)


# ITSC-376. Con el lote por defecto (1 M) y el tope por defecto (768 MiB) un tramo cabe solo
# bajo el tope: el estimador ya no cuenta las fronteras que `cover` trae por delante a 305 B.
def test_the_default_lot_fits_under_the_default_cap_and_does_not_warn(dense_runs):
    run = dense_runs["defaults"]
    assert run["lot_events"] == 1_000_000 and run["lots"] > DENSE_GROUPS, run
    assert 0 < run["inflight_alone_bytes_max"] <= run["inflight_bytes"], run
    assert run["warned"] is False, run
