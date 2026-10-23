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

PROBE = """
import json, sys, tracemalloc
import pyarrow as pa
from l3_dc_descriptors import summarize_month

tracemalloc.start()

l1, l2, l3 = sys.argv[1:4]
results = summarize_month(
    l1, l2, l3, (2017, 8), ["0.001", "0.002"], series_start=(2017, 8)
)
print(json.dumps({
    "rows": sum(r.rows for r in results.values()),
    "ticks": sum(r.reconciliation.closed for r in results.values()),
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
    out.append({k: details[k] for k in ("rss_after_trim_mib", "rss_peak_mib")})
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
# corridas) también cabe, y con él fue de 61 a 140. Solo vale para estas 10 unidades; con 60 el
# RSS tras el trim sigue subiendo ≈ 2 MiB por unidad (ADR-L3-16) y aquí no se afirma lo contrario.
UNITS_IN_FLIGHT_BYTES = (6 + 6) * UNITS_ROW_GROUP_BYTES
# Lo que detecta la falta del trim es el nivel del RSS en las últimas unidades, con y sin trim
# en la misma sesión: tres row groups (≈ 55 MiB) menos con el trim. En 16 corridas seguidas
# (10 vCPU, 8 GiB) la diferencia fue de 100 a 133 MiB: el margen es de ≈ 45 MiB contra los
# ± 15 MiB de ruido de una unidad, que al promediar cinco bajan a ≈ 7. Se comparan una contra
# otra, en la misma máquina, y no contra un número absoluto.
UNITS_TRIM_MIN_GAIN = 3 * UNITS_ROW_GROUP_BYTES
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
    # Con row groups en vuelo que dependen de la carrera entre el lector y el motor, el
    # pico se mueve unos MiB entre corridas; 30 row groups más en el evento sumarían ≈ 140 MB.
    assert longest["peak"] < long["peak"] + 2 * ROW_GROUP_BYTES
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


def test_rss_after_closing_each_unit_does_not_grow_with_the_units(unit_runs):
    """ITSC-371: con el asignador de la imagen glibc no devuelve al kernel lo liberado y, sin
    hacerlo al cerrar la unidad, el RSS crece unidad a unidad y un backfill largo muere por
    memoria (la causa exacta es una hipótesis, ver TRD-L3 ADR-L3-16). Las pruebas de arriba
    miden una unidad y no lo ven.
    """
    runs = unit_runs["trim"]
    # Lo que queda retenido no pasa de lo que la unidad tuvo en vuelo; ver UNITS_IN_FLIGHT_BYTES.
    assert drift(runs) <= UNITS_IN_FLIGHT_BYTES, runs
    # Y el trim recupera de verdad: las últimas unidades quedan más abajo que sin él; ver
    # UNITS_TRIM_MIN_GAIN. Esta comparación es la que falla cuando falta el trim.
    gain = tail_level(unit_runs["no-trim"]) - tail_level(runs)
    assert gain > UNITS_TRIM_MIN_GAIN, unit_runs
    # El pico de la última unidad lo acota la unidad más grande, no la suma de las anteriores.
    peaks = [r["rss_peak_mib"] for r in runs]
    assert peaks[-1] <= 1.3 * max(peaks[:-1]), peaks


def test_without_the_trim_the_rss_does_grow(unit_runs):
    """Contraprueba de la anterior: sin el trim el RSS al cerrar sí sube, así que la
    comparación de arriba no pasa en vacío.
    """
    assert drift(unit_runs["no-trim"]) > UNITS_TRIM_MIN_GAIN, unit_runs


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
# El pico baja al menos esta fracción de lo que cuestan las casillas de más. Medido (3 corridas, hilos
# del motor = 1): baja entre 0,50 y 0,56 de esa cuenta (225 a 260 MiB de 446 MiB), así que 0,3 deja
# un margen de casi la mitad bajo el mínimo medido.
DENSE_MIN_GAIN = 0.3

DENSE_UNIT = """
import json, sys
from l3_dc_descriptors import pipeline
from l3_dc_descriptors.context import RunContext, Unit

l1, l2, l3, cap, count, engine_threads = sys.argv[1:7]
ctx = RunContext(
    mode="backfill", run_id="run", image_version="test", series_start=(2017, 8),
    landing_root=l1, events_root=l2, summaries_root=l3, dq_root=f"{l3}/dq",
    threads=8, readers=2, inflight=4, writers=1, engine_threads=int(engine_threads),
    inflight_bytes=int(cap),
)
result = pipeline.process_unit(Unit(2017, 8), ctx, [100_000 * k for k in range(1, int(count) + 1)])
assert result.ok, result.error
details = result.timing.details()
keys = ("rss_peak_mib", "inflight_events_max", "inflight_bytes_max", "inflight_bytes", "events")
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
    """La misma unidad densa con K = 4 solo (sin tope de bytes) y con el tope de bytes."""
    root = tmp_path_factory.mktemp("dense")
    ticks = DENSE_GROUPS * DENSE_GROUP_TICKS
    write_flat_l1(root / "l1", ticks, DENSE_GROUP_TICKS)
    for k in range(1, DENSE_THETAS + 1):
        write_dense_theta(root / "events", 100_000 * k, ticks, 100 + 7 * k)
    runs = {}
    for name, cap in (("k_only", 1 << 40), ("bytes", DENSE_CAP)):
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
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **UNITS_ENV},
        )
        assert out.returncode == 0, out.stderr
        runs[name] = json.loads(out.stdout)
    return runs


def test_the_byte_cap_bounds_the_open_events_and_the_peak_rss(dense_runs):
    k_only, capped = dense_runs["k_only"], dense_runs["bytes"]
    assert k_only["events"] == capped["events"] > 7_000_000
    one_group = k_only["events"] // DENSE_GROUPS
    # Con el tope, lo abierto es un row group (más lo que `cover` trae por delante), no el mes.
    assert capped["inflight_events_max"] <= 1.2 * one_group, dense_runs
    # Sin él, con un motor más lento que el lector, se acumulan los row groups: es el problema 2
    # tal como lo describe la ADR (O(row groups en vuelo)), y el tope lo evita.
    assert k_only["inflight_events_max"] >= 1.5 * one_group, dense_runs
    # El RSS del pico baja lo que cuestan las casillas de más (SLOT_BYTES por evento).
    extra = k_only["inflight_events_max"] - capped["inflight_events_max"]
    gain = (k_only["rss_peak_mib"] - capped["rss_peak_mib"]) * MIB
    assert gain >= DENSE_MIN_GAIN * extra * SLOT_BYTES, dense_runs


def test_a_row_group_over_the_cap_is_recorded_and_the_unit_goes_on(dense_runs):
    """Con 120 MiB un solo row group denso pasa del tope: `inflight_bytes_max` lo dice y la
    unidad terminó (límite conocido, TRD-L3 ADR-L3-16).
    """
    capped = dense_runs["bytes"]
    assert capped["inflight_bytes_max"] > capped["inflight_bytes"] == DENSE_CAP
