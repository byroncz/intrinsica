"""Memoria del lector (TRD-L3 §7.4): un evento que cruza row groups se copia y suelta el row group.

La prueba mira los buffers de las tramas y no `pa.total_allocated_bytes()` a mitad de
la lectura: el lector de Parquet decodifica con hilos y suelta sus buffers con retraso,
así que ese contador oscila en varios MB.
"""

import gc
import json
import subprocess
import sys
from decimal import Decimal

import pyarrow as pa
import pytest
from dc_frames import read_frames
from frames_lake import MONTH, write_l1, write_l2

ROW_GROUP = 100_000
# Bytes de un row group decodificado: id, tiempo, precio y cantidad (8 + 8 + 16 + 16 B).
ROW_GROUP_BYTES = ROW_GROUP * 48


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


def test_crossing_event_is_copied_and_the_rest_are_slices(tmp_path):
    ticks = [{"id": i, "price": Decimal(1), "time": i} for i in range(4 * ROW_GROUP)]
    write_l1(tmp_path / "l1", ticks, ROW_GROUP)
    rows = [
        event(ROW_GROUP - 20, ROW_GROUP + 50, ROW_GROUP + 60),  # cruza el borde 0|1
        event(
            ROW_GROUP + 60, ROW_GROUP + 100, ROW_GROUP + 110
        ),  # dentro del row group 1
        event(2 * ROW_GROUP - 10, 3 * ROW_GROUP + 5, 3 * ROW_GROUP + 9),  # cruza tres
    ]
    write_l2(tmp_path / "l2", 100000, rows, 100)
    baseline = pa.total_allocated_bytes()

    crossing, inside, long = read_frames(
        "0.001", tmp_path / "l1", tmp_path / "l2", MONTH, MONTH
    )
    assert [len(e.confirmation) for e in (crossing, inside, long)] == [
        70,
        40,
        ROW_GROUP + 15,
    ]
    for frame in (crossing.confirmation, long.confirmation):
        for column in (frame.transact_time, frame.price, frame.quantity):
            # Un buffer propio del tamaño de la trama, no un trozo de un row group.
            assert column.buffers()[1].size <= len(frame) * 16 + 64
    for column in (inside.confirmation.transact_time, inside.confirmation.price):
        # Un slice sin copia: su buffer es el del row group completo.
        assert column.buffers()[1].size >= ROW_GROUP * 8

    # Al soltar los eventos no queda ningún row group retenido.
    del crossing, inside, long, frame, column
    gc.collect()  # el contador de Arrow se actualiza al soltar los ciclos pendientes
    assert pa.total_allocated_bytes() - baseline < 0.5 * ROW_GROUP_BYTES


# El pico de memoria de Arrow se mide en un proceso aparte: `max_memory()` del pool
# por defecto es el máximo desde que arrancó el proceso, y el lago ya lo habría subido.
PROBE = """
import json, sys
import pyarrow as pa
import pytest
from dc_frames import read_frames

l1, l2, chunks = sys.argv[1], sys.argv[2], sys.argv[3] == "chunks"
ticks = 0
for e in read_frames(["0.001", "0.002"], l1, l2, (2017, 8), (2017, 8), chunks=chunks):
    if chunks:
        ticks += sum(len(c) for c in e.confirmation) + sum(len(c) for c in e.overshoot)
    else:
        ticks += len(e.confirmation) + len(e.overshoot)
print(json.dumps({"ticks": ticks, "peak": pa.default_memory_pool().max_memory()}))
"""
GROUPS = 14
LONG = 10 * ROW_GROUP


def theta_events(shift: int, small: int) -> list[dict]:
    """`small` eventos dentro de un row group y uno de 10 row groups, tras ellos."""
    rows, position = [], 100 + shift
    for _ in range(small):
        rows.append(event(position, position + 50, position + 80))
        position += 100
    rows.append(event(position, position + LONG, position + LONG + 5))
    return rows


def peak(root, l2: str, mode: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(root / "l1"), str(root / l2), mode],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_chunks_peak_is_a_row_group_and_does_not_grow_with_events(tmp_path):
    ticks = [
        {"id": i, "price": Decimal(1), "time": i} for i in range(GROUPS * ROW_GROUP)
    ]
    write_l1(tmp_path / "l1", ticks, ROW_GROUP)
    for name, small in (("few", 10), ("many", 100)):
        # Dos θ en fan-out; cada uno con un evento que cruza 10 row groups.
        write_l2(tmp_path / name, 100000, theta_events(0, small), 100)
        write_l2(tmp_path / name, 200000, theta_events(7, small), 100)

    few, many = peak(tmp_path, "few", "chunks"), peak(tmp_path, "many", "chunks")
    whole = peak(tmp_path, "many", "event")
    assert many["ticks"] > few["ticks"] > 2 * LONG
    # O(row group): unos pocos row groups (el del pase, el anterior, el que se relee
    # en el evento largo y los buffers de decodificación), muy por debajo de los 10
    # row groups del evento largo.
    assert few["peak"] < 8 * ROW_GROUP_BYTES
    assert many["peak"] < 8 * ROW_GROUP_BYTES
    # Diez veces más eventos no suben el pico.
    assert many["peak"] < 1.25 * few["peak"]
    # El modo por evento, en cambio, concatena el evento largo.
    assert whole["ticks"] == many["ticks"]
    assert whole["peak"] > 15 * ROW_GROUP_BYTES


# La entrega por row group: un row group en vuelo y ninguno retenido, con cualquier
# número de eventos y aunque uno abarque 10 row groups. El consumidor suelta el elemento
# antes de pedir el siguiente.
#
# El lector de Parquet decodifica las columnas de un row group con hilos y suelta sus
# buffers con retraso; el máximo del pool depende entonces de cuántos se solapan. En 25
# corridas por lago (ITSC-333) el pico de `few` fue de 11,4 a 13,0 MB (cota: 14,4) y el de
# `many`, de 11,4 a 17,8 MB: la prueba fallaba 1 de cada 3 sin que el código cambiara. Con
# un solo hilo de CPU y de E/S el pico es el mismo en todas las corridas (10,1 MB, razón
# 1,002), y lo que se afirma (O(row group), sin crecer con los eventos) sigue a la vista.
ROW_GROUPS_PROBE = """
import json, sys
import pyarrow as pa
import pytest
from dc_frames import read_row_groups

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
l1, l2, readers = sys.argv[1], sys.argv[2], int(sys.argv[3])
ticks = events = 0
for delivery in read_row_groups(
    ["0.001", "0.002"], l1, l2, (2017, 8), (2017, 8), readers=readers
):
    ticks += len(delivery.frame)
    events += sum(len(t) for t in delivery.thetas)
    del delivery
print(json.dumps({
    "ticks": ticks, "events": events, "peak": pa.default_memory_pool().max_memory()
}))
"""


def row_groups_peak(root, l2: str, readers: int = 1) -> dict:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            ROW_GROUPS_PROBE,
            str(root / "l1"),
            str(root / l2),
            str(readers),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def decoded_row_groups(readers: int) -> int:
    """Cuántos row groups decodificados conviven con `readers` lectores.

    Con 1 el lector es secuencial: el que el consumidor tiene en la mano, y nada más.
    Con R > 1 hay R en camino (los que ya terminaron esperan su turno) más el de la mano.
    """
    return 1 if readers == 1 else readers + 1


@pytest.mark.parametrize("readers", [1, 2, 3])
def test_row_groups_peak_is_a_row_group_and_does_not_grow_with_events(
    tmp_path, readers
):
    ticks = [
        {"id": i, "price": Decimal(1), "time": i} for i in range(GROUPS * ROW_GROUP)
    ]
    write_l1(tmp_path / "l1", ticks, ROW_GROUP)
    for name, small in (("few", 10), ("many", 100)):
        write_l2(tmp_path / name, 100000, theta_events(0, small), 100)
        write_l2(tmp_path / name, 200000, theta_events(7, small), 100)

    few = row_groups_peak(tmp_path, "few", readers)
    many = row_groups_peak(tmp_path, "many", readers)
    # El pase se detiene en el row group donde cierra el último evento (el 10); cada tick
    # se entrega una vez y los eventos largos salen en cada row group que cruzan.
    assert few["ticks"] == many["ticks"] > LONG
    assert many["events"] > few["events"] > 2 * GROUPS
    # Los row groups decodificados que conviven más ~1,5 de buffers de decodificación de
    # lo que se lee (un solo read_row_group pica en ~2,1 row groups de 48 B por tick con
    # un hilo, o sea 1 + ~1,1). Con readers = 1 son 2,5; retener la entrega anterior sube
    # el pico a ~2,9: con 2,5 la prueba lo atrapa y aún deja 18 % de margen. Con R > 1 la
    # cuenta es R + 1 decodificados y medimos R + 2,1 (R = 2: 4,1; R = 3: 5,1), así que el
    # umbral es R + 2,5 y el margen sigue siendo de ~0,4.
    limit = (decoded_row_groups(readers) + 1.5) * ROW_GROUP_BYTES
    assert few["peak"] < limit
    assert many["peak"] < limit
    if readers == 1:
        chunks = peak(tmp_path, "many", "chunks")
        # Diez veces más eventos no suben el pico.
        assert many["peak"] < 1.25 * few["peak"]
        # Sin el row group anterior ni relecturas, por debajo del modo por chunks.
        assert many["peak"] < chunks["peak"]
