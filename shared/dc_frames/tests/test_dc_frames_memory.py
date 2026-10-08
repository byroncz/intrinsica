"""Memoria del lector (TRD-L3 §7.4): un evento que cruza row groups se copia y suelta el row group.

La prueba mira los buffers de las tramas y no `pa.total_allocated_bytes()` a mitad de
la lectura: el lector de Parquet decodifica con hilos y suelta sus buffers con retraso,
así que ese contador oscila en varios MB.
"""

import gc
from decimal import Decimal

import pyarrow as pa
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
