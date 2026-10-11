"""El pico de memoria de una pasada es O(row group) y no crece con los eventos (TRD-L3 §13.9, RNF-L3-01).

Mismo patrón que `test_dc_frames_memory.py`: el pico de Arrow se mide en un proceso
aparte, porque `max_memory()` del pool por defecto es el máximo desde que arrancó.
"""

import json
import subprocess
import sys
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from frames_lake import (
    DEC,
    L1_SCHEMA,
    MONTH,
    partition,
    write_carry_over,
    write_l2,
)

ROW_GROUP = 100_000
# Bytes de un row group decodificado: id, tiempo, precio y cantidad (8 + 8 + 16 + 16 B).
ROW_GROUP_BYTES = ROW_GROUP * 48
GROUPS = 14
LONG = 10 * ROW_GROUP
THETAS = (100_000, 200_000)

PROBE = """
import json, sys
import pyarrow as pa
from l3_dc_descriptors import summarize_month

l1, l2, l3 = sys.argv[1:4]
results = summarize_month(
    l1, l2, l3, (2017, 8), ["0.001", "0.002"], series_start=(2017, 8)
)
print(json.dumps({
    "rows": sum(r.rows for r in results.values()),
    "ticks": sum(r.reconciliation.closed for r in results.values()),
    "peak": pa.default_memory_pool().max_memory(),
}))
"""


def write_flat_l1(root, ticks: int) -> None:
    """Precio y cantidad 1 en todos los ticks, tiempo = id: una fase suma lo que mide."""
    ids = pa.array(np.arange(ticks, dtype=np.int64))
    one = pa.array([Decimal(1)] * ticks, DEC)
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
    pq.write_table(table, path, row_group_size=ROW_GROUP)


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


def chain(start: int, small: int) -> list[dict]:
    """`small` eventos dentro de un row group y uno de 10 row groups, encadenados:
    la referencia de cada uno es el extremo del anterior (sin ticks sin dueño).
    """
    rows, position = [], start
    for _ in range(small):
        rows.append(event(position, position + 50, position + 80))
        position += 80
    rows.append(event(position, position + LONG, position + LONG + 5))
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
    assert many["ticks"] > few["ticks"] > 2 * LONG
    # O(row group): unos pocos row groups (el del pase, el anterior, el que se relee
    # en el evento largo y los temporales de la agregación), muy por debajo de los 10
    # row groups del evento largo.
    assert few["peak"] < 8 * ROW_GROUP_BYTES
    assert many["peak"] < 8 * ROW_GROUP_BYTES
    # Casi veinte veces más eventos no suben el pico.
    assert many["peak"] < 1.25 * few["peak"]
