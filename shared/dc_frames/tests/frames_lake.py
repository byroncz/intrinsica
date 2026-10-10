"""Un lago local con los ticks y los eventos de los fixtures de dc_core, para probar el lector.

`ticks.csv` pasa a `consolidated.parquet` con el esquema de L1 y `events_v0.csv` a
`events.parquet` con el de L2. Los datos se eligen para que una trama se pueda comparar
con un oráculo en Python puro: `quantity` vale el `agg_trade_id` del tick (así los ids
de una trama se leen de ella sin que el lector los entregue) e `is_buyer_maker` vale
`id % 2 == 0`. El último evento de cada θ no tiene extremo (es el pendiente) y no
entra a `events.parquet`.
"""

import csv
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FIXTURES = Path(__file__).resolve().parents[3] / "shared/dc_core/tests/fixtures"
KEY = ("binance", "spot", "BTCUSDT")
MONTH = (2017, 8)
DEC = pa.decimal128(18, 8)
THETA_TYPE = pa.decimal128(9, 8)

L1_SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", DEC, nullable=False),
        pa.field("quantity", DEC, nullable=False),
        pa.field("first_trade_id", pa.int64(), nullable=False),
        pa.field("last_trade_id", pa.int64(), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
        pa.field("is_best_match", pa.bool_(), nullable=False),
    ]
)
EVENT_NAMES = ("reference", "confirm", "extreme")
EVENTS_SCHEMA = pa.schema(
    [
        *(
            pa.field(f"{name}_{field}", kind, nullable=False)
            for name in EVENT_NAMES
            for field, kind in (
                ("price", DEC),
                ("time", pa.int64()),
                ("agg_trade_id", pa.int64()),
            )
        ),
        pa.field("direction", pa.int8(), nullable=False),
        pa.field("theta", THETA_TYPE, nullable=False),
    ]
)


def read_ticks() -> list[dict]:
    with open(FIXTURES / "ticks.csv") as f:
        return [
            {
                "id": int(r["agg_trade_id"]),
                "price": Decimal(r["price"]),
                "time": int(r["transact_time"]),
            }
            for r in csv.DictReader(f)
        ]


def read_events() -> dict[int, list[dict]]:
    """Las filas cerradas de `events_v0.csv` por θ (× 10⁸), en orden; sin el pendiente."""
    by_theta: dict[int, list[dict]] = {}
    with open(FIXTURES / "events_v0.csv") as f:
        for r in csv.DictReader(f):
            by_theta.setdefault(int(r["theta"]), []).append(r)
    closed = {}
    for theta, rows in by_theta.items():
        *head, last = rows
        assert last["extreme_agg_trade_id"] == ""
        closed[theta] = head
    return closed


def theta_text(theta: int) -> str:
    return f"0.{theta:08d}"


def is_buyer_maker(tick_id: int) -> bool:
    return tick_id % 2 == 0


def partition(root: Path, month: tuple[int, int]) -> Path:
    return (
        root
        / f"provider={KEY[0]}/market={KEY[1]}/asset={KEY[2]}"
        / f"year={month[0]:04d}/month={month[1]:02d}"
    )


def write_l1(
    root: Path, ticks: list[dict], row_group_size: int, month: tuple[int, int] = MONTH
) -> Path:
    ids = pa.array([t["id"] for t in ticks], pa.int64())
    table = pa.table(
        {
            "agg_trade_id": ids,
            "price": pa.array([t["price"] for t in ticks], DEC),
            "quantity": pa.array([Decimal(t["id"]) for t in ticks], DEC),
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": pa.array([t["time"] for t in ticks], pa.int64()),
            "is_buyer_maker": pa.array([is_buyer_maker(t["id"]) for t in ticks]),
            "is_best_match": pa.array([True] * len(ticks)),
        },
        schema=L1_SCHEMA,
    )
    path = partition(root, month) / "consolidated.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def write_l2(
    root: Path,
    theta: int,
    rows: list[dict],
    row_group_size: int,
    month: tuple[int, int] = MONTH,
) -> Path:
    def column(name: str, kind: pa.DataType, cast) -> pa.Array:
        return pa.array([cast(r[name]) for r in rows], kind)

    arrays = []
    for field in EVENTS_SCHEMA:
        if field.name == "theta":
            arrays.append(pa.array([Decimal(theta).scaleb(-8)] * len(rows), field.type))
        elif field.name == "direction":
            arrays.append(column("direction", field.type, int))
        elif pa.types.is_decimal(field.type):
            arrays.append(column(field.name, field.type, Decimal))
        else:
            arrays.append(column(field.name, field.type, int))
    path = (
        partition(root, month).parent.parent
        / f"theta={theta_text(theta)}"
        / f"year={month[0]:04d}/month={month[1]:02d}/events.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_arrays(arrays, schema=EVENTS_SCHEMA),
        path,
        row_group_size=row_group_size,
    )
    return path


SECOND_MONTH = (2017, 9)


def boundary_ids(row: dict) -> tuple[int, int, int]:
    return tuple(int(row[f"{name}_agg_trade_id"]) for name in EVENT_NAMES)


def build_two_month_lake(
    base: Path, l1_row_group: int = 50, l2_row_group: int = 3
) -> tuple[Path, Path, list[dict], dict[int, list[dict]], int, set[tuple[int, int]]]:
    """L1 y L2 repartidos en `MONTH` y `SECOND_MONTH`, cortados por la mitad de los ticks.

    Devuelve `(raíz L1, raíz L2, ticks, eventos por θ, corte, tardíos)`. Los ticks con
    id ≤ `corte` están en el primer mes y los demás en el segundo. Un evento va a la
    partición del mes que lo cierra: la del segundo si su extremo supera el corte, y
    además el último evento de cada θ que termina antes del corte, que ADR-L2-06
    permite (la partición es el mes que confirma al siguiente, no el del extremo).
    `tardíos` son sus `(θ, confirm_agg_trade_id)`: eventos cuyo extremo es anterior al
    primer tick del mes de su partición.
    """
    ticks, events = read_ticks(), read_events()
    cut = ticks[len(ticks) // 2]["id"]
    l1_root, l2_root = base / "l1", base / "l2"
    write_l1(l1_root, [t for t in ticks if t["id"] <= cut], l1_row_group, MONTH)
    write_l1(l1_root, [t for t in ticks if t["id"] > cut], l1_row_group, SECOND_MONTH)
    late: set[tuple[int, int]] = set()
    for theta, rows in events.items():
        before = [r for r in rows if boundary_ids(r)[2] <= cut]
        after = [r for r in rows if boundary_ids(r)[2] > cut]
        late.add((theta, boundary_ids(before[-1])[1]))
        write_l2(l2_root, theta, before[:-1], l2_row_group, MONTH)
        write_l2(l2_root, theta, [before[-1], *after], l2_row_group, SECOND_MONTH)
    return l1_root, l2_root, ticks, events, cut, late


def build_lake(
    base: Path,
    l1_row_group: int = 500,
    l2_row_group: int = 3,
    ticks: list[dict] | None = None,
) -> tuple[Path, Path, list[dict], dict[int, list[dict]]]:
    """Arma L1 y L2 de 2017-08. Devuelve `(raíz L1, raíz L2, ticks, eventos por θ)`."""
    ticks = ticks if ticks is not None else read_ticks()
    events = read_events()
    l1_root, l2_root = base / "l1", base / "l2"
    write_l1(l1_root, ticks, l1_row_group)
    for theta, rows in events.items():
        write_l2(l2_root, theta, rows, l2_row_group)
    return l1_root, l2_root, ticks, events


FAMILY_SCHEMA = pa.schema(
    [
        pa.field("theta", THETA_TYPE, nullable=False),
        pa.field("confirm_agg_trade_id", pa.int64(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
        pa.field("w", DEC),
    ]
)


def family_batches(skeleton_path: str | Path) -> list[pa.RecordBatch]:
    """Un lote por row group del esqueleto, con sus claves y dos columnas derivadas.

    `n` vale el doble del id de confirmación y `w` lo mismo en decimal, con un nulo cada
    cuatro filas: una familia mínima que el escritor puede repetir sobre cualquier
    `events.parquet` o `summaries.parquet`.
    """
    parquet = pq.ParquetFile(skeleton_path)
    batches = []
    for i in range(parquet.num_row_groups):
        table = parquet.read_row_group(i, columns=["theta", "confirm_agg_trade_id"])
        ids = table.column("confirm_agg_trade_id").to_pylist()
        batches.append(
            pa.RecordBatch.from_pydict(
                {
                    "theta": table.column("theta").combine_chunks(),
                    "confirm_agg_trade_id": pa.array(ids, pa.int64()),
                    "n": pa.array([2 * i for i in ids], pa.int64()),
                    "w": pa.array(
                        [
                            None if n % 4 == 0 else Decimal(2 * i)
                            for n, i in enumerate(ids)
                        ],
                        DEC,
                    ),
                },
                schema=FAMILY_SCHEMA,
            )
        )
    return batches
