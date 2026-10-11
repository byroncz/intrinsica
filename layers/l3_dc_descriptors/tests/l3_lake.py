"""Un lago sintético para el resumidor: ticks reales, eventos del detector real.

Los ticks son los de `ticks.csv` (con cantidades y lados al azar, de semilla fija) y los
eventos y el carry-over de cada mes salen de `dc_pyo3`, el mismo detector de L2, mes a
mes y con `from_carry_over`, como corre L2. Así los eventos cumplen los invariantes de
TRD-L3 §7.2 por construcción y la conciliación de §9.1 tiene un carry-over verdadero.
"""

import itertools
import random
from array import array
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import dc_pyo3
import pyarrow as pa
import pyarrow.parquet as pq
from frames_lake import (
    KEY,
    L1_SCHEMA,
    partition,
    read_ticks,
    write_carry_over,
    write_l2,
)

THETAS = (100_000, 250_000, 500_000, 1_000_000, 2_000_000)
DEC = pa.decimal128(18, 8)
M1, M2, M3 = (2017, 8), (2017, 9), (2017, 10)


@dataclass(frozen=True)
class Tick:
    id: int
    time: int
    price: Decimal
    quantity: Decimal
    sold: bool  # is_buyer_maker


def fixture_ticks(
    seed: int = 7, drop: float = 0.0, price_factor: int = 1, quantity: int = 3
) -> list[Tick]:
    """Los ticks reales con cantidades de hasta `quantity` BTC y lado al azar.

    Con `drop` se borra esa fracción de ticks (menos el primero): deja huecos de
    `agg_trade_id`, también junto a las fronteras de los eventos. `price_factor`
    multiplica los precios (los eventos no cambian).
    """
    rng = random.Random(seed)
    ticks = []
    for i, t in enumerate(read_ticks()):
        if i and rng.random() < drop:
            continue
        amount = Decimal(rng.randint(1, quantity * 100_000_000)).scaleb(-8)
        ticks.append(
            Tick(
                t["id"],
                t["time"],
                t["price"] * price_factor,
                amount,
                rng.random() < 0.5,
            )
        )
    return ticks


def month_cuts(ticks: list[Tick], *positions: int) -> list[int]:
    """Posiciones de corte entre meses, corridas hasta un cambio de `transact_time`.

    El mes de un tick lo define su tiempo, así que un grupo de empate nunca cruza un
    corte.
    """
    cuts = []
    for position in positions:
        while ticks[position].time == ticks[position - 1].time:
            position += 1
        cuts.append(position)
    return cuts


def write_ticks(
    root: Path, ticks: list[Tick], month: tuple[int, int], row_group: int
) -> Path:
    ids = pa.array([t.id for t in ticks], pa.int64())
    table = pa.table(
        {
            "agg_trade_id": ids,
            "price": pa.array([t.price for t in ticks], DEC),
            "quantity": pa.array([t.quantity for t in ticks], DEC),
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": pa.array([t.time for t in ticks], pa.int64()),
            "is_buyer_maker": pa.array([t.sold for t in ticks]),
            "is_best_match": pa.array([True] * len(ticks)),
        },
        schema=L1_SCHEMA,
    )
    path = partition(root, month) / "consolidated.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group)
    return path


def _columns(ticks: list[Tick]):
    prices = b"".join(int(t.price.scaleb(8)).to_bytes(16, "little") for t in ticks)
    return (
        memoryview(prices),
        memoryview(array("q", [t.time for t in ticks])),
        memoryview(array("q", [t.id for t in ticks])),
    )


def _row(event: dc_pyo3.Event) -> dict:
    row = {"direction": event.direction}
    for name in ("reference", "confirm", "extreme"):
        price, time, tick_id = getattr(event, name)
        row[f"{name}_price"] = Decimal(price).scaleb(-8)
        row[f"{name}_time"] = time
        row[f"{name}_agg_trade_id"] = tick_id
    return row


@dataclass
class Lake:
    l1: Path
    l2: Path
    l3: Path
    months: list[tuple[int, int]]
    ticks: dict[tuple[int, int], list[Tick]]
    thetas: list[int]
    events: dict[tuple[int, tuple[int, int]], list[dict]]  # (θ, mes) → filas de L2
    carry: dict[tuple[int, tuple[int, int]], dc_pyo3.CarryOver]

    @property
    def theta_texts(self) -> list[str]:
        return [f"{Decimal(t).scaleb(-8):.8f}" for t in self.thetas]

    def all_ticks(self) -> list[Tick]:
        return [t for month in self.months for t in self.ticks[month]]


def build_lake(
    base: Path,
    ticks: list[Tick],
    cuts: list[int] = (),
    thetas: tuple[int, ...] = THETAS,
    l1_row_group: int = 400,
    l2_row_group: int = 5,
    first_month: tuple[int, int] = M1,
) -> Lake:
    """Escribe L1 (un archivo por mes), L2 (eventos y carry-over por θ y mes).

    `cuts` son posiciones en `ticks` (ver `month_cuts`): el mes `i` va de la posición
    `cuts[i-1]` a `cuts[i]`. Los eventos de un mes son los que el detector cierra
    mientras lo procesa (ADR-L2-06).
    """
    year, number = first_month
    edges = [0, *cuts, len(ticks)]
    months, by_month = [], {}
    for start, stop in itertools.pairwise(edges):
        month = (year, number)
        months.append(month)
        by_month[month] = ticks[start:stop]
        year, number = (year + 1, 1) if number == 12 else (year, number + 1)
    lake = Lake(
        base / "l1", base / "l2", base / "l3", months, by_month, list(thetas), {}, {}
    )

    fanout = dc_pyo3.FanOut(list(thetas))
    for month in months:
        batch = by_month[month]
        if batch:
            write_ticks(lake.l1, batch, month, l1_row_group)
        closed = fanout.feed_batch(*_columns(batch)) if batch else [[] for _ in thetas]
        tail = fanout.finish()
        for i, theta in enumerate(thetas):
            events = [*closed[i], *([tail[i]] if tail[i] is not None else [])]
            rows = [_row(e) for e in events]
            lake.events[theta, month] = rows
            write_l2(lake.l2, theta, rows, l2_row_group, month)
        for theta, carry in zip(thetas, fanout.carry_overs(), strict=True):
            lake.carry[theta, month] = carry
            pending = carry.pending
            write_carry_over(
                lake.l2,
                theta,
                month,
                carry.direction,
                pending_reference=pending[0][2] if pending else None,
                ext_high=carry.ext_high[2],
                ext_low=carry.ext_low[2],
            )
    return lake


__all__ = ["KEY", "M1", "M2", "M3", "THETAS", "Lake", "Tick", "build_lake"]
