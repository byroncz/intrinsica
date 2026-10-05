"""Un lago local armado con los fixtures de dc_core: L1 y L2 de 2017-08 para 5 θ.

`ticks.csv` pasa a un `consolidated.parquet` con el esquema de L1 y `events_v0.csv`
a `events.parquet` y `carry_over.parquet` con el de L2. El último evento de cada θ
no tiene extremo: es el pendiente, y el carry-over lleva como candidato el tick
de mayor precio (o menor, si baja) desde su confirmación.
"""

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from viz_helpers import FIXTURES
from viz_tiles.lake import (
    CARRY_OVER,
    CONSOLIDATED,
    EVENTS,
    events_rel,
    join,
    landing_rel,
)

DAY = "2017-08-18"
MONTH = (2017, 8)
KEY = ("binance", "spot", "BTCUSDT")
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


def _point(name: str, nullable: bool = False) -> list[pa.Field]:
    return [
        pa.field(f"{name}_price", DEC, nullable=nullable),
        pa.field(f"{name}_time", pa.int64(), nullable=nullable),
        pa.field(f"{name}_agg_trade_id", pa.int64(), nullable=nullable),
    ]


EVENTS_SCHEMA = pa.schema(
    [
        *_point("reference"),
        *_point("confirm"),
        *_point("extreme"),
        pa.field("direction", pa.int8(), nullable=False),
        pa.field("theta", THETA_TYPE, nullable=False),
    ]
)
CARRY_SCHEMA = pa.schema(
    [
        pa.field("provider", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset", pa.string(), nullable=False),
        pa.field("theta", THETA_TYPE, nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("month", pa.int32(), nullable=False),
        pa.field("state_version", pa.string(), nullable=False),
        pa.field("direction", pa.int8(), nullable=False),
        *_point("ext_high"),
        *_point("ext_low"),
        pa.field("has_pending_event", pa.bool_(), nullable=False),
        *_point("pending_reference", nullable=True),
        *_point("pending_confirm", nullable=True),
    ]
)


@dataclass(frozen=True)
class Roots:
    landing: Path
    events: Path
    tiles: Path
    dq: Path

    def env(self) -> dict[str, str]:
        return {
            "VIZ_LANDING_ROOT": str(self.landing),
            "VIZ_EVENTS_ROOT": str(self.events),
            "VIZ_TILES_ROOT": str(self.tiles),
            "VIZ_DQ_ROOT": str(self.dq),
        }


def roots(base: Path) -> Roots:
    return Roots(base / "landing", base / "events", base / "tiles", base / "dq")


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
    """Las filas de `events_v0.csv` por θ (× 10⁸), en orden."""
    by_theta: dict[int, list[dict]] = {}
    with open(FIXTURES / "events_v0.csv") as f:
        for r in csv.DictReader(f):
            by_theta.setdefault(int(r["theta"]), []).append(r)
    return by_theta


def theta_text(theta: int) -> str:
    return f"0.{theta:08d}"


def write_consolidated(
    path: Path, ticks: list[dict], row_group_size: int = 500
) -> None:
    n = len(ticks)
    ids = pa.array([t["id"] for t in ticks], pa.int64())
    table = pa.table(
        {
            "agg_trade_id": ids,
            "price": pa.array([t["price"] for t in ticks], DEC),
            "quantity": pa.array([Decimal(1)] * n, DEC),
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": pa.array([t["time"] for t in ticks], pa.int64()),
            "is_buyer_maker": pa.array([False] * n),
            "is_best_match": pa.array([True] * n),
        },
        schema=L1_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)


def candidate(ticks: list[dict], confirm_id: int, direction: int) -> dict:
    """El tick de mayor precio (menor si baja) desde la confirmación; el primero si empata."""
    after = [t for t in ticks if t["id"] >= confirm_id]
    pick = max if direction == 1 else min
    best = pick(t["price"] for t in after)
    return next(t for t in after if t["price"] == best)


def _decimal(value: str) -> Decimal:
    return Decimal(value)


def write_events(
    path: Path, theta: int, rows: list[dict], row_group_size: int = 3
) -> None:
    """`events.parquet` con las filas que ya tienen extremo, en row groups chicos."""
    table = pa.table(
        {
            "reference_price": pa.array(
                [_decimal(r["reference_price"]) for r in rows], DEC
            ),
            "reference_time": pa.array(
                [int(r["reference_time"]) for r in rows], pa.int64()
            ),
            "reference_agg_trade_id": pa.array(
                [int(r["reference_agg_trade_id"]) for r in rows], pa.int64()
            ),
            "confirm_price": pa.array(
                [_decimal(r["confirm_price"]) for r in rows], DEC
            ),
            "confirm_time": pa.array(
                [int(r["confirm_time"]) for r in rows], pa.int64()
            ),
            "confirm_agg_trade_id": pa.array(
                [int(r["confirm_agg_trade_id"]) for r in rows], pa.int64()
            ),
            "extreme_price": pa.array(
                [_decimal(r["extreme_price"]) for r in rows], DEC
            ),
            "extreme_time": pa.array(
                [int(r["extreme_time"]) for r in rows], pa.int64()
            ),
            "extreme_agg_trade_id": pa.array(
                [int(r["extreme_agg_trade_id"]) for r in rows], pa.int64()
            ),
            "direction": pa.array([int(r["direction"]) for r in rows], pa.int8()),
            "theta": pa.array([Decimal(theta).scaleb(-8)] * len(rows), THETA_TYPE),
        },
        schema=EVENTS_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)


def write_carry_over(
    path: Path,
    theta: int,
    month: tuple[int, int],
    *,
    direction: int,
    extreme: dict,
    pending: tuple[dict, dict] | None,
    other: dict | None = None,
) -> None:
    """`carry_over.parquet` de una fila.

    `extreme` es el extremo vigente de la tendencia (`{id, price, time}`);
    `pending` son la referencia y la confirmación del pendiente (filas del CSV).
    """
    other = other or extreme
    high, low = (extreme, other) if direction == 1 else (other, extreme)

    def point(name: str, p: dict | None, text: bool = False) -> dict:
        if p is None:
            return {
                f"{name}_price": None,
                f"{name}_time": None,
                f"{name}_agg_trade_id": None,
            }
        return {
            f"{name}_price": Decimal(p["price"]),
            f"{name}_time": int(p["time"]),
            f"{name}_agg_trade_id": int(p["id"]),
        }

    ref = conf = None
    if pending is not None:
        r, c = pending
        ref = {
            "price": r["reference_price"],
            "time": r["reference_time"],
            "id": r["reference_agg_trade_id"],
        }
        conf = {
            "price": c["confirm_price"],
            "time": c["confirm_time"],
            "id": c["confirm_agg_trade_id"],
        }
    row = {
        "provider": KEY[0],
        "market": KEY[1],
        "asset": KEY[2],
        "theta": Decimal(theta).scaleb(-8),
        "year": month[0],
        "month": month[1],
        "state_version": "test",
        "direction": direction,
        **point("ext_high", high),
        **point("ext_low", low),
        "has_pending_event": pending is not None,
        **point("pending_reference", ref),
        **point("pending_confirm", conf),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=CARRY_SCHEMA), path)


def events_path(r: Roots, theta: int, month: tuple[int, int], name: str) -> Path:
    return Path(join(r.events, events_rel(*KEY, theta_text(theta), month, name)))


def landing_path(r: Roots, month: tuple[int, int] = MONTH) -> Path:
    return Path(join(r.landing, landing_rel(*KEY, month)))


def build_lake(base: Path) -> tuple[Roots, list[dict], dict[int, dict]]:
    """Arma el lago de 2017-08. Devuelve las raíces, los ticks y por θ el pendiente.

    El pendiente de cada θ: `{"row": fila del CSV, "extreme": tick candidato,
    "direction": dirección}`.
    """
    r = roots(base)
    ticks = read_ticks()
    write_consolidated(landing_path(r), ticks)
    pending: dict[int, dict] = {}
    for theta, rows in read_events().items():
        *closed, last = rows
        assert last["extreme_agg_trade_id"] == ""
        direction = int(last["direction"])
        top = candidate(ticks, int(last["confirm_agg_trade_id"]), direction)
        write_events(events_path(r, theta, MONTH, EVENTS), theta, closed)
        write_carry_over(
            events_path(r, theta, MONTH, CARRY_OVER),
            theta,
            MONTH,
            direction=direction,
            extreme=top,
            pending=(last, last),
        )
        pending[theta] = {"row": last, "extreme": top, "direction": direction}
    return r, ticks, pending


__all__ = [
    "CARRY_OVER",
    "CONSOLIDATED",
    "DAY",
    "EVENTS",
    "KEY",
    "MONTH",
    "Roots",
    "build_lake",
]
