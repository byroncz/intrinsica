"""La fila de `summaries.parquet` de un evento cerrado (TRD-L3 §7.2, §8.1 paso 6).

`build_row` es el cierre escalar de un evento: recibe sus dos fases ya agregadas, comprueba
las cantidades y los invariantes de fila contra L2 y arma la fila. Lo usan el resumidor,
para el evento que cruza row groups y para explicar el primer fallo de un lote (el cierre
vectorizado de `rowgroup` solo decide si el lote está sano), y el oráculo por evento.
"""

from typing import NamedTuple

from dc_frames.lake import Month

from l3_dc_descriptors.errors import L2StateInconsistent
from l3_dc_descriptors.phase import PhaseSummary, close_phase, scaled
from l3_dc_descriptors.schema import COLUMN, WIDTH

# Las ocho columnas de L2 que L3 usa (§7.1). `reference_price` es el ancla de la fase de
# confirmación de la familia T; S no la lee.
EVENT_COLUMNS = (
    "reference_agg_trade_id",
    "confirm_agg_trade_id",
    "confirm_time",
    "extreme_agg_trade_id",
    "direction",
    "confirm_price",
    "extreme_price",
    "reference_price",
)


class EventRow(NamedTuple):
    """La fila de un evento y las dos fronteras de L2 que la conciliación necesita."""

    values: list[int]
    reference: int
    extreme: int


def build_row(
    theta: str, month: Month, l2: dict, confirmation, overshoot, instant
) -> EventRow:
    """Cierra las dos fases del evento y arma su fila, de `WIDTH` enteros.

    `l2` es la fila de L2 (`EVENT_COLUMNS`); `confirmation` y `overshoot` traen los
    agregados de cada fase y `instant` los del instante de confirmación (los atributos
    que usa `close_phase` y `ticks`, `high` y `low`). Los decimales quedan sin escalar
    (10⁻⁸) y los nulos del overshoot vacío, en 0. Lanza `PhaseQuantityMismatch` si las
    cantidades de una fase no cuadran y `L2StateInconsistent` si la fila contradice a los
    ticks.
    """
    reference = l2["reference_agg_trade_id"]
    confirm = l2["confirm_agg_trade_id"]
    extreme = l2["extreme_agg_trade_id"]
    direction = l2["direction"]
    if direction not in (1, -1):
        raise L2StateInconsistent.mismatch(
            theta, month, "direction", direction, "1 o -1"
        )

    c = close_phase(confirmation, theta, confirm, "confirmación")
    o = close_phase(overshoot, theta, confirm, "overshoot")

    up = direction == 1
    best, worst = (instant.high, instant.low) if up else (instant.low, instant.high)
    _check_invariants(
        theta,
        month,
        up=up,
        c=c,
        o=o,
        instant=(instant.ticks, best, worst),
        l2={
            "R": reference,
            "C": confirm,
            "E": extreme,
            "confirm_price": scaled(l2["confirm_price"]),
            "extreme_price": scaled(l2["extreme_price"]),
        },
    )

    row = [0] * WIDTH
    row[COLUMN["confirm_agg_trade_id"]] = confirm
    row[COLUMN["c_ticks"]] = c.ticks
    row[COLUMN["c_quantity_buy"]] = c.buy
    row[COLUMN["c_quantity_sell"]] = c.sell
    row[COLUMN["c_price_max"]] = c.high
    row[COLUMN["c_price_min"]] = c.low
    row[COLUMN["c_duration"]] = c.duration
    row[COLUMN["c_vwap"]] = c.vwap
    row[COLUMN["os_ticks"]] = o.ticks
    row[COLUMN["os_quantity_buy"]] = o.buy
    row[COLUMN["os_quantity_sell"]] = o.sell
    if o.ticks:
        row[COLUMN["os_price_max"]] = o.high
        row[COLUMN["os_price_min"]] = o.low
        row[COLUMN["os_duration"]] = o.duration
        row[COLUMN["os_vwap"]] = o.vwap
    row[COLUMN["confirm_instant_ticks"]] = instant.ticks
    row[COLUMN["confirm_instant_price_best"]] = best
    row[COLUMN["confirm_instant_price_worst"]] = worst
    return EventRow(row, reference, extreme)


def _check_invariants(
    theta: str,
    month: Month,
    *,
    up: bool,
    c: PhaseSummary,
    o: PhaseSummary,
    instant: tuple[int, int | None, int | None],
    l2: dict[str, int],
) -> None:
    """Los invariantes 1 a 7 de §7.2 contra la fila de L2 (§9.2). El 8 lo cierra
    `close_phase`.
    """

    def expect(ok: bool, field: str, got: object, wanted: object) -> None:
        if not ok:
            raise L2StateInconsistent.mismatch(theta, month, field, got, wanted)

    ticks, best, worst = instant
    price = l2["confirm_price"]
    # 1. c_ticks ≥ confirm_instant_ticks ≥ 1.
    expect(1 <= ticks <= c.ticks, "confirm_instant_ticks", ticks, f"1..{c.ticks}")
    # 2. peor ≤ confirm_price ≤ mejor en un upturn; al revés en un downturn.
    expect(
        (worst <= price <= best) if up else (best <= price <= worst),
        "confirm_price",
        price,
        f"entre {worst} y {best}",
    )
    # 3. El extremo de la confirmación está en el instante de confirmación.
    top = c.high if up else c.low
    expect(top == best, "c_price_max" if up else "c_price_min", top, best)
    # 4. El extremo del overshoot es `extreme_price`; sin overshoot, `confirm_price`.
    if o.ticks:
        edge = o.high if up else o.low
        expect(
            edge == l2["extreme_price"],
            "os_price_max" if up else "os_price_min",
            edge,
            l2["extreme_price"],
        )
    else:
        expect(
            l2["extreme_price"] == price, "extreme_price", l2["extreme_price"], price
        )
    # 5. os_ticks = 0 si y solo si E = C.
    expect(
        (o.ticks == 0) == (l2["E"] == l2["C"]),
        "os_ticks",
        o.ticks,
        "0" if l2["E"] == l2["C"] else "más de 0",
    )
    # 6. Sin huecos del proveedor las fases miden C − R y E − C; con ellos, menos.
    expect(c.ticks <= l2["C"] - l2["R"], "c_ticks", c.ticks, f"≤ {l2['C'] - l2['R']}")
    expect(o.ticks <= l2["E"] - l2["C"], "os_ticks", o.ticks, f"≤ {l2['E'] - l2['C']}")
    # 7. Duración 0 sii todos los ticks son del instante; VWAP entre mínimo y máximo.
    expect(
        (c.duration == 0) == (c.ticks == ticks),
        "c_duration",
        c.duration,
        "0" if c.ticks == ticks else "más de 0",
    )
    expect(c.low <= c.vwap <= c.high, "c_vwap", c.vwap, f"entre {c.low} y {c.high}")
    if o.ticks:
        expect(
            o.low <= o.vwap <= o.high,
            "os_vwap",
            o.vwap,
            f"entre {o.low} y {o.high}",
        )


__all__ = ["EVENT_COLUMNS", "EventRow", "build_row"]
