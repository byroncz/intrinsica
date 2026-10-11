"""La agregación ingenua de las pruebas: Python puro, sin Arrow compute (TRD-L3 §6.6).

Recorre los ticks de cada fase con un bucle y suma con `Decimal` de precisión amplia. No
comparte código con el resumidor: es lo que este debe reproducir.
"""

from decimal import ROUND_HALF_EVEN, Context, Decimal

WIDE = Context(prec=80)
EIGHT = Decimal("1e-8")


def ticks_of(frame) -> list[tuple[int, Decimal, Decimal, bool]]:
    """`(transact_time, price, quantity, is_buyer_maker)` de una trama del lector."""
    return list(
        zip(
            frame.transact_time.to_pylist(),
            frame.price.to_pylist(),
            frame.quantity.to_pylist(),
            frame.is_buyer_maker.to_pylist(),
            strict=True,
        )
    )


def phase(ticks: list) -> dict:
    """Los agregados de una fase; los de precio, duración y VWAP son `None` si no hay ticks."""
    buy = sell = notional = Decimal(0)
    for _, price, quantity, sold in ticks:
        if sold:
            sell = WIDE.add(sell, quantity)
        else:
            buy = WIDE.add(buy, quantity)
        notional = WIDE.add(notional, WIDE.multiply(price, quantity))
    out = {"ticks": len(ticks), "buy": buy, "sell": sell}
    if not ticks:
        return out | {"high": None, "low": None, "duration": None, "vwap": None}
    prices = [t[1] for t in ticks]
    vwap = WIDE.divide(notional, WIDE.add(buy, sell))
    return out | {
        "high": max(prices),
        "low": min(prices),
        "duration": ticks[-1][0] - ticks[0][0],
        "vwap": vwap.quantize(EIGHT, rounding=ROUND_HALF_EVEN, context=WIDE),
    }


def summary(event: dict, confirmation: list, overshoot: list) -> dict:
    """La fila esperada de `summaries.parquet` de un evento (`event` es su fila de L2)."""
    c, o = phase(confirmation), phase(overshoot)
    instant = [t[1] for t in confirmation if t[0] == event["confirm_time"]]
    up = event["direction"] == 1
    best, worst = (max(instant), min(instant)) if up else (min(instant), max(instant))
    return {
        "theta": event["theta"],
        "confirm_agg_trade_id": event["confirm_agg_trade_id"],
        "c_ticks": c["ticks"],
        "c_quantity_buy": c["buy"],
        "c_quantity_sell": c["sell"],
        "c_price_max": c["high"],
        "c_price_min": c["low"],
        "c_duration": c["duration"],
        "c_vwap": c["vwap"],
        "os_ticks": o["ticks"],
        "os_quantity_buy": o["buy"],
        "os_quantity_sell": o["sell"],
        "os_price_max": o["high"],
        "os_price_min": o["low"],
        "os_duration": o["duration"],
        "os_vwap": o["vwap"],
        "confirm_instant_ticks": len(instant),
        "confirm_instant_price_best": best,
        "confirm_instant_price_worst": worst,
    }


def from_reader(ev) -> dict:
    """Lo esperado de un `EventFrames` del lector en modo por evento."""
    return summary(
        ev.event.to_pylist()[0], ticks_of(ev.confirmation), ticks_of(ev.overshoot)
    )


def from_ticks(event: dict, ticks: list) -> dict:
    """Lo esperado de un evento filtrando la lista de ticks por id: ni el lector ni Arrow.

    `ticks` son objetos con `id`, `time`, `price`, `quantity` y `sold`; `event` es la fila
    de L2 con `theta` (Decimal).
    """
    r, c, e = (event[f"{n}_agg_trade_id"] for n in ("reference", "confirm", "extreme"))
    rows = [(t.time, t.price, t.quantity, t.sold, t.id) for t in ticks]
    confirmation = [x[:4] for x in rows if r < x[4] <= c]
    overshoot = [x[:4] for x in rows if c < x[4] <= e]
    return summary(event, confirmation, overshoot)
