"""Agregación de una fase con Arrow compute (TRD-L3 §8.1, paso 3).

Un acumulador recibe los trozos de una fase, uno por row group, y guarda solo
escalares de Python: el row group se puede soltar en cuanto se agregó. Las sumas son
exactas: `quantity` se suma en `decimal128(38,8)` por trozo y `price·quantity` en
`decimal128(38,16)`, y entre trozos se acumulan enteros de Python sin escalar, que no
desbordan ni redondean.
"""

from decimal import Context

import pyarrow as pa
import pyarrow.compute as pc
from dc_frames import Frame

from l3_dc_descriptors.errors import PhaseQuantityMismatch, PhaseValueError
from l3_dc_descriptors.schema import LIMIT, SCALE

# Los `Decimal` de 38 dígitos se escalan sin que el contexto por defecto (28) los redondee.
_WIDE = Context(prec=80)


def scaled(value, scale: int = SCALE) -> int:
    """El entero sin escalar de un `Decimal` con `scale` decimales."""
    return int(value.scaleb(scale, _WIDE))


def _sum(array: pa.Array) -> int:
    """La suma exacta de un array decimal, sin escalar; 0 si está vacío."""
    total = pc.sum(array, min_count=0)
    return scaled(total.as_py(), total.type.scale)


def _side_sum(quantity: pa.Array, mask: pa.Array) -> int:
    """La suma de `quantity` donde `mask` es verdadero."""
    return _sum(quantity.filter(mask))


def _min_max(prices: pa.Array) -> tuple[int, int]:
    both = pc.min_max(prices)
    return scaled(both["min"].as_py()), scaled(both["max"].as_py())


class Instant:
    """Los ticks de la confirmación con `transact_time = confirm_time` (ADR-L3-01)."""

    def __init__(self, time: int) -> None:
        self.time = time
        self.ticks = 0
        self.high: int | None = None
        self.low: int | None = None

    def add(self, frame: Frame) -> None:
        mask = pc.equal(frame.transact_time, self.time)
        ticks = pc.sum(mask, min_count=0).as_py()
        if not ticks:
            return
        low, high = _min_max(frame.price.filter(mask))
        self.ticks += ticks
        self.low = low if self.low is None else min(self.low, low)
        self.high = high if self.high is None else max(self.high, high)


class PhaseAcc:
    """El acumulador de una fase: conteo, cantidades, precios, tiempos y notional.

    Con `instant_time` agrega también el instante de confirmación (solo la fase de
    confirmación lo tiene).
    """

    def __init__(self, instant_time: int | None = None) -> None:
        self.ticks = 0
        self.buy = 0
        self.sell = 0
        self.total = 0
        self.notional = 0  # Σ price·quantity, escala 10⁻¹⁶
        self.high: int | None = None
        self.low: int | None = None
        self.first_time: int | None = None
        self.last_time: int | None = None
        self.instant = Instant(instant_time) if instant_time is not None else None

    def add(self, frame: Frame) -> None:
        count = len(frame)
        if not count:
            return
        quantity, sold = frame.quantity, frame.is_buyer_maker
        self.ticks += count
        self.total += _sum(quantity)
        self.sell += _side_sum(quantity, sold)
        self.buy += _side_sum(quantity, pc.invert(sold))
        self.notional += _sum(pc.multiply(frame.price, quantity))
        low, high = _min_max(frame.price)
        self.low = low if self.low is None else min(self.low, low)
        self.high = high if self.high is None else max(self.high, high)
        if self.first_time is None:
            self.first_time = frame.transact_time[0].as_py()
        self.last_time = frame.transact_time[count - 1].as_py()
        if self.instant is not None:
            self.instant.add(frame)

    def consume(self, frames) -> None:
        for frame in frames:
            self.add(frame)


class PhaseSummary:
    """Lo que una fase aporta a la fila. Los precios, la duración y el VWAP son nulos
    (`None`) si la fase no tiene ticks.
    """

    __slots__ = ("buy", "duration", "high", "low", "sell", "ticks", "vwap")

    def __init__(self, acc: PhaseAcc, vwap: int | None) -> None:
        self.ticks = acc.ticks
        self.buy = acc.buy
        self.sell = acc.sell
        self.high = acc.high
        self.low = acc.low
        self.duration = None if not acc.ticks else acc.last_time - acc.first_time
        self.vwap = vwap


def divide_half_even(numerator: int, denominator: int) -> int:
    """`numerator / denominator` redondeado al entero más cercano, empates al par."""
    quotient, rest = divmod(numerator, denominator)
    twice = 2 * rest
    if twice > denominator or (twice == denominator and quotient % 2):
        quotient += 1
    return quotient


def close_phase(
    acc: PhaseAcc, theta: str, confirm_agg_trade_id: int, name: str
) -> PhaseSummary:
    """Cierra una fase: comprueba `buy + sell` contra la suma de `quantity` (§7.2,
    invariante 8) y calcula el VWAP en decimal, redondeado half-even a 10⁻⁸.
    """
    if acc.buy + acc.sell != acc.total:
        raise PhaseQuantityMismatch(
            theta,
            confirm_agg_trade_id,
            name,
            _text(acc.total),
            _text(acc.buy + acc.sell),
        )
    if not acc.ticks:
        return PhaseSummary(acc, None)
    if acc.buy >= LIMIT or acc.sell >= LIMIT:
        raise PhaseValueError(
            theta,
            confirm_agg_trade_id,
            f"la cantidad de la fase {name} no cabe en DECIMAL(18,8)",
        )
    if acc.total <= 0:
        raise PhaseValueError(
            theta, confirm_agg_trade_id, f"la fase {name} no tiene cantidad"
        )
    # Σ price·quantity (10⁻¹⁶) / Σ quantity (10⁻⁸) queda en 10⁻⁸.
    return PhaseSummary(acc, divide_half_even(acc.notional, acc.total))


def _text(unscaled: int) -> str:
    sign, digits = ("-", -unscaled) if unscaled < 0 else ("", unscaled)
    whole, fraction = divmod(digits, 10**SCALE)
    return f"{sign}{whole}.{fraction:0{SCALE}d}"
