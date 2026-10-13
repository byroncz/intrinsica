"""Cierre de una fase (TRD-L3 §7.2, §8.1 paso 6).

Una fase llega agregada: conteo, cantidades de compra y venta, su suma total, `Σ price·
quantity` (10⁻¹⁶), precios y tiempos, todo en enteros de Python sin escalar, que no
desbordan ni redondean. `close_phase` comprueba que `buy + sell` sea la suma de `quantity`
y calcula el VWAP. Quien agrega es el resumidor por row group (`rowgroup`) y, como oráculo,
el camino por evento de la suite.
"""

from decimal import Context

from l3_dc_descriptors.errors import PhaseQuantityMismatch, PhaseValueError
from l3_dc_descriptors.schema import LIMIT, SCALE

# Los `Decimal` de 38 dígitos se escalan sin que el contexto por defecto (28) los redondee.
_PRECISION = 80


def scaled(value, scale: int = SCALE) -> int:
    """El entero sin escalar de un `Decimal` con `scale` decimales.

    El contexto es propio de cada llamada: un `Context` de módulo acumula banderas y
    lo compartirían los hilos (la agregación por θ debe poder correr en paralelo).
    """
    return int(value.scaleb(scale, Context(prec=_PRECISION)))


class PhaseSummary:
    """Lo que una fase aporta a la fila. Los precios, la duración y el VWAP son nulos
    (`None`) si la fase no tiene ticks.
    """

    __slots__ = ("buy", "duration", "high", "low", "sell", "ticks", "vwap")

    def __init__(self, acc, vwap: int | None) -> None:
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


def close_phase(acc, theta: str, confirm_agg_trade_id: int, name: str) -> PhaseSummary:
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
