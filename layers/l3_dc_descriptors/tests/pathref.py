"""La familia T de las pruebas: Python puro, sin el motor (TRD-L3 §7.5, T-01 a T-13).

Recorre los ticks de cada fase con un bucle y calcula con `Decimal` de precisión amplia
(`ln`) y `Fraction` (todo lo demás), sin redondear hasta el `float` final. No importa nada de
`l3_dc_descriptors` ni de `dc_pyo3`: es lo que el motor debe reproducir.

Una fase es una lista de ticks `(precio × 10⁸, transact_time en µs, lado)`, con lado +1 si
`is_buyer_maker` es falso y −1 si es verdadero. La confirmación `(R, C]` ancla en
`reference_price` y el overshoot `(C, E]` en `confirm_price`; los caminos y las sumas de
cuadrados incluyen el paso del ancla al primer tick, y huecos, duraciones y lados solo los ticks
de la fase.
"""

from bisect import bisect_right
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise

from dc_frames import read_frames

# Las 13 columnas de `path.parquet` tras las claves, en el orden del esquema.
COLUMNS = (
    "c_realized_var",
    "os_realized_var",
    "c_tortuosity",
    "os_tortuosity",
    "c_sign_autocorr",
    "os_sign_autocorr",
    "c_max_gap",
    "os_max_gap",
    "c_idle_fraction",
    "os_idle_fraction",
    "c_t25",
    "c_t50",
    "c_t75",
)
SILENCE_US = 10_000_000

type Tick = tuple[int, int, int]


def scaled(price: Decimal) -> int:
    """El precio como entero × 10⁸."""
    return int(price.scaleb(8))


def tick(time: int, price: Decimal, sold: bool) -> Tick:
    return scaled(price), time, -1 if sold else 1


def realized_var(anchor: int, ticks: list[Tick]) -> float:
    """Σ ln(p_i / p_{i−1})² desde el ancla; la escala del precio se cancela."""
    with localcontext() as context:
        context.prec = 60
        total, before = Decimal(0), anchor
        for price, _, _ in ticks:
            step = (Decimal(price) / Decimal(before)).ln()
            total += step * step
            before = price
        return float(total)


def tortuosity(anchor: int, ticks: list[Tick], net: int) -> float | None:
    """Σ|Δp| desde el ancla entre el desplazamiento neto `net`; nulo si es 0."""
    if net == 0:
        return None
    length, before = 0, anchor
    for price, _, _ in ticks:
        length += abs(price - before)
        before = price
    return float(Fraction(length, net))


def sign_autocorr(ticks: list[Tick]) -> float | None:
    """La autocorrelación muestral centrada de rezago 1 de los lados, con su media.

    Nula con menos de dos ticks o con todos los lados iguales (varianza 0).
    """
    sides = [side for _, _, side in ticks]
    if len(sides) < 2:
        return None
    mean = Fraction(sum(sides), len(sides))
    deviations = [side - mean for side in sides]
    variance = sum(d * d for d in deviations)
    if variance == 0:
        return None
    covariance = sum(a * b for a, b in pairwise(deviations))
    return float(covariance / variance)


def gaps(ticks: list[Tick]) -> list[int]:
    return [b[1] - a[1] for a, b in pairwise(ticks)]


def idle_fraction(ticks: list[Tick], silence_us: int) -> float | None:
    """(Σ de huecos mayores que `silence_us`) / duración de la fase; nula con duración 0."""
    duration = ticks[-1][1] - ticks[0][1]
    if duration == 0:
        return None
    idle = sum(gap for gap in gaps(ticks) if gap > silence_us)
    return float(Fraction(idle, duration))


def levels(
    ticks: list[Tick], reference: int, confirm: int, direction: int
) -> list[float | None]:
    """`(t* − t_1) / duración` de los niveles 25, 50 y 75 % del camino de `reference` a
    `confirm`: `t*` es el tiempo del primer tick con `4·(p − R)·d ≥ k·(C − R)·d`. Nulos con
    duración 0.
    """
    duration = ticks[-1][1] - ticks[0][1]
    if duration == 0:
        return [None, None, None]
    need = (confirm - reference) * direction
    found = []
    for k in (1, 2, 3):
        reached = next(
            (t for p, t, _ in ticks if 4 * (p - reference) * direction >= k * need),
            None,
        )
        if reached is None:
            raise ValueError(f"ningún tick de la confirmación alcanza el nivel {k}/4")
        found.append(float(Fraction(reached - ticks[0][1], duration)))
    return found


def path_row(
    confirmation: list[Tick],
    overshoot: list[Tick],
    *,
    reference: int,
    confirm: int,
    extreme: int,
    direction: int,
    silence_us: int = SILENCE_US,
) -> tuple:
    """Los 13 valores de un evento, en el orden de `COLUMNS`, con `None` donde son nulos.

    `reference`, `confirm` y `extreme` son `reference_price`, `confirm_price` y
    `extreme_price` de L2 (× 10⁸); el overshoot vacío anula todos los `os_*`.
    """
    if not confirmation:
        raise ValueError("la fase de confirmación no tiene ticks")
    c25, c50, c75 = levels(confirmation, reference, confirm, direction)
    c_tortuosity = tortuosity(reference, confirmation, abs(confirm - reference))
    c = (
        realized_var(reference, confirmation),
        c_tortuosity,
        sign_autocorr(confirmation),
        max(gaps(confirmation), default=0),
        idle_fraction(confirmation, silence_us),
    )
    if overshoot:
        o = (
            realized_var(confirm, overshoot),
            tortuosity(confirm, overshoot, abs(extreme - confirm)),
            sign_autocorr(overshoot),
            max(gaps(overshoot), default=0),
            idle_fraction(overshoot, silence_us),
        )
    else:
        o = (None,) * 5
    return (
        c[0],
        o[0],
        c[1],
        o[1],
        c[2],
        o[2],
        c[3],
        o[3],
        c[4],
        o[4],
        c25,
        c50,
        c75,
    )


class Series:
    """Los ticks de una serie ordenados por id, para sacar las fases `(R, C]` y `(C, E]`."""

    def __init__(self, ticks) -> None:
        """`ticks`: objetos con `id`, `time`, `price` (`Decimal`) y `sold` (como `l3_lake.Tick`)."""
        self._ids = [t.id for t in ticks]
        self._ticks = [tick(t.time, t.price, t.sold) for t in ticks]

    def between(self, low: int, high: int) -> list[Tick]:
        """Los ticks con `low < id ≤ high`."""
        return self._ticks[bisect_right(self._ids, low) : bisect_right(self._ids, high)]


def from_ticks(event: dict, series: Series, silence_us: int = SILENCE_US) -> tuple:
    """Los 13 valores de un evento filtrando la serie por id: ni el lector ni Arrow.

    `event` es la fila de L2 (`l3_lake.Lake.events`).
    """
    r, c, e = (event[f"{n}_agg_trade_id"] for n in ("reference", "confirm", "extreme"))
    return path_row(
        series.between(r, c),
        series.between(c, e),
        reference=scaled(event["reference_price"]),
        confirm=scaled(event["confirm_price"]),
        extreme=scaled(event["extreme_price"]),
        direction=event["direction"],
        silence_us=silence_us,
    )


def frame_ticks(frame) -> list[Tick]:
    """Los ticks de una trama del lector (`dc_frames.Frame`)."""
    return [
        tick(time, price, sold)
        for time, price, sold in zip(
            frame.transact_time.to_pylist(),
            frame.price.to_pylist(),
            frame.is_buyer_maker.to_pylist(),
            strict=True,
        )
    ]


def from_reader(ev, silence_us: int = SILENCE_US) -> tuple:
    """Los 13 valores de un `EventFrames` del lector en modo por evento."""
    event = ev.event.to_pylist()[0]
    return path_row(
        frame_ticks(ev.confirmation),
        frame_ticks(ev.overshoot),
        reference=scaled(event["reference_price"]),
        confirm=scaled(event["confirm_price"]),
        extreme=scaled(event["extreme_price"]),
        direction=event["direction"],
        silence_us=silence_us,
    )


def from_lake(
    l1_root, l2_root, month, thetas: list[str], silence_us: int = SILENCE_US
) -> dict[tuple[Decimal, int], tuple]:
    """Los 13 valores de cada evento de los θ en el mes, por `(θ, confirm_agg_trade_id)`,
    leyendo L1 y L2 con el lector de `dc_frames` (un evento a la vez).
    """
    values = [Decimal(t) for t in thetas]
    return {
        (ev.theta, ev.event.column("confirm_agg_trade_id")[0].as_py()): from_reader(
            ev, silence_us
        )
        for ev in read_frames(values, l1_root, l2_root, month, month)
    }
