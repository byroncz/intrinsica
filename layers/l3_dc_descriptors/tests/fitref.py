"""La familia A de las pruebas: Python puro, sin el motor (TRD-L3 §7.2.2, A-01 a A-09).

Recorre los ticks de cada fase con un bucle y calcula todo con `Fraction`, sin redondear hasta el
`float` final. No importa nada de `l3_dc_descriptors` ni de `dc_pyo3` y no comparte código con
el motor (que cuantiza a punto fijo y cierra con enteros de 256 bits): es lo que el motor debe
reproducir.

Una fase es una lista de ticks `(precio × 10⁸, transact_time en µs, orden)`, con `orden` el
`agg_trade_id` (o cualquier entero que crezca con la serie). Cada fase se mide en su propio
sistema de coordenadas, `(t₀, p₀) → (t_fin, p_fin)`, ambos de L2:

  c   (reference_time, reference_price) → (confirm_time, confirm_price)
  os  (confirm_time, confirm_price)     → (extreme_time, extreme_price)
  ev  (reference_time, reference_price) → (extreme_time, extreme_price), con los ticks de c y os

  A-01 a A-03  pendiente de mínimos cuadrados de `p / reference_price` sobre `t` en segundos
  A-04 a A-06  coeficiente `a` de `y = a·τ² + b·τ` sin intercepto, con `τ = (t − t₀)/(t_fin − t₀)`
               e `y = (p − p₀)/(p_fin − p₀)`
  A-07 a A-09  `(p_fin − p_prev) / (p_fin − p₀)`, con `p_prev` el precio del tick de mayor
               `(t, orden)` con `t < t_fin` (`p₀` si no hay)

Nulos: los `os_*` con el overshoot vacío; la pendiente con menos de 2 ticks o todos en el mismo
`t` (o `t_fin = t₀`); la aceleración, además, con `t_fin = t₀`, con `p_fin = p₀` y con un solo `τ`
positivo distinto (el sistema de 2×2 es singular); el salto, con `p_fin = p₀`.
"""

from bisect import bisect_right
from decimal import Decimal
from fractions import Fraction

from dc_frames import read_frames

# Las 9 columnas de `fit.parquet` tras las claves, en el orden del esquema.
COLUMNS = (
    "c_slope",
    "os_slope",
    "ev_slope",
    "c_accel",
    "os_accel",
    "ev_accel",
    "c_jump_share",
    "os_jump_share",
    "ev_jump_share",
)

type Tick = tuple[int, int, int]


def scaled(price: Decimal) -> int:
    """El precio como entero × 10⁸."""
    return int(price.scaleb(8))


def slope(ticks: list[Tick], reference: int, t0: int, t_fin: int) -> Fraction | None:
    """Pendiente de mínimos cuadrados de `p / reference` sobre `t` en segundos."""
    times = {t for _, t, _ in ticks}
    if len(ticks) < 2 or len(times) < 2 or t_fin <= t0:
        return None
    n = len(ticks)
    xs = [Fraction(t, 10**6) for _, t, _ in ticks]
    ys = [Fraction(p, reference) for p, _, _ in ticks]
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sxx = sum(x * x for x in xs)
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def acceleration(
    ticks: list[Tick], t0: int, p0: int, t_fin: int, p_fin: int
) -> Fraction | None:
    """El coeficiente `a` de `y = a·τ² + b·τ` por mínimos cuadrados."""
    later = {t for _, t, _ in ticks if t > t0}
    if len(ticks) < 2 or t_fin <= t0 or p_fin == p0 or len(later) < 2:
        return None
    if len({t for _, t, _ in ticks}) < 2:
        return None
    taus = [Fraction(t - t0, t_fin - t0) for _, t, _ in ticks]
    ys = [Fraction(p - p0, p_fin - p0) for p, _, _ in ticks]
    s2 = sum(x**2 for x in taus)
    s3 = sum(x**3 for x in taus)
    s4 = sum(x**4 for x in taus)
    sty = sum(x * y for x, y in zip(taus, ys, strict=True))
    s2y = sum(x**2 * y for x, y in zip(taus, ys, strict=True))
    determinant = s4 * s2 - s3 * s3
    if determinant == 0:
        return None
    return (s2y * s2 - sty * s3) / determinant


def jump_share(ticks: list[Tick], p0: int, t_fin: int, p_fin: int) -> Fraction | None:
    """`(p_fin − p_prev) / (p_fin − p₀)`; `p_prev` es el precio del último tick con `t < t_fin`
    por `(t, orden)`, o `p₀` si ninguno.
    """
    if p_fin == p0:
        return None
    before = [(t, order, p) for p, t, order in ticks if t < t_fin]
    p_prev = max(before)[2] if before else p0
    return Fraction(p_fin - p_prev, p_fin - p0)


def phase(
    ticks: list[Tick], anchor: tuple[int, int], close: tuple[int, int], reference: int
) -> tuple[Fraction | None, Fraction | None, Fraction | None]:
    """`(pendiente, aceleración, salto)` de una fase con ancla `(t₀, p₀)` y cierre
    `(t_fin, p_fin)`.
    """
    (t0, p0), (t_fin, p_fin) = anchor, close
    return (
        slope(ticks, reference, t0, t_fin),
        acceleration(ticks, t0, p0, t_fin, p_fin),
        jump_share(ticks, p0, t_fin, p_fin),
    )


def as_float(value: Fraction | None) -> float | None:
    return None if value is None else float(value)


def fit_row(
    confirmation: list[Tick],
    overshoot: list[Tick],
    *,
    reference: tuple[int, int],
    confirm: tuple[int, int],
    extreme: tuple[int, int],
) -> tuple:
    """Los 9 valores de un evento, en el orden de `COLUMNS`, con `None` donde son nulos.

    `reference`, `confirm` y `extreme` son `(tiempo, precio × 10⁸)` de L2; el overshoot vacío
    anula todos los `os_*`.
    """
    if not confirmation:
        raise ValueError("la fase de confirmación no tiene ticks")
    reference_price = reference[1]
    c = phase(confirmation, reference, confirm, reference_price)
    o = (
        phase(overshoot, confirm, extreme, reference_price)
        if overshoot
        else (None, None, None)
    )
    v = phase(confirmation + overshoot, reference, extreme, reference_price)
    values = (
        c[0], o[0], v[0],
        c[1], o[1], v[1],
        c[2], o[2], v[2],
    )  # fmt: skip
    return tuple(as_float(x) for x in values)


class Series:
    """Los ticks de una serie ordenados por id, para sacar las fases `(R, C]` y `(C, E]`."""

    def __init__(self, ticks) -> None:
        """`ticks`: objetos con `id`, `time` y `price` (`Decimal`), como `l3_lake.Tick`."""
        self._ids = [t.id for t in ticks]
        self._ticks = [(scaled(t.price), t.time, t.id) for t in ticks]

    def between(self, low: int, high: int) -> list[Tick]:
        """Los ticks con `low < id ≤ high`."""
        return self._ticks[bisect_right(self._ids, low) : bisect_right(self._ids, high)]


def from_ticks(event: dict, series: Series) -> tuple:
    """Los 9 valores de un evento filtrando la serie por id: ni el lector ni Arrow.

    `event` es la fila de L2 (`l3_lake.Lake.events`).
    """
    r, c, e = (event[f"{n}_agg_trade_id"] for n in ("reference", "confirm", "extreme"))
    return fit_row(
        series.between(r, c),
        series.between(c, e),
        reference=(event["reference_time"], scaled(event["reference_price"])),
        confirm=(event["confirm_time"], scaled(event["confirm_price"])),
        extreme=(event["extreme_time"], scaled(event["extreme_price"])),
    )


def frame_ticks(frame, start: int = 0) -> list[Tick]:
    """Los ticks de una trama del lector (`dc_frames.Frame`); el orden es la posición, que crece
    con la serie (`start` es el orden del primero).
    """
    return [
        (price, time, start + k)
        for k, (time, price) in enumerate(
            zip(
                frame.transact_time.to_pylist(),
                (scaled(p) for p in frame.price.to_pylist()),
                strict=True,
            )
        )
    ]


def from_reader(ev) -> tuple:
    """Los 9 valores de un `EventFrames` del lector en modo por evento."""
    event = ev.event.to_pylist()[0]
    confirmation = frame_ticks(ev.confirmation)
    overshoot = frame_ticks(ev.overshoot, len(confirmation))
    return fit_row(
        confirmation,
        overshoot,
        reference=(event["reference_time"], scaled(event["reference_price"])),
        confirm=(event["confirm_time"], scaled(event["confirm_price"])),
        extreme=(event["extreme_time"], scaled(event["extreme_price"])),
    )


def from_lake(
    l1_root, l2_root, month, thetas: list[str]
) -> dict[tuple[Decimal, int], tuple]:
    """Los 9 valores de cada evento de los θ en el mes, por `(θ, confirm_agg_trade_id)`,
    leyendo L1 y L2 con el lector de `dc_frames` (un evento a la vez).
    """
    values = [Decimal(t) for t in thetas]
    return {
        (ev.theta, ev.event.column("confirm_agg_trade_id")[0].as_py()): from_reader(ev)
        for ev in read_frames(values, l1_root, l2_root, month, month)
    }
