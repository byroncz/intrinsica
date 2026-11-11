"""La familia F de las pruebas: Python con `numpy.fft`, sin el motor (TRD-L3 §7.2.3, F-01 a F-04 y la ubicación de la energía).

Un oráculo por evento sobre las tramas de `dc_frames`, como `pathref.py` y `fitref.py`. No importa
nada de `l3_dc_descriptors` ni de `dc_pyo3` y no comparte código con el motor de Rust, que calcula
con `rustfft`; la tolerancia entre los dos es de 1e-9 en las bandas y de 1e-12 en la ubicación.

Una fase son los precios de sus ticks, enteros × 10⁸, y su ancla: `reference_price` de L2 para la
confirmación `(R, C]` y `confirm_price` para el overshoot `(C, E]`. Los retornos son
`r_k = ln(p_k / p_{k−1})`, con el primer paso desde el ancla, calculados con `Decimal` de precisión
amplia y redondeados a `float32` (el búfer del motor guarda `f32`, decisión del veredicto de la
Épica ITSC-410); de ahí en adelante todo es `float64`:

* Fourier: la serie se centra, se multiplica por la ventana de Hann periódica
  `0,5 − 0,5·cos(2π·j/n)`, `numpy.fft.rfft`, energía `|X_k|²` de los bins `k = 1..⌊n/2⌋` y fracción
  de cada banda de período `n/k` en ticks: [2, 4), [4, 8), [8, 16), [16, 32), [32, 64), [64, 128)
  y 128 o más.
* Haar: sobre los últimos `2^⌊log₂ n⌋` retornos, centrados, transformada ortonormal; la energía de
  un nivel es `Σ d²` y el séptimo nivel junta las escalas de 128 en adelante.
* Ubicación de la energía total (`c_energy_loc`, `os_energy_loc`): `Σ t_k·r_k² / Σ r_k²` con
  `t_k = (k − 0,5)/n` y `k = 1..n`, sobre los retornos sin centrar y sin ventana. Cae en (0, 1):
  0,5 si la energía se reparte pareja, menos si se concentra al inicio y más si al final. Es la
  fórmula de `total_location` de `l3_spectrum_probe.py`.

Una fase con menos de 16 retornos, un overshoot vacío, una banda o nivel sin bins y una serie con
todos los retornos iguales dejan sus valores de Fourier y Haar en `None`. La ubicación es `None`
con menos de 16 retornos, con el overshoot vacío y si todos los retornos son cero; una serie de
retornos iguales y no nulos no tiene Fourier pero sí ubicación (0,5).
"""

import math
from decimal import Decimal, localcontext

import numpy as np
from dc_frames import read_frames

N_MIN = 16
BANDS = 7

# Las 30 columnas de `spectrum.parquet` tras las claves, en el orden del esquema: las 28 de bandas
# y niveles, y al final las dos ubicaciones.
COLUMNS = (
    *(
        f"{phase}_{register}_{unit}_{band}"
        for register, unit in (("fourier", "band"), ("haar", "level"))
        for phase in ("c", "os")
        for band in range(1, BANDS + 1)
    ),
    "c_energy_loc",
    "os_energy_loc",
)

type Fractions = list[float | None]


def scaled(price: Decimal) -> int:
    """El precio como entero × 10⁸."""
    return int(price.scaleb(8))


def returns(anchor: int, prices: list[int]) -> np.ndarray:
    """Los retornos de la fase como `float64` de valores `float32`: el primero desde el ancla."""
    steps = []
    before = anchor
    with localcontext() as context:
        context.prec = 60
        for price in prices:
            steps.append(float((Decimal(price) / Decimal(before)).ln()))
            before = price
    return np.array(steps, dtype=np.float32).astype(np.float64)


def band_of(n: int, k: int) -> int:
    """La banda (1 a 7) del bin `k` de una serie de `n` valores: el período `n/k` en
    `[2^b, 2^(b+1))` para `b = 1..6`, o 128 o más.
    """
    for band in range(1, 7):
        if (1 << band) * k <= n < (1 << (band + 1)) * k:
            return band
    assert n >= 128 * k, (n, k)
    return 7


def fourier(x: np.ndarray) -> Fractions:
    """Las fracciones de energía por banda de Fourier de una serie de `n ≥ 16` valores."""
    n = len(x)
    if x.min() == x.max():
        return [None] * BANDS
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    spectrum = np.fft.rfft((x - x.mean()) * window)
    energy = np.abs(spectrum[1 : n // 2 + 1]) ** 2
    sums = np.zeros(BANDS)
    has = [False] * BANDS
    for k, value in enumerate(energy, start=1):
        band = band_of(n, k) - 1
        sums[band] += value
        has[band] = True
    total = float(energy.sum())
    return [float(sums[b]) / total if has[b] else None for b in range(BANDS)]


def haar(x: np.ndarray) -> Fractions:
    """Las fracciones de energía por nivel de Haar de los últimos `2^⌊log₂ n⌋` valores."""
    n = len(x)
    length = 1 << (n.bit_length() - 1)
    tail = x[n - length :]
    if tail.min() == tail.max():
        return [None] * BANDS
    approx = tail - tail.mean()
    levels = []
    while len(approx) > 1:
        even, odd = approx[0::2], approx[1::2]
        levels.append(float(np.sum(((even - odd) / math.sqrt(2)) ** 2)))
        approx = (even + odd) / math.sqrt(2)
    energy = [0.0] * BANDS
    for level, value in enumerate(levels):
        energy[min(level, BANDS - 1)] += value
    total = sum(energy)
    return [energy[b] / total if b < len(levels) else None for b in range(BANDS)]


def location(x: np.ndarray) -> float | None:
    """La ubicación en el tiempo de la energía total de una serie de `n` retornos:
    `Σ t_k·r_k² / Σ r_k²` con `t_k = (k − 0,5)/n`. `None` si la energía es cero.

    Caso a mano (`LOC_X` de `test_l3_spectrum_probe.py`): `[6, 0, 0, 0, 2, 0, −2, −6]` tiene
    `Σ r² = 80` y `Σ (k − 0,5)·r² = 18 + 18 + 26 + 270 = 332`, que sobre `n = 8` es `41,5`:
    la ubicación es `41,5/80`.
    """
    square = x * x
    energy = float(square.sum())
    if not energy > 0.0:
        return None
    t = (np.arange(len(x)) + 0.5) / len(x)
    return float(square @ t) / energy


def spectrum_row(
    confirmation: list[int],
    overshoot: list[int],
    *,
    reference: int,
    confirm: int,
) -> tuple:
    """Los 30 valores de un evento, en el orden de `COLUMNS`, con `None` donde son nulos.

    `confirmation` y `overshoot` son los precios de los ticks de cada fase (× 10⁸); `reference` y
    `confirm` son `reference_price` y `confirm_price` de L2 (× 10⁸), las anclas.
    """
    if not confirmation:
        raise ValueError("la fase de confirmación no tiene ticks")
    phases = []
    for prices, anchor in ((confirmation, reference), (overshoot, confirm)):
        if len(prices) < N_MIN:
            phases.append(([None] * BANDS, [None] * BANDS, None))
        else:
            x = returns(anchor, prices)
            phases.append((fourier(x), haar(x), location(x)))
    (c_fourier, c_haar, c_loc), (os_fourier, os_haar, os_loc) = phases
    return (*c_fourier, *os_fourier, *c_haar, *os_haar, c_loc, os_loc)


def frame_prices(frame) -> list[int]:
    """Los precios (× 10⁸) de una trama del lector (`dc_frames.Frame`)."""
    return [scaled(price) for price in frame.price.to_pylist()]


def from_reader(ev) -> tuple:
    """Los 30 valores de un `EventFrames` del lector en modo por evento."""
    event = ev.event.to_pylist()[0]
    return spectrum_row(
        frame_prices(ev.confirmation),
        frame_prices(ev.overshoot),
        reference=scaled(event["reference_price"]),
        confirm=scaled(event["confirm_price"]),
    )


def from_lake(
    l1_root, l2_root, month, thetas: list[str]
) -> dict[tuple[Decimal, int], tuple]:
    """Los 30 valores de cada evento de los θ en el mes, por `(θ, confirm_agg_trade_id)`,
    leyendo L1 y L2 con el lector de `dc_frames` (un evento a la vez). Un evento sobre el
    presupuesto no está aquí: el lector no lo distingue; las pruebas lo apartan.
    """
    values = [Decimal(t) for t in thetas]
    return {
        (ev.theta, ev.event.column("confirm_agg_trade_id")[0].as_py()): from_reader(ev)
        for ev in read_frames(values, l1_root, l2_root, month, month)
    }
