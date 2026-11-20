"""Sonda de la familia X (complejidad) de L3: Hurst y entropía de permutación (ITSC-432).

Decide si dos medidas baratas de complejidad dicen algo que L3 no diga ya. Sobre un θ y una lista
de meses mide, por fase (`c` = (R, C] y `os` = (C, E]), el exponente de Hurst por DFA-1 y la
entropía de permutación de una ventana de la fase, y las compara con dos nulos: el barajado de los
retornos y un sustituto que conserva su espectro (IAAFT), que es lo que la familia F ya guarda. Si
la medida no se distingue del sustituto, no ve nada que F no vea. El tamaño del efecto es la
distancia de Kolmogorov-Smirnov D; nunca se reporta un valor p. Además mide la redundancia
(Spearman) con las columnas que ya existen y el costo por punto. Qué medidas siguen lo decide el
arquitecto con las cifras; el script solo aplica la regla del veredicto.

Serie y ventana. Con `n` ticks y el ancla `p_0` (`p_R` en c, `p_C` en os, como F), los retornos son
`r_k = ln(p_k / p_{k−1})`, k = 1..n (el primer paso sale del ancla; los ceros se conservan). De
cada fase con `n ≥ --min-ticks` se usa un tramo seguido de a lo sumo `--window` retornos: los
últimos de c y los primeros de os. Con `w = min(n, --window)`, el tramo de c son `r_{n−w+1} … r_n`
y el de os, `r_1 … r_w`; los precios de la ventana son los `w` ticks que los cierran, y su `p_0` es
el tick anterior (el ancla si el tramo parte de ella). La memoria por fase abierta es O(ventana).

Hurst (DFA-1) sobre los retornos de la ventana. Perfil = suma acumulada de `r − media`. Escalas
`s` = 16, 32, 64… hasta `w/4`. En cada ventana no solapada de tamaño `s` se resta la recta de
mínimos cuadrados; `F(s)` es la raíz de la media de los cuadrados del residuo y `H` la pendiente
de `log F` contra `log s`. Es nulo (`nan`, y la fase se salta como `nulas`) si algún `F(s)` es 0.

Entropía de permutación de orden 4 y retardo 1 sobre los precios de la ventana: la frecuencia de
cada uno de los 4! = 24 órdenes de cuatro precios seguidos, con los empates ordenados por
aparición, `−Σ p ln p / ln 24`, en [0, 1]. Los precios monótonos dan 0.

Nulos, `--nulls` (10) de cada tipo por fase, con una sola semilla:
  - barajado: los retornos de la ventana permutados;
  - sustituto (IAAFT): parte del barajado y repite a lo sumo 20 veces imponer las amplitudes de
    Fourier de los retornos originales conservando las fases actuales y reasignar por rango los
    valores originales. Termina con la reasignación: conserva exactamente la distribución de los
    retornos. Se detiene antes si la reasignación ya no cambia.
  - los precios de un nulo son `p_0 · exp(suma acumulada de sus retornos)`.

D es la distancia exacta `sup |F_obs − F_sim|` entre el observado (una muestra de `N` fases) y el
simulado (`M = N · nulls` valores); `piso = 1,36 · √((N + M) / (N · M))`. Los cuantiles usan el
rango más cercano: con N valores, el cuantil q ocupa el lugar ⌈N·q⌉.

Redundancia, en la fase c: |ρ| de Spearman (rangos promedio en los empates) de cada medida contra
`c_fourier_band_1` a `c_fourier_band_7` (`spectrum.parquet`), `c_sign_autocorr` (`path.parquet`) y
`c_ticks` (`summaries.parquet`); se reporta el máximo. Una columna constante da ρ indefinida
(`nan`), que no cuenta como redundancia.

Veredicto, por medida y en la fase c (fijado antes de medir): `sigue` si y solo si D contra el
sustituto es ≥ 0,10 y ≥ `piso`, y su |ρ| máximo no pasa de 0,9; si no, `retirada`.

Tope de trabajo: `--max-phases` (600) fases por fase (c u os) y corrida. Si entre todos los meses hay
más fases con `n ≥ --min-ticks`, se toman espaciadas de forma pareja por posición: la j-ésima es la
de índice `⌊j · total / max⌋`. Un primer pase lee solo `summaries.parquet` para contar. Con la misma
semilla, los mismos archivos y los mismos argumentos el CSV es idéntico byte a byte; por eso el costo
(que depende del reloj) va al log y a la última línea, no al CSV.

Se saltan, y se cuentan: la fase bajo el mínimo, la de overshoot vacío (`os_ticks = 0`), la que el
tope no toma, el evento que arranca en el mes anterior y no cabe en el L1 del mes, y la fase nula
(algún `F(s)` en 0 en el observado o en un nulo).

Args:
  --theta 0.05  --months 2021-06,2024-06     (obligatorios; un θ y una lista de meses)
  --l1-root gs://<proyecto>-landing/l1       (por defecto, las del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2      OPS_RESULTS_URI; también rutas locales con la
  --l3-root gs://<proyecto>-dc-descriptors/l3  misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --min-ticks 1024   --window 10000   --max-phases 600
  --nulls 10   --seed <semilla fija del script>
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_complexity_probe_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

No importa ninguna capa: una importación cambiaría `uv.lock`. Abre por su ruta completa
`events.parquet` de L2, `summaries.parquet`, `path.parquet` y `spectrum.parquet` de L3 y el
`consolidated.parquet` del mes de L1, nunca un dataset sobre un prefijo, que pide un objeto sin
barra y la condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3,
«La trampa del 403 por prefijo»). Une las cuatro tablas por posición y comprueba que
`confirm_agg_trade_id` coincide.

Memoria: los ticks se leen un row group a la vez (`agg_trade_id` y `price`); los archivos por
evento, en lotes de `BATCH` filas. El estado por fase abierta es una ventana de a lo sumo
`--window + 1` precios; se libera al cerrar la fase. Los nulos de una fase suman `--nulls` copias de
la ventana por tipo, transitorias. Lo único que crece con la corrida son los resultados de las a lo
sumo `--max-phases` fases por fase.

Salida: una línea por mes al cerrarlo, las fases saltadas, el costo, la tabla del CSV y su ruta, una
línea de la fase os y, como última línea, `complexity probe: theta=… meses=… fases_c=… fases_os=…
D_hurst_sub=… D_pe_sub=… D_hurst_bar=… D_pe_bar=… piso_c=… rho_hurst=… rho_pe=… us_hurst=… us_pe=…
pared_s=… veredicto_hurst=sigue|retirada|fallo veredicto_pe=sigue|retirada|fallo`. `sub` es el
sustituto y `bar` el barajado; los D, `piso_c` y `rho_*` son los de la fase c (los de os van en el
CSV y en la línea `complexity probe os: …`); `us_*` son microsegundos por punto de la serie
observada.

Códigos: 0 si midió; 1 si falta algún archivo, la lectura falla, los archivos no cuadran entre sí o
no queda ninguna fase c medible; 2 por uso. Solo lee; lo único que escribe es el CSV.
"""

import argparse
import csv
import io
import math
import os
import re
import sys
import time
from collections import Counter
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

DEFAULT_SEED = 432_432
DEFAULT_NULLS = 10
DEFAULT_WINDOW = 10_000
DEFAULT_MIN_TICKS = 1_024
DEFAULT_MAX_PHASES = 600
# Umbrales de la regla de lectura, fijados antes de medir.
D_THRESHOLD = 0.10
RHO_THRESHOLD = 0.9
KS_FACTOR = 1.36
IAAFT_ITERATIONS = 20
DFA_FIRST_SCALE = 16
ORDER = 4
# Con la escala 16 y una segunda (32) hace falta `w / 4 ≥ 32`: la pendiente necesita dos escalas.
MIN_WINDOW = 128
# Filas por lote de los archivos por evento.
BATCH = 32_768
PHASES = ("c", "os")
MEASURES = ("hurst", "pe")
NULL_KINDS = ("barajado", "sustituto")
SHORT = {"barajado": "bar", "sustituto": "sub"}
QUANTILES = (5, 50, 95)
REDUNDANCY = (
    *(f"c_fourier_band_{b}" for b in range(1, 8)),
    "c_sign_autocorr",
    "c_ticks",
)
CSV_HEADER = ["seccion", "medida", "nulo", "fase", "estadistico", "valor"]
TICK_COLUMNS = ["agg_trade_id", "price"]
EVENT_COLUMNS = [
    "reference_price",
    "reference_agg_trade_id",
    "confirm_price",
    "confirm_agg_trade_id",
    "extreme_agg_trade_id",
]
SUMMARY_COLUMNS = ["confirm_agg_trade_id", "c_ticks", "os_ticks"]
PATH_COLUMNS = ["confirm_agg_trade_id", "c_sign_autocorr"]
SPECTRUM_COLUMNS = [
    "confirm_agg_trade_id",
    *(f"c_fourier_band_{b}" for b in range(1, 8)),
]
FILES = {
    "events": EVENT_COLUMNS,
    "summaries": SUMMARY_COLUMNS,
    "path": PATH_COLUMNS,
    "spectrum": SPECTRUM_COLUMNS,
}


class InputError(Exception):
    """Un archivo que falta o que no cuadra con los demás: sale con código 1."""


def log(message: str) -> None:
    print(message, flush=True)


def parse_theta(text: str) -> str:
    try:
        theta = Decimal(text)
        if not theta.is_finite() or theta <= 0:
            raise ValueError(f"--theta: {text!r} debe ser un número finito mayor que 0")
        # 1e30 no cabe en la precisión del contexto: también lanza InvalidOperation.
        scaled = theta.quantize(Decimal("1e-8"))
    except InvalidOperation:
        raise ValueError(f"--theta: {text!r} no es un θ válido") from None
    if scaled != theta:
        raise ValueError(f"θ {text} no cabe en 8 decimales, que es la partición")
    return format(scaled, "f")


def parse_months(text: str) -> list[str]:
    months = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        if re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", part) is None:
            raise ValueError(f"--months: {part!r} inválido: se espera YYYY-MM")
        months.append(part)
    if not months:
        raise ValueError("--months: falta al menos un mes YYYY-MM")
    if len(set(months)) != len(months):
        raise ValueError("--months: hay meses repetidos")
    return sorted(months)


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def series_root(root: str, key: tuple[str, str, str]) -> str:
    provider, market, asset = key
    return f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"


def l1_file(root: str, key: tuple[str, str, str], month: str) -> str:
    year, number = month.split("-")
    return f"{series_root(root, key)}/year={year}/month={number}/consolidated.parquet"


def partition_file(
    root: str, key: tuple[str, str, str], theta: str, month: str, name: str
) -> str:
    year, number = month.split("-")
    return f"{series_root(root, key)}/theta={theta}/year={year}/month={number}/{name}"


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        return pafs.FileSystem.from_uri(uri)
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def open_parquet(stack: ExitStack, uri: str) -> pq.ParquetFile:
    """Abre un Parquet por su ruta completa; `InputError` si no existe."""
    fs, path = open_fs(uri)
    if fs.get_file_info(path).type != pafs.FileType.File:
        raise InputError(f"falta el archivo {uri}")
    handle = stack.enter_context(fs.open_input_file(path))
    return pq.ParquetFile(handle)


def int64s(array: pa.Array) -> np.ndarray:
    """Una columna `int64` sin nulos, vista sin copia."""
    if array.null_count or not pa.types.is_int64(array.type):
        raise InputError("se esperaba una columna int64 sin nulos")
    words = np.frombuffer(array.buffers()[1], dtype="<i8")
    return words[array.offset : array.offset + len(array)]


def lows(array: pa.Array) -> np.ndarray:
    """Un `decimal128(18,8)` sin nulos como su entero sin escalar (la palabra baja)."""
    if (
        array.null_count
        or not pa.types.is_decimal128(array.type)
        or array.type.scale != 8
    ):
        raise InputError("se esperaba una columna decimal128(·, 8) sin nulos")
    words = np.frombuffer(array.buffers()[1], dtype="<i8")
    return words[2 * array.offset : 2 * (array.offset + len(array)) : 2]


def single(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    if isinstance(column, pa.Array):
        return column
    return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()


def tick_groups(parquet: pq.ParquetFile):
    """`(ids, precios)` de cada row group no vacío de L1, uno a la vez; precios en f64."""
    last = None
    for group in range(parquet.num_row_groups):
        if not parquet.metadata.row_group(group).num_rows:
            continue
        table = parquet.read_row_group(group, columns=TICK_COLUMNS)
        ids = int64s(single(table.column("agg_trade_id")))
        prices = lows(single(table.column("price"))).astype(np.float64)
        if (len(ids) > 1 and np.diff(ids).min() <= 0) or (
            last is not None and ids[0] <= last
        ):
            raise InputError(f"row group {group}: L1 debe venir ordenado")
        last = int(ids[-1])
        yield ids, prices
        del table, ids, prices


def column_rows(parquet: pq.ParquetFile, columns: list[str]):
    """Las filas de un archivo por evento como tuplas, un lote a la vez; `None` en los nulos.

    Los decimales salen como el entero sin escalar; los demás, como `int` o `float` de Python.
    """
    for batch in parquet.iter_batches(batch_size=BATCH, columns=columns):
        values = []
        for column in batch.columns:
            if pa.types.is_decimal128(column.type):
                values.append(lows(column).tolist())
            elif pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
                values.append(column.to_pylist())
            else:
                raise InputError(f"columna de tipo inesperado: {column.type}")
        yield from zip(*values, strict=True)


def event_rows(files: dict[str, pq.ParquetFile]):
    """Los cuatro archivos por evento, unidos por posición y con la clave comprobada."""
    for label, names in FILES.items():
        missing = set(names) - set(files[label].schema_arrow.names)
        if missing:
            raise InputError(f"{label}.parquet: faltan columnas {sorted(missing)}")
    joined = zip(
        *(column_rows(files[label], names) for label, names in FILES.items()),
        strict=True,
    )
    try:
        for event, summary, walk, spectrum in joined:
            if not event[3] == summary[0] == walk[0] == spectrum[0]:
                raise InputError(
                    "events, summaries, path y spectrum no están alineados: "
                    f"confirm_agg_trade_id {event[3]}, {summary[0]}, {walk[0]}, "
                    f"{spectrum[0]}"
                )
            yield event, summary, walk, spectrum
    except ValueError as exc:
        if "zip()" in str(exc):
            raise InputError(
                "events, summaries, path y spectrum tienen distinto número de filas"
            ) from exc
        raise


# ----------------------------------------------------------------------------------------------
# Las medidas: Hurst (DFA-1) y entropía de permutación
# ----------------------------------------------------------------------------------------------


def dfa_scales(n: int) -> list[int]:
    """Las escalas 16, 32, 64… hasta `n / 4`."""
    scales = []
    scale = DFA_FIRST_SCALE
    while scale <= n // 4:
        scales.append(scale)
        scale *= 2
    return scales


def hurst(returns: np.ndarray) -> np.ndarray:
    """H por DFA-1 de una serie `(n,)` o de varias `(g, n)`; `(g,)` con `nan` si algún F(s) es 0."""
    series = np.atleast_2d(returns)
    count, n = series.shape
    profile = np.cumsum(series - series.mean(axis=1, keepdims=True), axis=1)
    scales = dfa_scales(n)
    log_f = np.empty((count, len(scales)))
    zero = np.zeros(count, dtype=bool)
    for index, scale in enumerate(scales):
        windows = n // scale
        segments = profile[:, : windows * scale].reshape(count * windows, scale)
        # La recta de mínimos cuadrados de cada ventana, con el índice centrado.
        position = np.arange(scale) - (scale - 1) / 2
        slope = (segments @ position) / (position @ position)
        residual = segments - segments.mean(axis=1, keepdims=True)
        residual -= slope[:, None] * position
        residual *= residual
        square = residual.mean(axis=1).reshape(count, windows).mean(axis=1)
        zero |= square == 0
        with np.errstate(divide="ignore"):
            log_f[:, index] = 0.5 * np.log(square)
    log_s = np.log(np.array(scales, dtype=np.float64))
    log_s -= log_s.mean()
    with np.errstate(invalid="ignore"):  # -inf − (−inf) donde F(s) = 0
        centered = log_f - log_f.mean(axis=1, keepdims=True)
    result = (centered @ log_s) / (log_s @ log_s)
    result[zero] = np.nan
    return result


def permutation_entropy(prices: np.ndarray) -> np.ndarray:
    """Entropía de permutación (orden 4, retardo 1) de `(n,)` o `(g, n)`, normalizada a [0, 1].

    El orden de cuatro precios seguidos es el rango de cada uno entre los cuatro; en un empate
    el que apareció antes queda más abajo. Devuelve `(g,)`.
    """
    series = np.atleast_2d(prices)
    count, n = series.shape
    steps = n - ORDER + 1
    columns = [series[:, i : i + steps] for i in range(ORDER)]
    code = np.zeros((count, steps), dtype=np.int64)
    for i in range(ORDER):
        rank = np.zeros((count, steps), dtype=np.int64)
        for j in range(ORDER):
            if j < i:
                rank += columns[j] <= columns[i]
            elif j > i:
                rank += columns[j] < columns[i]
        code = code * ORDER + rank
    code += (np.arange(count) * ORDER**ORDER)[:, None]
    counts = np.bincount(code.ravel(), minlength=count * ORDER**ORDER)
    p = counts.reshape(count, ORDER**ORDER) / steps
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    entropy = 0.0 - terms.sum(axis=1)
    return np.clip(entropy / math.log(math.factorial(ORDER)), 0.0, 1.0)


def iaaft(shuffled: np.ndarray, original: np.ndarray) -> np.ndarray:
    """El sustituto IAAFT de cada fila de `shuffled` `(g, n)`, que se modifica y se devuelve.

    Parte del barajado y repite a lo sumo `IAAFT_ITERATIONS` veces: impone las amplitudes de
    Fourier de `original` conservando las fases actuales y reasigna por rango los valores de
    `original`. Siempre termina en la reasignación (las filas son una permutación exacta de
    `original`); se detiene antes si el orden de los rangos ya no cambia.
    """
    n = shuffled.shape[1]
    target = np.abs(np.fft.rfft(original))
    values = np.sort(original)
    order = np.argsort(shuffled, axis=1, kind="stable")
    for _ in range(IAAFT_ITERATIONS):
        spectrum = np.fft.rfft(shuffled, axis=1)
        magnitude = np.abs(spectrum)
        phase = np.divide(
            spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 0
        )
        candidate = np.fft.irfft(target * phase, n=n, axis=1)
        new_order = np.argsort(candidate, axis=1, kind="stable")
        if np.array_equal(new_order, order):
            break
        order = new_order
        np.put_along_axis(shuffled, order, np.broadcast_to(values, shuffled.shape), 1)
    return shuffled


class Result:
    """Lo medido en una ventana: observado `(2,)`, nulos `(nulls, 2)` por tipo y tiempos."""

    def __init__(self, observed, simulated, seconds, points):
        self.observed = observed  # [H, entropía]
        self.simulated = simulated  # {tipo de nulo: (nulls, 2)}
        self.seconds = seconds  # (hurst, pe) sobre la serie observada
        self.points = points

    def valid(self) -> bool:
        return bool(
            np.isfinite(self.observed).all()
            and all(np.isfinite(v).all() for v in self.simulated.values())
        )


def measure_window(buffer: np.ndarray, rng: np.random.Generator, nulls: int) -> Result:
    """Mide una ventana. `buffer` es `[p_0, p_1 … p_w]`: el precio previo y los `w` de la ventana.

    Retornos `r_k = ln(p_k / p_{k−1})` y entropía sobre `p_1 … p_w`. Cronometra solo el cálculo
    de cada medida sobre la serie observada.
    """
    returns = np.log(buffer[1:] / buffer[:-1])
    prices = buffer[1:]
    start = time.perf_counter()
    h = float(hurst(returns)[0])
    middle = time.perf_counter()
    entropy = float(permutation_entropy(prices)[0])
    end = time.perf_counter()

    shuffled = np.tile(returns, (nulls, 1))
    rng.permuted(shuffled, axis=1, out=shuffled)
    surrogate = iaaft(shuffled.copy(), returns)
    simulated = {}
    for kind, block in (("barajado", shuffled), ("sustituto", surrogate)):
        walk = np.cumsum(block, axis=1)
        np.exp(walk, out=walk)
        walk *= buffer[0]
        simulated[kind] = np.stack((hurst(block), permutation_entropy(walk)), axis=1)
    return Result(
        np.array([h, entropy]),
        simulated,
        (middle - start, end - middle),
        len(returns),
    )


# ----------------------------------------------------------------------------------------------
# Muestras, D, cuantiles, Spearman y veredicto
# ----------------------------------------------------------------------------------------------


class Samples:
    """Los valores observados y simulados de las fases medidas, en memoria reservada al abrir."""

    def __init__(self, capacity: dict[str, int], nulls: int) -> None:
        self.nulls = nulls
        self.n = dict.fromkeys(PHASES, 0)
        self.obs = {p: np.empty((capacity[p], 2)) for p in PHASES}
        self.sim = {
            (kind, p): np.empty((capacity[p] * nulls, 2))
            for kind in NULL_KINDS
            for p in PHASES
        }
        self.columns = np.empty((capacity["c"], len(REDUNDANCY)))
        # Costo: segundos de cada medida, puntos, y el mayor tiempo de una fase.
        self.seconds = [0.0, 0.0]
        self.worst = [0.0, 0.0]
        self.points = 0
        self.windows = 0

    def cost(self, result: Result) -> None:
        for i, seconds in enumerate(result.seconds):
            self.seconds[i] += seconds
            self.worst[i] = max(self.worst[i], seconds)
        self.points += result.points
        self.windows += 1

    def add(self, phase: str, result: Result, columns: list | None) -> None:
        i = self.n[phase]
        self.obs[phase][i] = result.observed
        for kind in NULL_KINDS:
            self.sim[kind, phase][i * self.nulls : (i + 1) * self.nulls] = (
                result.simulated[kind]
            )
        if phase == "c":
            self.columns[i] = [np.nan if v is None else v for v in columns]
        self.n[phase] = i + 1

    def observed(self, phase: str, measure: int) -> np.ndarray:
        return self.obs[phase][: self.n[phase], measure]

    def simulated(self, kind: str, phase: str, measure: int) -> np.ndarray:
        return self.sim[kind, phase][: self.n[phase] * self.nulls, measure]


def ks_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    """`sup |F_obs − F_sim|` exacto de dos muestras ordenadas, con cuentas enteras.

    Las dos distribuciones empíricas son continuas por la derecha y solo saltan en los valores
    de las muestras, así que el supremo se alcanza en alguno de ellos.
    """
    n, m = len(observed), len(simulated)
    best = 0
    for points in (observed, simulated):
        for start in range(0, len(points), 1 << 20):
            x = points[start : start + (1 << 20)]
            below_obs = np.searchsorted(observed, x, side="right").astype(np.int64)
            below_sim = np.searchsorted(simulated, x, side="right").astype(np.int64)
            best = max(best, int(np.abs(below_obs * m - below_sim * n).max()))
    return best / (n * m)


def floor_of(n: int, m: int) -> float:
    """El piso del tamaño de muestra: `1,36 · √((N + M) / (N · M))`."""
    return KS_FACTOR * math.sqrt((n + m) / (n * m))


def quantile(values: np.ndarray, percent: int) -> float:
    """Rango más cercano: el lugar ⌈N·q⌉ de los valores ya ordenados."""
    rank = -(-len(values) * percent // 100)
    return float(values[rank - 1])


def ranks(values: np.ndarray) -> np.ndarray:
    """Rangos (desde 1), con el promedio en los empates."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
    ends = np.append(starts[1:], len(values))
    result = np.empty(len(values))
    result[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """ρ de Spearman de los pares finitos; `nan` si falta variación o hay menos de 2 pares."""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return math.nan
    rx, ry = ranks(x[keep]), ranks(y[keep])
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return math.nan if denominator == 0 else float((rx * ry).sum()) / denominator


def max_abs(values) -> float:
    """Máximo de |ρ| sin contar los `nan`; `nan` si no queda ninguno."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.isnan(values).all():
        return math.nan
    return float(np.nanmax(np.abs(values)))


def distances(samples: Samples) -> dict:
    """D, N, M y piso por medida, fase y nulo: `{(medida, fase, nulo): (D, N, M, piso)}`."""
    result = {}
    for phase in PHASES:
        n = samples.n[phase]
        if not n:
            continue
        m = n * samples.nulls
        for measure, name in enumerate(MEASURES):
            observed = np.sort(samples.observed(phase, measure))
            for kind in NULL_KINDS:
                simulated = np.sort(samples.simulated(kind, phase, measure))
                result[name, phase, kind] = (
                    ks_distance(observed, simulated),
                    n,
                    m,
                    floor_of(n, m),
                )
    return result


def redundancy(samples: Samples) -> dict[tuple[str, str], float]:
    """ρ de cada medida contra cada columna que ya existe, en la fase c."""
    n = samples.n["c"]
    return {
        (name, column): spearman(
            samples.observed("c", measure), samples.columns[:n, index]
        )
        for measure, name in enumerate(MEASURES)
        for index, column in enumerate(REDUNDANCY)
    }


def verdict_of(d: float, floor: float, rho: float) -> str:
    """`sigue` si D ≥ 0,10, D ≥ piso y |ρ| máximo ≤ 0,9; si no, `retirada`.

    Un `nan` en ρ (columna constante o sin pares) no cuenta como redundancia; un `nan` en D no
    pasa.
    """
    if math.isnan(d) or d < D_THRESHOLD or d < floor:
        return "retirada"
    return "sigue" if math.isnan(rho) or rho <= RHO_THRESHOLD else "retirada"


# ----------------------------------------------------------------------------------------------
# Lectura de los meses
# ----------------------------------------------------------------------------------------------


def count_eligible(uri: str, min_ticks: int) -> Counter:
    """Las fases con `ticks ≥ min_ticks` de un `summaries.parquet`, por fase (solo lee 2 columnas)."""
    counts: Counter = Counter()
    with ExitStack() as stack:
        parquet = open_parquet(stack, uri)
        missing = {"c_ticks", "os_ticks"} - set(parquet.schema_arrow.names)
        if missing:
            raise InputError(f"summaries.parquet: faltan columnas {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=BATCH, columns=["c_ticks", "os_ticks"]
        ):
            counts["c"] += int((int64s(batch.column("c_ticks")) >= min_ticks).sum())
            counts["os"] += int((int64s(batch.column("os_ticks")) >= min_ticks).sum())
    return counts


def spaced(total: int, cap: int) -> set[int] | None:
    """Los índices que toma el tope: `⌊j · total / cap⌋`, j = 0..cap−1; `None` toma todas."""
    if total <= cap:
        return None
    return {(j * total) // cap for j in range(cap)}


class Plan:
    """Qué fases elegibles se miden: las espaciadas, por posición entre todos los meses."""

    def __init__(self, totals: Counter, cap: int) -> None:
        self.chosen = {p: spaced(totals[p], cap) for p in PHASES}
        self.capacity = {p: min(totals[p], cap) for p in PHASES}
        self.position: Counter = Counter()

    def take(self, phase: str) -> bool:
        index = self.position[phase]
        self.position[phase] += 1
        chosen = self.chosen[phase]
        return chosen is None or index in chosen


class Phase:
    """Una fase abierta: guarda solo la ventana. `c` los últimos precios; `os`, los primeros."""

    __slots__ = ("anchor", "buffer", "hi", "keep", "lo", "n", "name", "read", "seen")

    def __init__(self, name, lo, hi, anchor, n, window):
        self.name, self.lo, self.hi, self.n = name, lo, hi, n
        self.anchor = anchor
        # `w + 1` valores: el precio previo a la ventana y sus `w` precios.
        self.keep = min(n, window) + 1
        self.read = False
        self.seen = 0
        self.buffer = np.array([anchor])

    def feed(self, chunk: np.ndarray) -> None:
        """Suma un tramo de precios; en c conserva los últimos `keep`, en os los primeros."""
        if self.name == "c":
            excess = len(self.buffer) + len(chunk) - self.keep
            self.buffer = np.concatenate(
                (self.buffer[max(0, excess) :], chunk[max(0, len(chunk) - self.keep) :])
            )
        else:
            room = self.keep - len(self.buffer)
            if room > 0:
                self.buffer = np.concatenate((self.buffer, chunk[:room]))
        self.seen += len(chunk)


class Event:
    """El evento abierto: sus dos fases y qué fase se lee."""

    def __init__(self, row: tuple, plan: Plan, counters: Counter, ns) -> None:
        event, summary, walk, spectrum = row
        ref_price, self.ref_id, conf_price, self.conf_id, self.ext_id = event
        _, c_ticks, os_ticks = summary
        if (
            os_ticks < 0
            or c_ticks < 0
            or (os_ticks == 0) != (self.ext_id == self.conf_id)
        ):
            raise InputError(
                f"confirm_agg_trade_id {self.conf_id}: os_ticks={os_ticks} no cuadra con "
                "extreme_agg_trade_id"
            )
        # Las columnas contra las que se mide la redundancia, en el orden de `REDUNDANCY`.
        self.columns = [*spectrum[1:], walk[1], c_ticks]
        self.phases = [
            Phase("c", self.ref_id, self.conf_id, float(ref_price), c_ticks, ns.window),
            Phase(
                "os", self.conf_id, self.ext_id, float(conf_price), os_ticks, ns.window
            ),
        ]
        self.truncated = False
        for phase in self.phases:
            if phase.n == 0:
                counters[phase.name, "overshoot_vacio"] += 1
            elif phase.n < ns.min_ticks:
                counters[phase.name, "bajo_minimo"] += 1
            elif plan.take(phase.name):
                phase.read = True
            else:
                counters[phase.name, "tope"] += 1

    def drop(self) -> None:
        for phase in self.phases:
            phase.read = False


def measure_month(
    uris: dict[str, str],
    plan: Plan,
    samples: Samples,
    counters: Counter,
    rng: np.random.Generator,
    ns,
) -> None:
    """Recorre el mes una vez y suma sus fases a `samples` y los excluidos a `counters`."""
    with ExitStack() as stack:
        files = {label: open_parquet(stack, uri) for label, uri in uris.items()}
        rows = event_rows({k: files[k] for k in FILES})

        def next_event() -> Event | None:
            row = next(rows, None)
            return None if row is None else Event(row, plan, counters, ns)

        def close(event: Event, phase: Phase, first_id: int) -> None:
            if phase.seen != phase.n:
                if event.ref_id < first_id:
                    counters["evento_mes_anterior"] += 1
                    event.truncated = True
                    event.drop()  # L1 no trae el evento entero
                    return
                raise InputError(
                    f"confirm_agg_trade_id {event.conf_id}: L1 trae {phase.seen} ticks de "
                    f"{phase.name} y summaries dice {phase.n}"
                )
            result = measure_window(phase.buffer, rng, ns.nulls)
            phase.buffer, phase.read = None, False
            samples.cost(result)
            if result.valid():
                samples.add(phase.name, result, event.columns)
            else:
                counters[phase.name, "nulas"] += 1

        first_id = None
        current = next_event()
        for ids, prices in tick_groups(files["l1"]):
            if first_id is None:
                first_id = int(ids[0])
            last_id = int(ids[-1])
            while current is not None:
                for phase in current.phases:
                    if not phase.read:
                        continue
                    start = int(np.searchsorted(ids, phase.lo, side="right"))
                    stop = int(np.searchsorted(ids, phase.hi, side="right"))
                    if stop > start:
                        if phase.seen + (stop - start) > phase.n:
                            raise InputError(
                                f"confirm_agg_trade_id {current.conf_id}: L1 trae más ticks "
                                f"de {phase.name} que los {phase.n} de summaries"
                            )
                        phase.feed(prices[start:stop])
                    if phase.hi <= last_id:
                        close(current, phase, first_id)
                        if current.truncated:
                            break
                if current.ext_id > last_id:
                    break
                current = next_event()
            if current is None:
                break
        if current is not None or next(rows, None) is not None:
            raise InputError("L2 trae eventos más allá de los ticks del mes de L1")
        if first_id is None:
            raise InputError("el consolidated.parquet del mes no tiene ticks")


# ----------------------------------------------------------------------------------------------
# CSV y última línea
# ----------------------------------------------------------------------------------------------


def number(value: float) -> str:
    return format(value, ".17g")


def skipped_text(counters: Counter) -> str:
    parts = []
    for phase in PHASES:
        reasons = ("overshoot_vacio", "bajo_minimo", "tope", "nulas")
        parts.append(
            f"{phase}[" + " ".join(f"{r}={counters[phase, r]}" for r in reasons) + "]"
        )
    parts.append(f"evento_mes_anterior={counters['evento_mes_anterior']}")
    return " ".join(parts)


def csv_rows(
    samples: Samples,
    counters: Counter,
    dist: dict,
    rho: dict,
    verdicts: dict[str, str],
    config: dict,
) -> list[dict]:
    rows = []

    def add(section, measure, null, phase, statistic, value):
        rows.append(
            dict(
                zip(
                    CSV_HEADER,
                    (section, measure, null, phase, statistic, value),
                    strict=True,
                )
            )
        )

    for phase in PHASES:
        if not samples.n[phase]:
            continue
        for measure, name in enumerate(MEASURES):
            for source, values in (
                ("observado", samples.observed(phase, measure)),
                *(
                    (kind, samples.simulated(kind, phase, measure))
                    for kind in NULL_KINDS
                ),
            ):
                column = np.sort(values)
                for percent in QUANTILES:
                    add(
                        "cuantiles", name, source, phase, f"p{percent}",
                        number(quantile(column, percent)),
                    )  # fmt: skip
                add("cuantiles", name, source, phase, "N", len(column))
    for (name, phase, kind), (d, n, m, floor) in dist.items():
        add("D", name, kind, phase, "D", number(d))
        add("D", name, kind, phase, "N", n)
        add("D", name, kind, phase, "M", m)
        add("D", name, kind, phase, "piso", number(floor))
    for name in MEASURES:
        values = [rho[name, column] for column in REDUNDANCY]
        for column, value in zip(REDUNDANCY, values, strict=True):
            add("rho", name, "", "c", column, number(value))
        add("rho", name, "", "c", "max_abs_rho", number(max_abs(values)))
        add("veredicto", name, "", "c", "veredicto", verdicts[name])
    for phase in PHASES:
        for reason in ("overshoot_vacio", "bajo_minimo", "tope", "nulas"):
            add("excluidos", "", "", phase, reason, counters[phase, reason])
        add("muestras", "", "", phase, "N", samples.n[phase])
    add("excluidos", "", "", "", "evento_mes_anterior", counters["evento_mes_anterior"])
    for key, value in config.items():
        add("config", "", "", "", key, value)
    return rows


def write_csv(rows: list[dict], out: str) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if out.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(out)
        with fs.open_output_stream(path) as stream:
            stream.write(data)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)


def final_line(theta, months, wall, figures: dict | None = None) -> str:
    def cell(key, fmt="{:.6f}"):
        value = None if figures is None else figures.get(key)
        return "-" if value is None else fmt.format(value)

    def verdict(name):
        return "fallo" if figures is None else figures[f"veredicto_{name}"]

    return (
        f"complexity probe: theta={theta} meses={','.join(months)} "
        f"fases_c={cell('fases_c', '{}')} fases_os={cell('fases_os', '{}')} "
        f"D_hurst_sub={cell('D_hurst_sub')} D_pe_sub={cell('D_pe_sub')} "
        f"D_hurst_bar={cell('D_hurst_bar')} D_pe_bar={cell('D_pe_bar')} "
        f"piso_c={cell('piso_c')} rho_hurst={cell('rho_hurst')} rho_pe={cell('rho_pe')} "
        f"us_hurst={cell('us_hurst', '{:.3f}')} us_pe={cell('us_pe', '{:.3f}')} "
        f"pared_s={wall:.1f} veredicto_hurst={verdict('hurst')} "
        f"veredicto_pe={verdict('pe')}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    try:
        theta = parse_theta(ns.theta)
        months = parse_months(ns.months)
    except ValueError as exc:
        parser.error(str(exc))
    if ns.nulls < 1:
        parser.error("--nulls debe ser un entero mayor que 0")
    if ns.max_phases < 2:
        parser.error("--max-phases debe ser un entero de al menos 2")
    if ns.min_ticks < MIN_WINDOW or ns.window < MIN_WINDOW:
        parser.error(
            f"--min-ticks y --window deben ser de al menos {MIN_WINDOW}: el DFA "
            "necesita dos escalas"
        )
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_complexity_probe_{theta}.csv"

    name_project = project()
    if not (ns.l1_root and ns.l2_root and ns.l3_root) and name_project is None:
        parser.error(
            "sin las tres raíces y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    l1_root = ns.l1_root or f"gs://{name_project}-landing/l1"
    l2_root = ns.l2_root or f"gs://{name_project}-dc-events/l2"
    l3_root = ns.l3_root or f"gs://{name_project}-dc-descriptors/l3"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))

    uris = {
        month: {
            "l1": l1_file(l1_root, key, month),
            "events": partition_file(l2_root, key, theta, month, "events.parquet"),
            "summaries": partition_file(
                l3_root, key, theta, month, "summaries.parquet"
            ),
            "path": partition_file(l3_root, key, theta, month, "path.parquet"),
            "spectrum": partition_file(l3_root, key, theta, month, "spectrum.parquet"),
        }
        for month in months
    }
    started = time.perf_counter()
    log(
        f"complexity probe: θ={theta} meses={','.join(months)} nulls={ns.nulls} "
        f"ventana={ns.window} min_ticks={ns.min_ticks} max_phases={ns.max_phases} "
        f"semilla={ns.seed}"
    )
    counters: Counter = Counter()
    try:
        totals: Counter = Counter()
        for month in months:
            totals.update(count_eligible(uris[month]["summaries"], ns.min_ticks))
        log(
            f"fases con al menos {ns.min_ticks} ticks: c={totals['c']} os={totals['os']}"
        )
        plan = Plan(totals, ns.max_phases)
        samples = Samples(plan.capacity, ns.nulls)
        rng = np.random.default_rng(ns.seed)
        for month in months:
            before = dict(samples.n)
            month_counters: Counter = Counter()
            measure_month(uris[month], plan, samples, month_counters, rng, ns)
            counters.update(month_counters)
            log(
                f"mes {month}: fases_c={samples.n['c'] - before['c']} "
                f"fases_os={samples.n['os'] - before['os']} "
                f"saltadas {skipped_text(month_counters)}"
            )
        if samples.n["c"] == 0:
            raise InputError("ninguna fase c medible en los meses")
        dist = distances(samples)
        rho = redundancy(samples)
    except (InputError, OSError, ValueError, pa.ArrowException) as exc:
        log(f"FALLO {type(exc).__name__}: {exc}")
        log(final_line(theta, months, time.perf_counter() - started))
        return 1
    wall = time.perf_counter() - started

    max_rho = {name: max_abs([rho[name, c] for c in REDUNDANCY]) for name in MEASURES}
    verdicts = {
        name: verdict_of(
            dist[name, "c", "sustituto"][0],
            dist[name, "c", "sustituto"][3],
            max_rho[name],
        )
        for name in MEASURES
    }
    config = {
        "theta": theta,
        "meses": ",".join(months),
        "nulls": ns.nulls,
        "semilla": ns.seed,
        "ventana": ns.window,
        "min_ticks": ns.min_ticks,
        "max_phases": ns.max_phases,
    }
    rows = csv_rows(samples, counters, dist, rho, verdicts, config)
    log("  ".join(f"{h:<12}" for h in CSV_HEADER))
    for row in rows:
        log("  ".join(f"{row[h]!s:<12}" for h in CSV_HEADER))
    log(f"saltadas en total: {skipped_text(counters)}")
    per_point = [
        1e6 * s / samples.points if samples.points else math.nan
        for s in samples.seconds
    ]
    per_phase = [1e3 * s / max(samples.windows, 1) for s in samples.seconds]
    log(
        "costo (solo la serie observada): "
        + " ".join(
            f"{name}={us:.3f} us/punto ({ms:.3f} ms por fase, máx {1e3 * worst:.3f} ms)"
            for name, us, ms, worst in zip(
                MEASURES, per_point, per_phase, samples.worst, strict=True
            )
        )
    )
    try:
        write_csv(rows, out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")

    if samples.n["os"]:
        log(
            f"complexity probe os: fases_os={samples.n['os']} "
            + " ".join(
                f"D_{name}_{SHORT[kind]}={dist[name, 'os', kind][0]:.6f}"
                for kind in ("sustituto", "barajado")
                for name in MEASURES
            )
            + f" piso_os={dist['hurst', 'os', 'sustituto'][3]:.6f}"
        )
    else:
        log("complexity probe os: fases_os=0")
    figures = {
        "fases_c": samples.n["c"],
        "fases_os": samples.n["os"],
        "piso_c": dist["hurst", "c", "sustituto"][3],
        "veredicto_hurst": verdicts["hurst"],
        "veredicto_pe": verdicts["pe"],
        "us_hurst": per_point[0],
        "us_pe": per_point[1],
    }
    for name in MEASURES:
        figures[f"rho_{name}"] = max_rho[name]
        for kind in NULL_KINDS:
            figures[f"D_{name}_{SHORT[kind]}"] = dist[name, "c", kind][0]
    log(final_line(theta, months, wall, figures))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--l3-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--min-ticks", type=int, default=DEFAULT_MIN_TICKS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--max-phases", type=int, default=DEFAULT_MAX_PHASES)
    parser.add_argument("--nulls", type=int, default=DEFAULT_NULLS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
