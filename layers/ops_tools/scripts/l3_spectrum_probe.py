"""Sonda doble de la familia F (Frecuencia) de L3 (ITSC-411).

Decide qué registros de la familia F valen la pena construir. Sobre un θ y un mes del lago mide,
por evento y por fase (`c` = (R, C] y `os` = (C, E]), cuánta energía de la serie de ticks cae en
cada una de 7 bandas de período, con dos candidatos (decisión 1 de la Épica ITSC-410) y la
descomposición de Haar (decisión 2), y la compara con sus nulos. El tamaño del efecto es la
distancia de Kolmogorov-Smirnov D; nunca se reporta un valor p. Además mide la redundancia
(Spearman) con `c_accel` (familia A) y `c_t50` (familia T), la fracción de fases bajo 16 ticks por
θ y, solo con L2, el pico de ticks en vuelo de los 50 θ a la vez (decisión 6).

Series por fase, en tiempo-tick (sin remuestrear), con `n` ticks, el ancla `p_0` (`p_R` en c,
`p_C` en os, como T) y `p_1 … p_n` los precios de los ticks:

  retornos  r_k = ln(p_k / p_{k−1}),  k = 1..n   (el primer paso sale del ancla; los ceros se
            conservan)
  precios   s_k = (p_k − recta_k) / p_0,  k = 1..n,  con recta_k = p_0 + (p_n − p_0)·k/n
            (la recta del ancla, índice 0, al último tick, índice n; s_n = 0)

Espectro (decisión 3). La serie se centra (resta de la media), se multiplica por la ventana de
Hann periódica `0,5 − 0,5·cos(2π·j/n)`, j = 0..n−1, y se toma `numpy.fft.rfft`. La energía es
`|X_k|²` para k = 1..⌊n/2⌋ (el bin 0 queda fuera: la serie está centrada) y cada banda es la
fracción de la energía total de los k cuyo período `n/k` cae en ella:

  banda 1 [2, 4)   2 [4, 8)   3 [8, 16)   4 [16, 32)   5 [32, 64)   6 [64, 128)   7 [128, ∞)

(en ticks; suman 1). Con `n = 64` y una senoide de período 8, la ventana de Hann deja 2/3 en el bin
8, y 1/6 en cada vecino: 5/6 en la banda 3 y 1/6 en la 2.

Haar (decisión 2, convención de la hija 2). Sobre los últimos `2^⌊log₂ n⌋` retornos de la fase (los
más cercanos al cierre), centrados, se aplica la transformada de Haar ortonormal
(`a' = (x_2i + x_2i+1)/√2`, `d = (x_2i − x_2i+1)/√2`). La energía por nivel es `Σ d²`: los niveles
1 a 6 son las escalas 2, 4, 8, 16, 32 y 64 ticks y el séptimo junta las escalas de 128 en
adelante. Alinea el nivel j con la banda j de Fourier; suman la energía total por Parseval (la
serie centrada no deja energía en el promedio). Sin PyWavelets ni scipy.

Nulos (decisión 1), `--nulls` (20) por fase y evento, con una sola semilla:
  - retornos: los retornos de la fase barajados;
  - precios: el puente browniano de ITSC-398 en los `n` ticks de la fase,
    `ℓ_k = ln p_0 + (ln p_n − ln p_0)·k/n + σ·(W_k − (k/n)·W_n)`, con `σ² = c_realized_var / n`
    (c) o `os_realized_var / os_ticks` (os) de `path.parquet`, llevado a precios con la misma
    recta. Su extremo es el último tick observado;
  - Haar: la de los mismos retornos barajados.
D es la distancia exacta `sup |F_obs − F_sim|` de la fracción de energía en la banda más rápida
(banda 1, período 2 a 4) y en la más lenta (banda 7, 128 o más). Los simulados se guardan en
`float32` (≈ 3,7 KB por evento con 20 nulos y las dos fases); D se evalúa con el observado
redondeado a `float32` también, y los cuantiles del CSV son los de cada muestra.
Los cuantiles usan el rango más cercano: con N valores, el cuantil q ocupa el lugar ⌈N·q⌉.

Redundancia: ρ de Spearman (rangos promedio en los empates), banda a banda, entre las fracciones
de precios y las de retornos, entre las de Haar y las de Fourier de retornos, y entre cada banda de
cada candidato y `c_accel` (`fit.parquet`) y `c_t50` (`path.parquet`); en la fase c. La última
línea lleva el máximo de |ρ| de cada par. Una columna constante da ρ indefinida (`nan`), que no
cuenta como redundancia.

`f32_err` es el máximo de |fracción en f32 − fracción en f64| sobre las fases del mes y las 7
bandas de los dos candidatos (la FFT y las sumas, enteras en f32; decisión 5).

Umbral (decisión 1, fijado antes de medir): `veredicto=sigue` si y solo si algún candidato
(retornos o precios) tiene `D_fast ≥ 0,10` o `D_slow ≥ 0,10` sobre c y su |ρ| máximo contra
`c_accel` y `c_t50` no pasa de 0,9; si no, `retirada`. Qué registros entran lo decide el arquitecto
con las cifras.

Exclusiones, todas contadas en el CSV: la fase con menos de 16 ticks (decisión 4), el evento sobre
el presupuesto (`extreme_agg_trade_id − reference_agg_trade_id > 20 000 000`, decisión 6; sus dos
fases), la fase con energía cero (todos los retornos iguales, o una serie de precios sobre su
recta) y el evento que arranca en el mes anterior y no cabe en el L1 del mes (a lo sumo uno por mes
y θ). Una fase con `os_ticks = 0` no existe y se cuenta aparte.

Pase de solo L2 y L3 (decisiones 4 y 6), sin leer L1, sobre todos los θ que tienen archivos en el
mes:
  (a) pico de ticks en vuelo. Con `R`, `C` y `E` de `events.parquet`, la fase abierta de un θ en el
      tick `x` es c si `R < x ≤ C` (cuenta `x − R`) y os si `C < x ≤ E` (cuenta `x − C`); un evento
      sobre el presupuesto cuenta 0 porque no guarda retornos. El pico de la suma sobre θ está en
      un punto de corte, así que se recorre una mezcla ordenada de los archivos por lotes:
      estado O(θ) y memoria O(lote × θ). Se reporta en ticks y en bytes a 8 B y a 4 B por tick.
      Es una cota superior (los ids tienen huecos);
  (b) fracción de fases c y os con menos de 16 ticks por θ, de `summaries.parquet` del mes
      (`--month`); las fases os vacías (`os_ticks = 0`) cuentan como bajo 16 y se informan aparte.

Args:
  --theta 0.00098052  --month YYYY-MM          (obligatorios; un θ y un mes por corrida)
  --l1-root gs://<proyecto>-landing/l1         (por defecto, las del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2        OPS_RESULTS_URI; también rutas locales con la
  --l3-root gs://<proyecto>-dc-descriptors/l3   misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --nulls 20   --seed <semilla fija del script>
  --inflight-months 2017-12,2021-01,2023-03   (vacío: sin pase (a))
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_spectrum_probe_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

Con el mismo `--seed`, los mismos archivos y los mismos argumentos el CSV es idéntico byte a byte.

No importa ninguna capa: una importación cambiaría `uv.lock`. Abre por su ruta completa
`events.parquet` de L2, `summaries.parquet`, `path.parquet` y `fit.parquet` de L3 y el
`consolidated.parquet` del mes de L1, nunca un dataset sobre un prefijo, que pide un objeto sin
barra y la condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3,
«La trampa del 403 por prefijo»); los θ de los pases se descubren listando de forma recursiva.
Une las cuatro tablas por posición y comprueba que `confirm_agg_trade_id` coincide.

Memoria: los ticks se leen un row group a la vez (`agg_trade_id` y `price`); los cuatro archivos por
evento, en lotes de `BATCH` filas. El estado es la serie de la fase abierta (`n` precios en f64,
a lo sumo el presupuesto: 160 MB), que se libera al cerrar la fase; nunca el mes. Al cerrar, el
cálculo suma un trabajo transitorio de varias copias de la serie (retornos, centrada, espectro) y
los nulos se generan en grupos de a lo sumo `GROUP` valores. Medido: ≈ 2 ms por fase típica; una
fase de 2 M ticks sube el pico unos 130 MiB y una de 20 M ticks (el presupuesto) llega a 1,7 GiB
(≈ 9 veces la serie), dentro de los 8 GiB del job `ops-script`. Lo único que crece con el mes son
los resultados por evento.

Salida: la tabla del CSV, su ruta y, como última línea, `spectrum probe: mes=… theta=… eventos=…
D_ret_fast=… D_ret_slow=… D_price_fast=… D_price_slow=… rho_price_ret=… rho_haar_fourier=…
rho_ret_accel=… rho_ret_t50=… rho_price_accel=… rho_price_t50=… f32_err=… n_lt16=…
inflight_peak_ticks=… pared_s=… veredicto=sigue|retirada|fallo` (`eventos` son los N de la fase c;
`n_lt16` el número de fases c del θ bajo 16 ticks; `inflight_peak_ticks` el mayor de los meses).

Códigos: 0 si midió; 1 si falta alguno de los archivos, la lectura falla, los archivos no cuadran
entre sí o no queda ningún evento medible; 2 por uso. Solo lee; lo único que escribe es el CSV.
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

DEFAULT_SEED = 410_411
DEFAULT_NULLS = 20
DEFAULT_INFLIGHT_MONTHS = "2017-12,2021-01,2023-03"
# Umbrales de la decisión 1, fijados antes de medir.
D_THRESHOLD = 0.10
RHO_THRESHOLD = 0.9
# Decisión 4 (mínimo de ticks de una fase) y decisión 6 (presupuesto de un evento).
N_MIN = 16
BUDGET = 20_000_000
BANDS = 7
# Filas por lote de los archivos por evento; filas por lote de L2 en el pase de ticks en vuelo; y
# valores como máximo por grupo de nulos (8 MB en f64).
BATCH = 32_768
INFLIGHT_BATCH = 8_192
GROUP = 1 << 20
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
PHASES = ("c", "os")
CANDIDATES = ("retornos", "precios", "haar")
CSV_HEADER = ["seccion", "candidato", "fase", "banda", "fuente", "medida", "valor"]
TICK_COLUMNS = ["agg_trade_id", "price"]
EVENT_COLUMNS = [
    "reference_price",
    "reference_agg_trade_id",
    "confirm_price",
    "confirm_agg_trade_id",
    "extreme_agg_trade_id",
]
SUMMARY_COLUMNS = ["confirm_agg_trade_id", "c_ticks", "os_ticks"]
PATH_COLUMNS = ["confirm_agg_trade_id", "c_realized_var", "os_realized_var", "c_t50"]
FIT_COLUMNS = ["confirm_agg_trade_id", "c_accel"]
FILES = {
    "events": EVENT_COLUMNS,
    "summaries": SUMMARY_COLUMNS,
    "path": PATH_COLUMNS,
    "fit": FIT_COLUMNS,
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
            raise ValueError(f"--inflight-months: {part!r} inválido: se espera YYYY-MM")
        months.append(part)
    return months


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
    """Abre un Parquet por su ruta completa; `FileNotFoundError` si no existe."""
    fs, path = open_fs(uri)
    if fs.get_file_info(path).type != pafs.FileType.File:
        raise FileNotFoundError(uri)
    handle = stack.enter_context(fs.open_input_file(path))
    return pq.ParquetFile(handle)


def list_files(root: str) -> list[str]:
    """Todos los archivos bajo `root`, de forma recursiva: objetos por prefijo en GCS."""
    if not root.startswith("gs://"):
        return [str(p) for p in Path(root).rglob("*") if p.is_file()]
    from google.cloud import storage

    bucket, _, prefix = root.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    blobs = storage.Client().list_blobs(bucket, prefix=prefix)
    return [f"gs://{bucket}/{blob.name}" for blob in blobs]


def thetas_of(files: list[str], month: str, name: str) -> dict[str, str]:
    """`{θ: ruta}` de los archivos `name` del mes, entre los `files` listados."""
    year, number = month.split("-")
    pattern = re.compile(
        rf"/theta=(0\.\d{{8}})/year={year}/month={number}/{re.escape(name)}$"
    )
    found = {}
    for path in files:
        match = pattern.search(path)
        if match:
            found[match[1]] = path
    return dict(sorted(found.items()))


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
        for event, summary, walk, fit in joined:
            if not event[3] == summary[0] == walk[0] == fit[0]:
                raise InputError(
                    "events, summaries, path y fit no están alineados: "
                    f"confirm_agg_trade_id {event[3]}, {summary[0]}, {walk[0]}, {fit[0]}"
                )
            yield event, summary, walk, fit
    except ValueError as exc:
        if "zip()" in str(exc):
            raise InputError(
                "events, summaries, path y fit tienen distinto número de filas"
            ) from exc
        raise


# ----------------------------------------------------------------------------------------------
# Espectro, Haar y puente
# ----------------------------------------------------------------------------------------------


def hann(n: int, dtype=np.float64) -> np.ndarray:
    """La ventana de Hann periódica, `0,5 − 0,5·cos(2π·j/n)`."""
    window = np.arange(n, dtype=np.float64)
    window *= 2.0 * np.pi / n
    np.cos(window, out=window)
    window *= -0.5
    window += 0.5
    return window.astype(dtype, copy=False)


def band_slices(n: int) -> list[tuple[int, int]]:
    """Los índices `[lo, hi)` de la energía de los bins k = 1..⌊n/2⌋ (índice k − 1) por banda.

    La banda 1 es el período `n/k ∈ [2, 4)`, o sea `n/4 < k ≤ n/2`, y así hasta la 7, `k ≤ n/128`.
    """
    slices = [(n >> 2, n >> 1)]
    slices += [(n >> (b + 2), n >> (b + 1)) for b in range(1, 6)]
    slices.append((0, n >> 7))
    return slices


def fractions(series: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Fracción de la energía por banda de una serie `(n,)` o de varias `(g, n)`.

    Devuelve `(7,)` o `(g, 7)` con el dtype de la serie, y `nan` en las series de energía cero.
    """
    n = series.shape[-1]
    centered = series - series.mean(axis=-1, keepdims=True)
    centered *= window
    spectrum = np.fft.rfft(centered, axis=-1)
    del centered
    # Los bins k = 1..⌊n/2⌋ (índice k − 1): el bin 0 queda fuera.
    energy = spectrum[..., 1 : n // 2 + 1]
    energy = energy.real**2 + energy.imag**2
    del spectrum
    bands = np.stack(
        [
            np.sum(energy[..., lo:hi], axis=-1, dtype=energy.dtype)
            for lo, hi in band_slices(n)
        ],
        axis=-1,
    )
    del energy
    total = bands.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = bands / total
    flat = series.min(axis=-1) == series.max(axis=-1)
    result[np.broadcast_to(flat, result.shape[:-1])] = np.nan
    return result


def haar(series: np.ndarray) -> np.ndarray:
    """Fracción de la energía por nivel de Haar de los últimos `2^⌊log₂ n⌋` valores.

    `series` es `(n,)` o `(g, n)`; devuelve `(7,)` o `(g, 7)` (`nan` si la energía es cero).
    """
    n = series.shape[-1]
    length = 1 << (n.bit_length() - 1)
    tail = series[..., n - length :]
    flat = tail.min(axis=-1) == tail.max(axis=-1)
    approx = tail - tail.mean(axis=-1, keepdims=True)
    levels = []
    for _ in range(length.bit_length() - 1):
        even, odd = approx[..., 0::2], approx[..., 1::2]
        detail = (even - odd) / np.sqrt(2.0)
        levels.append(np.sum(detail * detail, axis=-1))
        approx = (even + odd) / np.sqrt(2.0)
    energy = np.zeros((*series.shape[:-1], BANDS))
    for index, level in enumerate(levels):
        energy[..., min(index, BANDS - 1)] += level
    total = energy.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = energy / total
    result[np.broadcast_to(flat, result.shape[:-1])] = np.nan
    return result


def group_size(n: int, nulls: int) -> int:
    return max(1, min(nulls, GROUP // n))


class PhaseResult:
    """Las fracciones de una fase: observadas `(7,)` y simuladas `(nulls, 7)` por candidato."""

    def __init__(self, nulls: int) -> None:
        self.obs: dict[str, np.ndarray] = {}
        self.sim = {c: np.empty((nulls, BANDS), np.float32) for c in CANDIDATES}
        self.f32_err = 0.0


def measure_phase(
    prices: np.ndarray,
    anchor: float,
    variance: float,
    rng: np.random.Generator,
    nulls: int,
) -> tuple[PhaseResult | None, list[str]]:
    """Calcula los tres candidatos de una fase de `n ≥ 16` ticks.

    Devuelve `(resultado, [])` o `(None, candidatos con energía cero)`. `prices` son los precios
    de los ticks en f64 y se leen, no se modifican.
    """
    n = len(prices)
    window = hann(n)
    out = PhaseResult(nulls)
    zero = []

    # Retornos: observados, Haar y barajados. El arreglo se libera antes de armar los precios.
    returns = np.empty(n)
    np.divide(prices[1:], prices[:-1], out=returns[1:])
    returns[0] = prices[0] / anchor
    np.log(returns, out=returns)
    out.obs["retornos"] = fractions(returns, window)
    out.obs["haar"] = haar(returns)
    out.f32_err = float(
        np.abs(
            fractions(returns.astype(np.float32), window.astype(np.float32)).astype(
                np.float64
            )
            - out.obs["retornos"]
        ).max(initial=0.0)
    )
    for start in range(0, nulls, group_size(n, nulls)):
        size = min(group_size(n, nulls), nulls - start)
        block = np.tile(returns, (size, 1))
        rng.permuted(block, axis=1, out=block)
        out.sim["retornos"][start : start + size] = fractions(block, window)
        out.sim["haar"][start : start + size] = haar(block)
        del block
    del returns

    # Precios: (p − recta) / p_0 con la recta del ancla (índice 0) al último tick (índice n).
    last = prices[-1]
    ratio = np.arange(1, n + 1, dtype=np.float64)
    ratio /= n
    line = ratio * (last - anchor)
    line += anchor
    series = prices - line
    series /= anchor
    out.obs["precios"] = fractions(series, window)
    out.f32_err = max(
        out.f32_err,
        float(
            np.abs(
                fractions(series.astype(np.float32), window.astype(np.float32)).astype(
                    np.float64
                )
                - out.obs["precios"]
            ).max(initial=0.0)
        ),
    )
    del series
    sigma = math.sqrt(variance / n)
    log_anchor, log_last = math.log(anchor), math.log(last)
    for start in range(0, nulls, group_size(n, nulls)):
        size = min(group_size(n, nulls), nulls - start)
        walk = rng.standard_normal((size, n))
        np.cumsum(walk, axis=1, out=walk)
        # ℓ_k = ℓ_0 + (ℓ_n − ℓ_0)·k/n + σ·(W_k − (k/n)·W_n), en el sitio.
        walk -= ratio * walk[:, -1:]
        walk *= sigma
        walk += log_anchor + (log_last - log_anchor) * ratio
        np.exp(walk, out=walk)
        walk -= line
        walk /= anchor
        out.sim["precios"][start : start + size] = fractions(walk, window)
        del walk
    del line, ratio

    for candidate in CANDIDATES:
        if np.isnan(out.obs[candidate]).any() or np.isnan(out.sim[candidate]).any():
            zero.append(candidate)
    return (None, zero) if zero else (out, [])


# ----------------------------------------------------------------------------------------------
# Muestras, D, cuantiles y Spearman
# ----------------------------------------------------------------------------------------------


class Samples:
    """Los valores observados y simulados, en memoria reservada al abrir."""

    def __init__(self, capacity: int, nulls: int) -> None:
        self.nulls = nulls
        self.n = dict.fromkeys(PHASES, 0)
        self.obs = {
            (c, p): np.empty((capacity, BANDS)) for c in CANDIDATES for p in PHASES
        }
        self.sim = {
            (c, p): np.empty((capacity * nulls, BANDS), np.float32)
            for c in CANDIDATES
            for p in PHASES
        }
        self.accel = np.empty(capacity)
        self.t50 = np.empty(capacity)
        self.f32_err = 0.0

    def add(self, phase: str, result: PhaseResult, accel, t50) -> None:
        i = self.n[phase]
        for candidate in CANDIDATES:
            self.obs[candidate, phase][i] = result.obs[candidate]
            self.sim[candidate, phase][i * self.nulls : (i + 1) * self.nulls] = (
                result.sim[candidate]
            )
        if phase == "c":
            self.accel[i] = np.nan if accel is None else accel
            self.t50[i] = np.nan if t50 is None else t50
        self.f32_err = max(self.f32_err, result.f32_err)
        self.n[phase] = i + 1

    def observed(self, candidate: str, phase: str) -> np.ndarray:
        return self.obs[candidate, phase][: self.n[phase]]

    def simulated(self, candidate: str, phase: str) -> np.ndarray:
        return self.sim[candidate, phase][: self.n[phase] * self.nulls]


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


def band_distance(samples: Samples, candidate: str, phase: str, band: int) -> float:
    """D de una banda (0 = la más rápida, 6 = la más lenta); ambas muestras en f32."""
    observed = np.sort(samples.observed(candidate, phase)[:, band].astype(np.float32))
    simulated = np.sort(samples.simulated(candidate, phase)[:, band])
    return ks_distance(observed, simulated)


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


PAIRS = {
    "precios_retornos": ("precios", "retornos"),
    "haar_fourier": ("haar", "retornos"),
    "retornos_accel": ("retornos", "accel"),
    "retornos_t50": ("retornos", "t50"),
    "precios_accel": ("precios", "accel"),
    "precios_t50": ("precios", "t50"),
}


def redundancy(samples: Samples) -> dict[tuple[str, str], np.ndarray]:
    """ρ por par, fase y banda. Los pares con `c_accel` y `c_t50` son solo de la fase c."""
    result = {}
    for pair, (left, right) in PAIRS.items():
        for phase in PHASES:
            if samples.n[phase] == 0:
                continue
            if right in ("accel", "t50"):
                if phase != "c":
                    continue
                other = (samples.accel if right == "accel" else samples.t50)[
                    : samples.n["c"]
                ]
                columns = [other] * BANDS
            else:
                columns = [samples.observed(right, phase)[:, b] for b in range(BANDS)]
            observed = samples.observed(left, phase)
            result[pair, phase] = np.array(
                [spearman(observed[:, b], columns[b]) for b in range(BANDS)]
            )
    return result


def max_abs(values: np.ndarray | None) -> float:
    if values is None or np.isnan(values).all():
        return math.nan
    return float(np.nanmax(np.abs(values)))


# ----------------------------------------------------------------------------------------------
# Lectura del mes
# ----------------------------------------------------------------------------------------------


class Phase:
    __slots__ = ("anchor", "buf", "filled", "hi", "lo", "n", "name", "read", "variance")

    def __init__(self, name, lo, hi, anchor, n, variance):
        self.name, self.lo, self.hi = name, lo, hi
        self.anchor, self.n, self.variance = anchor, n, variance
        self.read = False
        self.buf: np.ndarray | None = None
        self.filled = 0


class Event:
    """El evento abierto: sus dos fases y qué fase se lee."""

    def __init__(self, row: tuple, counters: Counter) -> None:
        event, summary, walk, fit = row
        ref_price, self.ref_id, conf_price, self.conf_id, self.ext_id = event
        _, c_ticks, os_ticks = summary
        _, c_var, os_var, self.c_t50 = walk
        self.accel = fit[1]
        if (
            os_ticks < 0
            or c_ticks < 0
            or (os_ticks == 0) != (self.ext_id == self.conf_id)
        ):
            raise InputError(
                f"confirm_agg_trade_id {self.conf_id}: os_ticks={os_ticks} no cuadra con "
                "extreme_agg_trade_id"
            )
        self.phases = [
            Phase("c", self.ref_id, self.conf_id, float(ref_price), c_ticks, c_var),
            Phase("os", self.conf_id, self.ext_id, float(conf_price), os_ticks, os_var),
        ]
        self.truncated = False
        if self.ext_id - self.ref_id > BUDGET:
            counters["eventos_sobre_presupuesto"] += 1
            return
        for phase in self.phases:
            if phase.n == 0:
                counters[phase.name, "vacias"] += 1
            elif phase.n < N_MIN:
                counters[phase.name, "cortas"] += 1
            else:
                if phase.variance is None:
                    raise InputError(
                        f"confirm_agg_trade_id {self.conf_id}: {phase.name} sin varianza "
                        "realizada en path.parquet"
                    )
                phase.read = True

    def drop(self) -> None:
        for phase in self.phases:
            phase.read = False
            phase.buf = None


def measure(uris: dict[str, str], seed: int, nulls: int) -> tuple[Samples, Counter]:
    """Recorre el mes una vez; devuelve las muestras y los conteos de excluidos."""
    rng = np.random.default_rng(seed)
    counters: Counter = Counter()
    with ExitStack() as stack:
        files = {}
        for label, uri in uris.items():
            try:
                files[label] = open_parquet(stack, uri)
            except FileNotFoundError:
                raise InputError(f"falta el archivo {uri}") from None
        samples = Samples(files["events"].metadata.num_rows, nulls)
        rows = event_rows({k: files[k] for k in FILES})

        def next_event() -> Event | None:
            row = next(rows, None)
            return None if row is None else Event(row, counters)

        def close(event: Event, phase: Phase, first_id: int) -> None:
            if phase.filled != phase.n:
                if event.ref_id < first_id:
                    counters["evento_mes_anterior"] += 1
                    event.truncated = True
                    event.drop()  # L1 no trae el evento entero
                    return
                raise InputError(
                    f"confirm_agg_trade_id {event.conf_id}: L1 trae {phase.filled} ticks de "
                    f"{phase.name} y summaries dice {phase.n}"
                )
            result, zero = measure_phase(
                phase.buf, phase.anchor, phase.variance, rng, nulls
            )
            phase.buf, phase.read = None, False
            if result is None:
                counters[phase.name, "energia_cero"] += 1
                for candidate in zero:
                    counters[phase.name, "energia_cero", candidate] += 1
            else:
                samples.add(phase.name, result, event.accel, event.c_t50)

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
                        if phase.buf is None:
                            phase.buf = np.empty(phase.n)
                        if phase.filled + (stop - start) > phase.n:
                            raise InputError(
                                f"confirm_agg_trade_id {current.conf_id}: L1 trae más ticks "
                                f"de {phase.name} que los {phase.n} de summaries"
                            )
                        phase.buf[phase.filled : phase.filled + stop - start] = prices[
                            start:stop
                        ]
                        phase.filled += stop - start
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
    if samples.n["c"] == 0:
        raise InputError("ningún evento medible en la fase c del mes")
    return samples, counters


# ----------------------------------------------------------------------------------------------
# Pase de solo L2 y L3
# ----------------------------------------------------------------------------------------------


def phase_stream(parquet: pq.ParquetFile):
    """Los pares `(a, b)` de las fases abiertas de un θ, en orden, por lotes.

    Una fase `(a, b]` cuenta `x − a` en el tick `x`. Un evento sobre el presupuesto no aporta
    fases (cuenta 0) y una fase vacía tampoco.
    """
    names = {"reference_agg_trade_id", "confirm_agg_trade_id", "extreme_agg_trade_id"}
    missing = names - set(parquet.schema_arrow.names)
    if missing:
        raise InputError(f"events.parquet: faltan columnas {sorted(missing)}")
    previous = None
    for batch in parquet.iter_batches(batch_size=INFLIGHT_BATCH, columns=sorted(names)):
        ref = int64s(batch.column("reference_agg_trade_id"))
        conf = int64s(batch.column("confirm_agg_trade_id"))
        ext = int64s(batch.column("extreme_agg_trade_id"))
        starts = np.stack((ref, conf), axis=1).reshape(-1)
        stops = np.stack((conf, ext), axis=1).reshape(-1)
        keep = np.repeat((ext - ref) <= BUDGET, 2) & (stops > starts)
        starts, stops = starts[keep], stops[keep]
        if not len(starts):
            continue
        if (previous is not None and starts[0] < previous) or (
            len(starts) > 1 and (starts[1:] < stops[:-1]).any()
        ):
            raise InputError(
                "events.parquet: las fases de un θ deben ir en orden y sin solaparse"
            )
        previous = int(stops[-1])
        yield starts, stops


def window_peak(
    starts: np.ndarray,
    stops: np.ndarray,
    open_starts: np.ndarray,
    open_stops: np.ndarray,
) -> int:
    """Pico de `Σ (x − a)` sobre las fases activas (`a < x ≤ b`) en los `x = b` de `stops`."""
    if not len(stops):
        return 0
    all_starts = np.concatenate((starts, open_starts))
    all_stops = np.concatenate((stops, open_stops))
    where = np.concatenate((all_starts + 1, all_stops + 1))
    ones = np.ones(len(all_starts), np.int64)
    order = np.argsort(where, kind="stable")
    where = where[order]
    count = np.cumsum(np.concatenate((ones, -ones))[order])
    total = np.cumsum(np.concatenate((all_starts, -all_starts))[order])
    index = np.searchsorted(where, stops, side="right") - 1
    return int((stops * count[index] - total[index]).max())


def inflight_peak(streams: list) -> int:
    """El pico de ticks en vuelo de varios θ a la vez, con memoria O(lote × θ).

    El pico está en un fin de fase. Cada ronda toma de cada θ las fases que terminan antes del
    horizonte (el menor «último fin» entre los θ con más lotes por leer) y evalúa en sus fines la
    suma sobre las fases de todos los θ: las tomadas y la primera que cada θ deja abierta.
    """
    empty = np.empty(0, np.int64)
    k = len(streams)
    buffers = [(empty, empty)] * k
    position = [0] * k
    more = [True] * k
    peak = 0
    while True:
        for i in range(k):
            while more[i] and position[i] >= len(buffers[i][0]):
                try:
                    buffers[i], position[i] = next(streams[i]), 0
                except StopIteration:
                    buffers[i], position[i], more[i] = (empty, empty), 0, False
        live = [i for i in range(k) if position[i] < len(buffers[i][0])]
        if not live:
            return peak
        bounds = [int(buffers[i][1][-1]) for i in live if more[i]]
        horizon = min(bounds) if bounds else None
        taken_starts, taken_stops, open_starts, open_stops = [], [], [], []
        for i in live:
            starts, stops = buffers[i]
            cut = (
                len(starts)
                if horizon is None
                else position[i]
                + int(np.searchsorted(stops[position[i] :], horizon, side="right"))
            )
            taken_starts.append(starts[position[i] : cut])
            taken_stops.append(stops[position[i] : cut])
            if cut < len(starts):
                open_starts.append(starts[cut : cut + 1])
                open_stops.append(stops[cut : cut + 1])
            position[i] = cut
        peak = max(
            peak,
            window_peak(
                np.concatenate(taken_starts),
                np.concatenate(taken_stops),
                np.concatenate(open_starts) if open_starts else empty,
                np.concatenate(open_stops) if open_stops else empty,
            ),
        )


def month_inflight(paths: dict[str, str]) -> int:
    with ExitStack() as stack:
        streams = []
        for uri in paths.values():
            try:
                streams.append(phase_stream(open_parquet(stack, uri)))
            except FileNotFoundError:
                raise InputError(f"falta el archivo {uri}") from None
        return inflight_peak(streams)


def short_fractions(path: str) -> dict[str, int]:
    """Eventos, fases c y os bajo 16 ticks y fases os vacías de un `summaries.parquet`."""
    counts = Counter()
    with ExitStack() as stack:
        try:
            parquet = open_parquet(stack, path)
        except FileNotFoundError:
            raise InputError(f"falta el archivo {path}") from None
        missing = {"c_ticks", "os_ticks"} - set(parquet.schema_arrow.names)
        if missing:
            raise InputError(f"summaries.parquet: faltan columnas {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=BATCH, columns=["c_ticks", "os_ticks"]
        ):
            c, o = int64s(batch.column("c_ticks")), int64s(batch.column("os_ticks"))
            counts["eventos"] += len(c)
            counts["c_lt16"] += int((c < N_MIN).sum())
            counts["os_lt16"] += int((o < N_MIN).sum())
            counts["os_vacias"] += int((o == 0).sum())
    return dict(counts)


# ----------------------------------------------------------------------------------------------
# CSV y última línea
# ----------------------------------------------------------------------------------------------


def number(value: float) -> str:
    return format(value, ".17g")


def number32(value: float) -> str:
    return format(float(np.float32(value)), ".9g")


def csv_rows(
    samples: Samples,
    counters: Counter,
    distances: dict,
    rho: dict,
    short: dict[str, dict],
    inflight: dict[str, int],
    nulls: int,
    seed: int,
) -> list[dict]:
    rows = []

    def add(section, candidate, phase, band, source, measure, value):
        rows.append(
            dict(
                zip(
                    CSV_HEADER,
                    (section, candidate, phase, band, source, measure, value),
                    strict=True,
                )
            )
        )

    for candidate in CANDIDATES:
        for phase in PHASES:
            if samples.n[phase] == 0:
                continue
            for band in range(BANDS):
                for source, values, fmt in (
                    ("observado", samples.observed(candidate, phase), number),
                    ("simulado", samples.simulated(candidate, phase), number32),
                ):
                    column = np.sort(values[:, band])
                    for percent in QUANTILES:
                        add(
                            "cuantiles", candidate, phase, band + 1, source,
                            f"p{percent}", fmt(quantile(column, percent)),
                        )  # fmt: skip
                    add(
                        "cuantiles",
                        candidate,
                        phase,
                        band + 1,
                        source,
                        "N",
                        len(column),
                    )
    for (candidate, phase), (fast, slow) in distances.items():
        add("D", candidate, phase, "", "ambas", "D_fast", number(fast))
        add("D", candidate, phase, "", "ambas", "D_slow", number(slow))
    for (pair, phase), values in rho.items():
        for band, value in enumerate(values, start=1):
            add("rho", pair, phase, band, "", "rho", number(value))
        add("rho", pair, phase, "", "", "max_abs_rho", number(max_abs(values)))
    add("f32", "", "", "", "", "f32_err", number(samples.f32_err))
    for theta, counts in short.items():
        events = counts["eventos"]
        add("fases_lt16", "", "", "", f"theta={theta}", "eventos", events)
        for phase in PHASES:
            lt16 = counts[f"{phase}_lt16"]
            add("fases_lt16", "", phase, "", f"theta={theta}", "lt16", lt16)
            add(
                "fases_lt16", "", phase, "", f"theta={theta}", "fraccion_lt16",
                number(lt16 / events if events else math.nan),
            )  # fmt: skip
        add("fases_lt16", "", "os", "", f"theta={theta}", "vacias", counts["os_vacias"])
    for month, peak in inflight.items():
        add("ticks_en_vuelo", "", "", "", f"mes={month}", "inflight_peak_ticks", peak)
        add("ticks_en_vuelo", "", "", "", f"mes={month}", "bytes_8B", peak * 8)
        add("ticks_en_vuelo", "", "", "", f"mes={month}", "bytes_4B", peak * 4)
    for phase in PHASES:
        add("excluidos", "", phase, "", "", "fases_cortas", counters[phase, "cortas"])
        add("excluidos", "", phase, "", "", "fases_vacias", counters[phase, "vacias"])
        add(
            "excluidos",
            "",
            phase,
            "",
            "",
            "energia_cero",
            counters[phase, "energia_cero"],
        )
        for candidate in CANDIDATES:
            add(
                "excluidos", candidate, phase, "", "", "energia_cero",
                counters[phase, "energia_cero", candidate],
            )  # fmt: skip
    for name in ("eventos_sobre_presupuesto", "evento_mes_anterior"):
        add("excluidos", "", "", "", "", name, counters[name])
    for phase in PHASES:
        add("muestras", "", phase, "", "", "N", samples.n[phase])
    add("config", "", "", "", "", "nulls", nulls)
    add("config", "", "", "", "", "semilla", seed)
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


def verdict_of(distances: dict, rho: dict) -> str:
    """`sigue` si algún candidato pasa el umbral de D sobre c y no repite a A ni a T."""
    for candidate, accel, t50 in (
        ("retornos", "retornos_accel", "retornos_t50"),
        ("precios", "precios_accel", "precios_t50"),
    ):
        fast, slow = distances[candidate, "c"]
        redundant = max(
            (
                v
                for v in (max_abs(rho.get((accel, "c"))), max_abs(rho.get((t50, "c"))))
                if not math.isnan(v)
            ),
            default=0.0,
        )
        if (fast >= D_THRESHOLD or slow >= D_THRESHOLD) and redundant <= RHO_THRESHOLD:
            return "sigue"
    return "retirada"


def final_line(month, theta, wall, verdict, figures: dict | None = None) -> str:
    def cell(key, fmt="{:.6f}"):
        value = None if figures is None else figures.get(key)
        return "-" if value is None else fmt.format(value)

    return (
        f"spectrum probe: mes={month} theta={theta} eventos={cell('eventos', '{}')} "
        f"D_ret_fast={cell('D_ret_fast')} D_ret_slow={cell('D_ret_slow')} "
        f"D_price_fast={cell('D_price_fast')} D_price_slow={cell('D_price_slow')} "
        f"rho_price_ret={cell('rho_price_ret')} rho_haar_fourier={cell('rho_haar_fourier')} "
        f"rho_ret_accel={cell('rho_ret_accel')} rho_ret_t50={cell('rho_ret_t50')} "
        f"rho_price_accel={cell('rho_price_accel')} rho_price_t50={cell('rho_price_t50')} "
        f"f32_err={cell('f32_err', '{:.3e}')} n_lt16={cell('n_lt16', '{}')} "
        f"inflight_peak_ticks={cell('inflight_peak_ticks', '{}')} "
        f"pared_s={wall:.1f} veredicto={verdict}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    if re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month) is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    try:
        theta = parse_theta(ns.theta)
        months = parse_months(ns.inflight_months)
    except ValueError as exc:
        parser.error(str(exc))
    if ns.nulls < 1:
        parser.error("--nulls debe ser un entero mayor que 0")
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_spectrum_probe_{ns.month}_{theta}.csv"

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
        "l1": l1_file(l1_root, key, ns.month),
        "events": partition_file(l2_root, key, theta, ns.month, "events.parquet"),
        "summaries": partition_file(l3_root, key, theta, ns.month, "summaries.parquet"),
        "path": partition_file(l3_root, key, theta, ns.month, "path.parquet"),
        "fit": partition_file(l3_root, key, theta, ns.month, "fit.parquet"),
    }
    started = time.perf_counter()
    log(f"spectrum probe: θ={theta} mes={ns.month} nulls={ns.nulls} semilla={ns.seed}")
    for label, uri in uris.items():
        log(f"  {label}: {uri}")
    try:
        samples, counters = measure(uris, ns.seed, ns.nulls)
        log(f"mes medido: {samples.n['c']} fases c, {samples.n['os']} fases os")

        distances = {}
        for candidate in ("retornos", "precios"):
            for phase in PHASES:
                if samples.n[phase]:
                    distances[candidate, phase] = (
                        band_distance(samples, candidate, phase, 0),
                        band_distance(samples, candidate, phase, BANDS - 1),
                    )
        rho = redundancy(samples)

        short = {}
        l3_files = list_files(series_root(l3_root, key))
        l3_summaries = thetas_of(l3_files, ns.month, "summaries.parquet")
        l3_summaries.setdefault(theta, uris["summaries"])
        for other, path in l3_summaries.items():
            short[other] = short_fractions(path)
        inflight = {}
        if months:
            l2_files = list_files(series_root(l2_root, key))
            for month in months:
                paths = thetas_of(l2_files, month, "events.parquet")
                if not paths:
                    raise InputError(f"L2 no tiene eventos de {month}")
                log(f"ticks en vuelo {month}: {len(paths)} θ")
                inflight[month] = month_inflight(paths)
    except (InputError, OSError, ValueError, pa.ArrowException) as exc:
        log(f"FALLO {type(exc).__name__}: {exc}")
        wall = time.perf_counter() - started
        log(final_line(ns.month, theta, wall, "fallo"))
        return 1
    wall = time.perf_counter() - started

    rows = csv_rows(
        samples, counters, distances, rho, short, inflight, ns.nulls, ns.seed
    )
    log("  ".join(f"{h:<12}" for h in CSV_HEADER))
    for row in rows:
        log("  ".join(f"{row[h]!s:<12}" for h in CSV_HEADER))
    try:
        write_csv(rows, out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")

    verdict = verdict_of(distances, rho)
    figures = {
        "eventos": samples.n["c"],
        "D_ret_fast": distances["retornos", "c"][0],
        "D_ret_slow": distances["retornos", "c"][1],
        "D_price_fast": distances["precios", "c"][0],
        "D_price_slow": distances["precios", "c"][1],
        "f32_err": samples.f32_err,
        "n_lt16": short[theta]["c_lt16"],
        "inflight_peak_ticks": max(inflight.values()) if inflight else None,
    }
    for label, pair, phase in (
        ("rho_price_ret", "precios_retornos", "c"),
        ("rho_haar_fourier", "haar_fourier", "c"),
        ("rho_ret_accel", "retornos_accel", "c"),
        ("rho_ret_t50", "retornos_t50", "c"),
        ("rho_price_accel", "precios_accel", "c"),
        ("rho_price_t50", "precios_t50", "c"),
    ):
        figures[label] = max_abs(rho[pair, phase])
    log(final_line(ns.month, theta, wall, verdict, figures))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--l3-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--nulls", type=int, default=DEFAULT_NULLS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inflight-months", default=DEFAULT_INFLIGHT_MONTHS)
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
