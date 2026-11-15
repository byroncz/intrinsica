"""Verificación cruzada de `spectrum.parquet` (familia F de L3): sus valores contra el oráculo (ITSC-422).

La sonda `l3_spectrum_check.py` (ITSC-414) comprueba la forma del archivo (esqueleto, rangos, sumas,
regla de nulos). Esta comprueba los **valores**: para un θ y un mes lee L1 (`consolidated.parquet`), L2
(`events.parquet`) y `spectrum.parquet`, recalcula en Python las 30 columnas de cada fase con el
oráculo de `l3_spectrum_probe.py` (ITSC-411 e ITSC-416) y compara columna por columna con lo que
escribió el motor. Mide: no cambia el motor ni ningún comportamiento de L3.

Fases, como las ve el motor (`spectrum.rs`): la fase `c` son los ticks de L1 en `(R, C]` con ancla
`reference_price`, y la fase `os` los de `(C, E]` con ancla `confirm_price`, con `R`, `C` y `E` =
`reference_`, `confirm_` y `extreme_agg_trade_id` de L2. Una fase de `n` ticks tiene `n` retornos: el
primero sale del ancla. Una fase con menos de `n_min` retornos (los ticks de `n_min` salen del pie de
`spectrum.parquet`, igual que `bands`, `budget_ticks`, `state_version` y `window`; ninguno es una
constante de la sonda) o vacía debe traer sus 15 columnas nulas, y eso también se compara.

Misma aritmética que el motor, no la de la sonda exploratoria. El oráculo (las funciones `hann`,
`band_slices`, `fractions`, `haar` y `total_location`, copiadas tal cual de `l3_spectrum_probe.py`: el
script se publica suelto en GCS y no puede importar a otro) se alimenta así:
  - El retorno es `Δ ln p` por `ln_1p(Δp / p_previo)` sobre los precios enteros de L1 (el decimal sin
    escalar), **guardado como `float32`** y llevado después a `float64`, igual que el búfer del motor.
    La sonda exploratoria calcula `ln(p_k / p_{k−1})` en `float64` sobre los precios en `float64` y
    nunca pasa por `float32`; la verificación sigue al motor. La diferencia solo cuenta en la energía
    cero de Haar (abajo).
  - La regla de nulos es la del contrato (TRD-L3 §7.2.3), que la sonda exploratoria no aplica: una
    banda o un nivel sin bins (`ticks < 2^b`) es nulo, no 0 (el oráculo da 0), y una serie sin energía
    es nula (el oráculo da `nan`; aquí es lo mismo).
  - Las sumas del oráculo son de NumPy (pareadas) y las del motor, de Neumaier y de `rustfft`: se
    comparan con la tolerancia de abajo, no bit a bit.

Comparación por columna, con tolerancia absoluta `TOL` = 1e-9 (son fracciones en [0, 1]): `comparadas`
(fases que entran), `fuera_tolerancia` (dos valores con `|a − b| > TOL`, o un `NaN` de cualquier lado),
`nulos_distintos` (uno nulo y el otro no) y `max_diff` (la mayor `|a − b|` entre los dos no nulos).

Energía cero de Haar. La cola de Haar de una fase con `n ≥ n_min` son sus últimos `2^⌊log₂ n⌋` retornos:
si es plana (todos iguales) el registro de Haar es nulo. La sonda exploratoria apartó esas fases; el
motor las deja nulas solo si la cola es plana **en `float32`**, y el oráculo exploratorio las decidía
en `float64`. Por fase, la sonda cuenta las colas planas con las dos aritméticas (`haar_cero_f64`, con
`ln(p_k / p_{k−1})` en `float64`, y `haar_cero_f32`, con los retornos del motor) y qué tiene el motor
en esas filas (nulo o valor). Una **discrepancia** es una fase cuyo Haar el motor deja nulo y alguna de
las dos aritméticas dice «no plana», o al revés; la sonda lista hasta 20 con `event_id`, fase, `n`, los
dos indicadores y los dos valores (el primer nivel de Haar del motor y del oráculo con la aritmética
del motor). Las discrepancias no entran al veredicto: lo que sí entra es que las columnas coincidan.

Veredicto (la regla se fijó antes de medir): `ok` si y solo si `fuera_tolerancia = 0` y
`nulos_distintos = 0` en las 30 columnas; si no, `revisar`. Si no queda ninguna fase por comparar o
falla una lectura, `fallo`: no se midió. Las cifras de energía cero no cambian el veredicto.

Args:
  --theta 0.00098052  --month 2023-03     (por defecto; un θ y un mes por corrida)
  --l1-root gs://<proyecto>-landing/l1     (la raíz de L1 y las de L2 y L3, donde viven `events.parquet`
  --l2-root gs://<proyecto>-dc-events/l2    y `spectrum.parquet`; por defecto, las del proyecto según
  --l3-root gs://<proyecto>-dc-descriptors/l3   OPS_RESULTS_URI; también rutas locales con la misma
                                                 disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --max-ticks 2000000   (las fases con más ticks se omiten y se cuentan; por defecto 2 000 000)
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_spectrum_xcheck_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

Omitidas y aparte. No se comparan, y se cuentan en el CSV: las fases con más de `--max-ticks` ticks
(`omitidas` de la última línea), las que empiezan antes del primer tick de L1 del mes (`antes_del_mes`,
no caben en el L1 del mes: a lo sumo una por mes y θ), las que terminan después del último
(`despues_del_mes`) y las de un evento sobre el presupuesto (`sobre_presupuesto`: `extreme − reference`
mayor que `budget_ticks`; el motor no leyó ningún tick).

Memoria (eficiencia de memoria ante todo): una fase a la vez, liberada al terminar. Como un θ tiene sus
fases en fila (el fin de una es el comienzo de la siguiente), L1 se lee un row group a la vez y de la
fase abierta solo se guardan sus precios enteros (8 B por tick, a lo sumo `--max-ticks`); al cerrarla se
derivan los retornos `float32` y la serie `float64` de a uno y cada arreglo se suelta antes del
siguiente. `events.parquet` y `spectrum.parquet` se leen en lotes de `BATCH` filas, en paralelo y por
posición con la clave comprobada. El pico es O(lote + `--max-ticks`), nunca el mes.

No importa ninguna capa: una importación cambiaría `uv.lock`. Abre los tres archivos por su ruta completa
(nunca un dataset sobre un prefijo, que pide un objeto sin barra y la condición IAM por prefijo de
`ops-script` lo niega con 403; runbook de operación de L3, «La trampa del 403 por prefijo»).

Salida: la tabla por columna y las cifras de energía cero, la ruta del CSV y, como última línea,
`spectrum xcheck: mes=… theta=… filas=… comparadas=… omitidas=… fuera_tolerancia=… nulos_distintos=…
max_diff=… haar_cero_f64=… haar_cero_f32=… veredicto=ok|revisar` (`filas` son las del `spectrum.parquet`;
`comparadas` y `omitidas`, fases; con `veredicto=fallo`, lo que no se pudo medir sale `-`).

Códigos: 0 con veredicto `ok`; 1 con `revisar` o `fallo` (falta un archivo, un metadato o una columna, los
archivos no cuadran entre sí o la lectura falla); 2 por uso. Solo lee; lo único que escribe es el CSV.
"""

import argparse
import csv
import io
import math
import os
import re
import sys
import time
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

TOL = 1e-9
DEFAULT_THETA = "0.00098052"
DEFAULT_MONTH = "2023-03"
DEFAULT_MAX_TICKS = 2_000_000
MAX_LISTED = 20
BANDS = 7
BATCH = 8_192
PHASES = ("c", "os")
# (registro, unidad) de F: Fourier por banda de período y Haar por nivel.
REGISTERS = (("fourier", "band"), ("haar", "level"))
LOC_COLUMNS = {"c": "c_energy_loc", "os": "os_energy_loc"}
META_KEYS = ("state_version", "n_min", "bands", "window", "budget_ticks")
TICK_COLUMNS = ["agg_trade_id", "price"]
EVENT_COLUMNS = [
    "reference_price",
    "reference_agg_trade_id",
    "confirm_price",
    "confirm_agg_trade_id",
    "extreme_agg_trade_id",
]
CSV_HEADER = [
    "seccion",
    "columna",
    "comparadas",
    "fuera_tolerancia",
    "nulos_distintos",
    "max_diff",
    "conteo",
    "detalle",
]


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


def parse_meta(meta: dict[str, str]) -> dict:
    """Los parámetros de F tal como el pie de `spectrum.parquet` los declara."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise InputError(f"faltan metadatos en spectrum.parquet: {', '.join(missing)}")
    try:
        params = {
            "n_min": int(meta["n_min"]),
            "bands": int(meta["bands"]),
            "budget_ticks": int(meta["budget_ticks"]),
        }
    except ValueError:
        raise InputError(
            "metadatos de spectrum.parquet no enteros: "
            f"n_min={meta['n_min']!r} bands={meta['bands']!r} "
            f"budget_ticks={meta['budget_ticks']!r}"
        ) from None
    if params["n_min"] < 2 or params["budget_ticks"] < 0:
        raise InputError(f"metadatos de spectrum.parquet fuera de rango: {params}")
    if params["bands"] != BANDS:
        raise InputError(
            f"el oráculo es de {BANDS} bandas y spectrum.parquet declara bands={params['bands']}"
        )
    return {**params, "state_version": meta["state_version"], "window": meta["window"]}


def group(phase: str, register: str) -> list[str]:
    """Las columnas de una fase y un registro, de la banda o el nivel 1 al último."""
    unit = dict(REGISTERS)[register]
    return [f"{phase}_{register}_{unit}_{b}" for b in range(1, BANDS + 1)]


def phase_columns(phase: str) -> list[str]:
    """Las 15 columnas de una fase: Fourier, Haar y la ubicación de su energía total."""
    return [*group(phase, "fourier"), *group(phase, "haar"), LOC_COLUMNS[phase]]


def data_columns() -> list[str]:
    """Las 30 columnas de datos de F en el orden del archivo."""
    columns = [c for r, _ in REGISTERS for p in PHASES for c in group(p, r)]
    return [*columns, *LOC_COLUMNS.values()]


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
    """`(ids, precios)` de cada row group no vacío de L1, uno a la vez; precios enteros sin escalar."""
    last = None
    for index in range(parquet.num_row_groups):
        if not parquet.metadata.row_group(index).num_rows:
            continue
        table = parquet.read_row_group(index, columns=TICK_COLUMNS)
        ids = int64s(single(table.column("agg_trade_id")))
        prices = lows(single(table.column("price")))
        if (len(ids) > 1 and np.diff(ids).min() <= 0) or (
            last is not None and ids[0] <= last
        ):
            raise InputError(f"row group {index}: L1 debe venir ordenado")
        last = int(ids[-1])
        yield ids, prices
        del table, ids, prices


def event_tuples(parquet: pq.ParquetFile):
    """Las filas de `events.parquet` como tuplas, un lote a la vez; precios como entero sin escalar."""
    missing = set(EVENT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise InputError(f"events.parquet: faltan columnas {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=BATCH, columns=EVENT_COLUMNS):
        values = []
        for column in batch.columns:
            if pa.types.is_decimal128(column.type):
                values.append(lows(column).tolist())
            elif pa.types.is_integer(column.type):
                values.append(column.to_pylist())
            else:
                raise InputError(f"columna de tipo inesperado: {column.type}")
        yield from zip(*values, strict=True)


def spectrum_rows(parquet: pq.ParquetFile, columns: list[str]):
    """Las filas de `spectrum.parquet` como diccionarios (`None` en los nulos), un lote a la vez."""
    for batch in parquet.iter_batches(
        batch_size=BATCH, columns=["confirm_agg_trade_id", *columns]
    ):
        yield from batch.to_pylist()


# ----------------------------------------------------------------------------------------------
# El oráculo: copiado de `l3_spectrum_probe.py` (ITSC-411 e ITSC-416)
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
    """Fracción de la energía por banda de una serie `(n,)`: `(7,)`, con `nan` si no hay energía."""
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
    """Fracción de la energía por nivel de Haar de los últimos `2^⌊log₂ n⌋` valores: `(7,)`."""
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


def total_location(returns: np.ndarray) -> np.ndarray:
    """Centro de masa de la energía de los `n` retornos: `Σ t_k·r_k² / Σ r_k²`, `t_k = (k − 0,5)/n`."""
    n = returns.shape[-1]
    moment = np.zeros(returns.shape[:-1])
    energy = np.zeros(returns.shape[:-1])
    for start in range(0, n, 1 << 20):
        part = returns[..., start : start + (1 << 20)]
        square = part * part
        moment += square @ ((np.arange(start, start + part.shape[-1]) + 0.5) / n)
        energy += np.sum(square, axis=-1)
        del square
    with np.errstate(divide="ignore", invalid="ignore"):
        return moment / energy


# ----------------------------------------------------------------------------------------------
# Retornos del motor y valores esperados de una fase
# ----------------------------------------------------------------------------------------------


def engine_returns(prices: np.ndarray, anchor: int) -> tuple[np.ndarray, bool]:
    """Los `n` retornos de la fase en `float32`, como el motor, y si su cola de Haar es plana en `float64`.

    `prices` son los precios enteros de los `n` ticks y `anchor` el entero del ancla. El motor calcula
    `ln_1p((p − p_previo) as f64 / p_previo as f64) as f32` (`log_return`); los enteros de un precio de
    mercado caben exactos en `float64`, así que la diferencia en `float64` es la misma. El segundo
    valor es la cola de Haar plana con la aritmética de la sonda exploratoria: `ln(p_k / p_{k−1})` en
    `float64`, sin pasar por `float32`.
    """
    n = len(prices)
    previous = np.empty(n)
    previous[0] = anchor
    previous[1:] = prices[:-1]
    current = prices.astype(np.float64)
    length = 1 << (n.bit_length() - 1)
    tail = np.log(current[n - length :] / previous[n - length :])
    flat_f64 = bool(tail.min() == tail.max())
    del tail
    current -= previous
    current /= previous
    del previous
    np.log1p(current, out=current)
    returns = current.astype(np.float32)
    del current
    return returns, flat_f64


def expected_columns(phase: str, returns: np.ndarray | None, n_min: int):
    """Los 15 valores de la fase según el oráculo (`None` = nulo) y si su cola de Haar es plana en `float32`.

    `returns` son los retornos `float32` del motor. Con menos de `n_min` retornos (o ninguno) todo es
    nulo. La banda o el nivel sin bins es nulo, no 0, y una serie sin energía es nula (`nan`).
    """
    nulls = [None] * (2 * BANDS + 1)
    if returns is None or len(returns) < n_min:
        return nulls, None
    n = len(returns)
    series = returns.astype(np.float64)
    length = 1 << (n.bit_length() - 1)
    flat_f32 = bool(returns[n - length :].min() == returns[n - length :].max())
    fourier = fractions(series, hann(n))
    slices = band_slices(n)
    values = [
        None if hi <= lo or math.isnan(x) else float(x)
        for (lo, hi), x in zip(slices, fourier, strict=True)
    ]
    levels = length.bit_length() - 1
    values += [
        None if level > levels or math.isnan(x) else float(x)
        for level, x in enumerate(haar(series), start=1)
    ]
    location = float(total_location(series))
    values.append(None if math.isnan(location) else location)
    del series
    return values, flat_f32


# ----------------------------------------------------------------------------------------------
# Fases abiertas y comparación
# ----------------------------------------------------------------------------------------------


class Phase:
    """Una fase `(lo, hi]` de un evento: los precios enteros de sus ticks y la fila del motor."""

    __slots__ = (
        "anchor",
        "before",
        "buf",
        "cap",
        "event_id",
        "hi",
        "lo",
        "long",
        "n",
        "name",
        "over",
        "spec",
    )

    def __init__(self, name, lo, hi, anchor, event_id, spec, over, limit):
        self.name, self.lo, self.hi, self.anchor = name, lo, hi, anchor
        self.event_id, self.spec, self.over = event_id, spec, over
        self.before = False
        self.long = False
        self.n = 0
        self.buf: np.ndarray | None = None
        # `n ≤ hi − lo` siempre (los ids son únicos): con `cap = hi − lo` nunca desborda.
        self.cap = min(hi - lo, limit)

    @property
    def skipped(self) -> bool:
        return self.before or self.over or self.long

    def take(self, prices: np.ndarray) -> None:
        count = len(prices)
        if not count or self.skipped:
            return
        self.n += count
        if self.n > self.cap:
            self.long, self.buf = True, None
            return
        if self.buf is None:
            self.buf = np.empty(self.cap, dtype=np.int64)
        self.buf[self.n - count : self.n] = prices


def phases_of(rows, budget: int, limit: int, counters: dict):
    """Las fases de cada evento en orden, con la fila del motor, comprobada la clave."""
    previous_hi = None
    for event, spec in rows:
        ref_price, ref_id, conf_price, conf_id, ext_id = event
        if spec["confirm_agg_trade_id"] != conf_id:
            raise InputError(
                "events y spectrum no están alineados: confirm_agg_trade_id "
                f"{conf_id}, {spec['confirm_agg_trade_id']}"
            )
        if previous_hi is not None and ref_id < previous_hi:
            raise InputError(
                f"confirm_agg_trade_id {conf_id}: las fases no son contiguas "
                f"(reference {ref_id} antes del extremo anterior {previous_hi})"
            )
        previous_hi = ext_id
        counters["filas"] += 1
        over = ext_id - ref_id > budget
        yield Phase("c", ref_id, conf_id, ref_price, conf_id, spec, over, limit)
        yield Phase("os", conf_id, ext_id, conf_price, conf_id, spec, over, limit)


class Tally:
    """Lo que se mide: por columna, y las cifras de energía cero de Haar."""

    def __init__(self) -> None:
        self.columns = {c: [0, 0, 0, 0.0] for c in data_columns()}
        self.counts: dict[str, int] = {}
        self.listed: list[dict] = []
        self.discrepancies = 0

    def add(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount

    def compare(self, phase: str, got: dict, wanted: list) -> None:
        for column, want in zip(phase_columns(phase), wanted, strict=True):
            have = got[column]
            cell = self.columns[column]
            cell[0] += 1
            if (have is None) != (want is None):
                cell[2] += 1
            elif have is not None:
                diff = abs(have - want)
                if not diff <= TOL:
                    cell[1] += 1
                cell[3] = max(cell[3], diff)

    def haar_zero(self, phase: Phase, got: dict, wanted: list, flats: tuple) -> None:
        """Cuenta las colas planas de Haar con las dos aritméticas y lo que tiene el motor."""
        flat_f64, flat_f32 = flats
        cols = group(phase.name, "haar")
        engine_null = all(got[c] is None for c in cols)
        for label, flat in (("f64", flat_f64), ("f32", flat_f32)):
            if flat:
                self.add(f"haar_cero_{label}")
                self.add(
                    f"haar_cero_{label}_motor_{'nulo' if engine_null else 'valor'}"
                )
        if flat_f64 != engine_null or flat_f32 != engine_null:
            self.discrepancies += 1
            if len(self.listed) < MAX_LISTED:
                self.listed.append(
                    {
                        "event_id": phase.event_id,
                        "fase": phase.name,
                        "n": phase.n,
                        "plano_f64": flat_f64,
                        "plano_f32": flat_f32,
                        "motor": got[cols[0]],
                        "oraculo": wanted[BANDS],
                    }
                )

    def totals(self) -> tuple[int, int, float]:
        out = sum(c[1] for c in self.columns.values())
        nulls = sum(c[2] for c in self.columns.values())
        worst = max((c[3] for c in self.columns.values()), default=0.0)
        return out, nulls, worst


def close_phase(phase: Phase, tally: Tally, n_min: int) -> None:
    """Compara una fase terminada con la fila del motor y suelta sus precios."""
    if phase.before:
        tally.add("antes_del_mes")
    elif phase.over:
        tally.add("sobre_presupuesto")
    elif phase.long:
        tally.add("omitidas_max_ticks")
    else:
        returns, flat_f64 = None, None
        if phase.n:
            returns, flat_f64 = engine_returns(phase.buf[: phase.n], phase.anchor)
        else:
            tally.add("fases_vacias")
        phase.buf = None
        wanted, flat_f32 = expected_columns(phase.name, returns, n_min)
        del returns
        if 0 < phase.n < n_min:
            tally.add("fases_cortas")
        tally.compare(phase.name, phase.spec, wanted)
        tally.add("comparadas")
        if flat_f32 is not None:
            tally.haar_zero(phase, phase.spec, wanted, (flat_f64, flat_f32))
    phase.buf = None


def measure(uris: dict[str, str], limit: int) -> tuple[Tally, dict]:
    """Recorre L1 una vez, una fase a la vez; devuelve lo medido y los parámetros del pie."""
    counters = {"filas": 0}
    with ExitStack() as stack:
        files = {}
        for label, uri in uris.items():
            try:
                files[label] = open_parquet(stack, uri)
            except FileNotFoundError:
                raise InputError(f"falta el archivo {uri}") from None
        footer = {
            k.decode(): v.decode()
            for k, v in (files["spectrum"].metadata.metadata or {}).items()
        }
        params = parse_meta(footer)
        columns = data_columns()
        absent = sorted(set(columns) - set(files["spectrum"].schema_arrow.names))
        if absent:
            raise InputError(f"a spectrum.parquet le faltan columnas: {absent}")
        if files["spectrum"].metadata.num_rows != files["events"].metadata.num_rows:
            raise InputError(
                f"spectrum tiene {files['spectrum'].metadata.num_rows} filas y events "
                f"{files['events'].metadata.num_rows}"
            )
        rows = zip(
            event_tuples(files["events"]),
            spectrum_rows(files["spectrum"], columns),
            strict=True,
        )
        phases = phases_of(rows, params["budget_ticks"], limit, counters)
        n_min = params["n_min"]
        tally = Tally()

        phase = next(phases, None)
        first_id = None
        for ids, prices in tick_groups(files["l1"]):
            if first_id is None:
                first_id = int(ids[0])
            last_id = int(ids[-1])
            while phase is not None:
                if phase.lo < first_id:
                    phase.before = True
                if not phase.skipped:
                    start = int(np.searchsorted(ids, phase.lo, side="right"))
                    stop = int(np.searchsorted(ids, phase.hi, side="right"))
                    phase.take(prices[start:stop])
                if phase.hi > last_id:
                    break
                close_phase(phase, tally, n_min)
                phase = next(phases, None)
            if phase is None:
                break
        if first_id is None:
            raise InputError("el consolidated.parquet del mes no tiene ticks")
        # Lo que queda termina después del último tick de L1 del mes.
        while phase is not None:
            phase.buf = None
            tally.add("despues_del_mes")
            phase = next(phases, None)
        tally.counts.update(counters)
    return tally, params


# ----------------------------------------------------------------------------------------------
# Salida
# ----------------------------------------------------------------------------------------------


def number(value: float) -> str:
    return f"{value:.3e}"


def csv_rows(tally: Tally) -> list[dict]:
    rows = []
    for column, (compared, out, nulls, worst) in tally.columns.items():
        rows.append(
            {
                "seccion": "columna",
                "columna": column,
                "comparadas": compared,
                "fuera_tolerancia": out,
                "nulos_distintos": nulls,
                "max_diff": number(worst),
            }
        )
    for name in INFO_COUNTS:
        rows.append(
            {
                "seccion": "informativo",
                "columna": name,
                "conteo": tally.counts.get(name, 0),
            }
        )
    rows.append(
        {
            "seccion": "informativo",
            "columna": "haar_discrepancias",
            "conteo": tally.discrepancies,
        }
    )
    for item in tally.listed:
        rows.append(
            {
                "seccion": "haar_discrepancia",
                "columna": item["event_id"],
                "detalle": (
                    f"fase={item['fase']} n={item['n']} plano_f64={int(item['plano_f64'])} "
                    f"plano_f32={int(item['plano_f32'])} motor={item['motor']!r} "
                    f"oraculo={item['oraculo']!r}"
                ),
            }
        )
    return rows


INFO_COUNTS = (
    "filas",
    "comparadas",
    "omitidas_max_ticks",
    "antes_del_mes",
    "despues_del_mes",
    "sobre_presupuesto",
    "fases_vacias",
    "fases_cortas",
    "haar_cero_f64",
    "haar_cero_f64_motor_nulo",
    "haar_cero_f64_motor_valor",
    "haar_cero_f32",
    "haar_cero_f32_motor_nulo",
    "haar_cero_f32_motor_valor",
)


def write_csv(rows: list[dict], out: str) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADER)
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


def final_line(month, theta, verdict, figures: dict | None = None) -> str:
    figures = figures or {}

    def cell(name, text=str):
        value = figures.get(name)
        return "-" if value is None else text(value)

    return (
        f"spectrum xcheck: mes={month} theta={theta} filas={cell('filas')} "
        f"comparadas={cell('comparadas')} omitidas={cell('omitidas')} "
        f"fuera_tolerancia={cell('fuera_tolerancia')} nulos_distintos={cell('nulos_distintos')} "
        f"max_diff={cell('max_diff', number)} haar_cero_f64={cell('haar_cero_f64')} "
        f"haar_cero_f32={cell('haar_cero_f32')} veredicto={verdict}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    if re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month) is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    try:
        theta = parse_theta(ns.theta)
    except ValueError as exc:
        parser.error(str(exc))
    if ns.max_ticks < 1:
        parser.error("--max-ticks debe ser un entero mayor que 0")
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_spectrum_xcheck_{ns.month}_{theta}.csv"

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
        "spectrum": partition_file(l3_root, key, theta, ns.month, "spectrum.parquet"),
    }
    started = time.perf_counter()
    log(
        f"spectrum xcheck: θ={theta} mes={ns.month} max_ticks={ns.max_ticks} tol={TOL:g}"
    )
    for label, uri in uris.items():
        log(f"  {label}: {uri}")
    try:
        tally, params = measure(uris, ns.max_ticks)
        if not tally.counts.get("comparadas"):
            raise InputError("ninguna fase quedó por comparar")
    except (InputError, OSError, ValueError, pa.ArrowException) as exc:
        log(f"FALLO {type(exc).__name__}: {exc}")
        log(final_line(ns.month, theta, "fallo"))
        return 1
    wall = time.perf_counter() - started
    log(
        f"state_version={params['state_version']} n_min={params['n_min']} "
        f"bands={params['bands']} window={params['window']} "
        f"budget_ticks={params['budget_ticks']} pared_s={wall:.1f}"
    )

    log(
        f"{'columna':<22} {'comparadas':>10} {'fuera_tol':>10} {'nulos_dist':>10} {'max_diff':>10}"
    )
    for column, (compared, bad, nulls, worst) in tally.columns.items():
        log(f"{column:<22} {compared:>10} {bad:>10} {nulls:>10} {number(worst):>10}")
    for info in INFO_COUNTS:
        log(f"{info:<32} {tally.counts.get(info, 0):>10}")
    log(f"{'haar_discrepancias':<32} {tally.discrepancies:>10}")
    for item in tally.listed:
        log(
            f"  discrepancia event_id={item['event_id']} fase={item['fase']} n={item['n']} "
            f"plano_f64={int(item['plano_f64'])} plano_f32={int(item['plano_f32'])} "
            f"motor={item['motor']!r} oraculo={item['oraculo']!r}"
        )
    try:
        write_csv(csv_rows(tally), out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")

    bad, nulls, worst = tally.totals()
    verdict = "ok" if bad == 0 and nulls == 0 else "revisar"
    figures = {
        "filas": tally.counts["filas"],
        "comparadas": tally.counts["comparadas"],
        "omitidas": tally.counts.get("omitidas_max_ticks", 0),
        "fuera_tolerancia": bad,
        "nulos_distintos": nulls,
        "max_diff": worst,
        "haar_cero_f64": tally.counts.get("haar_cero_f64", 0),
        "haar_cero_f32": tally.counts.get("haar_cero_f32", 0),
    }
    log(final_line(ns.month, theta, verdict, figures))
    return 0 if verdict == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", default=DEFAULT_THETA)
    parser.add_argument("--month", default=DEFAULT_MONTH)
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--l3-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
