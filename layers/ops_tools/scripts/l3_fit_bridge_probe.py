"""Sonda del puente browniano para la familia A (Forma) de L3 (ITSC-398).

Decide si `c_accel` dice algo que un paseo aleatorio condicionado no dice. Sobre un θ y un mes
del lago calcula, de cada evento, `c_accel` y `c_jump_share` observados y los compara con los
mismos registros sobre `--bridges` (20) puentes brownianos simulados por evento. El tamaño del
efecto es la distancia de Kolmogorov-Smirnov D entre las dos distribuciones empíricas; nunca se
reporta un valor p (con decenas de miles de eventos todo es significativo).

Registros (Épica ITSC-397, decisiones 3, 5 y 6). En la fase de confirmación (R, C] de un evento,
con `t_R, p_R` el ancla y `t_C, p_C` el cierre, ambos de L2:

  τ = (t − t_R) / (t_C − t_R)      y = (p − p_R) / (p_C − p_R)

  c_accel       coeficiente `a` del ajuste y = a·τ² + b·τ, sin intercepto, por mínimos
                cuadrados: a = (Στ²y·Στ² − Στy·Στ³) / (Στ⁴·Στ² − (Στ³)²)
  c_jump_share  (p_C − p_prev) / (p_C − p_R), con `p_prev` el precio del último tick con
                t < t_C; vale 1 si ningún tick es anterior a t_C

`c_accel` es nulo (decisión 6) con menos de 2 ticks, con todos los ticks en el mismo `t` o con
`t_C = t_R`; la sonda lo extiende al único caso que deja singular el sistema, un solo `τ`
positivo distinto (todo lo demás en `τ = 0`). Un evento con `c_accel` nulo queda fuera de las dos
muestras y se cuenta en `excluidos`. También se excluye, y se cuenta, el evento cuya referencia
cae antes del primer tick del mes y cuyos ticks no caben entero en el `consolidated.parquet`
del mes (a lo sumo uno por mes y θ): la sonda solo abre el L1 del mes.

Puente (decisión 2). Con `n = c_ticks` (de `summaries.parquet`), `σ² = c_realized_var / n` (de
`path.parquet`) y `ℓ = ln p`, el puente es, en los tiempos reales de los ticks del evento,

  ℓ_k = ℓ_R + (ℓ_C − ℓ_R)·k/n + σ·(W_k − (k/n)·W_n),    k = 1..n

con `W` un paseo gaussiano estándar. Se simulan en la misma pasada que los ticks, sin guardar los
tiempos del evento: los momentos del puente son lineales en `W_n`,

  Σ τ^j·y_k = Σ τ^j·k/n + (σ/d)·(Σ τ^j·W_k − W_n·Σ τ^j·k/n),    d = ℓ_C − ℓ_R,

así que por puente basta acumular `Σ τ^j·W_k` (j = 1, 2), el `W` del último tick anterior a
`t_C` y el `W` corriente, y aplicar `W_n` al cerrar. El registro del puente se calcula en las
coordenadas normalizadas del log-precio, `y = (ℓ − ℓ_R) / (ℓ_C − ℓ_R)`, que para un movimiento
de tamaño θ ≈ 10⁻³ difieren de las del precio en menos de θ/2: muy por debajo de la resolución
de D. El puente termina en `p_C` (k = n, `y = 1`); el último tick observado puede tener otro
precio si el grupo de empate de L2 lo dejó así. Observado: N valores; simulado: puentes·N.

D es la distancia exacta `sup |F_obs − F_sim|`, evaluada en todos los valores de las dos
muestras con cuentas enteras. Los cuantiles usan el rango más cercano: con N valores, el cuantil
q es el que ocupa el lugar ⌈N·q⌉ al ordenarlos.

Umbral (decisión 1, fijado antes de medir): `veredicto=sigue` si y solo si `D_accel ≥ 0,10`;
si no, `retirada`: la familia A no dice nada que un puente no diga.

Args:
  --theta 0.00098052  --month YYYY-MM          (obligatorios; un θ y un mes por corrida)
  --l1-root gs://<proyecto>-landing/l1         (por defecto, las del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2        OPS_RESULTS_URI; también rutas locales con la
  --l3-root gs://<proyecto>-dc-descriptors/l3   misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --bridges 20   --seed <semilla fija del script>
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_fit_bridge_probe_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

Con el mismo `--seed`, los mismos archivos y los mismos argumentos el CSV es idéntico byte a
byte: los normales salen de un solo generador de NumPy, evento por evento y tick por tick.

No importa ninguna capa: una importación cambiaría `uv.lock`. Abre por su ruta completa
`events.parquet` de L2, `summaries.parquet` y `path.parquet` de L3 y el `consolidated.parquet`
del mes de L1, nunca un dataset sobre un prefijo, que pide un objeto sin barra y la condición
IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3, «La trampa del 403
por prefijo»). Une las tres tablas por posición y comprueba que `confirm_agg_trade_id` coincide.

Memoria: los ticks se leen un row group a la vez y solo `agg_trade_id`, `transact_time` y
`price`; los tres archivos por evento se leen en lotes de `BATCH` filas; el estado del evento
abierto son unos pocos KB (momentos observados y, por puente, dos momentos y dos `W`). Un
segmento de ticks se procesa en trozos de `CHUNK`, así que el pico por tick es O(CHUNK · puentes).
Lo único que crece con el mes son los resultados: se necesitan todos los valores para que D y los
cuantiles sean exactos (≈ (1 + puentes)·16 B por evento, reservados al abrir sin tocar la página
hasta escribirla).

Salida: la tabla de cuantiles, la ruta del CSV y, como última línea, `fit probe: mes=… theta=…
eventos=… excluidos=… puentes=… semilla=… D_accel=… D_jump=… pared_s=…
veredicto=sigue|retirada|fallo` (`eventos` son los N de las muestras).

Códigos: 0 si midió; 1 si falta alguno de los cuatro archivos, la lectura falla, los archivos no
cuadran entre sí o no queda ningún evento medible; 2 por uso. Solo lee; lo único que escribe es
el CSV de `--out`.
"""

import argparse
import csv
import io
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

DEFAULT_SEED = 397_398
DEFAULT_BRIDGES = 20
# Umbral de la decisión 1, fijado antes de medir.
THRESHOLD = 0.10
# Filas por lote al leer los archivos por evento, y ticks por trozo de un segmento.
BATCH = 32_768
CHUNK = 65_536
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
REGISTERS = ("c_accel", "c_jump_share")
CSV_HEADER = ["registro", "fuente", "medida", "valor"]
TICK_COLUMNS = ["agg_trade_id", "transact_time", "price"]
EVENT_COLUMNS = [
    "reference_price",
    "reference_time",
    "reference_agg_trade_id",
    "confirm_price",
    "confirm_time",
    "confirm_agg_trade_id",
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


def l1_file(root: str, key: tuple[str, str, str], month: str) -> str:
    provider, market, asset = key
    year, number = month.split("-")
    return (
        f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/year={year}/month={number}/consolidated.parquet"
    )


def partition_file(
    root: str, key: tuple[str, str, str], theta: str, month: str, name: str
) -> str:
    provider, market, asset = key
    year, number = month.split("-")
    return (
        f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/theta={theta}/year={year}/month={number}/{name}"
    )


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
    """`(ids, times, prices)` de cada row group no vacío de L1, uno a la vez."""
    previous = None
    for group in range(parquet.num_row_groups):
        if not parquet.metadata.row_group(group).num_rows:
            continue
        table = parquet.read_row_group(group, columns=TICK_COLUMNS)
        ids = int64s(single(table.column("agg_trade_id")))
        times = int64s(single(table.column("transact_time")))
        prices = lows(single(table.column("price")))
        if len(ids) > 1 and (np.diff(ids).min() <= 0 or np.diff(times).min() < 0):
            raise InputError(f"row group {group}: L1 debe venir ordenado")
        if previous is not None and (ids[0] <= previous[0] or times[0] < previous[1]):
            raise InputError(f"row group {group}: L1 debe venir ordenado")
        previous = (int(ids[-1]), int(times[-1]))
        yield ids, times, prices
        del table, ids, times, prices


def column_rows(parquet: pq.ParquetFile, columns: list[str]):
    """Las filas de un archivo por evento como tuplas de enteros, un lote a la vez."""
    for batch in parquet.iter_batches(batch_size=BATCH, columns=columns):
        values = []
        for column in batch.columns:
            if pa.types.is_decimal128(column.type):
                values.append(lows(column).tolist())
            elif pa.types.is_integer(column.type) and not column.null_count or pa.types.is_floating(column.type) and not column.null_count:
                values.append(column.to_numpy(zero_copy_only=False).tolist())
            else:
                raise InputError(
                    f"columna {batch.schema.names[0]} con nulos o tipo raro"
                )
        yield from zip(*values, strict=True)


def event_rows(events: pq.ParquetFile, summaries: pq.ParquetFile, path: pq.ParquetFile):
    """Los tres archivos por evento, unidos por posición y con la clave comprobada."""
    names = {
        "events": set(EVENT_COLUMNS),
        "summaries": {"confirm_agg_trade_id", "c_ticks"},
        "path": {"confirm_agg_trade_id", "c_realized_var"},
    }
    for label, parquet in (
        ("events", events),
        ("summaries", summaries),
        ("path", path),
    ):
        missing = names[label] - set(parquet.schema_arrow.names)
        if missing:
            raise InputError(f"{label}.parquet: faltan columnas {sorted(missing)}")
    joined = zip(
        column_rows(events, EVENT_COLUMNS),
        column_rows(summaries, ["confirm_agg_trade_id", "c_ticks"]),
        column_rows(path, ["confirm_agg_trade_id", "c_realized_var"]),
        strict=True,
    )
    try:
        for event, summary, walk in joined:
            if not event[5] == summary[0] == walk[0]:
                raise InputError(
                    "events, summaries y path no están alineados: "
                    f"confirm_agg_trade_id {event[5]}, {summary[0]}, {walk[0]}"
                )
            yield event, summary[1], walk[1]
    except ValueError as exc:
        if "zip()" in str(exc):
            raise InputError(
                "events, summaries y path tienen distinto número de filas"
            ) from exc
        raise


class OpenEvent:
    """El evento cuya confirmación se está leyendo: momentos observados y de los puentes."""

    def __init__(self, event: tuple, ticks: int, realized_var: float, bridges: int):
        (self.pr, self.tr, self.ref_id, self.pc, self.tc, self.conf_id) = event
        self.n = ticks
        self.bridges = bridges
        self.span = self.tc - self.tr
        self.dp = self.pc - self.pr
        # Eventos sin registro por construcción: no se leen sus ticks.
        self.skip = ticks < 2 or self.span <= 0 or self.dp == 0
        self.found = 0
        self.t2 = self.t3 = self.t4 = self.sy1 = self.sy2 = 0.0
        self.pos_min: int | None = None
        self.pos_max: int | None = None
        self.prev_price: int | None = None
        if self.skip:
            return
        # σ/d con d = ln(p_C / p_R): el factor de los momentos del puente.
        self.scale = np.sqrt(realized_var / ticks) / np.log(self.pc / self.pr)
        self.a1 = np.zeros(bridges)
        self.a2 = np.zeros(bridges)
        self.pk1 = self.pk2 = 0.0
        self.walk = np.zeros(bridges)
        self.prev_walk = np.zeros(bridges)
        self.prev_k = 0

    def add(self, times: np.ndarray, prices: np.ndarray, rng: np.random.Generator):
        for start in range(0, len(times), CHUNK):
            self._chunk(
                times[start : start + CHUNK], prices[start : start + CHUNK], rng
            )

    def _chunk(self, times, prices, rng) -> None:
        m = len(times)
        tau = (times - self.tr).astype(np.float64) / self.span
        y = (prices - self.pr).astype(np.float64) / self.dp
        tau2 = tau * tau
        self.t2 += float(tau2.sum())
        self.t3 += float((tau2 * tau).sum())
        self.t4 += float((tau2 * tau2).sum())
        self.sy1 += float((tau * y).sum())
        self.sy2 += float((tau2 * y).sum())
        positive = times[times > self.tr]
        if len(positive):
            low, high = int(positive[0]), int(positive[-1])
            self.pos_min = low if self.pos_min is None else min(self.pos_min, low)
            self.pos_max = high if self.pos_max is None else max(self.pos_max, high)
        before = int(np.searchsorted(times, self.tc, side="left"))
        if before:
            self.prev_price = int(prices[before - 1])
        # Los puentes: W_k acumulado a través de los trozos con el normal de cada tick.
        walk = np.cumsum(rng.standard_normal((m, self.bridges)), axis=0)
        walk += self.walk
        ratio = (self.found + np.arange(1, m + 1)) / self.n
        self.a1 += tau @ walk
        self.a2 += tau2 @ walk
        self.pk1 += float(tau @ ratio)
        self.pk2 += float(tau2 @ ratio)
        if before:
            self.prev_walk = walk[before - 1].copy()
            self.prev_k = self.found + before
        self.walk = walk[-1].copy()
        self.found += m

    def result(self):
        """`(accel, jump, accel_puentes, jump_puentes)` o `None` si el evento se excluye."""
        if self.skip or self.found < 2 or self.pos_min == self.pos_max:
            return None
        det = self.t4 * self.t2 - self.t3 * self.t3
        if self.pos_min is None or det <= 0:
            return None
        accel = (self.sy2 * self.t2 - self.sy1 * self.t3) / det
        if self.prev_price is None:
            jump = 1.0
        else:
            jump = (self.pc - self.prev_price) / (self.pc - self.pr)
        # Momentos del puente: lineales en W_n, que es el W corriente al cerrar.
        s1 = self.pk1 + self.scale * (self.a1 - self.walk * self.pk1)
        s2 = self.pk2 + self.scale * (self.a2 - self.walk * self.pk2)
        accel_bridges = (s2 * self.t2 - s1 * self.t3) / det
        if self.prev_k == 0:
            jump_bridges = np.ones(self.bridges)
        else:
            ratio = self.prev_k / self.n
            y_prev = ratio + self.scale * (self.prev_walk - ratio * self.walk)
            jump_bridges = 1.0 - y_prev
        return accel, jump, accel_bridges, jump_bridges


class Samples:
    """Los valores observados y simulados, en memoria reservada al abrir."""

    def __init__(self, capacity: int, bridges: int) -> None:
        self.bridges = bridges
        self.n = 0
        self.excluded = 0
        self.observed = {r: np.empty(capacity) for r in REGISTERS}
        self.simulated = {r: np.empty((capacity, bridges)) for r in REGISTERS}

    def add(self, result: tuple) -> None:
        accel, jump, accel_bridges, jump_bridges = result
        self.observed["c_accel"][self.n] = accel
        self.observed["c_jump_share"][self.n] = jump
        self.simulated["c_accel"][self.n] = accel_bridges
        self.simulated["c_jump_share"][self.n] = jump_bridges
        self.n += 1

    def finish(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Los pares `(observado, simulado)` ordenados de cada registro; libera el resto."""
        pairs = {}
        for register in REGISTERS:
            observed = self.observed.pop(register)[: self.n]
            observed.sort()
            simulated = self.simulated.pop(register)[: self.n].reshape(-1)
            simulated.sort()
            pairs[register] = (observed, simulated)
        return pairs


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


def quantile(values: np.ndarray, percent: int) -> float:
    """Rango más cercano: el lugar ⌈N·q⌉ de los valores ya ordenados."""
    rank = -(-len(values) * percent // 100)
    return float(values[rank - 1])


def measure(
    uris: dict[str, str], seed: int, bridges: int
) -> tuple[Samples, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Recorre el mes una vez; devuelve las muestras ordenadas de los dos registros."""
    rng = np.random.default_rng(seed)
    with ExitStack() as stack:
        files = {}
        for label, uri in uris.items():
            try:
                files[label] = open_parquet(stack, uri)
            except FileNotFoundError:
                raise InputError(f"falta el archivo {uri}") from None
        capacity = files["events"].metadata.num_rows
        samples = Samples(capacity, bridges)
        events = event_rows(files["events"], files["summaries"], files["path"])

        def next_event() -> OpenEvent | None:
            row = next(events, None)
            return None if row is None else OpenEvent(row[0], row[1], row[2], bridges)

        def close(event: OpenEvent) -> None:
            if event.skip:
                samples.excluded += 1
                return
            if event.found != event.n:
                if event.ref_id < first_id:
                    samples.excluded += (
                        1  # arranca en el mes anterior: L1 no lo trae entero
                    )
                    return
                raise InputError(
                    f"confirm_agg_trade_id {event.conf_id}: L1 trae {event.found} ticks "
                    f"entre la referencia y la confirmación y summaries dice {event.n}"
                )
            result = event.result()
            if result is None:
                samples.excluded += 1
            else:
                samples.add(result)

        current = next_event()
        first_id = None
        for ids, times, prices in tick_groups(files["l1"]):
            if first_id is None:
                first_id = int(ids[0])
            last_id = int(ids[-1])
            while current is not None and current.ref_id < last_id:
                if not current.skip:
                    start = int(np.searchsorted(ids, current.ref_id, side="right"))
                    stop = int(np.searchsorted(ids, current.conf_id, side="right"))
                    if stop > start:
                        current.add(times[start:stop], prices[start:stop], rng)
                if current.conf_id > last_id:
                    break
                close(current)
                current = next_event()
            if current is None:
                break
        if current is not None or next(events, None) is not None:
            raise InputError("L2 trae eventos más allá de los ticks del mes de L1")
        if first_id is None:
            raise InputError("el consolidated.parquet del mes no tiene ticks")
    if samples.n == 0:
        raise InputError("ningún evento medible en el mes")
    return samples, samples.finish()


def csv_rows(
    samples: Samples,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    distances: dict[str, float],
    seed: int,
) -> list[dict]:
    def number(value: float) -> str:
        return format(value, ".17g")

    rows = []
    for register, (observed, simulated) in pairs.items():
        for source, values in (("observado", observed), ("simulado", simulated)):
            for percent in QUANTILES:
                rows.append(
                    {
                        "registro": register,
                        "fuente": source,
                        "medida": f"p{percent}",
                        "valor": number(quantile(values, percent)),
                    }
                )
            rows.append(
                {
                    "registro": register,
                    "fuente": source,
                    "medida": "N",
                    "valor": len(values),
                }
            )
    rows.append(
        {"registro": "c_accel", "fuente": "ambas", "medida": "D"}
        | {"valor": number(distances["c_accel"])}
    )
    rows.append(
        {"registro": "c_jump_share", "fuente": "ambas", "medida": "D"}
        | {"valor": number(distances["c_jump_share"])}
    )
    for name, value in (
        ("excluidos", samples.excluded),
        ("puentes", samples.bridges),
        ("semilla", seed),
    ):
        rows.append({"registro": "", "fuente": "", "medida": name, "valor": value})
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


def final_line(
    month, theta, bridges, seed, wall, verdict, samples=None, distances=None
) -> str:
    def cell(value, text="{}"):
        return "-" if value is None else text.format(value)

    return (
        f"fit probe: mes={month} theta={theta} "
        f"eventos={cell(samples and samples.n)} "
        f"excluidos={cell(samples and samples.excluded)} puentes={bridges} semilla={seed} "
        f"D_accel={cell(distances and distances['c_accel'], '{:.6f}')} "
        f"D_jump={cell(distances and distances['c_jump_share'], '{:.6f}')} "
        f"pared_s={wall:.1f} veredicto={verdict}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    if re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month) is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    try:
        theta = parse_theta(ns.theta)
    except ValueError as exc:
        parser.error(str(exc))
    if ns.bridges < 1:
        parser.error("--bridges debe ser un entero mayor que 0")
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_fit_bridge_probe_{ns.month}_{theta}.csv"

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
    }
    started = time.perf_counter()
    log(f"fit probe: θ={theta} mes={ns.month} puentes={ns.bridges} semilla={ns.seed}")
    for label, uri in uris.items():
        log(f"  {label}: {uri}")
    try:
        samples, pairs = measure(uris, ns.seed, ns.bridges)
        distances = {r: ks_distance(*pairs[r]) for r in REGISTERS}
    except (InputError, OSError, ValueError, pa.ArrowException) as exc:
        log(f"FALLO {type(exc).__name__}: {exc}")
        wall = time.perf_counter() - started
        log(final_line(ns.month, theta, ns.bridges, ns.seed, wall, "fallo"))
        return 1
    wall = time.perf_counter() - started

    rows = csv_rows(samples, pairs, distances, ns.seed)
    log(f"{'registro':<13} {'fuente':<10} {'medida':<8} valor")
    for row in rows:
        log(
            f"{row['registro']:<13} {row['fuente']:<10} {row['medida']:<8} {row['valor']}"
        )
    try:
        write_csv(rows, out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    verdict = "sigue" if distances["c_accel"] >= THRESHOLD else "retirada"
    log(
        final_line(
            ns.month, theta, ns.bridges, ns.seed, wall, verdict, samples, distances
        )
    )
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
    parser.add_argument("--bridges", type=int, default=DEFAULT_BRIDGES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
