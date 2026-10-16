"""Verificación cruzada de `summaries.parquet` de L3 contra una agregación independiente.

Cierra el criterio 7 de TRD-L3 §13 sobre datos reales. Para cada θ recorre con
`dc_frames.read_frames` (modo por chunks) las tramas de cada evento cerrado del mes, agrega
por fase con NumPy y compara fila a fila con `summaries.parquet` de la misma partición, por
`(theta, confirm_agg_trade_id)`: mismas filas, mismo orden y mismos valores en las 17
columnas de datos. No importa `l3_dc_descriptors`: la agregación es propia y comparte con el
resumidor solo el contrato de §7.2.

La agregación es exacta: conteos; `quantity` por lado según `is_buyer_maker`; máximo y mínimo
de precio; duración (último menos primer `transact_time` de la fase); los `confirm_instant_*`
(ticks de la confirmación con `transact_time = confirm_time`, mejor y peor precio según la
dirección); y el VWAP como `Σ price·quantity / Σ quantity`, con el numerador en entero de
Python por evento y redondeo half-even a 10⁻⁸. Los decimales se leen como el entero sin
escalar (la palabra baja del decimal de 128 bits) y las sumas de un trozo se parten en
limbos de 22 bits para que ningún producto ni suma desborde `int64`.

Args:
  --month YYYY-MM                    (obligatorio; la partición de L2 y de L3)
  --thetas 0.00100000,0.00250000 | all   (por defecto, el catálogo de `--catalog`; `all` toma
                                      los θ que tienen `summaries.parquet` del mes en L3)
  --catalog gs://<proyecto>-manifest/l2/thetas.yaml   (el catálogo único de θ; también local)
  --l1-root gs://<proyecto>-landing/l1          (por defecto, los del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2         OPS_RESULTS_URI; también rutas locales)
  --l3-root gs://<proyecto>-dc-descriptors/l3
  --provider binance --market spot --asset BTCUSDT
  --max-diffs 100                    (cuántas diferencias se imprimen por θ; las cuentas y la
                                      tabla las cuentan todas)

Un θ del catálogo sin `summaries.parquet` del mes es una diferencia (`falta el archivo`).
Una fila del oráculo sin su fila en L3, o al revés, también.

Salida: una línea por diferencia
`DIFERENCIA θ=… confirm_agg_trade_id=… columna=… oráculo=… L3=…`, la tabla por θ (filas
comparadas, diferencias, segundos) y, como última línea,
`check mes: mes=… thetas=… filas=… diferencias=… pared_s=… veredicto=ok|diferencias|fallo`.
Los segundos de un θ son su cómputo (agregación y comparación); la lectura de L1 y L2 se
comparte entre todos los θ y solo cuenta en la pared total. Con `OPS_RESULTS_URI` deja además
`l3_check_month_<mes>_<fecha>.csv` con la tabla.

Códigos: 0 sin diferencias; 1 con diferencias, o si el lector falla (`FramesError`); 2 por uso.

Solo lee. Lista de forma recursiva y abre cada archivo por su ruta: nunca `pyarrow.dataset`
sobre un prefijo, que pide el objeto `l3` sin barra y la condición IAM por prefijo de
`ops-script` lo niega con 403. Dónde corre: con `ops-script` (la service account lee `l1/`,
`l2/` y `l3/`) o en local con `uv run`.
"""

import argparse
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dc_frames import FramesError, read_frames
from dc_frames.lake import open_parquet
from pyutils import resolve_fs

SUMMARIES = "summaries.parquet"
EVENT_COLUMNS = ("confirm_agg_trade_id", "confirm_time", "direction")
# (nombre, clase): `dec` es un decimal con 8 decimales y `int` un entero. Las claves y las
# 17 columnas de datos de TRD-L3 §7.2, en su orden.
COLUMNS = (
    ("theta", "dec"),
    ("confirm_agg_trade_id", "int"),
    ("c_ticks", "int"),
    ("c_quantity_buy", "dec"),
    ("c_quantity_sell", "dec"),
    ("c_price_max", "dec"),
    ("c_price_min", "dec"),
    ("c_duration", "int"),
    ("c_vwap", "dec"),
    ("os_ticks", "int"),
    ("os_quantity_buy", "dec"),
    ("os_quantity_sell", "dec"),
    ("os_price_max", "dec"),
    ("os_price_min", "dec"),
    ("os_duration", "int"),
    ("os_vwap", "dec"),
    ("confirm_instant_ticks", "int"),
    ("confirm_instant_price_best", "dec"),
    ("confirm_instant_price_worst", "dec"),
)
NAMES = [name for name, _ in COLUMNS]
COL = {name: i for i, name in enumerate(NAMES)}
# Nulos juntos cuando el overshoot es vacío (`os_ticks` = 0).
NULLABLE = ("os_price_max", "os_price_min", "os_duration", "os_vwap")
LIMB = 22
MASK = (1 << LIMB) - 1
# Un producto de limbos pesa menos de 2⁴⁴: 2¹⁸ términos suman menos de 2⁶².
BLOCK = 1 << 18
# Bajo este largo, un bucle de Python cuesta menos que las llamadas de NumPy.
SMALL = 64
PROGRESS_EVERY_S = 60


def log(message: str) -> None:
    print(message, flush=True)


def lows(array: pa.Array) -> np.ndarray:
    """Los valores de un `decimal128(18,8)` sin escalar: la palabra baja, vista sin copia."""
    words = np.frombuffer(array.buffers()[1], dtype="<i8")
    return words[2 * array.offset : 2 * (array.offset + len(array)) : 2]


def int64s(array: pa.Array) -> np.ndarray:
    """Una columna `int64` sin nulos, vista sin copia."""
    words = np.frombuffer(array.buffers()[1], dtype="<i8")
    return words[array.offset : array.offset + len(array)]


def limbs(values: np.ndarray) -> list[np.ndarray]:
    """`values` en limbos de 22 bits (el último con signo): Σ limbo[i] · 2^(22 i) = values."""
    top = max(int(values.max()), -int(values.min()), 1)
    count = -(-top.bit_length() // LIMB)
    parts = [(values >> (LIMB * i)) & MASK for i in range(count - 1)]
    parts.append(values >> (LIMB * (count - 1)))
    return parts


def half_even(numerator: int, denominator: int) -> int:
    """`numerator / denominator` redondeado al entero más cercano, empates al par."""
    quotient, rest = divmod(numerator, denominator)
    twice = 2 * rest
    if twice > denominator or (twice == denominator and quotient % 2):
        quotient += 1
    return quotient


class Phase:
    """Una fase del evento: los agregados de sus trozos, en enteros de Python sin escalar."""

    def __init__(self, instant_time: int | None = None) -> None:
        self.ticks = 0
        self.buy = 0
        self.sell = 0
        self.notional = 0  # Σ price·quantity, escala 10⁻¹⁶
        self.high: int | None = None
        self.low: int | None = None
        self.first: int | None = None
        self.last: int | None = None
        self.instant_time = instant_time
        self.instant_ticks = 0
        self.instant_high: int | None = None
        self.instant_low: int | None = None

    def _range(self, high: int, low: int) -> None:
        self.high = high if self.high is None else max(self.high, high)
        self.low = low if self.low is None else min(self.low, low)

    def _instant(self, count: int, high: int, low: int) -> None:
        self.instant_ticks += count
        if self.instant_high is None:
            self.instant_high, self.instant_low = high, low
        else:
            self.instant_high = max(self.instant_high, high)
            self.instant_low = min(self.instant_low, low)

    def consume(self, frames) -> None:
        for frame in frames:
            if len(frame):
                self._add(frame)

    def _add(self, frame) -> None:
        n = len(frame)
        price, quantity = lows(frame.price), lows(frame.quantity)
        times = int64s(frame.transact_time)
        self.ticks += n
        if self.first is None:
            self.first = int(times[0])
        self.last = int(times[n - 1])
        if n <= SMALL:
            self._add_small(
                price.tolist(),
                quantity.tolist(),
                frame.is_buyer_maker.to_pylist(),
                times.tolist(),
            )
            return
        sold = frame.is_buyer_maker.to_numpy(zero_copy_only=False)
        self._range(int(price.max()), int(price.min()))
        for start in range(0, n, BLOCK):
            stop = start + BLOCK
            self._sums(price[start:stop], quantity[start:stop], sold[start:stop])
        if self.instant_time is not None:
            same = times == self.instant_time
            count = int(np.count_nonzero(same))
            if count:
                picked = price[same]
                self._instant(count, int(picked.max()), int(picked.min()))

    def _sums(self, price: np.ndarray, quantity: np.ndarray, sold: np.ndarray) -> None:
        """Las sumas exactas de un bloque: cantidad por lado y `Σ price·quantity`."""
        by_limb = limbs(quantity)
        mask = sold.astype(np.int64)
        for i, part in enumerate(by_limb):
            sell = int(np.dot(part, mask))
            self.sell += sell << (LIMB * i)
            self.buy += (int(part.sum()) - sell) << (LIMB * i)
        for i, left in enumerate(limbs(price)):
            for j, right in enumerate(by_limb):
                self.notional += int(np.dot(left, right)) << (LIMB * (i + j))

    def _add_small(self, price, quantity, sold, times) -> None:
        for p, q, s, t in zip(price, quantity, sold, times, strict=True):
            if s:
                self.sell += q
            else:
                self.buy += q
            self.notional += p * q
            if t == self.instant_time:
                self._instant(1, p, p)
        self._range(max(price), min(price))

    def vwap(self) -> int | None:
        total = self.buy + self.sell
        return half_even(self.notional, total) if total > 0 else None

    def duration(self) -> int:
        return self.last - self.first


class Oracle:
    """Las filas de un θ que se esperan, un row group de L3 a la vez."""

    def __init__(self, theta: str, max_diffs: int) -> None:
        self.theta = theta
        self.unscaled = int(Decimal(theta).scaleb(8))
        self.max_diffs = max_diffs
        self.rows = 0
        self.diffs = 0
        self.seconds = 0.0
        self.printed = 0
        self.l3: pq.ParquetFile | None = None
        self.sizes: list[
            tuple[int, int]
        ] = []  # (índice del row group, filas) pendientes
        self.buffer: np.ndarray | None = None
        self.fill = 0
        # Celdas que no caben en int64 o no existen (VWAP sin cantidad): siempre difieren.
        self.special: dict[tuple[int, int], object] = {}
        self.extra = 0  # eventos del oráculo sin fila en L3


def show(kind: str, value: object) -> str:
    if kind == "dec" and isinstance(value, int):
        sign, digits = ("-", -value) if value < 0 else ("", value)
        whole, fraction = divmod(digits, 10**8)
        return f"{sign}{whole}.{fraction:08d}"
    return str(value)


def report(lane: Oracle, row: int, column: int, expected: object, got: object) -> None:
    lane.diffs += 1
    if lane.printed >= lane.max_diffs:
        return
    lane.printed += 1
    kind = COLUMNS[column][1]
    ident = lane.buffer[row, COL["confirm_agg_trade_id"]]
    log(
        f"DIFERENCIA θ={lane.theta} confirm_agg_trade_id={int(ident)} "
        f"columna={NAMES[column]} oráculo={show(kind, expected)} L3={show(kind, got)}"
    )


def column_values(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Los valores sin escalar de una columna de L3 y su validez, sin copiar los buffers."""
    chunked = table.column(name)
    array = chunked.combine_chunks() if chunked.num_chunks != 1 else chunked.chunk(0)
    if pa.types.is_decimal(array.type):
        values = lows(array)
    else:
        values = int64s(array)
    if array.null_count:
        valid = array.is_valid().to_numpy(zero_copy_only=False)
    else:
        valid = np.ones(len(array), dtype=bool)
    return values, valid


def compare(lane: Oracle, group: int) -> None:
    """Compara las `lane.fill` filas del oráculo con las primeras del row group `group`."""
    n = lane.fill
    table = lane.l3.read_row_group(group)
    expected = lane.buffer[:n]
    present = expected[:, COL["os_ticks"]] > 0
    for column, name in enumerate(NAMES):
        got, valid = column_values(table, name)
        got, valid = got[:n], valid[:n]
        want = expected[:, column]
        wanted_valid = present if name in NULLABLE else np.ones(n, dtype=bool)
        good = (wanted_valid == valid) & (~wanted_valid | (want == got))
        for row, special_column in lane.special:
            if special_column == column and row < n:
                good[row] = True  # la celda especial se reporta aparte
        for row in np.flatnonzero(~good):
            row = int(row)
            report(
                lane,
                row,
                column,
                int(want[row]) if wanted_valid[row] else None,
                int(got[row]) if valid[row] else None,
            )
    for (row, column), value in lane.special.items():
        if row < n:
            got, valid = column_values(table, NAMES[column])
            report(lane, row, column, value, int(got[row]) if valid[row] else None)
    lane.rows += n
    lane.special.clear()


def flush(lane: Oracle) -> None:
    """Compara el row group de L3 que le toca al buffer y lo suelta."""
    group, _ = lane.sizes.pop(0)
    compare(lane, group)
    lane.buffer, lane.fill = None, 0


def add_event(lane: Oracle, values: list[int | None]) -> None:
    """Pone la fila de un evento en el buffer del row group en curso."""
    if lane.buffer is None:
        if not lane.sizes:
            lane.extra += 1  # L3 no tiene más filas: se cuenta al cerrar
            return
        lane.buffer = np.empty((lane.sizes[0][1], len(COLUMNS)), dtype=np.int64)
    row = lane.fill
    try:
        lane.buffer[row] = [0 if value is None else value for value in values]
    except OverflowError:
        for column, value in enumerate(values):
            try:
                lane.buffer[row, column] = 0 if value is None else value
            except OverflowError:
                lane.buffer[row, column] = 0
                lane.special[row, column] = value
    if values[COL["c_vwap"]] is None:
        lane.special[row, COL["c_vwap"]] = None
    lane.fill += 1
    if lane.fill == len(lane.buffer):
        flush(lane)


def summarize(event, lane: Oracle) -> list[int | None]:
    """La fila esperada de un evento (consume sus dos iteradores de fase)."""
    batch = event.event
    confirm_id = batch.column("confirm_agg_trade_id")[0].as_py()
    confirm_time = batch.column("confirm_time")[0].as_py()
    up = batch.column("direction")[0].as_py() == 1
    confirmation = Phase(confirm_time)
    confirmation.consume(event.confirmation)
    overshoot = Phase()
    overshoot.consume(event.overshoot)
    has_overshoot = overshoot.ticks > 0
    high, low = confirmation.instant_high, confirmation.instant_low
    best, worst = (high, low) if up else (low, high)
    return [
        lane.unscaled,
        confirm_id,
        confirmation.ticks,
        confirmation.buy,
        confirmation.sell,
        confirmation.high,
        confirmation.low,
        confirmation.duration(),
        confirmation.vwap(),
        overshoot.ticks,
        overshoot.buy,
        overshoot.sell,
        overshoot.high if has_overshoot else None,
        overshoot.low if has_overshoot else None,
        overshoot.duration() if has_overshoot else None,
        overshoot.vwap() if has_overshoot else None,
        confirmation.instant_ticks,
        best,
        worst,
    ]


def finish(lane: Oracle) -> None:
    """Cierra el θ: compara el tramo parcial y cuenta las filas de L3 que el oráculo no tiene."""
    if lane.fill:
        group, size = lane.sizes.pop(0)
        compare(lane, group)
        lane.sizes.insert(0, (group, size - lane.fill))
        lane.buffer, lane.fill = None, 0
    missing = sum(size for _, size in lane.sizes)
    if lane.extra:
        lane.diffs += 1
        log(
            f"DIFERENCIA θ={lane.theta} filas: el oráculo tiene {lane.extra} "
            f"evento(s) más que L3 ({lane.rows + lane.extra} contra {lane.rows})"
        )
    if missing:
        lane.diffs += 1
        log(
            f"DIFERENCIA θ={lane.theta} filas: L3 tiene {missing} fila(s) más que "
            f"el oráculo ({lane.rows + missing} contra {lane.rows})"
        )


def list_files(root: str) -> list[str]:
    """Todos los archivos bajo `root`, de forma recursiva: objetos por prefijo en GCS."""
    if not root.startswith("gs://"):
        return [str(p) for p in Path(root).rglob("*") if p.is_file()]
    from google.cloud import storage

    bucket, _, prefix = root.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    blobs = storage.Client().list_blobs(bucket, prefix=prefix)
    return [f"gs://{bucket}/{blob.name}" for blob in blobs]


def listed_thetas(l3_root: str, key: tuple[str, str, str], month: str) -> list[str]:
    provider, market, asset = key
    series = f"{l3_root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
    year, number = month.split("-")
    pattern = re.compile(
        rf"/theta=(0\.\d{{8}})/year={year}/month={number}/{re.escape(SUMMARIES)}$"
    )
    found = (pattern.search(path) for path in list_files(series))
    return sorted({m[1] for m in found if m})


def read_catalog(source: str) -> list[str]:
    """Los θ del catálogo de L2 (`scale: 100000000` y `- <round(θ × 10⁸)>`) como texto."""
    fs, resolved = resolve_fs(source)
    with fs.open_input_stream(resolved) as stream:
        text = stream.read().decode()
    if not re.search(r"^scale:\s*100000000\s*(?:#.*)?$", text, re.MULTILINE):
        raise ValueError(f"{source}: `scale` debe ser 100000000")
    integers = [
        int(m[1]) for m in re.finditer(r"^\s*-\s*(\d+)\s*(?:#.*)?$", text, re.MULTILINE)
    ]
    if not integers:
        raise ValueError(f"{source}: sin θ")
    return [f"{Decimal(i).scaleb(-8):.8f}" for i in sorted(integers)]


def parse_thetas(text: str) -> list[str]:
    thetas = []
    for part in text.split(","):
        try:
            theta = Decimal(part)
        except InvalidOperation:
            raise ValueError(f"--thetas: {part!r} no es un número") from None
        scaled = theta.quantize(Decimal("1e-8"))
        if scaled != theta:
            raise ValueError(f"θ {part} no cabe en 8 decimales, que es la partición")
        thetas.append(format(scaled, "f"))
    return thetas


def project_roots() -> tuple[str, str, str, str]:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        raise ValueError("no sé de qué proyecto es: falta OPS_RESULTS_URI")
    project = match[1]
    return (
        f"gs://{project}-landing/l1",
        f"gs://{project}-dc-events/l2",
        f"gs://{project}-dc-descriptors/l3",
        f"gs://{project}-manifest/l2/thetas.yaml",
    )


def l3_file(root: str, key: tuple[str, str, str], theta: str, month: str) -> str:
    provider, market, asset = key
    year, number = month.split("-")
    return (
        f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/theta={theta}/year={year}/month={number}/{SUMMARIES}"
    )


def open_lane(lane: Oracle, path: str) -> str | None:
    """Abre `summaries.parquet` del θ; devuelve por qué no se puede comparar, o `None`."""
    try:
        parquet = open_parquet(path, "summaries.parquet")
    except FramesError:
        return f"falta el archivo {path}"
    missing = [name for name in NAMES if name not in parquet.schema_arrow.names]
    if missing:
        parquet.close()
        return f"faltan columnas en {path}: {', '.join(missing)}"
    lane.l3 = parquet
    lane.sizes = [
        (i, parquet.metadata.row_group(i).num_rows)
        for i in range(parquet.metadata.num_row_groups)
        if parquet.metadata.row_group(i).num_rows
    ]
    return None


def save_csv(month: str, lanes: list[Oracle]) -> None:
    """Deja la tabla por θ en `OPS_RESULTS_URI`; si no se puede, la salida estándar basta."""
    uri = os.environ.get("OPS_RESULTS_URI")
    if not uri:
        return
    try:
        from google.cloud import storage

        bucket, _, prefix = uri.removeprefix("gs://").partition("/")
        stamp = time.strftime("%Y%m%dT%H%M%S")
        name = f"{prefix}l3_check_month_{month}_{stamp}.csv"
        rows = ["theta,filas,diferencias,segundos"]
        rows += [f"{x.theta},{x.rows},{x.diffs},{x.seconds:.1f}" for x in lanes]
        storage.Client().bucket(bucket).blob(name).upload_from_string(
            "\n".join(rows) + "\n"
        )
        log(f"tabla por θ en gs://{bucket}/{name}")
    except Exception as exc:  # noqa: BLE001 - el check no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en results/: {type(exc).__name__}: {exc}")


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    month = re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month)
    if month is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    period = (int(month[1]), int(month[2]))
    key = (ns.provider, ns.market, ns.asset)
    try:
        if not (ns.l1_root and ns.l2_root and ns.l3_root and (ns.thetas or ns.catalog)):
            l1_default, l2_default, l3_default, catalog_default = project_roots()
        else:
            l1_default = l2_default = l3_default = catalog_default = ""
        l1_root = ns.l1_root or l1_default
        l2_root = ns.l2_root or l2_default
        l3_root = ns.l3_root or l3_default
        if ns.thetas == "all":
            thetas = listed_thetas(l3_root, key, ns.month)
            if not thetas:
                parser.error(f"--thetas all: L3 no tiene ningún θ en {ns.month}")
        elif ns.thetas:
            thetas = parse_thetas(ns.thetas)
        else:
            thetas = read_catalog(ns.catalog or catalog_default)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if len(set(thetas)) != len(thetas):
        parser.error("--thetas trae un θ repetido")

    log(
        f"check mes: {ns.month}, {len(thetas)} θ, l1={l1_root}, l2={l2_root}, l3={l3_root}"
    )
    started = time.perf_counter()
    lanes = [Oracle(theta, ns.max_diffs) for theta in thetas]
    by_theta = {Decimal(lane.theta): lane for lane in lanes}
    active = []
    for lane in lanes:
        path = l3_file(l3_root, key, lane.theta, ns.month)
        reason = open_lane(lane, path)
        if reason is None:
            active.append(lane)
        else:
            lane.diffs += 1
            log(f"DIFERENCIA θ={lane.theta} {reason}")

    error = None
    events = 0
    last_progress = time.perf_counter()
    try:
        if active:
            frames = read_frames(
                [Decimal(lane.theta) for lane in active],
                l1_root,
                l2_root,
                period,
                period,
                *key,
                chunks=True,
                columns=EVENT_COLUMNS,
            )
            for event in frames:
                lane = by_theta[event.theta]
                begun = time.perf_counter()
                add_event(lane, summarize(event, lane))
                # Suelta el evento antes de que el lector entregue el siguiente.
                del event
                lane.seconds += time.perf_counter() - begun
                events += 1
                if begun - last_progress >= PROGRESS_EVERY_S:
                    last_progress = begun
                    log(f"  … {begun - started:.0f} s: eventos={events}")
            for lane in active:
                begun = time.perf_counter()
                finish(lane)
                lane.seconds += time.perf_counter() - begun
    except FramesError as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for lane in active:
            if lane.l3 is not None:
                lane.l3.close()
    wall = time.perf_counter() - started

    log(f"{'θ':>12} {'filas':>10} {'diferencias':>12} {'segundos':>9}")
    for lane in lanes:
        log(f"{lane.theta:>12} {lane.rows:>10} {lane.diffs:>12} {lane.seconds:>9.1f}")
    save_csv(ns.month, lanes)
    rows = sum(lane.rows for lane in lanes)
    diffs = sum(lane.diffs for lane in lanes)
    if error is not None:
        log(f"FALLO lector: {error}")
    verdict = "fallo" if error else ("diferencias" if diffs else "ok")
    log(
        f"check mes: mes={ns.month} thetas={len(thetas)} filas={rows} "
        f"diferencias={diffs} pared_s={wall:.1f} veredicto={verdict}"
    )
    return 0 if verdict == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", required=True)
    parser.add_argument("--thetas")
    parser.add_argument("--catalog")
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--l3-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--max-diffs", type=int, default=100)
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
