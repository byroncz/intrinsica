"""Sonda del lector de tramas (`shared/dc_frames`): pared, memoria y bytes sobre un mes real.

Recorre `read_frames` para uno o varios θ y mide lo que decide el criterio de la Épica
E5 (TRD-L3 §10.3, ADR-L3-09): pared por fase, RSS pico, bytes leídos de
`consolidated.parquet` y `events.parquet`, decodificaciones de row group, ticks por
segundo y, por θ, eventos y ticks por fase. Al final compara con los umbrales y sale con
1 si alguno no se cumple o si el lector falla.

Args:
  --theta 0.00100000 | a,b,c | all   (obligatorio; `all` toma los θ que tienen L2 en
                                      todo el rango)
  --from YYYY-MM --to YYYY-MM        (meses de L2, inclusivos; `--to` por defecto = `--from`)
  --l1-root gs://<proyecto>-landing/l1    (por defecto, los del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2   OPS_RESULTS_URI; también rutas locales)
  --provider binance --market spot --asset BTCUSDT
  --max-wall-s N                     (umbral de pared; por defecto 300 con un θ, 1200 con más)
  --max-bytes-ratio 1.1              (con más de un θ, los bytes leídos de los archivos de L1 del
                                      rango no pasan de esto × su tamaño; los de meses anteriores,
                                      donde cae la referencia del primer evento, se reportan aparte)
  --modo chunks|evento|row-groups    (cómo consume el lector; por defecto `chunks`, el modo de
                                      las pasadas de L3: cada fase llega en trozos de un row
                                      group y la sonda cuenta los ticks de cada trozo.
                                      `row-groups` es `read_row_groups`, la entrega de TRD-L3
                                      §7.4 que usa el resumidor de L3)
  --readers N                        (solo `--modo row-groups`; por defecto 1: cuántos row groups
                                      se leen y decodifican a la vez, ITSC-362)
  --sin-pre-buffer                   (abre los archivos sin `pre_buffer`, como antes de ITSC-362:
                                      la línea base para comparar)
  --max-event-ticks 20000000         (solo en `--modo evento`: presupuesto de ticks por evento,
                                      ADR-L3-13; uno mayor llega sin tramas y se cuenta como
                                      `over_budget`)
  --big-event-ticks 100000          (desde cuántos ticks un evento entra a la suma de
                                      eventos abiertos a la vez)
  --sin-contar-io                    (no envuelve los archivos: la pared es la del lector
                                      tal cual, pero sin bytes leídos ni fase de lectura)

RSS (§10.3, umbral 2, con un solo θ): 1 GiB en modo chunks, que no retiene ningún evento; en
modo evento, 1 GiB más el presupuesto de ticks (`--max-event-ticks` × 40 B por tick), que es lo
que puede pesar el evento más grande que el lector entrega entero.

`--modo row-groups` no arma tramas por evento: cuenta por θ los eventos que cierran en cada row
group y los ticks de sus fases (`R..C` y `C..E`) a partir de las posiciones que entrega el
lector. No admite relecturas: cada row group se decodifica una vez. Con `--readers` > 1 el lector
puede decodificar de más, a lo sumo `readers` row groups tras el del último evento, que no se
entregan; la sonda los cuenta como decodificados.

Row groups: TRD-L3 §10.3 pide tantas decodificaciones como row groups del mes. El lector salta
por estadísticas los que ningún evento necesita (el tramo tras el último evento cerrado) y la
sonda reporta cuántos se saltaron. En modo evento exige que ninguno se decodifique más de una
vez. En modo chunks admite las relecturas de los eventos de más de dos row groups (el lector
retiene dos), las acota con los trozos de cada evento y resta de los bytes del rango solo las
de los archivos del rango: las de meses anteriores ya quedan fuera de ellos.
Los bytes incluyen los pies de Parquet, que el lector abre más de una vez por mes: en un
archivo real pesan nada, en uno de fixtures dominan (por eso `--max-bytes-ratio`).

Las fases se miden así (con `--readers` > 1 los hilos se solapan: la sonda solo reporta el tiempo
acumulado dentro de `read_row_group` y la pared total, que es lo que decide). Lectura: tiempo dentro de las lecturas de archivo (la sonda envuelve
cada archivo para contarlas; serializa las lecturas de un row group, así que la pared
puede ser algo más alta que la del lector sin envolver: `--sin-contar-io` da esa pared).
Decodificación: lo que tarda `read_row_group` sin su lectura. Selección: el resto
(`searchsorted`, slices, validaciones y el conteo de la propia sonda).

Última línea: `sonda lector: thetas=… meses=… eventos=… ticks_l1=… pared_s=… lectura_s=…
decodificación_s=… selección_s=… rss_pico_mib=… arrow_pool=… arrow_max_mib=… bytes_l1=…
ticks_s=… readers=… pre_buffer=si|no s_por_row_group=… s_read_row_group_l1=… modo=chunks|evento|row-groups over_budget=… veredicto=ok|excedido|fallo`.
`s_por_row_group` es la pared total (incluye L2, la selección y los row groups de meses anteriores)
dividida por los row groups de L1 del mes; `s_read_row_group_l1` es el tiempo medio de una llamada
a `read_row_group` de L1, que es lo que mide la línea base de ≈ 2,3 s (con `--readers` > 1 es la
latencia de una llamada, con los hilos compitiendo).
`arrow_pool` es el backend del pool de Arrow (`mimalloc`, `jemalloc` o `system`) y
`arrow_pico_mib` el máximo del pool de Arrow en todo el proceso (`max_memory()`) y `arrow_max_mib` el máximo de `pa.total_allocated_bytes()` visto con cada trozo en mano (modo
chunks) o con el evento entregado (modo evento) y al cerrar cada evento. Separan
lo que Arrow tiene asignado de lo que pesa el proceso (`rss_pico_mib`): si el RSS se dispara y
`arrow_max_mib` no, el pico no es de datos del lector (pool que no devuelve páginas, entorno).
Con `OPS_RESULTS_URI` deja además `l3_probe_reader_<fecha>.csv` con la tabla por θ.

Solo lee. Dónde corre: con `ops-script` (la service account lee `l1/` y `l2/`) o en local con
`uv run`.
"""

import argparse
import os
import re
import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from dc_frames import FramesError, FramesInputError, lake, read_frames, read_row_groups
from pyutils import resolve_fs

HISTORY_TICKS = 4_051_000_000  # TRD-L3 §10.3, criterio 6
GIB = 1 << 30
MIB = 1 << 20
# 3 columnas de 8 y 16 B más un bit: lo que pesa un tick en una trama.
BYTES_PER_TICK = 8 + 16 + 16 + 1 / 8
# Un row group de L1 decodificado: id, tiempo, precio y cantidad (8 + 8 + 16 + 16 B).
ROW_GROUP_BYTES_PER_TICK = 48
PROGRESS_EVERY_S = 30


@dataclass
class Stats:
    """Lo que miden los archivos envueltos; lo llenan `_CountedFile` y `_ProbeFile`."""

    count_io: bool = True
    pre_buffer: bool = True
    # Con `readers` > 1 varios hilos llaman a `read_row_group` a la vez.
    lock: threading.Lock = field(default_factory=threading.Lock)
    io_s: float = 0.0
    io_in_rg_s: float = 0.0
    rg_s: float = 0.0
    # La parte de `rg_s` que es de `consolidated.parquet`: la que la línea base de 2,3 s mide.
    rg_l1_s: float = 0.0
    bytes: dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0})
    # ruta de L1 -> bytes leídos de ese archivo
    l1_read: dict[str, int] = field(default_factory=dict)
    expected: dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0})
    decodes: dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0})
    l1_ticks: int = 0
    # Ticks del row group de L1 más grande decodificado.
    l1_group_ticks_max: int = 0
    # ruta de L1 -> (bytes del archivo, row groups, decodificaciones)
    l1_files: dict[str, list[int]] = field(default_factory=dict)
    l1_seen: set[tuple[str, int]] = field(default_factory=set)
    l1_repeated: int = 0
    # Bytes por metadatos de esas decodificaciones repetidas, por ruta de L1 (solo se restan
    # los de los archivos del rango), y cota de las que el modo por chunks admite: los
    # trozos de cada evento más allá de los dos que retiene el lector.
    l1_repeated_bytes: dict[str, int] = field(default_factory=dict)
    reread_allowed: int = 0
    started: float = 0.0
    last_progress: float = 0.0
    events: int = 0
    # Máximo de `pa.total_allocated_bytes()` visto con cada trozo o evento en mano y al cerrar cada evento.
    arrow_max: int = 0


STATS = Stats()


def rss_bytes() -> int:
    """El RSS pico del proceso. En Linux, `VmHWM`: `ru_maxrss` hereda el pico del padre al hacer exec."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def sample_arrow() -> None:
    STATS.arrow_max = max(STATS.arrow_max, pa.total_allocated_bytes())


def log(message: str) -> None:
    print(message, flush=True)


class _CountedFile:
    """Un archivo de solo lectura que cuenta los bytes y el tiempo de cada lectura.

    Lleva la posición él mismo (`_position`) y nunca le pasa al archivo interno un seek a un
    offset ≥ tamaño: el archivo de GCS reabre la lectura del objeto en cada seek y GCS
    responde "Request range not satisfiable" a cualquier offset igual o mayor que el tamaño.
    """

    def __init__(self, inner, kind: str, path: str, size: int) -> None:
        self._inner = inner
        self._kind = kind
        self._path = path
        self._size = size
        self._position = 0
        # Dónde está el archivo interno; solo se mueve cuando una lectura lo necesita.
        self._inner_position = 0

    def read(self, nbytes: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        start = time.perf_counter()
        if self._inner_position != self._position:
            self._inner.seek(self._position, 0)
            self._inner_position = self._position
        data = (
            self._inner.read()
            if nbytes is None or nbytes < 0
            else self._inner.read(nbytes)
        )
        self._position += len(data)
        self._inner_position += len(data)
        elapsed = time.perf_counter() - start
        # Con `--readers` > 1 varios hilos leen a la vez: un `+=` sin candado se pierde.
        with STATS.lock:
            STATS.io_s += elapsed
            STATS.bytes[self._kind] += len(data)
            if self._kind == "l1":
                STATS.l1_read[self._path] = STATS.l1_read.get(self._path, 0) + len(data)
        return data

    def seek(self, position: int, whence: int = 0) -> int:
        # `pa.PythonFile` pide el tamaño con `seek(0, 2)`. Se calcula la posición nueva sin
        # tocar el archivo interno: `read` lo posiciona justo antes de leer.
        if whence == 0:
            target = position
        elif whence == 1:
            target = self._position + position
        elif whence == 2:
            target = self._size + position
        else:
            raise ValueError(f"whence inválido: {whence}")
        if target < 0:
            raise OSError(f"seek a una posición negativa: {target}")
        self._position = target
        return target

    def tell(self) -> int:
        return self._position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def close(self) -> None:
        self._inner.close()


class _ProbeFile(pq.ParquetFile):
    """`ParquetFile` que mide cada `read_row_group`: tiempo, filas y bytes esperados."""

    def __init__(self, source, kind: str, path: str, **kwargs) -> None:
        super().__init__(source, **kwargs)
        self._kind = kind
        self._probe_path = path

    def close(self, force: bool = False) -> None:
        super().close(force=True)

    def read_row_group(self, i, columns=None, **kwargs):
        io_before = STATS.io_s
        start = time.perf_counter()
        table = super().read_row_group(i, columns=columns, **kwargs)
        elapsed = time.perf_counter() - start
        group = self.metadata.row_group(i)
        names = self.schema_arrow.names
        size = sum(
            group.column(names.index(name)).total_compressed_size
            for name in columns or names
        )
        with STATS.lock:
            STATS.rg_s += elapsed
            if self._kind == "l1":
                STATS.rg_l1_s += elapsed
            STATS.io_in_rg_s += STATS.io_s - io_before
            STATS.decodes[self._kind] += 1
            STATS.expected[self._kind] += size
            if self._kind == "l1":
                STATS.l1_ticks += table.num_rows
                STATS.l1_group_ticks_max = max(STATS.l1_group_ticks_max, table.num_rows)
                STATS.l1_files[self._probe_path][2] += 1
                if (self._probe_path, i) in STATS.l1_seen:
                    STATS.l1_repeated += 1
                    STATS.l1_repeated_bytes[self._probe_path] = (
                        STATS.l1_repeated_bytes.get(self._probe_path, 0) + size
                    )
                STATS.l1_seen.add((self._probe_path, i))
        progress()
        return table


def probe_open_parquet(
    path: str,
    what: str,
    metadata: pq.FileMetaData | None = None,
    pre_buffer: bool = False,
) -> pq.ParquetFile:
    """Sustituto de `dc_frames.lake.open_parquet` que mide; hace lo mismo que el original.

    Con `metadata` es la apertura de un hilo del lector (ver `lake.open_parquet`): no
    comprueba que el archivo exista, pero sí necesita su tamaño para envolverlo.
    """
    fs, resolved = resolve_fs(path)
    info = fs.get_file_info(resolved) if metadata is None or STATS.count_io else None
    if metadata is None and info.type != pafs.FileType.File:
        raise FramesInputError(f"falta {what}: {path}")
    kind = "l1" if path.endswith(lake.CONSOLIDATED) else "l2"
    options = {"pre_buffer": pre_buffer and STATS.pre_buffer, "metadata": metadata}
    if STATS.count_io:
        source = pa.PythonFile(
            _CountedFile(fs.open_input_file(resolved), kind, path, info.size),
            mode="r",
        )
        parquet = _ProbeFile(source, kind, path, **options)
    else:
        parquet = _ProbeFile(resolved, kind, path, filesystem=fs, **options)
    with STATS.lock:
        if kind == "l1" and metadata is None and path not in STATS.l1_files:
            STATS.l1_files[path] = [info.size, parquet.num_row_groups, 0]
    return parquet


def progress() -> None:
    now = time.perf_counter()
    if now - STATS.last_progress < PROGRESS_EVERY_S:
        return
    STATS.last_progress = now
    log(
        f"  … {now - STATS.started:.0f} s: row groups de L1={STATS.decodes['l1']} "
        f"ticks_l1={STATS.l1_ticks} eventos={STATS.events} rss={rss_bytes() / MIB:.0f} MiB"
    )


@dataclass
class ThetaStats:
    events: int = 0
    confirmation: int = 0
    overshoot: int = 0
    biggest_ticks: int = 0
    biggest_bytes: int = 0
    # Eventos sobre el presupuesto (modo evento) y el mayor de ellos, por ids de L2 (cota).
    over_budget: int = 0
    over_budget_max: int = 0


def frame_bytes(*frames) -> int:
    return sum(
        array.nbytes
        for frame in frames
        for array in (
            frame.transact_time,
            frame.price,
            frame.quantity,
            frame.is_buyer_maker,
        )
    )


def tally(
    event, per_theta: dict[Decimal, ThetaStats], bigs: list[tuple[int, int, int]], ns
) -> None:
    """Suma los ticks de un evento a su θ. Una función aparte para que, al volver, no quede
    ninguna referencia a las tramas (ni a sus trozos) antes de pedir el siguiente evento.
    """
    row = event.event
    reference = row.column("reference_agg_trade_id")[0].as_py()
    extreme = row.column("extreme_agg_trade_id")[0].as_py()
    stats = per_theta.get(event.theta)
    if stats is None:
        stats = per_theta[event.theta] = ThetaStats()
    stats.events += 1
    STATS.events += 1
    if event.over_budget:
        stats.over_budget += 1
        stats.over_budget_max = max(stats.over_budget_max, extreme - reference)
        return
    confirmation = overshoot = size = 0
    if ns.modo == "chunks":
        # Cada fase es un iterador de trozos; se consume antes del siguiente evento.
        chunks = 0
        for frame in event.confirmation:
            chunks += 1
            confirmation += len(frame)
            size += frame_bytes(frame)
            sample_arrow()
        for frame in event.overshoot:
            chunks += 1
            overshoot += len(frame)
            size += frame_bytes(frame)
            sample_arrow()
        # Un evento de k row groups da a lo sumo k + 1 trozos (el del borde entre fases
        # sale en ambas) y el lector relee k - 2 de ellos. La cota `chunks - 2` es holgada:
        # llega a k - 1 cuando la confirmación cae en un row group releído.
        STATS.reread_allowed += max(0, chunks - 2)
    else:
        confirmation = len(event.confirmation.transact_time)
        overshoot = len(event.overshoot.transact_time)
        size = frame_bytes(event.confirmation, event.overshoot)
        sample_arrow()
    ticks = confirmation + overshoot
    stats.confirmation += confirmation
    stats.overshoot += overshoot
    if ticks > stats.biggest_ticks:
        stats.biggest_ticks = ticks
        stats.biggest_bytes = size
    if ticks >= ns.big_event_ticks:
        bigs.append((reference, extreme, ticks))


def tally_row_group(delivery, per_theta: dict[Decimal, ThetaStats]) -> None:
    """Suma a cada θ los eventos que cierran en el row group y los ticks de sus fases.

    Las fases salen de las posiciones: confirmación `[R, C)` y overshoot `[C, E)` dentro del
    row group. Una función aparte para que, al volver, la sonda no retenga el elemento.
    """
    for te in delivery.thetas:
        stats = per_theta.get(te.theta)
        if stats is None:
            stats = per_theta[te.theta] = ThetaStats()
        closing = int(np.count_nonzero(~te.ends_after))
        stats.events += closing
        STATS.events += closing
        stats.confirmation += int((te.positions[:, 1] - te.positions[:, 0]).sum())
        stats.overshoot += int((te.positions[:, 2] - te.positions[:, 1]).sum())


def default_roots() -> tuple[str, str]:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit(
            "sin --l1-root o --l2-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    project = match[1]
    return f"gs://{project}-landing/l1", f"gs://{project}-dc-events/l2"


def parse_month(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if match is None or not 1 <= int(match[2]) <= 12:
        sys.exit(f"mes inválido {text!r}: se espera YYYY-MM")
    return int(match[1]), int(match[2])


EVENTS_FILE = re.compile(
    r"/theta=(0\.\d{8})/year=(\d{4})/month=(\d{2})/events\.parquet$"
)


def lake_thetas(l2_root: str, key: tuple[str, str, str], first, last) -> list[str]:
    """Los θ de L2 que tienen `events.parquet` en todos los meses de `first`..`last`.

    Se listan los archivos de forma recursiva y no los directorios `theta=`: en GCS un
    prefijo no es una entrada, así que un listado no recursivo no los devuelve.
    """
    fs, base = resolve_fs(l2_root)
    provider, market, asset = key
    series = f"{base.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
    selector = pafs.FileSelector(series, recursive=True, allow_not_found=True)
    present: dict[str, set[str]] = {}
    for info in fs.get_file_info(selector):
        match = EVENTS_FILE.search(info.path)
        if match is not None and info.type == pafs.FileType.File:
            present.setdefault(match[1], set()).add(f"{match[2]}-{match[3]}")
    needed = {f"{y:04d}-{m:02d}" for y, m in lake.months(first, last)}
    complete = sorted(theta for theta, found in present.items() if needed <= found)
    if len(complete) != len(present):
        skipped = sorted(set(present) - set(complete))
        log(f"θ sin L2 en todo el rango, omitidos: {', '.join(skipped)}")
    return complete


def parse_thetas(text: str, l2_root, key, first, last) -> list[str]:
    if text == "all":
        thetas = lake_thetas(l2_root, key, first, last)
        if not thetas:
            sys.exit("--theta all: ningún θ tiene L2 en todo el rango")
        return thetas
    thetas = []
    for part in text.split(","):
        theta = Decimal(part)
        scaled = theta.quantize(Decimal("1e-8"))
        if scaled != theta:
            sys.exit(f"θ {part} no cabe en 8 decimales, que es la partición de L2")
        thetas.append(format(scaled, "f"))
    return thetas


def open_sweep(bigs: list[tuple[int, int, int]]) -> int:
    """La mayor suma de ticks de eventos grandes abiertos a la vez (cota; ver el informe)."""
    if not bigs:
        return 0
    starts, ends, ticks = (
        np.array(column, dtype=np.int64) for column in zip(*bigs, strict=True)
    )
    position = np.concatenate([starts, ends])
    delta = np.concatenate([ticks, -ticks])
    # A igual posición, primero los cierres: un evento que termina donde otro empieza no se suma.
    order = np.lexsort((delta, position))
    return int(np.cumsum(delta[order]).max())


def run(ns: argparse.Namespace) -> int:
    default_l1, default_l2 = (
        (None, None) if ns.l1_root and ns.l2_root else default_roots()
    )
    l1_root = ns.l1_root or default_l1
    l2_root = ns.l2_root or default_l2
    key = (ns.provider, ns.market, ns.asset)
    first = parse_month(ns.start)
    last = parse_month(ns.end or ns.start)
    thetas = parse_thetas(ns.theta, l2_root, key, first, last)
    multi = len(thetas) > 1
    max_wall = ns.max_wall_s if ns.max_wall_s is not None else (1200 if multi else 300)
    months = [f"{y:04d}-{m:02d}" for y, m in lake.months(first, last)]
    log(
        f"sonda lector: {len(thetas)} θ, meses {months[0]}..{months[-1]}, l1={l1_root}, l2={l2_root}"
    )

    STATS.count_io = not ns.sin_contar_io
    STATS.pre_buffer = not ns.sin_pre_buffer
    original_open_parquet = lake.open_parquet
    lake.open_parquet = probe_open_parquet
    per_theta: dict[Decimal, ThetaStats] = {}
    bigs: list[tuple[int, int, int]] = []
    error = None
    STATS.started = STATS.last_progress = time.perf_counter()
    try:
        if ns.modo == "row-groups":
            for delivery in read_row_groups(
                thetas, l1_root, l2_root, first, last, *key, readers=ns.readers
            ):
                tally_row_group(delivery, per_theta)
                sample_arrow()
                # Suelta el row group antes de que el lector entregue el siguiente.
                del delivery
        else:
            options = {"chunks": True}
            if ns.modo == "evento":
                options = {"max_event_ticks": ns.max_event_ticks}
            for event in read_frames(
                thetas, l1_root, l2_root, first, last, *key, **options
            ):
                tally(event, per_theta, bigs, ns)
                sample_arrow()
                # Suelta el evento antes de que el lector entregue el siguiente.
                del event
    except FramesError as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        lake.open_parquet = original_open_parquet
    wall = time.perf_counter() - STATS.started

    decode = STATS.rg_s - STATS.io_in_rg_s
    reading = STATS.io_s if STATS.count_io else None
    selection = wall - STATS.rg_s - (STATS.io_s - STATS.io_in_rg_s)
    if ns.readers > 1:
        # Los hilos se solapan: `rg_s` suma el tiempo de cada llamada y puede superar la
        # pared, así que no se resta de ella.
        selection = 0.0
        phases = f"tiempo acumulado en read_row_group de los {ns.readers} hilos={STATS.rg_s:.1f} s"
    elif STATS.count_io:
        phases = f"lectura={STATS.io_s:.1f} s, decodificación={decode:.1f} s, selección={selection:.1f} s"
    else:
        phases = f"lectura+decodificación={STATS.rg_s:.1f} s, selección={wall - STATS.rg_s:.1f} s"

    header = f"{'θ':>12} {'eventos':>10} {'ticks_confirmación':>19} {'ticks_overshoot':>16} {'evento_máx_ticks':>17} {'evento_máx_MiB':>15} {'over_budget':>12}"
    lines = [
        f"{theta:>12} {s.events:>10} {s.confirmation:>19} {s.overshoot:>16} "
        f"{s.biggest_ticks:>17} {s.biggest_bytes / MIB:>15.1f} {s.over_budget:>12}"
        for theta, s in sorted(per_theta.items())
    ]
    log(header)
    for line in lines:
        log(line)
    save_csv(header, lines)

    target = {lake.l1_path(l1_root, key, m) for m in lake.months(first, last)}
    month_files = [v for path, v in STATS.l1_files.items() if path in target]
    month_bytes = sum(v[0] for v in month_files)
    month_groups = sum(v[1] for v in month_files)
    month_decodes = sum(v[2] for v in month_files)
    earlier_decodes = sum(
        v[2] for path, v in STATS.l1_files.items() if path not in target
    )
    month_read = sum(n for path, n in STATS.l1_read.items() if path in target)
    earlier_read = STATS.bytes["l1"] - month_read
    delivered = sum(s.confirmation + s.overshoot for s in per_theta.values())
    events = sum(s.events for s in per_theta.values())
    rate = STATS.l1_ticks / wall if wall else 0.0
    per_row_group = wall / (month_decodes or STATS.decodes["l1"] or 1)
    # Lo que mide la línea base de 2,3 s: el `read_row_group` de L1, sin L2 ni selección.
    per_l1_call = STATS.rg_l1_s / (STATS.decodes["l1"] or 1)
    biggest = max((s.biggest_bytes for s in per_theta.values()), default=0)
    big_open = open_sweep(bigs)
    small_open = len(thetas) * (ns.big_event_ticks - 1)
    rss = rss_bytes()
    over_budget = sum(s.over_budget for s in per_theta.values())
    over_budget_max = max((s.over_budget_max for s in per_theta.values()), default=0)
    if ns.modo == "evento":
        limit = GIB + int(ns.max_event_ticks * BYTES_PER_TICK)
        limit_text = f"1 GiB + presupuesto de {ns.max_event_ticks} ticks"
    else:
        limit = GIB
        limit_text = "1 GiB (modo chunks)"

    log(f"pared por fase: {phases}; total {wall:.1f} s")
    log(
        f"ticks de L1 decodificados={STATS.l1_ticks} ({rate / 1e6:.2f} M/s); "
        f"ticks entregados={delivered} ({delivered / wall / 1e6 if wall else 0:.2f} M/s); eventos={events}"
    )
    log(
        f"row groups de L1: decodificados del mes={month_decodes} de {month_groups} "
        f"(saltados por estadísticas={month_groups - month_decodes}), "
        f"de meses anteriores={earlier_decodes}, repetidos={STATS.l1_repeated}; "
        f"de L2={STATS.decodes['l2']}"
    )
    if STATS.count_io:
        log(
            f"bytes leídos: L1={STATS.bytes['l1']} (del rango {month_read} = "
            f"{month_read / max(month_bytes, 1):.3f} × los {month_bytes} de sus archivos; "
            f"de meses anteriores {earlier_read}), L2={STATS.bytes['l2']}; "
            f"por metadatos L1={STATS.expected['l1']}"
        )
    else:
        log(
            f"bytes por metadatos (no medidos): L1={STATS.expected['l1']}, L2={STATS.expected['l2']}"
        )
    log(
        f"RSS pico={rss / MIB:.0f} MiB; evento más grande={biggest / MIB:.1f} MiB; "
        f"suma máxima de eventos abiertos ≥{ns.big_event_ticks} ticks={big_open} "
        f"({big_open * BYTES_PER_TICK / MIB:.0f} MiB), los menores aportan a lo sumo {small_open}"
    )
    if ns.modo == "evento":
        log(
            f"eventos sobre el presupuesto de {ns.max_event_ticks} ticks (over_budget)={over_budget}; "
            f"el mayor tiene a lo sumo {over_budget_max} ticks (por ids de L2)"
        )
    arrow_peak = pa.default_memory_pool().max_memory()
    if ns.modo == "row-groups":
        # TRD-L3 §10.3: (K + 1,5) row groups decodificados, con K = 1 si `readers` = 1 y
        # `readers` + 1 si no. Es lo medido con el fixture, no una cota: con las `readers`
        # decodificaciones solapadas el peor caso es 1 + `readers` × 2,1 row groups. Es solo informativo: el pool de Arrow cuenta también los row
        # groups de eventos de L2 de cada θ, que la prueba de memoria de dc_frames no tiene.
        held = 1 if ns.readers == 1 else ns.readers + 1
        reader_limit = (
            (held + 1.5) * STATS.l1_group_ticks_max * ROW_GROUP_BYTES_PER_TICK
        )
        log(
            f"pico del pool de Arrow={arrow_peak / MIB:.0f} MiB; umbral del lector "
            f"({held} + 1,5) × row group de {STATS.l1_group_ticks_max * ROW_GROUP_BYTES_PER_TICK / MIB:.0f} MiB "
            f"= {reader_limit / MIB:.0f} MiB: {'bajo' if arrow_peak <= reader_limit else 'SOBRE'} el umbral"
        )
    if rate:
        log(
            f"extrapolación al histórico ({HISTORY_TICKS // 1_000_000} M de ticks) a esa tasa: "
            f"{HISTORY_TICKS / rate / 3600:.1f} h"
        )

    checks: list[tuple[str, bool]] = []
    if error is None:
        checks.append((f"pared {wall:.0f} s ≤ {max_wall:.0f} s", wall <= max_wall))
        if not multi:
            checks.append(
                (
                    f"RSS pico {rss / MIB:.0f} MiB ≤ {limit_text} ({limit / MIB:.0f} MiB)",
                    rss <= limit,
                )
            )
        if multi or ns.modo == "row-groups":
            # El modo por chunks relee los row groups que sobran de un evento de más de dos
            # (TRD-L3 §10.3, umbral 3): esas relecturas se miden aparte y no cuentan en los bytes.
            allowed = STATS.reread_allowed if ns.modo == "chunks" else 0
            checks.append(
                (
                    (
                        f"row groups de L1 decodificados más de una vez: {STATS.l1_repeated} "
                        f"(relecturas admitidas de eventos de más de dos row groups: {allowed})"
                    ),
                    STATS.l1_repeated <= allowed,
                )
            )
            if STATS.count_io:
                counted = month_read - (
                    sum(
                        n
                        for path, n in STATS.l1_repeated_bytes.items()
                        if path in target
                    )
                    if ns.modo == "chunks"
                    else 0
                )
                checks.append(
                    (
                        f"bytes de L1 del rango {counted} ≤ {ns.max_bytes_ratio} × {month_bytes}",
                        counted <= ns.max_bytes_ratio * month_bytes,
                    )
                )
    failed = error is not None
    if error is not None:
        log(f"FALLO lector: {error}")
    for text, ok in checks:
        log(f"{'ok' if ok else 'FALLO'} {text}")
        failed |= not ok
    verdict = "fallo" if error is not None else ("excedido" if failed else "ok")
    log(
        f"sonda lector: thetas={len(thetas)} meses={months[0]}..{months[-1]} eventos={events} "
        f"ticks_l1={STATS.l1_ticks} pared_s={wall:.1f} lectura_s={'n/d' if reading is None or ns.readers > 1 else f'{reading:.1f}'} "
        f"decodificación_s={'n/d' if reading is None or ns.readers > 1 else f'{decode:.1f}'} "
        f"selección_s={'n/d' if ns.readers > 1 else f'{selection:.1f}'} "
        f"rss_pico_mib={rss / MIB:.0f} arrow_pool={pa.default_memory_pool().backend_name} "
        f"arrow_max_mib={STATS.arrow_max / MIB:.1f} arrow_pico_mib={arrow_peak / MIB:.1f} bytes_l1={STATS.bytes['l1'] if STATS.count_io else 'n/d'} "
        f"ticks_s={rate:.0f} readers={ns.readers} pre_buffer={'si' if STATS.pre_buffer else 'no'} "
        f"s_por_row_group={per_row_group:.2f} s_read_row_group_l1={per_l1_call:.2f} modo={ns.modo} over_budget={over_budget} veredicto={verdict}"
    )
    return 1 if failed else 0


def save_csv(header: str, lines: list[str]) -> None:
    """Deja la tabla por θ en `OPS_RESULTS_URI`; si no se puede, la salida estándar basta."""
    uri = os.environ.get("OPS_RESULTS_URI")
    if not uri:
        return
    try:
        from google.cloud import storage

        bucket, _, prefix = uri.removeprefix("gs://").partition("/")
        name = f"{prefix}l3_probe_reader_{time.strftime('%Y%m%dT%H%M%S')}.csv"
        body = "\n".join(",".join(row.split()) for row in [header, *lines]) + "\n"
        storage.Client().bucket(bucket).blob(name).upload_from_string(body)
        log(f"tabla por θ en gs://{bucket}/{name}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en results/: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end")
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--max-bytes-ratio", type=float, default=1.1)
    parser.add_argument(
        "--modo", choices=("chunks", "evento", "row-groups"), default="chunks"
    )
    parser.add_argument("--readers", type=int, default=1)
    parser.add_argument("--sin-pre-buffer", action="store_true")
    parser.add_argument("--max-event-ticks", type=int, default=20_000_000)
    parser.add_argument("--big-event-ticks", type=int, default=100_000)
    parser.add_argument("--sin-contar-io", action="store_true")
    ns = parser.parse_args(argv)
    if ns.readers < 1:
        parser.error("--readers debe ser al menos 1")
    if ns.readers > 1 and ns.modo != "row-groups":
        parser.error("--readers solo aplica con --modo row-groups")
    return run(ns)


if __name__ == "__main__":
    sys.exit(main())
