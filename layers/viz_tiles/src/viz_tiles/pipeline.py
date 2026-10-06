"""Un mes de días: decide qué días rehacer, lee L1 una vez y escribe cada día.

TRD-viz §7.8 y §8.1. El orden respeta la eficiencia de memoria: el `input_hash`
se calcula con los metadatos de los archivos (nunca se lee un tick para decidir);
los días al día se saltan; los demás salen de **una** pasada por el
`consolidated.parquet` del mes, row group a row group, y cada día cierra cuando
los ticks cambian de día. Cada `events.parquet` de θ se lee **una vez por mes**
(no una por día). En RAM: un row group de L1, un tramo de `ticks.bin` (≈ 330 KB;
cada tramo se escribe al objeto en cuanto se llena) y, por θ, los eventos del mes en
arreglos de NumPy.
"""

import calendar
import logging
import resource
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
from dq import Finding, Severity

from viz_tiles import findings
from viz_tiles.chain import ThetaMonth, theta_month
from viz_tiles.context import RunContext
from viz_tiles.contract import (
    DAY_US,
    EVENTS_FILE,
    MAX_THETAS,
    PAGE_FILE,
    TICKS_FILE,
    TILES_VERSION,
    price_scale,
)
from viz_tiles.events import EventsBuffer, event_rows
from viz_tiles.lake import (
    CARRY_OVER,
    EVENTS,
    TICK_COLUMNS,
    EventsIndex,
    InputError,
    Month,
    MonthEvents,
    column_bounds,
    events_rel,
    exists,
    input_hash,
    join,
    landing_rel,
    object_stat,
    open_parquet,
    read_month_events,
    validate_ticks,
)
from viz_tiles.ticks import (
    DayTicks,
    PriceUnrepresentable,
    TicksAccumulator,
    day_start_us,
)
from viz_tiles.write import (
    ThetaEvents,
    TicksFile,
    advance_latest_from_files,
    day_sizes,
    find_index,
    write_day,
)

logger = logging.getLogger(__name__)

# Lecturas simultáneas de metadatos de objeto (tamaño y CRC32C).
STAT_WORKERS = 8

_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def days_of_month(month: Month) -> list[date]:
    year, number = month
    return [
        date(year, number, d)
        for d in range(1, calendar.monthrange(year, number)[1] + 1)
    ]


def _day_of(us: int) -> date:
    return date.fromordinal(_EPOCH_ORDINAL + us // DAY_US)


def _span_days(month: Month, bounds: list[tuple[int, int] | None]) -> list[date]:
    """Los días del mes que el `consolidated.parquet` cubre, de su primer a su último tick.

    Un mes completo no pide los días anteriores al inicio de la serie (agosto de
    2017 arranca el 17) ni los posteriores al último tick. Un hueco *dentro* del
    rango sí es un día pedido: sin ticks, deja `input_missing`. Sin estadísticas
    en algún row group no se puede acotar y se piden todos.
    """
    days = days_of_month(month)
    if not bounds or any(b is None for b in bounds):
        return days
    first, last = _day_of(min(b[0] for b in bounds)), _day_of(max(b[1] for b in bounds))
    return [d for d in days if first <= d <= last]


class _Stats:
    """`(tamaño, CRC32C)` de objetos por ruta relativa, con caché y en paralelo."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[int, str]] = {}

    def fetch(self, paths: dict[str, str]) -> dict[str, tuple[int, str]]:
        """Las huellas de `paths` (relativa → absoluta); pide solo las que faltan."""
        missing = {rel: path for rel, path in paths.items() if rel not in self._cache}
        with ThreadPoolExecutor(max_workers=STAT_WORKERS) as pool:
            for rel, stat in zip(
                missing, pool.map(object_stat, missing.values()), strict=True
            ):
                self._cache[rel] = stat
        return {rel: self._cache[rel] for rel in paths}


def _probe(day: date, ticks: int, started: float, **extra: object) -> None:
    """Línea de cierre de un día: ticks, pared y RSS pico (KiB en Linux → MiB).

    `wall_s` es el tiempo del día desde que su primer tick entra al acumulador
    hasta que se escribe (el criterio de aceptación: un mes en menos de 10 min).
    """
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    wall_s = max(0.1, round(time.monotonic() - started, 1))
    tail = "".join(f" {name}={value}" for name, value in extra.items())
    logger.info(
        "sonda: unit=%s ticks=%d wall_s=%.1f rss_mib=%d%s",
        day.isoformat(),
        ticks,
        wall_s,
        rss_mib,
        tail,
    )


class _Month:
    """El estado de un mes en curso: los θ listos, sus eventos y los hallazgos."""

    def __init__(
        self,
        ctx: RunContext,
        month: Month,
        ready: list[ThetaMonth],
        missing: list[str],
        out: list[Finding],
    ) -> None:
        self.ctx = ctx
        self.month = month
        self.ready = ready
        self.missing = missing
        self.out = out
        self.scale = price_scale(ctx.asset)
        # Los eventos del mes de cada θ, leídos la primera vez que un día los pide.
        self._events: dict[str, MonthEvents] = {}

    def _key(self, day: date) -> dict:
        c = self.ctx
        return {
            "root": c.tiles_root,
            "provider": c.provider,
            "market": c.market,
            "asset": c.asset,
            "day": day,
        }

    def summary(self, day: date, index: dict, skipped: bool, started: float) -> None:
        """`tiles_summary` y sonda de un día, escrito ahora o ya al día."""
        sizes = day_sizes(
            self.ctx.tiles_root, self.ctx.provider, self.ctx.market, self.ctx.asset, day
        )
        thetas = index["thetas"]
        self.out.append(
            findings.tiles_summary(
                self.ctx,
                day,
                ticks=index["ticks"],
                skipped=skipped,
                tiles_version=index["tiles_version"],
                input_hash=index["input_hash"],
                content_hash=index["content_hash"],
                objects=len(sizes),
                size=sum(sizes.values()),
                ticks_bytes=sizes.get(TICKS_FILE, 0),
                events_bytes=sizes.get(EVENTS_FILE, 0),
                page_bytes=sizes.get(PAGE_FILE, 0),
                thetas=[t["theta"] for t in thetas],
                provisional_thetas=[
                    t["theta"] for t in thetas if t["provisional_from_s"] is not None
                ],
                missing_thetas=index["missing_thetas"],
            )
        )
        _probe(
            day,
            index["ticks"],
            started,
            ticks_bytes=sizes.get(TICKS_FILE, 0),
            page_bytes=sizes.get(PAGE_FILE, 0),
            **({"skipped": "true"} if skipped else {}),
        )

    def build(self, day: date, digest: str, ticks: DayTicks, started: float) -> None:
        """Eventos por θ de un día (TRD-viz §8.1, paso 4): escritura y resumen."""
        ctx = self.ctx
        if ticks.ticks == 0:
            self.out.append(
                findings.input_missing(
                    ctx, day, "ticks", path=join(ctx.landing_root, self._landing())
                )
            )
            return
        if len(self.ready) > MAX_THETAS:
            raise ValueError(f"más de {MAX_THETAS} θ no caben en un uint8")
        if ticks.rounded:
            self.out.append(
                findings.price_rounded(
                    ctx,
                    day,
                    count=ticks.rounded,
                    max_abs_delta_int=ticks.max_abs_delta_int,
                )
            )
        events = EventsBuffer()
        thetas = [self._theta_events(state, day, ticks, events) for state in self.ready]
        write_day(
            ctx.tiles_root,
            provider=ctx.provider,
            market=ctx.market,
            asset=ctx.asset,
            day=day,
            ticks=ticks,
            events=events,
            thetas=thetas,
            missing_thetas=self.missing,
            input_hash=digest,
            image_version=ctx.image_version,
        )
        index = find_index(**self._key(day))
        self.summary(day, index, False, started)

    def _landing(self) -> str:
        c = self.ctx
        return landing_rel(c.provider, c.market, c.asset, self.month)

    def _month_events(self, theta: str) -> MonthEvents:
        """Los eventos del θ en el mes: se leen de `events.parquet` una sola vez."""
        if theta not in self._events:
            c = self.ctx
            path = join(
                c.events_root,
                events_rel(c.provider, c.market, c.asset, theta, self.month, EVENTS),
            )
            self._events[theta] = read_month_events(path)
        return self._events[theta]

    def _theta_events(
        self,
        state: ThetaMonth,
        day: date,
        ticks: DayTicks,
        buffer: EventsBuffer,
    ) -> ThetaEvents:
        # Los eventos que tocan el día son los de ids entre su primer y su último tick.
        first, last = ticks.first_agg_trade_id, ticks.last_agg_trade_id
        events = self._month_events(state.theta).touching(first, last)
        tail = state.tail
        # La cola entra solo si el día tiene ticks posteriores a la referencia del pendiente.
        pending = (
            tail if tail is not None and tail.reference_agg_trade_id < last else None
        )
        provisional_from_s = state.provisional_from_s(day)
        rows = event_rows(
            events, pending, provisional_from_s is not None, day_start_us(day)
        )
        del events  # las filas del θ ya son los bytes de `events.bin`
        buffer.add(rows)
        count = len(rows)
        del rows  # el buffer del día es la única copia de los eventos
        return ThetaEvents(
            theta=state.theta,
            events=count,
            provisional_from_s=provisional_from_s,
        )


class _DayStream:
    """Reparte los lotes de ticks entre los días por construir, en orden.

    Hay un solo acumulador vivo. Un día cierra cuando llega un lote cuyo último
    tick ya es del día siguiente o posterior; el mismo lote se ofrece entonces al
    día que sigue (el acumulador ignora los ticks que no son suyos).
    """

    def __init__(
        self,
        pending: list[tuple[date, str]],
        scale: int,
        close: Callable[[date, str, DayTicks | PriceUnrepresentable, float], None],
        open_file: Callable[[date], TicksFile],
    ) -> None:
        self._pending = pending
        self._scale = scale
        self._close = close
        self._open_file = open_file
        self._pos = -1
        self.acc: TicksAccumulator | None = None
        self._file: TicksFile | None = None
        self._started = 0.0
        self._end = 0
        self._advance()

    def _advance(self) -> None:
        self._pos += 1
        if self._pos >= len(self._pending):
            self.acc = None
            return
        day = self._pending[self._pos][0]
        self._file = self._open_file(day)
        self.acc = TicksAccumulator(day, self._scale, self._file)
        self._started = time.monotonic()
        self._end = day_start_us(day) + DAY_US

    def _finish(self) -> None:
        day, digest = self._pending[self._pos]
        acc, self.acc = self.acc, None
        file, self._file = self._file, None
        try:
            ticks = acc.finish()
        except PriceUnrepresentable as exc:
            file.discard()  # un día que no se escribe no deja su `ticks.bin`
            self._close(day, digest, exc, self._started)
        except BaseException:
            file.close()
            raise
        else:
            file.close()
            self._close(day, digest, ticks, self._started)
        self._advance()

    def abort(self) -> None:
        """Corta el día en curso (la corrida falló): cierra su `ticks.bin` sin indexarlo."""
        self.acc = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def feed(self, batch: pa.RecordBatch) -> None:
        top = batch.column("transact_time")[-1].as_py()
        while self.acc is not None:
            self.acc.update(batch)
            if top < self._end:
                return
            self._finish()

    @property
    def done(self) -> bool:
        return self.acc is None

    def drain(self) -> None:
        """Cierra los días que quedan: los ticks se acabaron."""
        while self.acc is not None:
            self._finish()


def _row_groups(
    bounds: list[tuple[int, int] | None], spans: list[tuple[int, int]]
) -> Iterable[int]:
    """Row groups cuyo rango de `transact_time` toca algún día por construir."""
    for i, b in enumerate(bounds):
        if b is None or any(b[0] < end and b[1] >= start for start, end in spans):
            yield i


def _read_batches(parquet: pq.ParquetFile, groups: Iterable[int]):
    """Los ticks de esos row groups, uno a la vez, soltando cada lote al consumirlo."""
    for i in groups:
        table = parquet.read_row_group(i, columns=list(TICK_COLUMNS), use_threads=False)
        batches = table.to_batches()
        del table
        while batches:
            batch = batches.pop(0)
            if batch.num_rows:
                yield batch
            del batch


def process_month(
    ctx: RunContext, month: Month, days: list[date] | None, lake: EventsIndex
) -> bool:
    """Procesa `days` del mes (todos los de L1 si es `None`). False si dejó un error.

    Los hallazgos del mes se emiten juntos, en una llamada (TRD-viz §9.4).
    """
    out: list[Finding] = []
    try:
        _run(ctx, month, days, lake, out)
    finally:
        findings.emit(ctx, out)
    return not any(f.severity is Severity.ERROR for f in out)


def _run(
    ctx: RunContext,
    month: Month,
    days: list[date] | None,
    lake: EventsIndex,
    out: list[Finding],
) -> None:
    label = f"{month[0]:04d}-{month[1]:02d}"
    requested = days or days_of_month(month)
    landing_file = landing_rel(ctx.provider, ctx.market, ctx.asset, month)
    landing = join(ctx.landing_root, landing_file)
    if not exists(landing):
        logger.error("unidad %s: falta %s", label, landing)
        out.append(
            findings.input_missing(
                ctx, requested[0], "l1", path=landing, days=len(requested)
            )
        )
        return
    thetas = lake.thetas(month)
    if not any(lake.has(month, t, EVENTS) for t in thetas):
        prefix = join(
            ctx.events_root,
            f"provider={ctx.provider}/market={ctx.market}/asset={ctx.asset}",
        )
        logger.error("unidad %s: L2 no tiene events.parquet en %s", label, prefix)
        out.append(
            findings.input_missing(
                ctx,
                requested[0],
                "events",
                path=prefix,
                month=label,
                days=len(requested),
            )
        )
        return

    try:
        parquet = open_parquet(landing)
        validate_ticks(parquet, landing)
    except InputError as exc:
        out.append(
            findings.input_missing(
                ctx, requested[0], exc.what, days=len(requested), **exc.details
            )
        )
        return
    with parquet:
        bounds = column_bounds(parquet, "transact_time")
        days = days or _span_days(month, bounds)
        _process(
            ctx, month, days, lake, thetas, landing_file, landing, parquet, bounds, out
        )


def _theta_states(
    ctx: RunContext,
    month: Month,
    days: list[date],
    lake: EventsIndex,
    thetas: list[str],
    out: list[Finding],
) -> tuple[list[ThetaMonth], list[str]]:
    """Los θ con entrada completa en el mes y los que no (`missing_thetas`).

    Un θ sin `events.parquet` o sin `carry_over.parquet` en el mes, o con la
    cadena rota, no aporta eventos a `events.bin`: deja `input_missing` y el día
    se escribe con los demás (TRD-viz §9.1).
    """
    ready: list[ThetaMonth] = []
    missing: list[str] = []
    for theta in thetas:
        try:
            for what, name in (("events", EVENTS), ("carry_over", CARRY_OVER)):
                if not lake.has(month, theta, name):
                    path = join(
                        ctx.events_root,
                        events_rel(
                            ctx.provider, ctx.market, ctx.asset, theta, month, name
                        ),
                    )
                    raise InputError(what, f"falta {path}", theta=theta, path=path)
            ready.append(
                theta_month(
                    str(ctx.events_root),
                    lake,
                    ctx.provider,
                    ctx.market,
                    ctx.asset,
                    theta,
                    month,
                )
            )
        except InputError as exc:
            logger.warning("θ=%s sin entrada completa en el mes: %s", theta, exc)
            missing.append(theta)
            out.append(
                findings.input_missing(
                    ctx, days[0], exc.what, days=len(days), **exc.details
                )
            )
    return ready, missing


def _process(
    ctx, month, days, lake, thetas, landing_file, landing, parquet, bounds, out
) -> None:
    ready, missing = _theta_states(ctx, month, days, lake, thetas, out)
    if not ready:
        logger.error("unidad %d-%02d: ningún θ con entrada completa", *month)
        return

    # Archivos de entrada del mes: el consolidado y los de L2 de cada θ.
    base_paths = {landing_file: landing}
    for theta in thetas:
        for name in (EVENTS, CARRY_OVER):
            if lake.has(month, theta, name):
                rel = events_rel(
                    ctx.provider, ctx.market, ctx.asset, theta, month, name
                )
                base_paths[rel] = join(ctx.events_root, rel)
    stats = _Stats()
    base = stats.fetch(base_paths)
    # La cadena de carry-overs solo entra en el hash de los días con cola provisional.
    chain_paths = {
        rel: join(ctx.events_root, rel)
        for state in ready
        if any(state.affects(d) for d in days)
        for rel in state.chain
    }
    chain = stats.fetch(chain_paths)

    pending: list[tuple[date, str]] = []
    skipped: list[dict] = []
    month_state = _Month(ctx, month, ready, missing, out)
    for day in days:
        files = dict(base)
        for state in ready:
            if state.affects(day):
                files |= {rel: chain[rel] for rel in state.chain}
        digest = input_hash(TILES_VERSION, day, files)
        started = time.monotonic()
        index = find_index(ctx.tiles_root, ctx.provider, ctx.market, ctx.asset, day)
        if (
            not ctx.force
            and index is not None
            and index["input_hash"] == digest
            and index["tiles_version"] == TILES_VERSION
        ):
            logger.info("unidad %s: al día (input_hash %s), se salta", day, digest[:12])
            month_state.summary(day, index, True, started)
            skipped.append(index)
        else:
            pending.append((day, digest))

    def advance_skipped() -> None:
        # Un día al día no pasa por `write_day`: si un run anterior cayó antes de
        # avanzar `latest.*`, nadie más lo haría (ITSC-318). Va después de escribir
        # los días pendientes, para que un fallo de `latest` no los deje sin escribir;
        # solo avanza, y solo si está atrasado: si un día pendiente posterior ya lo
        # adelantó, no renderiza nada.
        if skipped:
            advance_latest_from_files(
                ctx.tiles_root, max(skipped, key=lambda i: i["day"])
            )

    if not pending:
        advance_skipped()
        return

    def close(day: date, digest: str, result, started: float) -> None:
        if isinstance(result, PriceUnrepresentable):
            out.append(
                findings.price_unrepresentable(
                    ctx,
                    day,
                    price_scale=result.price_scale,
                    max_price_int=result.max_price_int,
                )
            )
            return
        month_state.build(day, digest, result, started)

    def open_file(day: date) -> TicksFile:
        return TicksFile(ctx.tiles_root, ctx.provider, ctx.market, ctx.asset, day)

    spans = [(day_start_us(d), day_start_us(d) + DAY_US) for d, _ in pending]
    stream = _DayStream(pending, month_state.scale, close, open_file)
    try:
        for batch in _read_batches(parquet, _row_groups(bounds, spans)):
            stream.feed(batch)
            del batch
            if stream.done:
                break
        stream.drain()
    except BaseException:
        stream.abort()
        raise
    advance_skipped()
