"""Un mes de tiles: decide qué días rehacer, lee L1 una vez y escribe cada día.

TRD-viz §7.8 y §8.1. El orden respeta la eficiencia de memoria: el `input_hash`
se calcula con los metadatos de los archivos (nunca se lee un tick para decidir);
los días al día se saltan; los demás salen de **una** pasada por el
`consolidated.parquet` del mes, row group a row group, y cada día cierra cuando
los ticks cambian de día. En RAM: un row group de L1, los acumuladores del día en
curso y, por θ, los eventos que tocan el día.
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
from viz_tiles.contract import DAY_US, FINEST, TILES_VERSION, price_scale
from viz_tiles.direction import direction_tiles
from viz_tiles.events import ConfirmAccumulator, EventsBuffer, event_rows
from viz_tiles.lake import (
    CARRY_OVER,
    EVENTS,
    TICK_COLUMNS,
    EventsIndex,
    InputError,
    Month,
    column_bounds,
    events_rel,
    exists,
    input_hash,
    join,
    landing_rel,
    object_stat,
    open_parquet,
    read_events,
    validate_ticks,
)
from viz_tiles.reduce import (
    DayReduction,
    M4Accumulator,
    PriceUnrepresentable,
    day_start_us,
)
from viz_tiles.write import ThetaTiles, day_objects, find_index, write_day

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
    """Línea de cierre de un día: ticks, pared y RSS pico (KiB en Linux → MiB)."""
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
    """El estado de un mes en curso: los θ listos y los hallazgos que va juntando."""

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
        objects, size = day_objects(
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
                objects=objects,
                size=size,
                levels=index["levels"],
                thetas=[t["theta"] for t in thetas],
                provisional_thetas=[
                    t["theta"] for t in thetas if t["provisional_from_s"] is not None
                ],
                missing_thetas=index["missing_thetas"],
            )
        )
        _probe(day, index["ticks"], started, **({"skipped": "true"} if skipped else {}))

    def build(
        self, day: date, digest: str, reduction: DayReduction, started: float
    ) -> None:
        """Pasada por θ de un día (TRD-viz §8.1, paso 4): dirección, escritura y resumen."""
        ctx = self.ctx
        if reduction.ticks == 0:
            self.out.append(
                findings.input_missing(
                    ctx, day, "ticks", path=join(ctx.landing_root, self._landing())
                )
            )
            return
        if reduction.rounded:
            self.out.append(
                findings.price_rounded(
                    ctx,
                    day,
                    count=reduction.rounded,
                    max_abs_delta_int=reduction.max_abs_delta_int,
                )
            )
        # Los eventos que tocan el día son los de ids entre el primer y el último
        # tick que fija el estado de una columna.
        ids = reduction.last_ids[FINEST]
        valid = ids[ids >= 0]
        lo, hi = int(valid.min()), int(valid.max())
        confirmations = ConfirmAccumulator()
        events = EventsBuffer()
        thetas = [
            self._theta_tiles(state, day, reduction, lo, hi, confirmations, events)
            for state in self.ready
        ]
        write_day(
            ctx.tiles_root,
            provider=ctx.provider,
            market=ctx.market,
            asset=ctx.asset,
            day=day,
            reduction=reduction,
            confirmations=confirmations.finish(),
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

    def _theta_tiles(
        self,
        state: ThetaMonth,
        day: date,
        reduction: DayReduction,
        lo: int,
        hi: int,
        confirmations: ConfirmAccumulator,
        buffer: EventsBuffer,
    ) -> ThetaTiles:
        c = self.ctx
        path = join(
            c.events_root,
            events_rel(c.provider, c.market, c.asset, state.theta, self.month, EVENTS),
        )
        events = read_events(path, lo, hi)
        tail = state.tail
        # La cola entra solo si el día tiene ticks posteriores a la referencia del pendiente.
        pending = (
            tail if tail is not None and tail.reference_agg_trade_id < hi else None
        )
        directions = direction_tiles(reduction.last_ids, events, pending)
        provisional_from_s = state.provisional_from_s(day)
        rows, confirm_us = event_rows(
            events, pending, provisional_from_s is not None, day_start_us(day)
        )
        del events  # las filas del θ ya son los tiles: la tabla no se necesita más
        confirmations.add(confirm_us)
        buffer.add(rows)
        count = len(rows)
        del rows  # el buffer del día es la única copia de los eventos
        return ThetaTiles(
            theta=state.theta,
            events=count,
            provisional_from_s=provisional_from_s,
            direction=directions,
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
        close: Callable[[date, str, DayReduction, float], None],
    ) -> None:
        self._pending = pending
        self._scale = scale
        self._close = close
        self._pos = -1
        self.acc: M4Accumulator | None = None
        self._started = 0.0
        self._end = 0
        self._advance()

    def _advance(self) -> None:
        self._pos += 1
        if self._pos >= len(self._pending):
            self.acc = None
            return
        day = self._pending[self._pos][0]
        self.acc = M4Accumulator(day, self._scale)
        self._started = time.monotonic()
        self._end = day_start_us(day) + DAY_US

    def _finish(self) -> None:
        day, digest = self._pending[self._pos]
        acc, self.acc = self.acc, None
        try:
            reduction = acc.finish()
        except PriceUnrepresentable as exc:
            self._close(day, digest, exc, self._started)
        else:
            self._close(day, digest, reduction, self._started)
        self._advance()

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
    cadena rota, no tiene bloque en `dir-<w>.u8`: deja `input_missing` y el día
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
        else:
            pending.append((day, digest))
    if not pending:
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

    spans = [(day_start_us(d), day_start_us(d) + DAY_US) for d, _ in pending]
    stream = _DayStream(pending, month_state.scale, close)
    for batch in _read_batches(parquet, _row_groups(bounds, spans)):
        stream.feed(batch)
        del batch
        if stream.done:
            break
    stream.drain()
