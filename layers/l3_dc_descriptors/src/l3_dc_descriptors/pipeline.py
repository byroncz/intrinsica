"""La unidad de L3: un mes para una lista de θ, de la landing al hallazgo (TRD-L3 §8.1).

`process_unit` es una función de sus argumentos: no lee ni escribe estado de módulo, no
supone que antes corrió otro mes (ADR-L3-12) y devuelve todo lo que dejó (hallazgos y
tiempos) en un `Result`. Por eso se puede invocar desde una tarea aparte, o varias a la
vez, sin que se pisen; quien decide el orden y el paralelismo es la CLI (ITSC-335).

Un fallo no se lanza: vuelve en el `Result` con su hallazgo, si lo tiene, para que quien
llama lo emita y decida qué hacer con el rango.
"""

import cProfile
import time
from contextlib import closing
from dataclasses import dataclass, replace
from decimal import Decimal

import pyarrow.parquet as pq
from dc_frames import FramesError, l2_path
from dc_frames.lake import l1_path, partition_path
from dq import Finding
from pyutils import ContentHasher, resolve_fs

from l3_dc_descriptors import findings
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.cpu import cpu_limit, process_cpu_s, throttled_s
from l3_dc_descriptors.errors import SummarizeError
from l3_dc_descriptors.frontier import theta_text
from l3_dc_descriptors.l2 import CARRY_OVER
from l3_dc_descriptors.landing import missing_input
from l3_dc_descriptors.profiling import top_functions
from l3_dc_descriptors.summarize import ThetaResult, summarize_month
from l3_dc_descriptors.timing import (
    Phases,
    RowGroupCount,
    ThetaStats,
    Timing,
    count_row_groups,
    reset_rss_peak,
    rss_peak_bytes,
)


@dataclass(frozen=True)
class Result:
    """Lo que dejó la unidad. `error` es `None` si el mes se escribió; si no, el motivo.

    `findings` son los hallazgos para emitir: en un éxito, un `summaries_summary` por θ
    y el `unit_timing`; en un fallo, el hallazgo del error (ninguno si el error no tiene
    `check_type`).
    """

    findings: tuple[Finding, ...]
    error: str | None = None
    timing: Timing | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def file_hash(path: str) -> str:
    """El `content_hash` de un Parquet, un row group a la vez.

    Es el mismo que L2 deja en `events_summary` (`ContentHasher` sobre el esquema del
    archivo), sin cargar el archivo entero: el pico es un row group.
    """
    fs, resolved = resolve_fs(path)
    with closing(pq.ParquetFile(resolved, filesystem=fs)) as parquet:
        hasher = ContentHasher(parquet.schema_arrow)
        for i in range(parquet.num_row_groups):
            table = parquet.read_row_group(i)
            for batch in table.to_batches():
                hasher.update(batch)
            del table
        return hasher.hexdigest()


def _theta(value: Decimal) -> int:
    return int(value.scaleb(8))


def process_unit(unit: Unit, ctx: RunContext, thetas: list[int]) -> Result:
    """Escribe el mes de `unit` para `thetas` (θ × 10⁸) y devuelve sus hallazgos.

    Con `ctx.profile` la unidad corre bajo `cProfile` y suma el hallazgo `unit_profile`;
    sin él no se crea ningún perfilador.
    """
    if not ctx.profile:
        return _process_unit(unit, ctx, thetas)
    profiler = cProfile.Profile()
    result = profiler.runcall(_process_unit, unit, ctx, thetas)
    if result.timing is None:
        return result
    profile = findings.unit_profile(
        ctx, unit, wall_s=result.timing.wall_s, top=top_functions(profiler)
    )
    return replace(result, findings=(*result.findings, profile))


def _process_unit(unit: Unit, ctx: RunContext, thetas: list[int]) -> Result:
    started = time.perf_counter()
    cpu_started = process_cpu_s()
    missing = missing_input(str(ctx.landing_root), unit.key, unit.at)
    if missing is not None:
        check_type, details = missing
        path = details["expected_path"]
        return Result(
            (findings.input_missing(ctx, unit, check_type, details),),
            error=f"{check_type}: {path}",
        )

    phases = Phases()
    counts = RowGroupCount(l1_path(ctx.landing_root, unit.key, unit.at))
    scope = "unit" if reset_rss_peak() else "process"
    throttled = throttled_s()
    try:
        with count_row_groups(counts):
            written = summarize_month(
                ctx.landing_root,
                ctx.events_root,
                ctx.summaries_root,
                unit.at,
                [theta_text(t) for t in thetas],
                series_start=ctx.series_start,
                provider=unit.provider,
                market=unit.market,
                asset=unit.asset,
                phases=phases,
                threads=ctx.threads,
                prefetch=ctx.prefetch,
            )
    except SummarizeError as error:
        found = findings.from_error(ctx, unit, error)
        return Result((found,) if found else (), error=str(error))
    except (
        FramesError
    ) as error:  # L1 o L2 fuera de contrato: sin hallazgo propio (§9.3)
        return Result((), error=str(error))

    summaries, stats = [], []
    for value, result in written.items():
        theta = _theta(value)
        summaries.append((theta, result, _lineage(ctx, unit, theta, phases)))
        stats.append(_stats(theta, result))
    wall_s = time.perf_counter() - started
    cpu_user_s, cpu_system_s = (
        end - start for end, start in zip(process_cpu_s(), cpu_started, strict=True)
    )
    after = throttled_s()
    timing = Timing(
        wall_s=wall_s,
        phases=phases,
        rss_peak_bytes=rss_peak_bytes(),
        rss_scope=scope,
        row_groups=counts,
        thetas=tuple(sorted(stats, key=lambda s: s.theta)),
        cpu=cpu_limit(),
        cpu_throttled_s=(
            None if throttled is None or after is None else max(0.0, after - throttled)
        ),
        cpu_user_s=cpu_user_s,
        cpu_system_s=cpu_system_s,
    )
    return Result(
        (
            *(
                findings.summaries_summary(
                    ctx,
                    unit,
                    theta=theta_text(theta),
                    rows=result.rows,
                    content_hash=result.content_hash,
                    l2_events_content_hash=events_hash,
                    l2_carry_over_content_hash=carry_hash,
                    unassigned_ticks=result.reconciliation.unassigned,
                )
                for theta, result, (events_hash, carry_hash) in sorted(
                    summaries, key=lambda item: item[0]
                )
            ),
            findings.unit_timing(ctx, unit, timing),
        ),
        timing=timing,
    )


def _stats(theta: int, result: ThetaResult) -> ThetaStats:
    reconciliation = result.reconciliation
    return ThetaStats(
        theta_text(theta),
        result.rows,
        reconciliation.closed,
        reconciliation.before_month,
    )


def _lineage(
    ctx: RunContext, unit: Unit, theta: int, phases: Phases
) -> tuple[str, str]:
    """Los `content_hash` de las entradas de L2 que usó el θ: `events` y `carry_over`."""
    started = time.perf_counter()
    text = theta_text(theta)
    events = file_hash(l2_path(ctx.events_root, unit.key, text, unit.at))
    carry = file_hash(
        partition_path(ctx.events_root, unit.key, text, unit.at, CARRY_OVER)
    )
    phases.lineage_s += time.perf_counter() - started
    return events, carry
