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
from dataclasses import dataclass, replace
from decimal import Decimal

from dc_frames import FramesError
from dc_frames.lake import l1_path
from dq import Finding

from l3_dc_descriptors import findings
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.cpu import cpu_limit, kernel_counts, process_cpu_s, throttled_s
from l3_dc_descriptors.errors import SummarizeError
from l3_dc_descriptors.frontier import theta_text
from l3_dc_descriptors.landing import missing_input
from l3_dc_descriptors.memory import (
    DEFAULT_SAMPLE_S,
    DEFAULT_TRIM_MIN_S,
    DEFAULT_TRIM_RSS_MIB,
    MemorySampler,
    MemSample,
    mmap_threshold,
    release_to_kernel,
    take_sample,
    trim_threshold,
)
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

    `findings` son los hallazgos para emitir: en un éxito, un `summaries_summary` por θ (de
    la familia de `ctx.family`) y el `unit_timing`; en un fallo, el hallazgo del error
    (ninguno si el error no tiene `check_type`).
    """

    findings: tuple[Finding, ...]
    error: str | None = None
    timing: Timing | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _theta(value: Decimal) -> int:
    return int(value.scaleb(8))


def process_unit(
    unit: Unit, ctx: RunContext, thetas: list[int], spawn_s: float | None = None
) -> Result:
    """Escribe el mes de `unit` para `thetas` (θ × 10⁸) y devuelve sus hallazgos.

    Con `ctx.profile` la unidad corre bajo `cProfile` y suma el hallazgo `unit_profile`;
    sin él no se crea ningún perfilador. `spawn_s` es lo que tardó en arrancar el proceso
    hijo que la corre (`unit_process`), si lo hay.
    """
    if not ctx.profile:
        return _process_unit(unit, ctx, thetas, spawn_s)
    profiler = cProfile.Profile()
    result = profiler.runcall(_process_unit, unit, ctx, thetas, spawn_s)
    if result.timing is None:
        return result
    profile = findings.unit_profile(
        ctx, unit, wall_s=result.timing.wall_s, top=top_functions(profiler)
    )
    return replace(result, findings=(*result.findings, profile))


def _process_unit(
    unit: Unit, ctx: RunContext, thetas: list[int], spawn_s: float | None
) -> Result:
    started = time.perf_counter()
    cpu_started = process_cpu_s()
    kernel_started = kernel_counts()
    missing = missing_input(str(ctx.landing_root), unit.key, unit.at)
    if missing is not None:
        check_type, details = missing
        path = details["expected_path"]
        return Result(
            (findings.input_missing(ctx, unit, check_type, details),),
            error=f"{check_type}: {path}",
        )

    phases = Phases()
    counts = RowGroupCount(
        l1_path(ctx.landing_root, unit.key, unit.at), meter=phases.read
    )
    scope = "unit" if reset_rss_peak() else "process"
    throttled = throttled_s()
    trimmed: tuple[int | None, int | None] = (None, None)
    close_samples: tuple[MemSample | None, MemSample | None] = (None, None)
    # Arranca antes del primer empuje y para al cerrar la unidad (ITSC-375).
    phases.sampler = MemorySampler(
        DEFAULT_SAMPLE_S if ctx.memory_sample_s is None else ctx.memory_sample_s,
        started,
        DEFAULT_TRIM_RSS_MIB if ctx.trim_rss_mib is None else ctx.trim_rss_mib,
        DEFAULT_TRIM_MIN_S if ctx.trim_min_s is None else ctx.trim_min_s,
    )
    phases.sampler.start()
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
                readers=ctx.readers,
                inflight=ctx.inflight,
                writers=ctx.writers,
                engine_threads=ctx.engine_threads,
                inflight_bytes=ctx.inflight_bytes,
                lot_events=ctx.lot_events,
                meta_workers=ctx.meta_workers,
                family=ctx.family,
            )
    except SummarizeError as error:
        found = findings.from_error(ctx, unit, error)
        return Result((found,) if found else (), error=str(error))
    except (
        FramesError
    ) as error:  # L1 o L2 fuera de contrato: sin hallazgo propio (§9.3)
        return Result((), error=str(error))
    finally:
        # Con éxito o con error, y ya cerrados los pools de la unidad (ITSC-371). Las dos
        # muestras del cierre (ITSC-375) rodean al trim: lo que baja entre ellas es lo que
        # glibc retenía y el programa ya había liberado.
        phases.sampler.stop()
        before_trim = take_sample(started)
        trimmed = release_to_kernel()
        close_samples = (before_trim, take_sample(started))

    summaries, stats = [], []
    for value, result in written.items():
        theta = _theta(value)
        summaries.append((theta, result))
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
        kernel=kernel_counts().since(kernel_started),
        mmap_threshold=mmap_threshold(),
        trim_threshold=trim_threshold(),
        rss_before_trim_bytes=trimmed[0],
        rss_after_trim_bytes=trimmed[1],
        spawn_s=spawn_s,
        mem_peak=phases.sampler.peak,
        mem_close_before_trim=close_samples[0],
        mem_close_after_trim=close_samples[1],
        mem_samples=phases.sampler.ticks,
        trims=phases.sampler.trims,
        trim_s=phases.sampler.trim_s,
        trim_rss_mib=phases.sampler.trim_rss_mib,
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
                    l2_events_content_hash=result.l2_events_content_hash,
                    l2_carry_over_content_hash=result.l2_carry_over_content_hash,
                    unassigned_ticks=(
                        None
                        if result.reconciliation is None
                        else result.reconciliation.unassigned
                    ),
                )
                for theta, result in sorted(summaries, key=lambda item: item[0])
            ),
            findings.unit_timing(ctx, unit, timing),
        ),
        timing=timing,
    )


def _stats(theta: int, result: ThetaResult) -> ThetaStats:
    return ThetaStats(
        theta_text(theta), result.rows, result.closed, result.before_month
    )
