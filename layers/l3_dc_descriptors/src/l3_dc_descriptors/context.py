"""Los datos con que arranca una unidad: qué mes es y en qué ejecución va.

Son datos inmutables: la unidad de L3 es una función de ellos (`pipeline.process_unit`)
y no guarda nada entre meses (ADR-L3-12), así que puede correr en cualquier hilo o tarea.
"""

from dataclasses import dataclass
from pathlib import Path

from dc_frames.lake import Month

from l3_dc_descriptors.schema import FAMILY


@dataclass(frozen=True)
class Unit:
    """Un mes de un activo."""

    year: int
    month: int
    provider: str = "binance"
    market: str = "spot"
    asset: str = "BTCUSDT"

    def __str__(self) -> str:
        return (
            f"{self.provider}/{self.market}/{self.asset}/"
            f"{self.year:04d}-{self.month:02d}"
        )

    @property
    def key(self) -> tuple[str, str, str]:
        """Proveedor, mercado y activo: la clave que usan el lector y el resumidor."""
        return self.provider, self.market, self.asset

    @property
    def at(self) -> Month:
        return self.year, self.month

    def with_month(self, month: Month) -> Unit:
        return Unit(*month, self.provider, self.market, self.asset)


@dataclass(frozen=True)
class RunContext:
    mode: str
    run_id: str
    image_version: str
    # Primer mes de la serie de L2: el único sin mes anterior. Se declara, no se infiere.
    series_start: Month
    landing_root: str | Path
    events_root: str | Path
    summaries_root: str | Path
    dq_root: str | Path
    family: str = FAMILY
    # `L3_PROFILE=1`: la unidad corre bajo cProfile y emite `unit_profile`.
    profile: bool = False
    # `L3_THREADS`, `L3_READERS`, `L3_INFLIGHT`, `L3_WRITERS`, `L3_ENGINE_THREADS` y
    # `L3_INFLIGHT_BYTES`: pool de E/S (`None`: `ceil(3 × vCPU)`), lectores de L1 (R), row
    # groups en vuelo en el motor (K), escritores (W), hilos del motor (`None`: los vCPU) y
    # tope de bytes de eventos abiertos. `None` es el valor por defecto de `summarize`.
    threads: int | None = None
    readers: int | None = None
    inflight: int | None = None
    writers: int | None = None
    engine_threads: int | None = None
    inflight_bytes: int | None = None
    # `L3_META_WORKERS`: hilos del pool de metadatos (compuerta, apertura de `events.parquet` y de
    # los escritores, pie de L1 y publicación; ITSC-369). `None` es el valor por defecto de
    # `summarize`.
    meta_workers: int | None = None
    # `L3_LOT_EVENTS`: eventos que cabe en un tramo, la parte de un row group de L1 que se empuja
    # al motor (ITSC-373). `None` es el valor por defecto de `summarize` y 0 desactiva el corte.
    lot_events: int | None = None
    # `L3_UNIT_PROCESS` (por defecto 1): en backfill cada unidad corre en un proceso hijo y su
    # memoria muere con ella (ITSC-372). `monthly` procesa una sola unidad y no lo usa.
    unit_process: bool = True
    # `L3_MEMORY_SAMPLE_S`: período del hilo que muestrea el RSS y guarda la muestra del pico
    # (ITSC-375). `None` es `memory.DEFAULT_SAMPLE_S`.
    memory_sample_s: float | None = None
    # `L3_TRIM_RSS_MIB` y `L3_TRIM_MIN_S`: marca de agua del `malloc_trim(0)` periódico del
    # muestreador (0 lo desactiva) y segundos mínimos entre dos trims (ITSC-376). `None` es
    # `memory.DEFAULT_TRIM_RSS_MIB` y `memory.DEFAULT_TRIM_MIN_S`.
    trim_rss_mib: int | None = None
    trim_min_s: float | None = None
