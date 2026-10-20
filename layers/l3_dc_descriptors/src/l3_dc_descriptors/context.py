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
    # `L3_THREADS` y `L3_PREFETCH`: hilos del pool de θ (`None`: `ceil(3 × vCPU)`) y row
    # groups de ventaja del lector (`None`: el valor por defecto de `summarize`).
    threads: int | None = None
    prefetch: int | None = None
