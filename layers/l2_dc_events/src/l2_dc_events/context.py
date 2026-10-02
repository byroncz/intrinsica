"""Los dos datos con que arranca una unidad: qué mes es y en qué ejecución va."""

from dataclasses import dataclass
from pathlib import Path

from l2_dc_events.landing import READ_AHEAD


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

    def previous(self) -> Unit:
        """El mes anterior del mismo activo: de donde sale el carry-over."""
        year, month = divmod(self.year * 12 + self.month - 2, 12)
        return Unit(year, month + 1, self.provider, self.market, self.asset)


@dataclass(frozen=True)
class RunContext:
    mode: str
    run_id: str
    image_version: str
    # Primer mes de la serie: el único que arranca en frío (ADR-L2-08). Se
    # declara y no se infiere: un carry-over que falta no es "primer mes".
    series_start: tuple[int, int]
    landing_root: str | Path
    events_root: str | Path
    dq_root: str | Path
    # Row groups de la landing que el lector pide por delante (`landing.read_batches`).
    read_ahead: int = READ_AHEAD
