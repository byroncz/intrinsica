"""Los datos con que arranca una ejecución: dónde lee, dónde escribe y qué activo."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    run_id: str
    image_version: str
    landing_root: str | Path
    events_root: str | Path
    tiles_root: str | Path
    dq_root: str | Path
    asset: str = "BTCUSDT"
    provider: str = "binance"
    market: str = "spot"
    # Regenera lo seleccionado aunque el `input_hash` no haya cambiado.
    force: bool = False
