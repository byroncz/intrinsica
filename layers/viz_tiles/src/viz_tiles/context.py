"""Los datos con que arranca una ejecución: dónde lee, dónde escribe y qué activo."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    run_id: str
    image_version: str
    tiles_root: str | Path
    dq_root: str | Path
    # El modo `render` no lee L1 ni L2 y no los necesita.
    landing_root: str | Path | None = None
    events_root: str | Path | None = None
    mode: str = "tiles"
    asset: str = "BTCUSDT"
    provider: str = "binance"
    market: str = "spot"
    # Regenera lo seleccionado aunque el `input_hash` no haya cambiado.
    force: bool = False
