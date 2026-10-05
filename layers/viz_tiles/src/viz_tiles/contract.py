"""Constantes del contrato de tiles de viz.

Contrato en `docs/data-contracts.md` ("Tiles de viz") y TRD-viz §7. Una prueba
(`tests/test_tiles_contract_doc.py`) rompe el CI si esa sección se desvía de
estas constantes.
"""

from dataclasses import dataclass

TILES_VERSION = "1.0.0"

DAY_US = 86_400_000_000
DAY_S = 86_400

# Niveles de zoom: columnas en que se divide el día UTC completo (ADR-VZ-08).
# Potencias de 2 para que la columna de un tick se calcule con enteros y para
# que M4 sea componible: un nivel grueso sale de dos columnas del fino.
LEVELS = (128, 256, 512, 1024, 2048, 4096)
FINEST = LEVELS[-1]

# Escala de price y quantity en L1: DECIMAL(18, 8).
PRICE_SCALE = 10**8

# Estados del tile de dirección (uint8).
STATE_NONE = 0
STATE_CONFIRM_UP = 1
STATE_OVERSHOOT_UP = 2
STATE_CONFIRM_DOWN = 3
STATE_OVERSHOOT_DOWN = 4
STATES = {
    STATE_NONE: "sin evento",
    STATE_CONFIRM_UP: "confirmación alza",
    STATE_OVERSHOOT_UP: "overshoot alza",
    STATE_CONFIRM_DOWN: "confirmación baja",
    STATE_OVERSHOOT_DOWN: "overshoot baja",
}


@dataclass(frozen=True)
class TileFile:
    """Un tipo de archivo de tile: nombre, tipo de valor y valores por columna."""

    kind: str
    template: str
    dtype: str
    per_column: int


# `price-<w>.f32` guarda dos bloques de 4w valores: el tiempo y el precio de los
# cuatro puntos M4 de cada columna.
TILE_FILES = (
    TileFile("price", "price-{w}.f32", "<f4", 8),
    TileFile("volume", "volume-{w}.f32", "<f4", 1),
    TileFile("dir", "dir-{w}-{theta}.u8", "u1", 1),
)
FILE_BY_KIND = {f.kind: f for f in TILE_FILES}

INDEX_FILE = "index.json"
LATEST_FILE = "latest.json"

# Campos de `index.json` en orden de escritura, con su tipo JSON.
INDEX_FIELDS = (
    ("tiles_version", "string"),
    ("provider", "string"),
    ("market", "string"),
    ("asset", "string"),
    ("day", "string"),
    ("t0", "integer"),
    ("ticks", "integer"),
    ("levels", "array"),
    ("price", "object"),
    ("volume", "object"),
    ("thetas", "array"),
    ("missing_thetas", "array"),
    ("input_hash", "string"),
    ("content_hash", "string"),
    ("generated_at", "string"),
    ("image_version", "string"),
)

# Campos de `latest.json`.
LATEST_FIELDS = ("tiles_version", "provider", "market", "asset", "day")


def tile_name(kind: str, w: int, theta: str | None = None) -> str:
    """Nombre del archivo de `kind` en el nivel `w` (y el θ, si es de dirección)."""
    return FILE_BY_KIND[kind].template.format(w=w, theta=theta)
