"""Constantes del contrato de tiles de viz.

Contrato en `docs/data-contracts.md` ("Tiles de viz") y TRD-viz §7. Una prueba
(`tests/test_tiles_contract_doc.py`) rompe el CI si esa sección se desvía de
estas constantes.
"""

from dataclasses import dataclass

TILES_VERSION = "1.0.0"

DAY_US = 86_400_000_000
DAY_S = 86_400
DAY_MS = 86_400_000

# Niveles de zoom: columnas en que se divide el día UTC completo (ADR-VZ-08).
# Potencias de 2 para que la columna de un tick se calcule con enteros y para
# que M4 sea componible: un nivel grueso sale de dos columnas del fino.
LEVELS = (128, 256, 512, 1024, 2048, 4096)
FINEST = LEVELS[-1]

# Escala de price y quantity en L1: DECIMAL(18, 8).
L1_SCALE = 10**8

# Unidades de precio del tile por unidad de la cotización (`price_scale`): fija
# por activo e igual a su tick (TRD-viz §7.3). Nunca se elige por día: un precio
# fuera del tick se redondea al tick más cercano y se avisa (`price_rounded`).
PRICE_SCALE_BY_ASSET = {"BTCUSDT": 100}

# Centinela de `p` en una columna sin ticks (reemplaza al NaN del float).
EMPTY_PRICE = -(2**31)
INT32_MAX = 2**31 - 1

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


# `price-<w>.i32` guarda dos bloques de 4w enteros: el tiempo (uint32, ms desde
# el inicio del día) y el precio (int32, unidades de 1/price_scale) de los cuatro
# puntos M4 de cada columna. El tiempo no pasa de 86 400 000, así que uint32 e
# int32 dan los mismos bytes y el arreglo en RAM es uno solo, `<i4`.
# `dir-<w>.u8` guarda un bloque de `w` bytes por θ, en el orden de `thetas` del
# índice: `per_column` es por θ.
TILE_FILES = (
    TileFile("price", "price-{w}.i32", "<i4", 8),
    TileFile("volume", "volume-{w}.f32", "<f4", 1),
    TileFile("dir", "dir-{w}.u8", "u1", 1),
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
    ("price_scale", "integer"),
    ("ticks", "integer"),
    ("levels", "array"),
    ("price", "object"),
    ("volume", "object"),
    ("dir", "object"),
    ("thetas", "array"),
    ("missing_thetas", "array"),
    ("input_hash", "string"),
    ("content_hash", "string"),
    ("generated_at", "string"),
    ("image_version", "string"),
)

# Campos de `latest.json`.
LATEST_FIELDS = ("tiles_version", "provider", "market", "asset", "day")


def tile_name(kind: str, w: int) -> str:
    """Nombre del archivo de `kind` en el nivel `w`."""
    return FILE_BY_KIND[kind].template.format(w=w)


def price_scale(asset: str) -> int:
    """`price_scale` del activo. Un activo sin tick declarado no se reduce."""
    try:
        return PRICE_SCALE_BY_ASSET[asset]
    except KeyError:
        raise ValueError(
            f"sin price_scale declarado para {asset}: agrégalo a PRICE_SCALE_BY_ASSET"
        ) from None
