"""Constantes del contrato de la página de un día de viz.

Contrato en `docs/data-contracts.md` ("Tiles de viz") y TRD-viz §7. Una prueba
(`tests/test_tiles_contract_doc.py`) rompe el CI si esa sección se desvía de
estas constantes.
"""

# 2.0.0: la página del día lleva los ticks (`ticks.bin`, deltas en varint) y los
# eventos exactos (`events.bin`); el navegador deriva precio, volumen y
# confirmaciones por píxel al dibujar. Desaparecen los arreglos por nivel
# (`price-<w>`, `volume-<w>`, `dir-<w>`, `count-<w>`, `confirms-<w>`, `simul-<w>`)
# y la noción de nivel (TRD-viz §7 y ADR-VZ-14).
TILES_VERSION = "2.0.0"

DAY_US = 86_400_000_000
DAY_S = 86_400
DAY_MS = 86_400_000

# Escala de price y quantity en L1: DECIMAL(18, 8).
L1_SCALE = 10**8

# Unidades de precio de `ticks.bin` por unidad de la cotización (`price_scale`):
# fija por activo e igual a su tick (TRD-viz §7.3). Nunca se elige por día: un
# precio fuera del tick se redondea al tick más cercano y se avisa (`price_rounded`).
PRICE_SCALE_BY_ASSET = {"BTCUSDT": 100}

INT32_MAX = 2**31 - 1

# `ticks.bin`: tres secciones de enteros varint (LEB128, 7 bits por byte, el bit
# alto marca que sigue otro byte), cada una con `ticks` valores y en este orden:
#   1. Δtiempo en ms desde el tick anterior (el primero, desde el inicio del día).
#   2. Δprecio en unidades de 1/price_scale, en zigzag (el primero, desde 0).
#   3. cantidad en unidades de 10⁻⁸ (sin signo).
# Los ticks van en el orden del consolidado de L1: `transact_time` y, dentro de
# un mismo instante, `agg_trade_id`.
TICKS_FILE = "ticks.bin"
TICK_SECTIONS = ("dt_ms", "dprice_zigzag", "quantity_1e8")

# `events.bin`: cuatro secciones de `N` valores (referencia, confirmación y
# extremo en int32 y un byte de banderas), con `N` el total de eventos. El θ `k`
# ocupa de `events_offset` a `events_offset + events - 1` en cada sección.
EVENTS_FILE = "events.bin"
EVENT_BYTES = 13

# Banderas de un evento (uint8).
FLAG_UP = 1  # alza; sin el bit, baja
FLAG_PROVISIONAL = 2  # cola del carry-over: su extremo es un candidato vigente
FLAG_REF_CLIPPED = 4  # la referencia es anterior al día: se recortó a 0
FLAG_CONFIRM_CLIPPED = 8  # la confirmación cae fuera del día: se recortó a su borde
FLAG_EXTREME_CLIPPED = 16  # el extremo cae fuera del día: se recortó a su borde
FLAGS = {
    FLAG_UP: "alza",
    FLAG_PROVISIONAL: "provisional",
    FLAG_REF_CLIPPED: "referencia recortada",
    FLAG_CONFIRM_CLIPPED: "confirmación recortada",
    FLAG_EXTREME_CLIPPED: "extremo recortado",
}
# Máximo de θ de un día: el navegador los guarda en un byte (`Uint8Array`).
MAX_THETAS = 255

INDEX_FILE = "index.json"
LATEST_FILE = "latest.json"
# La página autocontenida del día y su copia para el último día (TRD-viz §6.5).
PAGE_FILE = "index.html"
LATEST_PAGE_FILE = "latest.html"

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
    ("first_agg_trade_id", "integer"),
    ("last_agg_trade_id", "integer"),
    ("ticks_file", "string"),
    ("events", "string"),
    ("page", "string"),
    ("thetas", "array"),
    ("missing_thetas", "array"),
    ("input_hash", "string"),
    ("content_hash", "string"),
    ("generated_at", "string"),
    ("image_version", "string"),
)

# Campos de `latest.json`.
LATEST_FIELDS = ("tiles_version", "provider", "market", "asset", "day")


def price_scale(asset: str) -> int:
    """`price_scale` del activo. Un activo sin tick declarado no se reduce."""
    try:
        return PRICE_SCALE_BY_ASSET[asset]
    except KeyError:
        raise ValueError(
            f"sin price_scale declarado para {asset}: agrégalo a PRICE_SCALE_BY_ASSET"
        ) from None
