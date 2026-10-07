#!/usr/bin/env bash
# Prueba de humo de la imagen: arma un lago local con el día real 2017-08-18
# (4 735 ticks de shared/dc_core/tests/fixtures/ticks.csv y sus eventos de
# events_v0.csv para cinco θ), corre --mode tiles sobre ese día y verifica el
# índice, ticks.bin y events.bin, la página index.html y el Parquet de hallazgos; luego
# corre --mode render, que no lee L1 ni L2. Uso: smoke.sh <imagen>
#
# La entrada no viene de L1 ni de L2: los CSV se convierten con el pyarrow de la
# propia imagen (el mismo que corre en producción), sin red ni otra imagen.
# Cada θ trae el pendiente de su último evento en carry_over.parquet, con el
# tick de mayor precio (o menor) desde su confirmación como extremo vigente.
set -euo pipefail

image="${1:?uso: smoke.sh <imagen>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixtures="$root/shared/dc_core/tests/fixtures"
data="$(mktemp -d)"
trap 'docker run --rm -v "$data:/data" --entrypoint rm "$image" -rf /data/landing /data/events /data/tiles /data/dq; rm -rf "$data"' EXIT

# El contenedor corre como root: crear las rutas desde ahí evita que el trap
# tenga que borrar archivos ajenos.
docker run --rm -i -v "$data:/data" -v "$fixtures:/fixtures:ro" \
  --entrypoint python "$image" - <<'PY'
import csv
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

dec = pa.decimal128(18, 8)
theta_type = pa.decimal128(9, 8)
series = "provider=binance/market=spot/asset=BTCUSDT"

with open("/fixtures/ticks.csv") as f:
    ticks = [
        (int(r["agg_trade_id"]), Decimal(r["price"]), int(r["transact_time"]))
        for r in csv.DictReader(f)
    ]
n = len(ticks)
ids = pa.array([t[0] for t in ticks], pa.int64())
# Esquema de OUTPUT_SCHEMA de L1 (layers/l1_ingest/src/l1_ingest/schema.py).
l1 = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", dec, nullable=False),
        pa.field("quantity", dec, nullable=False),
        pa.field("first_trade_id", pa.int64(), nullable=False),
        pa.field("last_trade_id", pa.int64(), nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
        pa.field("is_best_match", pa.bool_(), nullable=False),
    ]
)
table = pa.table(
    {
        "agg_trade_id": ids,
        "price": pa.array([t[1] for t in ticks], dec),
        "quantity": pa.array([Decimal(1)] * n, dec),
        "first_trade_id": ids,
        "last_trade_id": ids,
        "transact_time": pa.array([t[2] for t in ticks], pa.int64()),
        "is_buyer_maker": pa.array([False] * n),
        "is_best_match": pa.array([True] * n),
    },
    schema=l1,
)
landing = Path(f"/data/landing/{series}/year=2017/month=08/consolidated.parquet")
landing.parent.mkdir(parents=True)
pq.write_table(table, landing, row_group_size=1000)
print(f"consolidated.parquet: {n} ticks")


def point(name, nullable=False):
    return [
        pa.field(f"{name}_price", dec, nullable=nullable),
        pa.field(f"{name}_time", pa.int64(), nullable=nullable),
        pa.field(f"{name}_agg_trade_id", pa.int64(), nullable=nullable),
    ]


# Esquemas de EVENTS_SCHEMA y CARRY_OVER_SCHEMA de L2 (l2_dc_events/schema.py).
events_schema = pa.schema(
    [
        *point("reference"),
        *point("confirm"),
        *point("extreme"),
        pa.field("direction", pa.int8(), nullable=False),
        pa.field("theta", theta_type, nullable=False),
    ]
)
carry_schema = pa.schema(
    [
        pa.field("provider", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset", pa.string(), nullable=False),
        pa.field("theta", theta_type, nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("month", pa.int32(), nullable=False),
        pa.field("state_version", pa.string(), nullable=False),
        pa.field("direction", pa.int8(), nullable=False),
        *point("ext_high"),
        *point("ext_low"),
        pa.field("has_pending_event", pa.bool_(), nullable=False),
        *point("pending_reference", nullable=True),
        *point("pending_confirm", nullable=True),
    ]
)

by_theta = {}
with open("/fixtures/events_v0.csv") as f:
    for row in csv.DictReader(f):
        by_theta.setdefault(int(row["theta"]), []).append(row)

for theta, rows in by_theta.items():
    *closed, pending = rows
    assert pending["extreme_agg_trade_id"] == "", "el último evento debe ser el pendiente"
    folder = Path(f"/data/events/{series}/theta=0.{theta:08d}/year=2017/month=08")
    folder.mkdir(parents=True)
    cols = {}
    for name in ("reference", "confirm", "extreme"):
        cols[f"{name}_price"] = pa.array([Decimal(r[f"{name}_price"]) for r in closed], dec)
        cols[f"{name}_time"] = pa.array([int(r[f"{name}_time"]) for r in closed], pa.int64())
        cols[f"{name}_agg_trade_id"] = pa.array(
            [int(r[f"{name}_agg_trade_id"]) for r in closed], pa.int64()
        )
    cols["direction"] = pa.array([int(r["direction"]) for r in closed], pa.int8())
    cols["theta"] = pa.array([Decimal(theta).scaleb(-8)] * len(closed), theta_type)
    pq.write_table(
        pa.table(cols, schema=events_schema), folder / "events.parquet", row_group_size=64
    )

    direction = int(pending["direction"])
    confirm_id = int(pending["confirm_agg_trade_id"])
    after = [t for t in ticks if t[0] >= confirm_id]
    best = (max if direction == 1 else min)(t[1] for t in after)
    top = next(t for t in after if t[1] == best)
    ext = {"price": top[1], "time": top[2], "id": top[0]}

    def carry_point(name, p):
        return {
            f"{name}_price": p["price"],
            f"{name}_time": p["time"],
            f"{name}_agg_trade_id": p["id"],
        }

    def from_row(prefix):
        return {
            "price": Decimal(pending[f"{prefix}_price"]),
            "time": int(pending[f"{prefix}_time"]),
            "id": int(pending[f"{prefix}_agg_trade_id"]),
        }

    ref, conf = from_row("reference"), from_row("confirm")
    other = ref  # el otro extremo de la tendencia: basta con que exista
    high, low = (ext, other) if direction == 1 else (other, ext)
    carry = {
        "provider": "binance",
        "market": "spot",
        "asset": "BTCUSDT",
        "theta": Decimal(theta).scaleb(-8),
        "year": 2017,
        "month": 8,
        "state_version": "smoke",
        "direction": direction,
        **carry_point("ext_high", high),
        **carry_point("ext_low", low),
        "has_pending_event": True,
        **carry_point("pending_reference", ref),
        **carry_point("pending_confirm", conf),
    }
    pq.write_table(
        pa.Table.from_pylist([carry], schema=carry_schema), folder / "carry_over.parquet"
    )
print(f"events.parquet y carry_over.parquet: {len(by_theta)} θ")
PY

run_tiles() {
  docker run --rm -v "$data:/data" \
    -e VIZ_LANDING_ROOT=/data/landing \
    -e VIZ_EVENTS_ROOT=/data/events \
    -e VIZ_TILES_ROOT=/data/tiles \
    -e VIZ_DQ_ROOT=/data/dq \
    "$image" --mode tiles --day 2017-08-18 "$@" 2>&1
}

run_tiles | tee "$data/run.log"
grep -q 'sonda: unit=2017-08-18 ticks=4735 ' "$data/run.log" \
  || { echo "::error::falta la línea sonda del día"; exit 1; }
grep -q '"check_type": "tiles_summary"' "$data/run.log" \
  || { echo "::error::falta el hallazgo tiles_summary en el log"; exit 1; }

# El índice, los 4 objetos con su tamaño y el Parquet de hallazgos.
docker run --rm -v "$data:/data" --entrypoint python "$image" - <<'PY'
import base64
import csv
import gzip
import hashlib
import json
import re
import struct
from pathlib import Path

import pyarrow.parquet as pq

day = Path("/data/tiles/provider=binance/market=spot/asset=BTCUSDT/day=2017-08-18")
index = json.loads((day / "index.json").read_text())
assert index["ticks"] == 4735, index["ticks"]
assert index["tiles_version"] == "2.1.0", index["tiles_version"]
assert index["page"] == "index.html", index["page"]
assert (index["ticks_file"], index["events"]) == ("ticks.bin", "events.bin")
assert index["first_agg_trade_id"] == 3089 and index["last_agg_trade_id"] == 7823, index
assert len(index["thetas"]) == 5 and index["missing_thetas"] == [], index["thetas"]
assert all(t["provisional_from_s"] is not None for t in index["thetas"]), "cola provisional"
files = sorted(p.name for p in day.iterdir())
assert files == ["events.bin", "index.html", "index.json", "ticks.bin"], files
total = sum(t["events"] for t in index["thetas"])
assert (day / index["events"]).stat().st_size == 25 * total, total
assert [t["events_offset"] for t in index["thetas"]][0] == 0


def varints(raw, pos, count):
    """`count` enteros varint (LEB128) de `raw` desde `pos`, y la posición que sigue."""
    out = []
    for _ in range(count):
        value, shift = 0, 0
        while True:
            byte = raw[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if byte < 0x80:
                break
        out.append(value)
    return out, pos


# ticks.bin: tramos (cabecera de cuatro uint32 y tres secciones de varint) que ocupan el
# archivo entero y decodifican a los 4 735 ticks del día (ticks.csv: id, precio y tiempo;
# la cantidad de la prueba es 1). Un día tan corto cabe en un solo tramo.
assert index["ticks_chunk"] == 65_536, index["ticks_chunk"]
raw = (day / "ticks.bin").read_bytes()
dt, dprice, quantity = [], [], []
pos = 0
while pos < len(raw):
    count, *sizes = struct.unpack_from("<4I", raw, pos)
    pos += 16
    assert 0 < count <= index["ticks_chunk"], count
    for column, size in zip((dt, dprice, quantity), sizes):
        values, end = varints(raw, pos, count)
        assert end == pos + size, (end, pos, size)
        column += values
        pos = end
assert pos == len(raw), (pos, len(raw))
assert len(dt) == index["ticks"], len(dt)
assert set(quantity) == {10**8}, set(quantity)
times, prices = [], []
t = p = 0
for a, z in zip(dt, dprice):
    t += a
    p += (z >> 1) ^ -(z & 1)
    times.append(t)
    prices.append(p)
assert times == sorted(times) and 0 <= times[0] and times[-1] < 86_400_000
assert min(prices) > 0

# events.bin: siete secciones (tres int32 de tiempo, tres uint32 de posición de tick y las
# banderas). Los ids del fixture son consecutivos desde el primero del día, así que la
# posición de cada punto es `agg_trade_id − first_agg_trade_id` (TRD-viz §7.5).
events = (day / index["events"]).read_bytes()
tick_sections = struct.unpack_from(f"<{3 * total}I", events, 12 * total)
assert all(p < index["ticks"] for p in tick_sections), "posición fuera de ticks.bin"
with open("/fixtures/events_v0.csv") as f:
    first = next(csv.DictReader(f))
assert int(first["theta"]) == int(index["thetas"][0]["theta"].split(".")[1]), first
for k, name in enumerate(("reference", "confirm", "extreme")):
    wanted = int(first[f"{name}_agg_trade_id"]) - index["first_agg_trade_id"]
    assert tick_sections[k * total] == wanted, (name, tick_sections[k * total], wanted)
latest = json.loads(Path("/data/tiles/latest.json").read_text())
assert latest["day"] == "2017-08-18", latest

# La página: un solo documento con los dos archivos dentro, que reproducen el
# content_hash del índice, y su copia latest.html.
html = (day / "index.html").read_text()
assert (Path("/data/tiles/latest.html")).read_text() == html
assert "@@" not in html and 'name="viz-render"' in html
match = re.search(r"window\.VIZ_DATA=(\{.*\});\n</script>", html, re.S)
data = json.loads(match.group(1))
assert data["index"] == index and data["tiles_version"] == index["tiles_version"]
assert sorted(data["files"]) == ["events.bin", "ticks.bin"], sorted(data["files"])
digest = hashlib.sha256()
for name in sorted(data["files"]):
    raw = base64.b64decode(data["files"][name])
    assert raw == (day / name).read_bytes(), name
    digest.update(name.encode() + b"\0" + raw)
assert digest.hexdigest() == index["content_hash"], "content_hash de la página"
print("página OK:", len(html), "B sin comprimir,", len(gzip.compress(html.encode())), "B en gzip")

rows = [
    row
    for path in Path("/data/dq").rglob("*.parquet")
    for row in pq.read_table(path).to_pylist()
]
(summary,) = [r for r in rows if r["check_type"] == "tiles_summary"]
assert (summary["layer"], summary["mode"], summary["stage"]) == ("viz", "tiles", "canonical")
details = json.loads(summary["details"])
assert details["day"] == "2017-08-18" and details["skipped"] is False, details
assert details["content_hash"] == index["content_hash"]
print("índice y archivos OK, content_hash", index["content_hash"][:16])
PY

# Segunda corrida: el día está al día y no se reescribe. Con --force se rehace
# con el mismo contenido.
first="$(grep -o -m1 '"content_hash": "[0-9a-f]*"' "$data/run.log")"
run_tiles | tee "$data/second.log"
grep -q 'al día' "$data/second.log" || { echo "::error::la segunda corrida debía saltar el día"; exit 1; }
run_tiles --force | tee "$data/forced.log"
[ "$(grep -o -m1 '"content_hash": "[0-9a-f]*"' "$data/forced.log")" = "$first" ] \
  || { echo "::error::--force cambió el content_hash"; exit 1; }

# Modo render: regenera la página desde ticks.bin y events.bin, sin L1 ni L2. Con las mismas
# entradas la página sale idéntica; sin --force, el día está al día y se salta.
run_render() {
  docker run --rm -v "$data:/data" \
    -e VIZ_TILES_ROOT=/data/tiles \
    -e VIZ_DQ_ROOT=/data/dq \
    "$image" --mode render --day 2017-08-18 "$@" 2>&1
}
page="$data/tiles/provider=binance/market=spot/asset=BTCUSDT/day=2017-08-18/index.html"
before="$(sha256sum "$page")"
run_render | tee "$data/render.log"
grep -q 'al día' "$data/render.log" || { echo "::error::render debía saltar la página al día"; exit 1; }
run_render --force | tee "$data/render-forced.log"
[ "$(sha256sum "$page")" = "$before" ] || { echo "::error::render --force cambió la página"; exit 1; }
echo "humo OK: $first"
