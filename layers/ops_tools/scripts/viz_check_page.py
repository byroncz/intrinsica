"""Herramienta local. Invariantes de un día de viz: `ticks.bin`, `events.bin` e `index.json` cuadran.

Comprueba, sin leer L1 ni L2, lo que el contrato del día promete (TRD-viz §7.2,
§7.3 y §7.5) y que la vista da por hecho:

  1. `index.json` trae todos los campos, la `tiles_version` 2.x y los θ en orden
     con `events_offset` acumulado (`indice_*`, `offset_incorrecto`).
  2. `ticks.bin` decodifica: cada tramo con sus tres secciones exactas, todos
     llenos salvo el último, `ticks` suma lo del índice y el precio es positivo
     (`ticks_dañado`, `ticks_cuenta`, `precio_no_positivo`).
  3. `events.bin` pesa 25 B por evento del índice (`events_tamano`).
  4. Por θ, evento a evento: tiempos en `[0, 86 400 000]` y ordenados
     (`tiempo_fuera_del_dia`, `tiempos_desordenados`); la posición de cada tick
     existe y su tiempo es el del evento, y un punto recortado lleva el centinela y
     al revés (`tick_fuera_de_rango`, `tick_no_coincide`, `recorte_inconsistente`);
     `referencia < confirmación <= extremo` en ticks (`ticks_desordenados`); el
     extremo de un evento es la referencia del siguiente, en tiempo y en tick
     (`cadena_rota`); solo la cola, el último evento de un θ, puede ser provisional
     (`provisional_fuera_de_cola`). No juzga la semántica del detector (sentido,
     umbral, precios): eso es de L2 y de su seam-check.
  5. Si el día es el de `latest.json`: `latest.html` pesa lo que `index.html`
     (`latest_distinto`; ITSC-318 y ITSC-320). Solo con `--tiles-root`.

Args (todos opcionales):
  --tiles-root gs://<proyecto>-viz/tiles   (por defecto, el del proyecto; también
                                            una ruta local con la misma disposición)
  --dir <directorio>                        (un directorio con index.json, ticks.bin y
                                            events.bin sueltos, p. ej. descargados)
  --day YYYY-MM-DD                          (por defecto, el de latest.json)
  --provider binance --market spot --asset BTCUSDT

Imprime hasta 50 líneas `FALLO <código>: <detalle>`. Última línea:
`día AAAA-MM-DD: ticks=…; θ=…; eventos=…; fallos=…`. Sale con 1 si hay algún fallo.

Solo lee. `index.html` y `latest.html` no se leen (van con `Content-Encoding: gzip` y
Arrow los descomprimiría): solo se compara su tamaño guardado.

Dónde corre: `uv run` en local o con `gcloud` desde Cloud Shell. NO corre como
`ops-script`: esa cuenta no tiene permiso sobre el bucket viz (solo `l1/` y `l2/`) y
fallaría con 403. No es `viz_check_day.py` (el de `ops-script`, que valida L2 contra L1).
"""

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

DAY_MS = 86_400_000
OUTSIDE = 0xFFFFFFFF
EVENT_BYTES = 25
HEADER = struct.Struct("<4I")
UP, PROVISIONAL, REF_CLIPPED, CONFIRM_CLIPPED, EXTREME_CLIPPED = 1, 2, 4, 8, 16
INDEX_FIELDS = [
    "tiles_version",
    "provider",
    "market",
    "asset",
    "day",
    "t0",
    "price_scale",
    "ticks",
    "ticks_chunk",
    "first_agg_trade_id",
    "last_agg_trade_id",
    "ticks_file",
    "events",
    "page",
    "thetas",
    "missing_thetas",
    "input_hash",
    "content_hash",
    "generated_at",
    "image_version",
]
MAX_PRINTED = 50


class Store:
    """Lee objetos bajo una raíz local o `gs://`, sin importar pyarrow si es local."""

    def __init__(self, root: str) -> None:
        self.fs = None
        self.base = root.rstrip("/")
        if root.startswith("gs://"):
            import pyarrow.fs as pafs

            self.fs, self.base = pafs.FileSystem.from_uri(root.rstrip("/"))

    def read(self, rel: str) -> bytes:
        path = f"{self.base}/{rel}"
        if self.fs is None:
            return Path(path).read_bytes()
        with self.fs.open_input_stream(path) as stream:
            return stream.read()

    def size(self, rel: str) -> int | None:
        """Bytes guardados del objeto (comprimidos si lleva `Content-Encoding: gzip`)."""
        path = f"{self.base}/{rel}"
        if self.fs is None:
            p = Path(path)
            return p.stat().st_size if p.is_file() else None
        import pyarrow.fs as pafs

        info = self.fs.get_file_info(path)
        return info.size if info.type == pafs.FileType.File else None


def default_tiles_root() -> str:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit(
            "sin --tiles-root ni --dir y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    return f"gs://{match[1]}-viz/tiles"


def varints(raw: memoryview, start: int, size: int, count: int) -> list[int]:
    """Los `count` enteros varint (LEB128) de `raw[start:start + size]`, exactos."""
    out: list[int] = []
    value = shift = 0
    for byte in raw[start : start + size]:
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            out.append(value)
            value = shift = 0
        else:
            shift += 7
            if shift > 63:
                raise ValueError("un varint de más de 10 bytes")
    if shift or len(out) != count:
        raise ValueError(f"se esperaban {count} varint y hay {len(out)}")
    return out


def decode_ticks(raw: bytes, chunk_max: int) -> tuple[list[int], list[int]]:
    """`(tiempo en ms, precio)` de cada tick de `ticks.bin`, o `ValueError` si no cuadra."""
    view = memoryview(raw)
    time_ms: list[int] = []
    price: list[int] = []
    last_time = last_price = pos = 0
    short = False
    while pos < len(raw):
        if short:
            raise ValueError("un tramo que no es el último no está lleno")
        if len(raw) - pos < HEADER.size:
            raise ValueError("cabecera de tramo truncada")
        count, *sizes = HEADER.unpack_from(raw, pos)
        pos += HEADER.size
        if not 0 < count <= chunk_max:
            raise ValueError(f"un tramo de {count} ticks no cabe en {chunk_max}")
        if sum(sizes) > len(raw) - pos:
            raise ValueError("tramo truncado")
        short = count < chunk_max
        dt = varints(view, pos, sizes[0], count)
        dp = varints(view, pos + sizes[0], sizes[1], count)
        varints(view, pos + sizes[0] + sizes[1], sizes[2], count)  # cantidades
        pos += sum(sizes)
        for a, z in zip(dt, dp, strict=True):
            last_time += a
            last_price += (z >> 1) ^ -(z & 1)
            time_ms.append(last_time)
            price.append(last_price)
    return time_ms, price


class Report:
    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []

    def fail(self, code: str, detail: str) -> None:
        self.failures.append((code, detail))


def check_index(index: dict, report: Report) -> bool:
    """Los campos, la versión y los θ del índice. `False` si no se puede seguir."""
    missing = [f for f in INDEX_FIELDS if f not in index]
    if missing:
        report.fail("indice_incompleto", f"faltan {', '.join(missing)}")
        return False
    if not str(index["tiles_version"]).startswith("2."):
        report.fail(
            "indice_version", f"tiles_version {index['tiles_version']}, se esperaba 2.x"
        )
        return False
    names = [t["theta"] for t in index["thetas"]]
    if names != sorted(names) or len(set(names)) != len(names):
        report.fail("indice_thetas", "los θ no están en orden creciente y únicos")
    offset = 0
    for t in index["thetas"]:
        if t["events_offset"] != offset:
            report.fail(
                "offset_incorrecto",
                f"θ={t['theta']}: events_offset {t['events_offset']}, se esperaba {offset}",
            )
        offset += t["events"]
    return True


def check_theta(
    theta: dict,
    cols: dict[str, tuple],
    ticks_time: list[int],
    report: Report,
) -> None:
    """Los invariantes de los eventos de un θ (`cols`: las siete secciones de `events.bin`)."""
    lo = theta["events_offset"]
    name = theta["theta"]
    n_ticks = len(ticks_time)
    previous = None  # (tiempo y tick del extremo) del evento anterior
    last = lo + theta["events"] - 1
    for i in range(lo, last + 1):
        times = [cols[k][i] for k in ("reference", "confirm", "extreme")]
        pos = [cols[k][i] for k in ("reference_tick", "confirm_tick", "extreme_tick")]
        flags = cols["flags"][i]
        clipped = [
            bool(flags & f) for f in (REF_CLIPPED, CONFIRM_CLIPPED, EXTREME_CLIPPED)
        ]
        where = f"θ={name} evento {i - lo + 1}/{theta['events']}"
        if any(not 0 <= t <= DAY_MS for t in times):
            report.fail("tiempo_fuera_del_dia", f"{where}: {times}")
        if not times[0] <= times[1] <= times[2]:
            report.fail("tiempos_desordenados", f"{where}: {times}")
        for point, t, p, cut in zip(
            ("referencia", "confirmación", "extremo"), times, pos, clipped, strict=True
        ):
            if cut != (p == OUTSIDE):
                report.fail(
                    "recorte_inconsistente",
                    f"{where}: {point} recortado={cut}, tick={p}",
                )
            elif p != OUTSIDE:
                if p >= n_ticks:
                    report.fail(
                        "tick_fuera_de_rango",
                        f"{where}: {point} en el tick {p} de {n_ticks}",
                    )
                elif ticks_time[p] != t:
                    report.fail(
                        "tick_no_coincide",
                        f"{where}: {point} a los {t} ms, el tick {p} está a los {ticks_time[p]} ms",
                    )
        if all(p != OUTSIDE and p < n_ticks for p in pos):
            ref, confirm, extreme = pos
            if not ref < confirm <= extreme:
                report.fail("ticks_desordenados", f"{where}: ticks {pos}")
        if flags & PROVISIONAL and i != last:
            report.fail("provisional_fuera_de_cola", where)
        if previous is not None:
            (p_time, p_tick) = previous
            if (times[0], pos[0]) != (p_time, p_tick):
                report.fail(
                    "cadena_rota",
                    f"{where}: la referencia ({times[0]} ms, tick {pos[0]}) no es el extremo "
                    f"del anterior ({p_time} ms, tick {p_tick})",
                )
        previous = (times[2], pos[2])


def check_latest(root: Store, day: str, series: str, report: Report) -> str:
    """`latest.html` pesa lo mismo que el `index.html` del día al que apunta `latest.json`."""
    latest = json.loads(root.read("latest.json"))
    if latest["day"] != day:
        return f"latest apunta a {latest['day']}: no se compara con {day}"
    page = root.size(f"{series}/day={day}/index.html")
    copy = root.size("latest.html")
    if page is None or copy is None or page != copy:
        report.fail(
            "latest_distinto", f"latest.html={copy} B, index.html del día={page} B"
        )
        return "latest distinto"
    return f"latest.html = index.html del {day}: {copy} B"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles-root")
    parser.add_argument("--dir")
    parser.add_argument("--day")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    ns = parser.parse_args(argv)

    root = None
    if ns.dir:
        store, day = Store(ns.dir), ns.day
    else:
        base = (ns.tiles_root or default_tiles_root()).rstrip("/")
        root = Store(base)
        day = ns.day or json.loads(root.read("latest.json"))["day"]
        series = f"provider={ns.provider}/market={ns.market}/asset={ns.asset}"
        store = Store(f"{base}/{series}/day={day}")
    index = json.loads(store.read("index.json"))
    day = day or index.get("day", "?")

    report = Report()
    ticks = events = 0
    if check_index(index, report):
        try:
            time_ms, price = decode_ticks(
                store.read(index["ticks_file"]), index["ticks_chunk"]
            )
        except ValueError as exc:
            report.fail("ticks_dañado", str(exc))
            time_ms = price = []
        ticks = len(time_ms)
        if time_ms:
            if ticks != index["ticks"]:
                report.fail(
                    "ticks_cuenta", f"index.json declara {index['ticks']} y hay {ticks}"
                )
            if min(price) <= 0:
                report.fail("precio_no_positivo", f"precio mínimo {min(price)}")
            if time_ms[-1] >= DAY_MS:
                report.fail(
                    "tiempo_fuera_del_dia",
                    f"el último tick está a los {time_ms[-1]} ms",
                )
        events = sum(t["events"] for t in index["thetas"])
        raw = store.read(index["events"])
        if len(raw) != EVENT_BYTES * events:
            report.fail(
                "events_tamano",
                f"events.bin pesa {len(raw)} B, se esperaban {EVENT_BYTES * events}",
            )
        elif time_ms:
            names = ("reference", "confirm", "extreme")
            cols = {}
            for k, name in enumerate(names):
                cols[name] = struct.unpack_from(f"<{events}i", raw, 4 * events * k)
                cols[f"{name}_tick"] = struct.unpack_from(
                    f"<{events}I", raw, 4 * events * (k + 3)
                )
            cols["flags"] = raw[24 * events :]
            for theta in index["thetas"]:
                check_theta(theta, cols, time_ms, report)

    note = (
        check_latest(
            root,
            day,
            f"provider={ns.provider}/market={ns.market}/asset={ns.asset}",
            report,
        )
        if root
        else None
    )
    for code, detail in report.failures[:MAX_PRINTED]:
        print(f"FALLO {code}: {detail}")
    if len(report.failures) > MAX_PRINTED:
        print(f"... y {len(report.failures) - MAX_PRINTED} fallos más")
    if note:
        print(note)
    print(
        f"día {day}: ticks={ticks}; θ={len(index.get('thetas', []))}; "
        f"eventos={events}; fallos={len(report.failures)}"
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
