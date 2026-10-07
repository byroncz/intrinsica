"""Sonda del tamaño de `ticks.bin`: cuánto pesan los ticks de un día de viz.

La página de un día lleva sus ticks (TRD-viz §7.3, ADR-VZ-14) y su presupuesto es
de 4 MB en gzip el 2026-09-30. Esta sonda lee `ticks.bin` e `index.json` del día y
mide lo que decide ese presupuesto: bytes por tick, bytes de cada una de las tres
secciones, el archivo en gzip (nivel 9, como la página) y en base64 más gzip (como
va dentro de ella), `events.bin` en gzip y el tamaño guardado de `index.html` si
está (con `--dir` suele faltar: el presupuesto queda `sin_página`). No decodifica
valores: cuenta los varint de cada sección contra lo que declara la cabecera de
cada tramo, así que un archivo dañado sale como `FALLO`.

Args (todos opcionales):
  --tiles-root gs://<proyecto>-viz/tiles   (por defecto, el del proyecto; también
                                            una ruta local con la misma disposición)
  --dir <directorio>                        (un directorio con index.json y ticks.bin
                                            sueltos, p. ej. descargados con gcloud)
  --day YYYY-MM-DD                          (por defecto, el de latest.json)
  --provider binance --market spot --asset BTCUSDT
  --page-budget-mb 4                        (la página en gzip debe pesar menos)

Última línea: `sonda ticks: día=… ticks=… bytes=… gzip=… base64_gzip=…
events_gzip=… página=… presupuesto=ok|excedido|sin_página`. Sale con 1 si el archivo está dañado o la página supera
el presupuesto.

Solo lee: no escribe nada. La service account de `ops-script` aún no puede leer el
bucket viz: con `gs://` corre desde Cloud Shell, o descarga el día y usa `--dir`.
"""

import argparse
import base64
import gzip
import json
import os
import re
import struct
import sys
from pathlib import Path

HEADER = struct.Struct("<4I")
SECTIONS = ("dt_ms", "dprice_zigzag", "quantity_1e8")


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


def chunks(raw: bytes, chunk_max: int):
    """Los tramos de `ticks.bin`: `(ticks, [bytes de cada sección])`, validando la forma.

    Lanza `ValueError` si una cabecera se trunca, un tramo declara más de `chunk_max`
    o una sección no tiene exactamente tantos varint como ticks.
    """
    view = memoryview(raw)
    pos = 0
    while pos < len(raw):
        if len(raw) - pos < HEADER.size:
            raise ValueError("cabecera de tramo truncada")
        count, *sizes = HEADER.unpack_from(raw, pos)
        pos += HEADER.size
        if not 0 < count <= chunk_max:
            raise ValueError(f"un tramo de {count} ticks no cabe en {chunk_max}")
        if sum(sizes) > len(raw) - pos:
            raise ValueError("tramo truncado")
        parts = []
        for name, size in zip(SECTIONS, sizes, strict=True):
            section = view[pos : pos + size]
            ends = sum(1 for b in section if b < 0x80)
            if ends != count or (size and section[size - 1] >= 0x80):
                raise ValueError(
                    f"la sección {name} trae {ends} varint y declara {count}"
                )
            pos += size
            parts.append(section)
        yield count, parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles-root")
    parser.add_argument("--dir")
    parser.add_argument("--day")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--page-budget-mb", type=float, default=4.0)
    ns = parser.parse_args(argv)

    if ns.dir:
        store, day = Store(ns.dir), ns.day
    else:
        root = (ns.tiles_root or default_tiles_root()).rstrip("/")
        day = ns.day or json.loads(Store(root).read("latest.json"))["day"]
        series = f"provider={ns.provider}/market={ns.market}/asset={ns.asset}"
        store = Store(f"{root}/{series}/day={day}")
    index = json.loads(store.read("index.json"))
    day = day or index["day"]
    raw = store.read(index["ticks_file"])

    per_section = [bytearray() for _ in SECTIONS]
    ticks = n_chunks = 0
    last_full = True
    try:
        for count, parts in chunks(raw, index["ticks_chunk"]):
            if not last_full:  # un tramo no lleno solo puede ser el último
                raise ValueError("un tramo que no es el último no está lleno")
            last_full = count == index["ticks_chunk"]
            ticks += count
            n_chunks += 1
            for acc, part in zip(per_section, parts, strict=True):
                acc += part
        if ticks != index["ticks"]:
            raise ValueError(f"index.json declara {index['ticks']} ticks y hay {ticks}")
    except ValueError as exc:
        print(f"FALLO ticks_dañado: {exc}")
        print(f"sonda ticks: día={day} dañado")
        return 1

    packed = len(gzip.compress(raw, compresslevel=9, mtime=0))
    encoded = len(gzip.compress(base64.b64encode(raw), compresslevel=9, mtime=0))
    page = store.size(index["page"])
    budget = int(ns.page_budget_mb * 1_000_000)

    print(
        f"día {day}: ticks={ticks} tramos={n_chunks} bytes={len(raw)} "
        f"b_por_tick={len(raw) / ticks:.2f}"
    )
    for name, section in zip(SECTIONS, per_section, strict=True):
        size = len(section)
        packed_section = len(gzip.compress(section, compresslevel=9, mtime=0))
        print(
            f"sección {name}: {size} B ({size / ticks:.2f} B por tick), "
            f"{packed_section} B en gzip"
        )
    print(f"ticks.bin en gzip: {packed} B ({packed / len(raw):.1%} del archivo)")
    print(f"ticks.bin en base64 y gzip (dentro de la página): {encoded} B")
    events = None
    if store.size(index["events"]) is not None:
        events_raw = store.read(index["events"])
        events = len(gzip.compress(events_raw, compresslevel=9, mtime=0))
        print(f"events.bin: {len(events_raw)} B, {events} B en gzip")
    ok = page is None or page <= budget
    status = "sin_página" if page is None else ("ok" if ok else "excedido")
    print(
        f"sonda ticks: día={day} ticks={ticks} bytes={len(raw)} gzip={packed} "
        f"base64_gzip={encoded} events_gzip={events} página={page} "
        f"presupuesto={status}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
