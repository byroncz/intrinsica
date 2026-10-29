"""Sonda de los huecos entre ticks de L1 sobre un mes (ITSC-392).

Mide la diferencia entre `transact_time` consecutivos del `consolidated.parquet` de un mes,
incluidos los huecos entre un row group y el siguiente, y deja un CSV con ticks, huecos,
cuantiles (p50, p90, p99, p99,9 y p99,99), máximo y cuántos huecos pasan de 1 s, 10 s, 60 s
y 600 s. Con las cifras de varios meses se fija `silence_us`, el umbral de silencio de la
familia T de L3.

Cuantil: rango más cercano. Con N huecos, el cuantil q es el hueco que ocupa el lugar
⌈N·q⌉ al ordenarlos de menor a mayor. Es exacto, no aproximado, y siempre es un hueco real.

Args:
  --month YYYY-MM                                  (obligatorio)
  --landing-root gs://<proyecto>-landing/l1        (por defecto, el del proyecto según
                                                    OPS_RESULTS_URI; también una ruta local con
                                                    la misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_gap_probe_<mes>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

No importa ninguna capa: una importación cambiaría `uv.lock`. Abre el `consolidated.parquet`
del mes por su ruta completa, nunca un dataset sobre un prefijo, que pide un objeto sin barra
y la condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3,
«La trampa del 403 por prefijo»).

Memoria: lee un row group a la vez y solo `transact_time`. Los huecos de hasta 1 s caen en un
histograma exacto de una cuenta por µs (10⁶ + 1 enteros, 8 MB) y los mayores se guardan uno a
uno, que son pocos. El pico es O(row group), no O(mes). Al final lee `agg_trade_id` de un solo
row group, el del tick que cierra el hueco máximo.

Salida: la tabla de medidas, la ruta del CSV y, como última línea, `gap probe: mes=… ticks=…
p50_us=… p90_us=… p99_us=… p999_us=… p9999_us=… max_us=… ge_1s=… ge_10s=… ge_60s=… ge_600s=…
pared_s=… veredicto=ok|fallo`.

Códigos: 0 si midió; 1 sin `consolidated.parquet`, si la lectura falla o si `transact_time`
decrece; 2 por uso. Solo lee; lo único que escribe es el CSV de `--out`.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

SMALL_US = (
    1_000_000  # los huecos de hasta 1 s van al histograma; los mayores, uno a uno
)
QUANTILES = (
    ("p50_us", 50, 100),
    ("p90_us", 90, 100),
    ("p99_us", 99, 100),
    ("p999_us", 999, 1000),
    ("p9999_us", 9999, 10000),
)
THRESHOLDS = (
    ("ge_1s", 1_000_000),
    ("ge_10s", 10_000_000),
    ("ge_60s", 60_000_000),
    ("ge_600s", 600_000_000),
)
CSV_HEADER = ["medida", "valor", "legible", "agg_trade_id", "transact_time"]


def log(message: str) -> None:
    print(message, flush=True)


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        return pafs.FileSystem.from_uri(uri)
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def human(us: int) -> str:
    """Un intervalo en µs como segundos, minutos u horas."""
    seconds = us / 1_000_000
    if seconds < 60:
        return f"{seconds:.6f}".rstrip("0").rstrip(".") + " s"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} h"


def utc(us: int) -> str:
    moment = datetime.fromtimestamp(us // 1_000_000, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S") + f".{us % 1_000_000:06d}Z"


class Gaps:
    """Los huecos del mes, acumulados un row group a la vez."""

    def __init__(self) -> None:
        self.ticks = 0
        self.small = np.zeros(SMALL_US + 1, dtype=np.int64)
        self.large: list[np.ndarray] = []
        self.previous: int | None = None
        self.max_gap = -1
        self.max_group = -1
        self.max_row = -1
        self.max_time = -1

    def add(self, group: int, times: np.ndarray) -> None:
        """Suma un row group: `times` son sus `transact_time`, ya ordenados."""
        if not len(times):
            return
        self.ticks += len(times)
        if self.previous is None:
            gaps, first = (
                np.diff(times),
                1,
            )  # el primer tick del mes no cierra ningún hueco
        else:
            gaps, first = np.diff(times, prepend=self.previous), 0
        self.previous = int(times[-1])
        if not len(gaps):
            return
        if int(gaps.min()) < 0:
            raise ValueError(
                f"row group {group}: transact_time decrece; L1 debe venir ordenado"
            )
        row = int(np.argmax(gaps))  # el primero, si el máximo se repite
        if int(gaps[row]) > self.max_gap:
            self.max_gap = int(gaps[row])
            self.max_group = group
            self.max_row = row + first
            self.max_time = int(times[row + first])
        is_small = gaps <= SMALL_US
        self.small += np.bincount(gaps[is_small], minlength=SMALL_US + 1)
        self.large.append(gaps[~is_small])

    @property
    def count(self) -> int:
        return max(self.ticks - 1, 0)

    def summary(self) -> dict:
        """Cuantiles, máximo y conteos exactos; libera el histograma al terminar."""
        large = np.sort(np.concatenate(self.large)) if self.large else np.empty(0, int)
        self.large = []
        n_small = self.count - len(large)
        cumulative = np.cumsum(self.small)
        result: dict = {"ticks": self.ticks, "huecos": self.count}
        for name, numerator, denominator in QUANTILES:
            if not self.count:
                result[name] = None
                continue
            rank = -(-self.count * numerator // denominator)  # ⌈N·q⌉ sin flotantes
            if rank <= n_small:
                result[name] = int(np.searchsorted(cumulative, rank, side="left"))
            else:
                result[name] = int(large[rank - n_small - 1])
        result["max_us"] = self.max_gap if self.count else None
        at_one_second = int(self.small[SMALL_US])
        for name, threshold in THRESHOLDS:
            if not self.count:
                result[name] = 0
            elif threshold == SMALL_US:
                result[name] = at_one_second + len(large)
            else:
                result[name] = int(len(large) - np.searchsorted(large, threshold))
        self.small = np.empty(0, dtype=np.int64)
        return result


def measure(path: str) -> tuple[dict, int | None]:
    """Recorre el archivo; devuelve el resumen y el `agg_trade_id` del tick del máximo."""
    fs, resolved = open_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        raise FileNotFoundError(path)
    gaps = Gaps()
    with fs.open_input_file(resolved) as source:
        parquet = pq.ParquetFile(source)
        for group in range(parquet.num_row_groups):
            column = parquet.read_row_group(group, columns=["transact_time"]).column(0)
            if not pa.types.is_int64(column.type) or column.null_count:
                raise ValueError("transact_time debe ser int64 sin nulos")
            times = column.to_numpy()
            del column
            gaps.add(group, times)
            del times
        trade_id = None
        if gaps.count:
            ids = parquet.read_row_group(gaps.max_group, columns=["agg_trade_id"])
            trade_id = ids.column(0)[gaps.max_row].as_py()
    return gaps.summary() | {"max_time": gaps.max_time}, trade_id


def csv_rows(summary: dict, trade_id: int | None) -> list[dict]:
    def blank(value):
        return "" if value is None else value

    rows = [
        {"medida": "ticks", "valor": summary["ticks"]},
        {"medida": "huecos", "valor": summary["huecos"]},
    ]
    for name in [q[0] for q in QUANTILES] + ["max_us"]:
        value = summary[name]
        row = {
            "medida": name,
            "valor": blank(value),
            "legible": "" if value is None else human(value),
        }
        if name == "max_us" and value is not None:
            row["agg_trade_id"] = blank(trade_id)
            row["transact_time"] = summary["max_time"]
        rows.append(row)
    rows += [{"medida": name, "valor": summary[name]} for name, _ in THRESHOLDS]
    return rows


def write_csv(rows: list[dict], out: str) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if out.startswith("gs://"):
        fs, path = pafs.FileSystem.from_uri(out)
        with fs.open_output_stream(path) as stream:
            stream.write(data)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)


def final_line(month: str, wall: float, verdict: str, summary: dict | None) -> str:
    """Las medidas de la última línea; con `fallo` salen `-`."""
    names = ["ticks"] + [q[0] for q in QUANTILES] + ["max_us"]
    names += [t[0] for t in THRESHOLDS]
    values = {n: "-" if summary is None else summary[n] for n in names}
    cells = " ".join(f"{n}={'-' if v is None else v}" for n, v in values.items())
    return f"gap probe: mes={month} {cells} pared_s={wall:.1f} veredicto={verdict}"


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    month = re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month)
    if month is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    name = f"l3_gap_probe_{ns.month}.csv"
    name_project = project()
    if not ns.landing_root and name_project is None:
        parser.error(
            "sin --landing-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    root = ns.landing_root or f"gs://{name_project}-landing/l1"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))
    path = (
        f"{root.rstrip('/')}/provider={ns.provider}/market={ns.market}/asset={ns.asset}"
        f"/year={month[1]}/month={month[2]}/consolidated.parquet"
    )

    started = time.perf_counter()
    log(f"gap probe: mes={ns.month} archivo={path}")
    try:
        summary, trade_id = measure(path)
    except FileNotFoundError:
        log(f"FALLO falta el archivo {path}")
    except (OSError, ValueError, pa.ArrowException) as exc:
        log(f"FALLO no se pudo leer {path}: {type(exc).__name__}: {exc}")
    else:
        wall = time.perf_counter() - started
        rows = csv_rows(summary, trade_id)
        log(f"{'medida':<12} {'valor':>16}  legible")
        for row in rows:
            log(f"{row['medida']:<12} {row['valor']!s:>16}  {row.get('legible', '')}")
        if summary["huecos"]:
            log(
                f"el hueco máximo cierra en agg_trade_id={trade_id} "
                f"transact_time={summary['max_time']} ({utc(summary['max_time'])})"
            )
        try:
            write_csv(rows, out)
            log(f"tabla en {out}")
        except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
            log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
        log(final_line(ns.month, wall, "ok", summary))
        return 0
    log(final_line(ns.month, time.perf_counter() - started, "fallo", None))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", required=True)
    parser.add_argument("--landing-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
