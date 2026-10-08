"""Sonda de empates: cuántos `transact_time` distintos hay dentro de un milisegundo de L1.

L2 agrupa los ticks por `transact_time` en µs (ADR-L2-03) y la página de viz por ms
(TRD-viz §7.5). El 2026-09-30 12:40:26.980 hay 4 090 ticks en un mismo milisegundo; de
si comparten un µs o solo el ms depende cuántos quedan en la confirmación de un evento y
cuántos en su overshoot. Esta sonda lo mide sobre `consolidated.parquet` y `events.parquet`:

  1. Los `transact_time` distintos del milisegundo, con cuántos ticks tiene cada uno y su
     rango de `agg_trade_id`.
  2. Para cada θ que confirma un evento en ese milisegundo (`confirm_time` dentro de él),
     su grupo de empate: los ticks de L1 con `transact_time = confirm_time`, cuántos están
     en `agg_trade_id ≤ C` (el tick de confirmación y los anteriores) y cuántos después
     de `C`, y de esos cuántos caen en el overshoot `(C, E]` del evento.

Args (todos opcionales):
  --at 2026-09-30T12:40:26.980        (el milisegundo, en UTC; el mes sale de aquí)
  --l1-root gs://<proyecto>-landing/l1    (por defecto, los del proyecto según
  --l2-root gs://<proyecto>-dc-events/l2   OPS_RESULTS_URI; también rutas locales)
  --provider binance --market spot --asset BTCUSDT
  --expected-ticks 4090               (los ticks del milisegundo que dice la card)
  --max-listed 40                     (cuántos `transact_time` distintos se listan)

Última línea: `sonda empates: ms=… ticks=… distintos=… max_por_tt=… thetas=… max_empate=…`.
Sale con 1 si el milisegundo no tiene ticks o falta algún archivo. Solo lee: dónde corre,
con `ops-script` (lee `l1/` y `l2/`) o en local con `uv run`.
"""

import argparse
import os
import re
import sys
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyutils import resolve_fs

THETA_DIR = re.compile(r"theta=(0\.\d{8})")


def default_roots() -> tuple[str, str]:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit(
            "sin --l1-root o --l2-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    return f"gs://{match[1]}-landing/l1", f"gs://{match[1]}-dc-events/l2"


def micros(moment: datetime) -> int:
    delta = moment - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def render(microseconds: int) -> str:
    moment = datetime.fromtimestamp(microseconds // 1_000_000, UTC)
    return f"{moment:%Y-%m-%d %H:%M:%S}.{microseconds % 1_000_000:06d}"


def open_file(path: str) -> pq.ParquetFile | None:
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.File:
        return None
    return pq.ParquetFile(resolved, filesystem=fs)


def read_window(
    parquet: pq.ParquetFile, column: str, columns: list[str], low: int, high: int
):
    """Las filas con `low ≤ column < high`, leyendo solo los row groups que lo permiten.

    Un row group se salta si sus estadísticas de `column` no tocan la ventana; sin
    estadísticas se lee, porque no se puede descartar.
    """
    index = parquet.schema_arrow.get_field_index(column)
    tables = []
    for i in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(i)
        stats = group.column(index).statistics
        if group.num_rows == 0 or (
            stats is not None
            and stats.has_min_max
            and (stats.max < low or stats.min >= high)
        ):
            continue
        table = parquet.read_row_group(i, columns=columns)
        mask = (table[column].to_numpy() >= low) & (table[column].to_numpy() < high)
        tables.append(table.filter(pa.array(mask)))
    return pa.concat_tables(tables) if tables else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--at", default="2026-09-30T12:40:26.980")
    parser.add_argument("--l1-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--expected-ticks", type=int, default=4090)
    parser.add_argument("--max-listed", type=int, default=40)
    ns = parser.parse_args(argv)

    moment = datetime.fromisoformat(ns.at).replace(tzinfo=UTC)
    low = micros(moment)
    low -= low % 1000
    high = low + 1000
    default_l1, default_l2 = (
        (None, None) if ns.l1_root and ns.l2_root else default_roots()
    )
    l1_root = ns.l1_root or default_l1
    l2_root = ns.l2_root or default_l2
    series = f"provider={ns.provider}/market={ns.market}/asset={ns.asset}"
    year_month = f"year={moment.year:04d}/month={moment.month:02d}"

    l1_path = f"{l1_root.rstrip('/')}/{series}/{year_month}/consolidated.parquet"
    l1 = open_file(l1_path)
    if l1 is None:
        print(f"FALLO l1_faltante: {l1_path}")
        return 1
    ticks = read_window(
        l1, "transact_time", ["agg_trade_id", "transact_time"], low, high
    )
    l1.close()
    if ticks is None or ticks.num_rows == 0:
        print(f"FALLO sin_ticks: ningún tick en el ms {render(low)[:-3]} de {l1_path}")
        return 1
    ids = ticks["agg_trade_id"].to_numpy()
    times = ticks["transact_time"].to_numpy()
    distinct, counts = np.unique(times, return_counts=True)

    print(
        f"ms {render(low)[:-3]} UTC: ticks={len(times)} (la card dice {ns.expected_ticks}), "
        f"transact_time distintos={len(distinct)}"
    )
    print(f"{'transact_time':>28} {'ticks':>7} {'id_mín':>14} {'id_máx':>14}")
    for value, count in list(zip(distinct, counts, strict=True))[: ns.max_listed]:
        picked = ids[times == value]
        print(
            f"{render(int(value)):>28} {count:>7} {picked.min():>14} {picked.max():>14}"
        )
    if len(distinct) > ns.max_listed:
        print(f"  … {len(distinct) - ns.max_listed} más sin listar")
    histogram = dict(zip(*np.unique(counts, return_counts=True), strict=True))
    print(
        "ticks por transact_time -> cuántos transact_time: "
        + ", ".join(f"{k}→{v}" for k, v in sorted(histogram.items()))
    )

    fs, base = resolve_fs(l2_root)
    root = f"{base.rstrip('/')}/{series}"
    thetas = sorted(
        m[1]
        for info in fs.get_file_info(pafs.FileSelector(root))
        if (m := THETA_DIR.fullmatch(info.base_name)) is not None
    )
    print(
        f"{'θ':>12} {'confirm_time':>28} {'C':>14} {'grupo_empate':>13} {'hasta_C':>8} "
        f"{'después_de_C':>13} {'en_overshoot':>13}"
    )
    columns = ["confirm_time", "confirm_agg_trade_id", "extreme_agg_trade_id"]
    confirming = 0
    biggest = 0
    missing = []
    for theta in thetas:
        path = (
            f"{l2_root.rstrip('/')}/{series}/theta={theta}/{year_month}/events.parquet"
        )
        events = open_file(path)
        if events is None:
            missing.append(theta)
            continue
        rows = read_window(events, "confirm_time", columns, low, high)
        events.close()
        if rows is None:
            continue
        for confirm_time, confirm_id, extreme_id in zip(
            rows["confirm_time"].to_pylist(),
            rows["confirm_agg_trade_id"].to_pylist(),
            rows["extreme_agg_trade_id"].to_pylist(),
            strict=True,
        ):
            confirming += 1
            tie = ids[times == confirm_time]
            until = int(np.count_nonzero(tie <= confirm_id))
            after = len(tie) - until
            overshoot = int(np.count_nonzero((tie > confirm_id) & (tie <= extreme_id)))
            biggest = max(biggest, len(tie))
            print(
                f"{theta:>12} {render(confirm_time):>28} {confirm_id:>14} {len(tie):>13} "
                f"{until:>8} {after:>13} {overshoot:>13}"
            )
    if missing:
        print(f"θ sin events.parquet del mes (no se midieron): {', '.join(missing)}")
    print(
        f"sonda empates: ms={render(low)[:-3]} ticks={len(times)} distintos={len(distinct)} "
        f"max_por_tt={int(counts.max())} thetas={confirming} max_empate={biggest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
