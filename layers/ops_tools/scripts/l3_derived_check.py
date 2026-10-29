"""Sonda de la vista `derived` (familia D de L3) sobre un θ y un mes del lago (ITSC-386).

Ejecuta `derived.sql` en DuckDB embebido sobre el `summaries.parquet` de L3 y el
`events.parquet` de L2 de una sola partición (un θ y un mes) y deja un CSV con:
  - por cada una de las 16 columnas DOUBLE de D: filas, nulos, mínimo, mediana y máximo;
  - por cada una de las 3 banderas: filas y cuántas son verdaderas;
  - por cada invariante: cuántas filas lo violan (0 es lo esperado).

Invariantes: filas de la vista = filas de `summaries`; unión 1:1 por
`(theta, confirm_agg_trade_id)` (sin filas huérfanas ni claves repetidas);
|c_flow_imbalance| ≤ 1; |os_flow_imbalance| ≤ 1; c_range ≥ 0; c_instant_bias ≥ 0 (las
desigualdades con tolerancia 1e-9 por el redondeo de DOUBLE); la cota de confirmación, que
sale de `summaries` unido a `events` porque D no trae `c_price_max` ni `c_price_min`: en un
upturn (c_price_max − reference_price) / reference_price ≥ θ y en un downturn
(reference_price − c_price_min) / reference_price ≥ θ; atomic_sweep ⇔ c_duration = 0 ⇔
c_tick_rate nulo; empty_overshoot ⇔ todos los os_* de D nulos; banderas sin nulos; columnas
DOUBLE sin ±inf ni NaN; y sin nulos en las columnas que D declara «nunca nulas». `c_range ≥ θ`
no es invariante: el tick de referencia no pertenece a la fase (TRD-L3 §7.1, consecuencia 3).

Args:
  --theta 0.00100000  --month YYYY-MM          (obligatorios; un θ y un mes por corrida)
  --summaries-root gs://<proyecto>-dc-descriptors/l3   (por defecto, los del proyecto según
  --events-root gs://<proyecto>-dc-events/l2            OPS_RESULTS_URI; también rutas locales
                                                        con la misma disposición hive)
  --sql <ruta o gs://…/derived.sql>   (por defecto, con ops-script
                                       gs://<proyecto>-ops/scripts/derived.sql y, en local,
                                       shared/dc_views/src/dc_views/derived.sql del repo)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_derived_check_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

`derived_version` sale de la cabecera del `.sql` (`-- derived_version: X.Y.Z`). No importa
`dc_views` ni ninguna capa: la imagen de `ops_tools` no trae `dc_views` y una importación
cambiaría `uv.lock`; por eso el `.sql` se sube aparte a `scripts/`.

Abre los dos archivos de la partición por su ruta completa y los registra en DuckDB como
datasets de PyArrow de un solo archivo: nunca un dataset sobre un prefijo, que pide el objeto
`l3` sin barra y la condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de
operación de L3, «La trampa del 403 por prefijo»). Todas las cuentas son consultas de
agregación en DuckDB: ninguna fila pasa por Python. El pico es O(partición): las medianas
retienen los valores de las 16 columnas.

Salida: la tabla de invariantes y banderas, la ruta del CSV y, como última línea,
`derived check: mes=… theta=… derived_version=… filas=… atomic_sweep=… empty_overshoot=…
provider_gap=… violaciones=… pared_s=… veredicto=ok|violaciones|fallo`.

Códigos: 0 sin violaciones; 1 con violaciones, sin alguno de los dos archivos o si DuckDB
falla; 2 por uso. Solo lee; lo único que escribe es el CSV de `--out`.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import duckdb
import pyarrow.dataset as ds
import pyarrow.fs as pafs

TOL = 1e-9
DOUBLES = (
    "c_instant_bias",
    "c_flow_imbalance",
    "os_flow_imbalance",
    "c_tick_rate",
    "os_tick_rate",
    "c_quantity_rate",
    "os_quantity_rate",
    "c_mean_trade",
    "os_mean_trade",
    "c_range",
    "os_range",
    "c_vwap_distance",
    "os_vwap_distance",
    "c_speed",
    "os_speed",
    "ev_speed",
)
FLAGS = ("atomic_sweep", "empty_overshoot", "provider_gap")
OS_COLUMNS = (
    "os_flow_imbalance",
    "os_tick_rate",
    "os_quantity_rate",
    "os_mean_trade",
    "os_range",
    "os_vwap_distance",
    "os_speed",
)
# Las columnas que la tabla de «Vista derived» de data-contracts.md marca «nunca» nulas.
NEVER_NULL = (
    "c_instant_bias",
    "c_flow_imbalance",
    "c_mean_trade",
    "c_range",
    "c_vwap_distance",
)
KEY = "(theta, confirm_agg_trade_id)"
SQL_IN_REPO = "shared/dc_views/src/dc_views/derived.sql"
CSV_HEADER = [
    "seccion",
    "columna",
    "filas",
    "nulos",
    "min",
    "mediana",
    "max",
    "verdaderas",
    "violaciones",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_theta(text: str) -> str:
    try:
        theta = Decimal(text)
        if not theta.is_finite() or theta <= 0:
            raise ValueError(f"--theta: {text!r} debe ser un número finito mayor que 0")
        # 1e30 no cabe en la precisión del contexto: también lanza InvalidOperation.
        scaled = theta.quantize(Decimal("1e-8"))
    except InvalidOperation:
        raise ValueError(f"--theta: {text!r} no es un θ válido") from None
    if scaled != theta:
        raise ValueError(f"θ {text} no cabe en 8 decimales, que es la partición")
    return format(scaled, "f")


def project() -> str | None:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    return match[1] if match else None


def partition_file(
    root: str, key: tuple[str, str, str], theta: str, month: str, name: str
) -> str:
    provider, market, asset = key
    year, number = month.split("-")
    return (
        f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/theta={theta}/year={year}/month={number}/{name}"
    )


def open_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """El sistema de archivos y la ruta de un `gs://` o de una ruta local."""
    if uri.startswith("gs://"):
        return pafs.FileSystem.from_uri(uri)
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def open_dataset(uri: str) -> ds.Dataset:
    """Un dataset de PyArrow de ese solo archivo; `FileNotFoundError` si no existe."""
    fs, path = open_fs(uri)
    if fs.get_file_info(path).type != pafs.FileType.File:
        raise FileNotFoundError(uri)
    return ds.dataset(path, filesystem=fs, format="parquet")


def read_sql(source: str) -> str:
    fs, path = open_fs(source)
    with fs.open_input_stream(path) as stream:
        return stream.read().decode("utf-8")


def derived_version(sql: str) -> str:
    match = re.match(r"--\s*derived_version:\s*(\S+)", sql)
    if match is None:
        raise ValueError(
            "la primera línea del .sql debe ser `-- derived_version: X.Y.Z`"
        )
    return match[1]


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def stats_query() -> str:
    """Una pasada sobre `derived` unida a `summaries`: estadísticas e invariantes de D."""
    items = ["count(*) AS filas"]
    for c in DOUBLES:
        items += [
            f"count({c}) AS {c}__n",
            f"min({c}) AS {c}__min",
            f"median({c}) AS {c}__med",
            f"max({c}) AS {c}__max",
            f"count(*) FILTER (WHERE NOT isfinite({c})) AS {c}__nofinito",
        ]
    for f in FLAGS:
        items += [
            f"count({f}) AS {f}__n",
            f"count(*) FILTER (WHERE {f}) AS {f}__true",
        ]
    all_os_null = " AND ".join(f"{c} IS NULL" for c in OS_COLUMNS)
    items += [
        f"count(*) FILTER (WHERE abs(c_flow_imbalance) > 1 + {TOL}) AS v_c_flow",
        f"count(*) FILTER (WHERE abs(os_flow_imbalance) > 1 + {TOL}) AS v_os_flow",
        f"count(*) FILTER (WHERE c_range < -{TOL}) AS v_c_range",
        f"count(*) FILTER (WHERE c_instant_bias < -{TOL}) AS v_c_bias",
        "count(*) FILTER (WHERE atomic_sweep IS DISTINCT FROM (c_duration = 0)) "
        "AS v_atomic_duration",
        "count(*) FILTER (WHERE atomic_sweep IS DISTINCT FROM (c_tick_rate IS NULL)) "
        "AS v_atomic_rate",
        f"count(*) FILTER (WHERE empty_overshoot IS DISTINCT FROM ({all_os_null})) "
        "AS v_empty",
        "count(*) FILTER (WHERE "
        + " OR ".join(f"{f} IS NULL" for f in FLAGS)
        + ") AS v_flags_null",
        "count(*) FILTER (WHERE "
        + " OR ".join(f"{c} IS NULL" for c in NEVER_NULL)
        + ") AS v_never_null",
    ]
    return (
        f"SELECT {', '.join(items)} FROM derived AS d JOIN summaries AS s USING {KEY}"
    )


# D no trae c_price_max ni c_price_min: la cota sale de summaries unido a events. La unión
# es completa para contar también las filas sin pareja de cada lado.
BOUND_QUERY = f"""
SELECT
    count(*) FILTER (WHERE e.reference_price IS NULL) AS sin_events,
    count(*) FILTER (WHERE s.c_ticks IS NULL) AS sin_summaries,
    count(*) FILTER (WHERE e.direction = 1 AND
        (s.c_price_max::DOUBLE - e.reference_price::DOUBLE) / e.reference_price::DOUBLE
            < theta::DOUBLE * (1 - {TOL})) AS v_up,
    count(*) FILTER (WHERE e.direction = -1 AND
        (e.reference_price::DOUBLE - s.c_price_min::DOUBLE) / e.reference_price::DOUBLE
            < theta::DOUBLE * (1 - {TOL})) AS v_down
FROM summaries AS s FULL JOIN events AS e USING {KEY}
"""


def key_counts(con: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int]:
    """Filas de la tabla y filas que sobran por claves repetidas."""
    rows, distinct = con.execute(
        f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT theta, "
        f"confirm_agg_trade_id FROM {table})) FROM {table}"
    ).fetchone()
    return rows, rows - distinct


def measure(con: duckdb.DuckDBPyConnection) -> dict:
    """Todas las cuentas de la partición; ninguna fila sale de DuckDB."""
    cursor = con.execute(stats_query())
    stats = dict(
        zip([d[0] for d in cursor.description], cursor.fetchone(), strict=True)
    )
    cursor = con.execute(BOUND_QUERY)
    bound = dict(
        zip([d[0] for d in cursor.description], cursor.fetchone(), strict=True)
    )
    summaries_rows, summaries_dups = key_counts(con, "summaries")
    _events_rows, events_dups = key_counts(con, "events")
    stats["no_finitos"] = sum(stats[f"{c}__nofinito"] for c in DOUBLES)
    stats["summaries_rows"] = summaries_rows
    invariants = {
        "filas_vista_igual_summaries": abs(stats["filas"] - summaries_rows),
        "union_1a1": (
            bound["sin_events"] + bound["sin_summaries"] + summaries_dups + events_dups
        ),
        "c_flow_imbalance_acotado": stats["v_c_flow"],
        "os_flow_imbalance_acotado": stats["v_os_flow"],
        "c_range_no_negativo": stats["v_c_range"],
        "c_instant_bias_no_negativo": stats["v_c_bias"],
        "cota_de_confirmacion": bound["v_up"] + bound["v_down"],
        "atomic_sweep_igual_c_duration_cero": stats["v_atomic_duration"],
        "atomic_sweep_igual_c_tick_rate_nulo": stats["v_atomic_rate"],
        "empty_overshoot_igual_os_nulos": stats["v_empty"],
        "banderas_sin_nulos": stats["v_flags_null"],
        "columnas_nunca_nulas": stats["v_never_null"],
        "doubles_finitos": stats["no_finitos"],
    }
    return {"stats": stats, "invariants": invariants}


def csv_rows(result: dict) -> list[dict]:
    stats = result["stats"]
    rows = []
    for c in DOUBLES:
        rows.append(
            {
                "seccion": "doble",
                "columna": c,
                "filas": stats["filas"],
                "nulos": stats["filas"] - stats[f"{c}__n"],
                "min": stats[f"{c}__min"],
                "mediana": stats[f"{c}__med"],
                "max": stats[f"{c}__max"],
            }
        )
    for f in FLAGS:
        rows.append(
            {
                "seccion": "bandera",
                "columna": f,
                "filas": stats["filas"],
                "verdaderas": stats[f"{f}__true"],
            }
        )
    for name, count in result["invariants"].items():
        rows.append({"seccion": "invariante", "columna": name, "violaciones": count})
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


def final_line(
    month, theta, version, wall, verdict, stats=None, violations=None
) -> str:
    def cell(value):
        return "-" if value is None else value

    return (
        f"derived check: mes={month} theta={theta} derived_version={version} "
        f"filas={cell(stats and stats['filas'])} "
        f"atomic_sweep={cell(stats and stats['atomic_sweep__true'])} "
        f"empty_overshoot={cell(stats and stats['empty_overshoot__true'])} "
        f"provider_gap={cell(stats and stats['provider_gap__true'])} "
        f"violaciones={cell(violations)} pared_s={wall:.1f} veredicto={verdict}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    month = re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month)
    if month is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    try:
        theta = parse_theta(ns.theta)
    except ValueError as exc:
        parser.error(str(exc))
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_derived_check_{ns.month}_{theta}.csv"

    name_project = project()
    if not (ns.summaries_root and ns.events_root) and name_project is None:
        parser.error(
            "sin --summaries-root o --events-root y sin OPS_RESULTS_URI: no sé de qué "
            "proyecto es"
        )
    summaries_root = ns.summaries_root or f"gs://{name_project}-dc-descriptors/l3"
    events_root = ns.events_root or f"gs://{name_project}-dc-events/l2"
    if ns.sql:
        sql_source = ns.sql
    elif name_project:
        sql_source = f"gs://{name_project}-ops/scripts/derived.sql"
    else:
        # Subido a `scripts/`, el archivo no vive en el repo y `parents[3]` no existe.
        sql_source = str(Path(__file__).resolve().parents[3] / SQL_IN_REPO)
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))

    started = time.perf_counter()

    def fail(message: str, version: str = "-") -> int:
        log(f"FALLO {message}")
        log(
            final_line(ns.month, theta, version, time.perf_counter() - started, "fallo")
        )
        return 1

    try:
        sql = read_sql(sql_source)
        version = derived_version(sql)
    except (ValueError, OSError) as exc:
        return fail(f"no se pudo leer {sql_source}: {exc}")
    paths = {
        "summaries": partition_file(
            summaries_root, key, theta, ns.month, "summaries.parquet"
        ),
        "events": partition_file(events_root, key, theta, ns.month, "events.parquet"),
    }
    log(
        f"derived check: θ={theta} mes={ns.month} derived_version={version} "
        f"sql={sql_source} summaries={paths['summaries']} events={paths['events']}"
    )
    datasets = {}
    missing = []
    for table, path in paths.items():
        try:
            datasets[table] = open_dataset(path)
        except FileNotFoundError:
            missing.append(f"falta el archivo {path}")
    if missing:
        return fail("; ".join(missing), version)

    con = duckdb.connect()
    try:
        for table, dataset in datasets.items():
            con.register(table, dataset)
        con.execute(sql)
        result = measure(con)
    except (duckdb.Error, OSError) as exc:
        return fail(f"{type(exc).__name__}: {exc}", version)
    finally:
        con.close()
    wall = time.perf_counter() - started

    stats, invariants = result["stats"], result["invariants"]
    log(f"{'invariante':<40} {'violaciones':>12}")
    for invariant, count in invariants.items():
        log(f"{invariant:<40} {count:>12}")
    log(f"{'bandera':<40} {'verdaderas':>12}")
    for flag in FLAGS:
        log(f"{flag:<40} {stats[f'{flag}__true']:>12}")
    try:
        write_csv(csv_rows(result), out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    violations = sum(invariants.values())
    verdict = "violaciones" if violations else "ok"
    log(final_line(ns.month, theta, version, wall, verdict, stats, violations))
    return 1 if violations else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--summaries-root")
    parser.add_argument("--events-root")
    parser.add_argument("--sql")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
