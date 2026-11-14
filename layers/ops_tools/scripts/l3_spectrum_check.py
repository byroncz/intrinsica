"""Sonda de `spectrum.parquet` (familia F de L3) sobre un θ y un mes del lago (ITSC-414).

Abre el `spectrum.parquet` y el `summaries.parquet` de L3 y el `events.parquet` de L2 de una sola
partición (un θ y un mes) y deja un CSV con:
  - por cada una de las columnas de datos de F (las que trae el esquema del archivo: 30 con
    `bands` = 7 en `state_version` 1.1.0 y 1.2.0): filas, nulos, mínimo, mediana y máximo;
  - por cada invariante: cuántas filas lo violan (0 es lo esperado);
  - por cada cifra informativa: su conteo (no suma a las violaciones).

Invariantes (TRD-L3 §7.2.3, decisiones 4, 6 y 8 de la Épica ITSC-410). Una fase es `c`
(confirmación) u `os` (overshoot), con `ticks` = `c_ticks` u `os_ticks` de `summaries`; un registro
es `fourier` (bandas de período) o `haar` (niveles):
  - esqueleto: filas de `spectrum` = filas de `summaries`; unión 1:1 por `(theta,
    confirm_agg_trade_id)` con `summaries` y con `events` (sin huérfanas ni claves repetidas);
    mismos row groups y mismas filas por row group según los pies de `spectrum` y `summaries`;
  - `fracciones_en_0_1`: cada fracción no nula está en [0, 1], también `c_energy_loc` y
    `os_energy_loc`, la ubicación de la energía total de cada fase;
  - `suma_bandas_<fase>_<registro>`: las fracciones no nulas de una fase y un registro suman 1
    con tolerancia 1e-9 (si hay alguna no nula); `*_energy_loc` no entra: no es una fracción
    que se reparta;
  - presupuesto: `nulos_presupuesto` cuenta los eventos con `extreme_agg_trade_id −
    reference_agg_trade_id > budget_ticks` que no traen todas las columnas de datos nulas. Un evento sobre el
    presupuesto no se calcula (decidido con la fila de L2, antes de leer un tick);
  - regla de nulos, solo en los eventos que no superan el presupuesto y con lo que `summaries`
    permite decidir (`n_min` y `bands` salen de los metadatos):
      · `nulos_<fase>_<registro>_valor_donde_nulo`: una columna no nula donde la regla manda nula,
        que es toda la fase si `ticks < n_min` y, con `ticks ≥ n_min`, la banda o el nivel `b` sin
        bins, o sea `ticks < 2^b` (la 5 con menos de 32, la 6 con menos de 64 y la 7 con menos de
        128 retornos; la cola de Haar de `2^⌊log₂ ticks⌋` retornos da las mismas cotas);
      · `nulos_<fase>_<registro>_nulo_donde_valor`: con `ticks ≥ n_min`, un registro con algunas
        columnas con bins nulas y otras no. Con bins, o están todas o ninguna: ninguna es la serie
        sin energía (abajo);
      · `nulos_<fase>_energy_loc_valor_donde_nulo`: `<fase>_energy_loc` no nula con
        `ticks < n_min`. Con `ticks ≥ n_min` es nula solo si Σ r² = 0 (abajo);
  - `doubles_finitos`: ni `±inf` ni `NaN` en las columnas de datos.

Lo que la sonda no puede decidir porque no ve los ticks se cuenta aparte, como información, y no
suma a `violaciones`:
  - `energia_cero_<fase>_<registro>`: con `ticks ≥ n_min`, el registro entero nulo: la serie (o la
    cola de Haar) no tiene energía porque todos sus retornos son iguales (quinta causa de nulo del
    contrato);
  - `energia_total_cero_<fase>`: con `ticks ≥ n_min`, `<fase>_energy_loc` nula: todos los retornos
    de la fase son cero (Σ r² = 0), causa de nulo que `summaries` no deja decidir;
  - `fase_corta_c` y `fase_corta_os`: fases con `ticks < n_min`, que explican sus nulos;
  - `sobre_presupuesto`: eventos sobre `budget_ticks` (el `over_budget` de la última línea).

Args:
  --theta 0.00098052  --month YYYY-MM          (obligatorios; un θ y un mes por corrida)
  --summaries-root gs://<proyecto>-dc-descriptors/l3   (la raíz de L3, donde viven
  --l2-root gs://<proyecto>-dc-events/l2                `spectrum.parquet` y `summaries.parquet`;
                                                        y la de L2, para `events.parquet`. Por
                                                        defecto, las del proyecto según
                                                        OPS_RESULTS_URI; también rutas locales con
                                                        la misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_spectrum_check_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

`state_version`, `n_min`, `bands`, `window` y `budget_ticks` salen de los metadatos de
`spectrum.parquet`, leídos con PyArrow; la sonda usa esos valores y no constantes propias. No
importa ninguna capa: una importación cambiaría `uv.lock`.

Abre los tres archivos por su ruta completa y los registra en DuckDB como datasets de PyArrow de
un solo archivo: nunca un dataset sobre un prefijo, que pide el objeto `l3` sin barra y la
condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3, «La
trampa del 403 por prefijo»). Todas las cuentas son consultas de agregación en DuckDB: ninguna
fila pasa por Python. El pico es O(partición): las medianas retienen los valores de las columnas.

Salida: la tabla de invariantes y de cifras informativas, la ruta del CSV y, como última línea,
`spectrum check: mes=… theta=… state_version=… filas=… violaciones=… over_budget=… pared_s=…
veredicto=ok|violaciones|fallo`.

Códigos: 0 sin violaciones; 1 con violaciones, sin alguno de los tres archivos, con metadatos que
faltan o no se leen, o si DuckDB o la lectura de los pies falla; 2 por uso. Solo lee; lo único
que escribe es el CSV de `--out`.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from itertools import zip_longest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

TOL = 1e-9
# Ticks de cada fase, con las filas de `summaries` (s).
PHASES = {"c": "s.c_ticks", "os": "s.os_ticks"}
# (registro, unidad) de F: Fourier por banda de período y Haar por nivel. El orden de las columnas
# del archivo es registro, fase, banda.
REGISTERS = (("fourier", "band"), ("haar", "level"))
# La ubicación en el tiempo de la energía total de cada fase (desde `state_version` 1.1.0): una
# columna por fase, después de los registros. Si el archivo no la trae (1.0.0), no se verifica.
LOC_COLUMNS = {"c": "c_energy_loc", "os": "os_energy_loc"}
KEYS = ("theta", "confirm_agg_trade_id")
META_KEYS = ("state_version", "n_min", "bands", "window", "budget_ticks")
CSV_HEADER = [
    "seccion",
    "columna",
    "filas",
    "nulos",
    "min",
    "mediana",
    "max",
    "violaciones",
    "conteo",
]
BOTH = "s.confirm_agg_trade_id IS NOT NULL AND e.confirm_agg_trade_id IS NOT NULL"


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


def read_footer(uri: str) -> tuple[list[int], dict[str, str]]:
    """Filas por row group y metadatos clave-valor del pie de un archivo; sin leer datos."""
    fs, path = open_fs(uri)
    with fs.open_input_file(path) as handle:
        meta = pq.ParquetFile(handle).metadata
        groups = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        extra = {k.decode(): v.decode() for k, v in (meta.metadata or {}).items()}
    return groups, extra


def check_out(out: str) -> None:
    """Con `ops-script`, un `gs://` fuera de OPS_RESULTS_URI da 403 al final: falla ya."""
    results = os.environ.get("OPS_RESULTS_URI", "")
    if out.startswith("gs://") and results and not out.startswith(results):
        raise ValueError(
            f"--out {out} no está bajo {results}; el job solo crea objetos ahí"
        )


def parse_meta(meta: dict[str, str]) -> dict:
    """Los parámetros de F tal como el pie de `spectrum.parquet` los declara."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"faltan metadatos en spectrum.parquet: {', '.join(missing)}")
    try:
        params = {
            "n_min": int(meta["n_min"]),
            "bands": int(meta["bands"]),
            "budget_ticks": int(meta["budget_ticks"]),
        }
    except ValueError:
        raise ValueError(
            "metadatos de spectrum.parquet no enteros: "
            f"n_min={meta['n_min']!r} bands={meta['bands']!r} "
            f"budget_ticks={meta['budget_ticks']!r}"
        ) from None
    if params["n_min"] < 2 or params["bands"] < 1 or params["budget_ticks"] < 0:
        raise ValueError(f"metadatos de spectrum.parquet fuera de rango: {params}")
    return {**params, "state_version": meta["state_version"], "window": meta["window"]}


def group(phase: str, register: str, bands: int) -> list[str]:
    """Las columnas de una fase y un registro, de la banda o el nivel 1 al último."""
    unit = dict(REGISTERS)[register]
    return [f"{phase}_{register}_{unit}_{b}" for b in range(1, bands + 1)]


def data_columns(bands: int, with_loc: bool = True) -> list[str]:
    """Las columnas de datos de F en el orden del archivo."""
    columns = [
        column
        for register, _ in REGISTERS
        for phase in PHASES
        for column in group(phase, register, bands)
    ]
    return [*columns, *LOC_COLUMNS.values()] if with_loc else columns


def schema_columns(names: list[str], bands: int) -> list[str]:
    """Las columnas de datos que el esquema trae; `ValueError` si faltan o sobran.

    El número no es una constante de la sonda: sale del esquema, con las de la ubicación de la
    energía como opcionales. Una columna que la sonda no conoce no se verificaría: falla.
    """
    for with_loc in (True, False):
        expected = data_columns(bands, with_loc)
        if set(names) - set(KEYS) == set(expected):
            absent = [k for k in KEYS if k not in names]
            if absent:
                raise ValueError(f"a spectrum.parquet le faltan columnas: {', '.join(absent)}")
            return expected
    expected = set(data_columns(bands))
    absent = sorted(expected - set(names))
    extra = sorted(set(names) - expected - set(KEYS))
    raise ValueError(
        "el esquema de spectrum.parquet no es el esperado con "
        f"bands={bands}: faltan {absent or '-'}; sobran {extra or '-'}"
    )


def stats_query(columns: list[str]) -> str:
    """Una pasada sobre `spectrum` sola: estadísticas por columna y valores no finitos."""
    items = ["count(*) AS filas"]
    for c in columns:
        items += [
            f"count({c}) AS {c}__n",
            f"min({c}) AS {c}__min",
            f"median({c}) AS {c}__med",
            f"max({c}) AS {c}__max",
            f"count(*) FILTER (WHERE NOT isfinite({c})) AS {c}__nofinito",
        ]
    return f"SELECT {', '.join(items)} FROM spectrum"


def v(condition: str) -> str:
    """Cuenta las filas con pareja en `summaries` y `events` que cumplen `condition`."""
    return f"count(*) FILTER (WHERE ({BOTH}) AND ({condition}))"


def any_of(clauses: list[str]) -> str:
    return "(" + " OR ".join(clauses) + ")"


def join_query(params: dict) -> str:
    """Invariantes e informativas que cruzan `spectrum` con `summaries` y `events`, en una pasada."""
    bands, n_min = params["bands"], params["n_min"]
    over = f"(e.extreme_agg_trade_id - e.reference_agg_trade_id) > {params['budget_ticks']}"
    within = f"NOT ({over})"
    every = params["columns"]
    loc = {p: c for p, c in LOC_COLUMNS.items() if c in every}
    items = {
        "sin_summaries": "count(*) FILTER (WHERE s.confirm_agg_trade_id IS NULL)",
        "sin_events": "count(*) FILTER (WHERE e.confirm_agg_trade_id IS NULL)",
        "fracciones_en_0_1": v(
            any_of([f"f.{c} IS NOT NULL AND (f.{c} < 0 OR f.{c} > 1)" for c in every])
        ),
    }
    for register, _ in REGISTERS:
        for phase in PHASES:
            cols = group(phase, register, bands)
            total = " + ".join(f"coalesce(f.{c}, 0)" for c in cols)
            valued = any_of([f"f.{c} IS NOT NULL" for c in cols])
            items[f"suma_bandas_{phase}_{register}"] = v(
                f"{valued} AND abs(({total}) - 1) > {TOL}"
            )
    items["nulos_presupuesto"] = v(
        f"({over}) AND {any_of([f'f.{c} IS NOT NULL' for c in every])}"
    )
    informative = {"sobre_presupuesto": v(over)}
    for register, _ in REGISTERS:
        for phase, ticks in PHASES.items():
            cols = group(phase, register, bands)
            # Una columna no nula donde la regla manda nula: toda la fase si `ticks < n_min` y,
            # con `ticks ≥ n_min`, la banda o el nivel `b` sin bins (`ticks < 2^b`).
            extra = any_of(
                [
                    f"f.{c} IS NOT NULL AND ({ticks} < {n_min} OR {ticks} < {2**b})"
                    for b, c in enumerate(cols, start=1)
                ]
            )
            items[f"nulos_{phase}_{register}_valor_donde_nulo"] = v(
                f"{within} AND {extra}"
            )
            # Con bins y `ticks ≥ n_min`, o están todas las columnas o ninguna.
            with_bins = [(c, f"{ticks} >= {2**b}") for b, c in enumerate(cols, start=1)]
            some_null = any_of([f"({k} AND f.{c} IS NULL)" for c, k in with_bins])
            some_value = any_of([f"({k} AND f.{c} IS NOT NULL)" for c, k in with_bins])
            items[f"nulos_{phase}_{register}_nulo_donde_valor"] = v(
                f"{within} AND {ticks} >= {n_min} AND {some_null} AND {some_value}"
            )
            informative[f"energia_cero_{phase}_{register}"] = v(
                f"{within} AND {ticks} >= {n_min} AND NOT {some_value}"
            )
    for phase, column in loc.items():
        ticks = PHASES[phase]
        # Con `ticks < n_min` la ubicación es nula; con `ticks ≥ n_min` solo es nula con Σ r² = 0.
        items[f"nulos_{phase}_energy_loc_valor_donde_nulo"] = v(
            f"{within} AND f.{column} IS NOT NULL AND {ticks} < {n_min}"
        )
        informative[f"energia_total_cero_{phase}"] = v(
            f"{within} AND {ticks} >= {n_min} AND f.{column} IS NULL"
        )
    for phase, ticks in PHASES.items():
        informative[f"fase_corta_{phase}"] = v(f"{within} AND {ticks} < {n_min}")
    items.update({f"info_{name}": expr for name, expr in informative.items()})
    select = ", ".join(f"{expr} AS {name}" for name, expr in items.items())
    return (
        f"SELECT {select} FROM spectrum AS f "
        "LEFT JOIN summaries AS s ON f.theta = s.theta "
        "AND f.confirm_agg_trade_id = s.confirm_agg_trade_id "
        "LEFT JOIN events AS e ON f.theta = e.theta "
        "AND f.confirm_agg_trade_id = e.confirm_agg_trade_id"
    )


def anti_count(con: duckdb.DuckDBPyConnection, left: str, right: str) -> int:
    """Filas de `left` sin pareja en `right` por `(theta, confirm_agg_trade_id)`."""
    return con.execute(
        f"SELECT count(*) FROM {left} AS a ANTI JOIN {right} AS b "
        "ON a.theta = b.theta AND a.confirm_agg_trade_id = b.confirm_agg_trade_id"
    ).fetchone()[0]


def key_counts(con: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int]:
    """Filas de la tabla y filas que sobran por claves repetidas."""
    rows, distinct = con.execute(
        f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT theta, "
        f"confirm_agg_trade_id FROM {table})) FROM {table}"
    ).fetchone()
    return rows, rows - distinct


def row_group_mismatches(
    spectrum_groups: list[int], summaries_groups: list[int]
) -> int:
    """Row groups que difieren en posición, por filas, o que sobran de un lado."""
    return sum(a != b for a, b in zip_longest(spectrum_groups, summaries_groups))


def as_dict(cursor: duckdb.DuckDBPyConnection) -> dict:
    return dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=True))


def measure(
    con: duckdb.DuckDBPyConnection,
    params: dict,
    spectrum_groups: list[int],
    summaries_groups: list[int],
) -> dict:
    """Todas las cuentas de la partición; ninguna fila sale de DuckDB."""
    columns = params["columns"]
    stats = as_dict(con.execute(stats_query(columns)))
    joined = as_dict(con.execute(join_query(params)))
    spectrum_rows, spectrum_dups = key_counts(con, "spectrum")
    summaries_rows, summaries_dups = key_counts(con, "summaries")
    _events_rows, events_dups = key_counts(con, "events")
    invariants = {
        "filas_spectrum_igual_summaries": abs(spectrum_rows - summaries_rows),
        "union_1a1": (
            joined.pop("sin_summaries")
            + anti_count(con, "summaries", "spectrum")
            + spectrum_dups
            + summaries_dups
        ),
        "union_events_1a1": (
            joined.pop("sin_events")
            + anti_count(con, "events", "spectrum")
            + events_dups
        ),
        "row_groups_iguales": row_group_mismatches(spectrum_groups, summaries_groups),
    }
    informative = {}
    for name, count in joined.items():
        (informative if name.startswith("info_") else invariants)[name] = count
    invariants["doubles_finitos"] = sum(stats[f"{c}__nofinito"] for c in columns)
    return {
        "columns": columns,
        "stats": stats,
        "invariants": invariants,
        "informative": {k.removeprefix("info_"): n for k, n in informative.items()},
    }


def csv_rows(result: dict) -> list[dict]:
    stats = result["stats"]
    rows = []
    for c in result["columns"]:
        rows.append(
            {
                "seccion": "columna",
                "columna": c,
                "filas": stats["filas"],
                "nulos": stats["filas"] - stats[f"{c}__n"],
                "min": stats[f"{c}__min"],
                "mediana": stats[f"{c}__med"],
                "max": stats[f"{c}__max"],
            }
        )
    for name, count in result["invariants"].items():
        rows.append({"seccion": "invariante", "columna": name, "violaciones": count})
    for name, count in result["informative"].items():
        rows.append({"seccion": "informativo", "columna": name, "conteo": count})
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
    month, theta, version, wall, verdict, rows=None, violations=None, over=None
) -> str:
    def cell(value):
        return "-" if value is None else value

    return (
        f"spectrum check: mes={month} theta={theta} state_version={cell(version)} "
        f"filas={cell(rows)} violaciones={cell(violations)} over_budget={cell(over)} "
        f"pared_s={wall:.1f} veredicto={verdict}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    if re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month) is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    try:
        theta = parse_theta(ns.theta)
    except ValueError as exc:
        parser.error(str(exc))
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_spectrum_check_{ns.month}_{theta}.csv"

    name_project = project()
    if not (ns.summaries_root and ns.l2_root) and name_project is None:
        parser.error(
            "sin --summaries-root o --l2-root y sin OPS_RESULTS_URI: no sé de qué "
            "proyecto es"
        )
    l3_root = ns.summaries_root or f"gs://{name_project}-dc-descriptors/l3"
    l2_root = ns.l2_root or f"gs://{name_project}-dc-events/l2"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))

    started = time.perf_counter()

    def fail(message: str, version=None) -> int:
        log(f"FALLO {message}")
        log(
            final_line(ns.month, theta, version, time.perf_counter() - started, "fallo")
        )
        return 1

    paths = {
        "spectrum": partition_file(l3_root, key, theta, ns.month, "spectrum.parquet"),
        "summaries": partition_file(l3_root, key, theta, ns.month, "summaries.parquet"),
        "events": partition_file(l2_root, key, theta, ns.month, "events.parquet"),
    }
    log(
        f"spectrum check: θ={theta} mes={ns.month} spectrum={paths['spectrum']} "
        f"summaries={paths['summaries']} events={paths['events']}"
    )
    datasets = {}
    missing = []
    for table, path in paths.items():
        try:
            datasets[table] = open_dataset(path)
        except FileNotFoundError:
            missing.append(f"falta el archivo {path}")
        except OSError as exc:
            missing.append(f"no se pudo abrir {path}: {type(exc).__name__}: {exc}")
    if missing:
        return fail("; ".join(missing))

    try:
        spectrum_groups, spectrum_meta = read_footer(paths["spectrum"])
        summaries_groups, _ = read_footer(paths["summaries"])
    except (OSError, pa.ArrowException) as exc:
        return fail(f"no se pudo leer el pie: {type(exc).__name__}: {exc}")
    version = spectrum_meta.get("state_version")
    try:
        params = parse_meta(spectrum_meta)
    except ValueError as exc:
        return fail(str(exc), version)
    try:
        params["columns"] = schema_columns(datasets["spectrum"].schema.names, params["bands"])
    except ValueError as exc:
        return fail(str(exc), version)
    log(
        f"spectrum check: state_version={params['state_version']} n_min={params['n_min']} "
        f"bands={params['bands']} window={params['window']} "
        f"budget_ticks={params['budget_ticks']} columnas_de_datos={len(params['columns'])}"
    )

    con = duckdb.connect()
    try:
        for table, dataset in datasets.items():
            con.register(table, dataset)
        result = measure(con, params, spectrum_groups, summaries_groups)
    except (duckdb.Error, OSError, pa.ArrowException) as exc:
        return fail(f"{type(exc).__name__}: {exc}", version)
    finally:
        con.close()
    wall = time.perf_counter() - started

    stats, invariants = result["stats"], result["invariants"]
    log(f"{'invariante':<48} {'violaciones':>12}")
    for invariant, count in invariants.items():
        log(f"{invariant:<48} {count:>12}")
    log(f"{'informativo (no suma a violaciones)':<48} {'conteo':>12}")
    for name_, count in result["informative"].items():
        log(f"{name_:<48} {count:>12}")
    try:
        write_csv(csv_rows(result), out)
        log(f"tabla en {out}")
    except Exception as exc:  # noqa: BLE001 - la sonda no falla por no poder guardar el CSV
        log(f"no se guardó la tabla en {out}: {type(exc).__name__}: {exc}")
    violations = sum(invariants.values())
    verdict = "violaciones" if violations else "ok"
    log(
        final_line(
            ns.month,
            theta,
            params["state_version"],
            wall,
            verdict,
            stats["filas"],
            violations,
            result["informative"]["sobre_presupuesto"],
        )
    )
    return 1 if violations else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--summaries-root")
    parser.add_argument("--l2-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
