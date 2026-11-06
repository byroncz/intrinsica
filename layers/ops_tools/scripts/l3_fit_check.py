"""Sonda de `fit.parquet` (familia A de L3) sobre un θ y un mes del lago (ITSC-401).

Abre el `fit.parquet` y el `summaries.parquet` de L3 y el `events.parquet` de L2 de una sola
partición (un θ y un mes) y deja un CSV con:
  - por cada una de las 9 columnas de datos de A: filas, nulos, mínimo, mediana y máximo; y,
    para las tres `*_accel`, los cuantiles 1, 5, 95 y 99 %;
  - por cada invariante: cuántas filas lo violan (0 es lo esperado);
  - por cada cifra informativa: su conteo (no suma a las violaciones).

Invariantes (TRD-L3 §7.2.2; las desigualdades con tolerancia 1e-9 por el redondeo de DOUBLE):
  - esqueleto: filas de `fit` = filas de `summaries`; unión 1:1 por `(theta,
    confirm_agg_trade_id)` con `summaries` y con `events` (sin huérfanas ni claves repetidas);
    mismos row groups y mismas filas por row group según los pies de `fit` y `summaries`;
  - leyes de una fila (decisión 7 de la Épica ITSC-397, corregida el 2026-11-06): `0 <
    c_jump_share ≤ 1`, `os_jump_share ≥ 0` y las cuatro leyes de `ev_jump_share`, por caso y
    decidibles con `summaries` y `events` (TRD-L3 §7.2.2, «Leyes de una fila de A»):
      · (1) `os_ticks = 0`: `ev_jump_share = c_jump_share`;
      · (2) `os_ticks > 0` y `os_duration > 0`: `0 ≤ ev_jump_share ≤ 1 + θ`;
      · (3) `os_ticks > 0`, `os_duration = 0` y `extreme_time = confirm_time`: `0 <
        ev_jump_share ≤ 1 + θ`;
      · (4) `os_ticks > 0`, `os_duration = 0` y `extreme_time > confirm_time`: `p_prev` es un
        tick del instante de confirmación, así que `(p_E − confirm_instant_price_best) /
        (p_E − p_R) ≤ ev_jump_share ≤ (p_E − confirm_instant_price_worst) / (p_E − p_R)`
        (la misma fórmula vale en un upturn y en un downturn); puede ser negativo;
  - la regla de nulos, una invariante por columna, con lo que `summaries` y `events` permiten
    decidir. Fases: `c` = (`reference_time`, `confirm_time`), `os` = (`confirm_time`,
    `extreme_time`), `ev` = (`reference_time`, `extreme_time`); `ticks` = `c_ticks`, `os_ticks`,
    `c_ticks + os_ticks`:
      · `*_slope` es nula si `ticks < 2` o si el cierre es el instante del ancla; con eso la
        columna no puede traer valor (una `*_slope` no nula donde la regla manda nula es violación);
      · `*_accel`, además, si el precio de cierre es el del ancla (`p_fin = p_0`);
      · `*_jump_share` es nula **exactamente** si `p_fin = p_0` (o `os_ticks = 0` en `os_*`);
        nula donde no toca y no nula donde toca son violación;
  - sin `±inf` ni `NaN` en las nueve columnas.

Lo que la sonda no puede decidir porque no ve los ticks se cuenta aparte, como información, y no
suma a `violaciones`:
  - `mismo_t_*`: un `*_slope` nulo con ticks suficientes y tiempos distintos; es el caso «todos
    los ticks en el mismo `t`». El número de la última línea es la suma de las tres fases;
  - `accel_sin_explicar_*`: un `*_accel` nulo con `*_slope` no nulo, ticks suficientes, tiempos
    distintos y `p_fin ≠ p_0`; lo explican un solo `τ` positivo distinto o la singularidad
    numérica del sistema normal tras cuantizar (`det ≤ 0`). Esperado: cerca de 0;
  - `ev_caso_1_*` a `ev_caso_4_*`: cuántos eventos caen en cada caso de las leyes de
    `ev_jump_share` (los cuatro casos parten todas las filas);
  - `abs_*_accel_mayor_2`: filas con `|*_accel| > 2` y su fracción; `|a| ≤ 2` no es una ley
    (TRD-L3 §7.2.2, «Lo que no es una ley»).

Args:
  --theta 0.00098052  --month YYYY-MM          (obligatorios; un θ y un mes por corrida)
  --summaries-root gs://<proyecto>-dc-descriptors/l3   (la raíz de L3, donde viven `fit.parquet`
  --l2-root gs://<proyecto>-dc-events/l2                y `summaries.parquet`; y la de L2, para
                                                        `events.parquet`. Por defecto, las del
                                                        proyecto según OPS_RESULTS_URI; también
                                                        rutas locales con la misma disposición
                                                        hive)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_fit_check_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

`state_version` sale de los metadatos de `fit.parquet`, leídos con PyArrow. No importa ninguna
capa: una importación cambiaría `uv.lock`.

Abre los tres archivos por su ruta completa y los registra en DuckDB como datasets de PyArrow de
un solo archivo: nunca un dataset sobre un prefijo, que pide el objeto `l3` sin barra y la
condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3, «La
trampa del 403 por prefijo»). Todas las cuentas son consultas de agregación en DuckDB: ninguna
fila pasa por Python. El pico es O(partición): las medianas retienen los valores de las 9
columnas.

Salida: la tabla de invariantes y de cifras informativas, la ruta del CSV y, como última línea,
`fit check: mes=… theta=… state_version=… filas=… violaciones=… mismo_t=… pared_s=…
veredicto=ok|violaciones|fallo`.

Códigos: 0 sin violaciones; 1 con violaciones, sin alguno de los tres archivos o si DuckDB o la
lectura de los pies falla; 2 por uso. Solo lee; lo único que escribe es el CSV de `--out`.
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
# Las 9 columnas de datos de A, en el orden del esquema.
COLUMNS = (
    "c_slope",
    "os_slope",
    "ev_slope",
    "c_accel",
    "os_accel",
    "ev_accel",
    "c_jump_share",
    "os_jump_share",
    "ev_jump_share",
)
ACCELS = ("c_accel", "os_accel", "ev_accel")
QUANTILES = (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))
# Ancla y cierre de cada fase (TRD-L3 §7.2.2) y sus ticks, con las filas de `events` (e) y de
# `summaries` (s).
PHASES = {
    "c": {
        "t0": "e.reference_time",
        "tf": "e.confirm_time",
        "p0": "e.reference_price",
        "pf": "e.confirm_price",
        "ticks": "s.c_ticks",
    },
    "os": {
        "t0": "e.confirm_time",
        "tf": "e.extreme_time",
        "p0": "e.confirm_price",
        "pf": "e.extreme_price",
        "ticks": "s.os_ticks",
    },
    "ev": {
        "t0": "e.reference_time",
        "tf": "e.extreme_time",
        "p0": "e.reference_price",
        "pf": "e.extreme_price",
        "ticks": "(s.c_ticks + s.os_ticks)",
    },
}
CSV_HEADER = [
    "seccion",
    "columna",
    "filas",
    "nulos",
    "min",
    "mediana",
    "max",
    *(name for name, _ in QUANTILES),
    "violaciones",
    "conteo",
    "fraccion",
]
BOTH = "s.confirm_agg_trade_id IS NOT NULL AND e.confirm_agg_trade_id IS NOT NULL"
KEY = "(theta, confirm_agg_trade_id)"


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


def stats_query() -> str:
    """Una pasada sobre `fit` sola: estadísticas por columna y valores no finitos."""
    items = ["count(*) AS filas"]
    for c in COLUMNS:
        items += [
            f"count({c}) AS {c}__n",
            f"min({c}) AS {c}__min",
            f"median({c}) AS {c}__med",
            f"max({c}) AS {c}__max",
            f"count(*) FILTER (WHERE NOT isfinite({c})) AS {c}__nofinito",
        ]
    for c in ACCELS:
        items += [f"quantile_cont({c}, {q}) AS {c}__{name}" for name, q in QUANTILES]
        items.append(f"count(*) FILTER (WHERE abs({c}) > 2) AS {c}__mayor_2")
    return f"SELECT {', '.join(items)} FROM fit"


def v(condition: str) -> str:
    """Cuenta las filas con pareja en `summaries` y `events` que cumplen `condition`."""
    return f"count(*) FILTER (WHERE ({BOTH}) AND ({condition}))"


def slope_is_null(phase: str) -> str:
    """Cuándo la regla manda `*_slope` nula con lo que ven `summaries` y `events`."""
    p = PHASES[phase]
    return f"{p['ticks']} < 2 OR {p['tf']} = {p['t0']}"


def accel_is_null(phase: str) -> str:
    p = PHASES[phase]
    return f"({slope_is_null(phase)}) OR {p['pf']} = {p['p0']}"


def jump_is_null(phase: str) -> str:
    p = PHASES[phase]
    empty = " OR s.os_ticks = 0" if phase == "os" else ""
    return f"{p['pf']} = {p['p0']}{empty}"


# Los cuatro casos de `ev_jump_share` (TRD-L3 §7.2.2), con las filas de `events` (e) y de
# `summaries` (s). Parten todas las filas por `os_ticks`, `os_duration` y `extreme_time`.
EV_CASES = {
    "1_sin_overshoot": "s.os_ticks = 0",
    "2_overshoot_con_duracion": "s.os_ticks > 0 AND s.os_duration > 0",
    "3_overshoot_en_el_instante": (
        "s.os_ticks > 0 AND s.os_duration = 0 AND e.extreme_time = e.confirm_time"
    ),
    "4_instante_de_confirmacion": (
        "s.os_ticks > 0 AND s.os_duration = 0 AND e.extreme_time > e.confirm_time"
    ),
}


def ev_jump_share_laws(theta: str) -> dict[str, str]:
    """Una invariante por caso de `ev_jump_share`. Solo mira filas con el valor no nulo: la
    regla de nulos ya dice cuándo debe serlo. Los precios son DECIMAL; la razón va en DOUBLE.
    """
    ev = "f.ev_jump_share"
    span = "(e.extreme_price - e.reference_price)::DOUBLE"
    lower = f"(e.extreme_price - s.confirm_instant_price_best)::DOUBLE / {span}"
    upper = f"(e.extreme_price - s.confirm_instant_price_worst)::DOUBLE / {span}"
    bound = f"1 + {theta}::DOUBLE + {TOL}"
    cases = list(EV_CASES.values())
    return {
        "ev_jump_share_caso_1": v(
            f"({cases[0]}) AND {ev} IS NOT NULL AND f.c_jump_share IS NOT NULL "
            f"AND abs({ev} - f.c_jump_share) > {TOL}"
        ),
        "ev_jump_share_caso_2": v(
            f"({cases[1]}) AND {ev} IS NOT NULL AND ({ev} < -{TOL} OR {ev} > {bound})"
        ),
        "ev_jump_share_caso_3": v(
            f"({cases[2]}) AND {ev} IS NOT NULL AND ({ev} <= 0 OR {ev} > {bound})"
        ),
        "ev_jump_share_caso_4": v(
            f"({cases[3]}) AND {ev} IS NOT NULL AND e.extreme_price <> e.reference_price "
            f"AND ({ev} < {lower} - {TOL} OR {ev} > {upper} + {TOL})"
        ),
    }


def join_query(theta: str) -> str:
    """Invariantes e informativas que cruzan `fit` con `summaries` y `events`, en una pasada."""
    items = {
        "sin_summaries": "count(*) FILTER (WHERE s.confirm_agg_trade_id IS NULL)",
        "sin_events": "count(*) FILTER (WHERE e.confirm_agg_trade_id IS NULL)",
        "c_jump_share_en_0_1": v(f"f.c_jump_share <= 0 OR f.c_jump_share > 1 + {TOL}"),
        "os_jump_share_no_negativo": v(f"f.os_jump_share < -{TOL}"),
        **ev_jump_share_laws(theta),
    }
    for phase in PHASES:
        items[f"nulos_{phase}_slope"] = v(
            f"f.{phase}_slope IS NOT NULL AND ({slope_is_null(phase)})"
        )
    for phase in PHASES:
        items[f"nulos_{phase}_accel"] = v(
            f"f.{phase}_accel IS NOT NULL AND ({accel_is_null(phase)})"
        )
    for phase in PHASES:
        items[f"nulos_{phase}_jump_share"] = v(
            f"(f.{phase}_jump_share IS NULL) IS DISTINCT FROM ({jump_is_null(phase)})"
        )
    for phase in PHASES:
        items[f"info_mismo_t_{phase}_slope"] = v(
            f"f.{phase}_slope IS NULL AND NOT ({slope_is_null(phase)})"
        )
    for phase in PHASES:
        items[f"info_accel_sin_explicar_{phase}_accel"] = v(
            f"f.{phase}_accel IS NULL AND f.{phase}_slope IS NOT NULL "
            f"AND NOT ({accel_is_null(phase)})"
        )
    for name, case in EV_CASES.items():
        items[f"info_ev_caso_{name}"] = v(case)
    select = ", ".join(f"{expr} AS {name}" for name, expr in items.items())
    return (
        f"SELECT {select} FROM fit AS f "
        f"LEFT JOIN summaries AS s ON f.theta = s.theta "
        "AND f.confirm_agg_trade_id = s.confirm_agg_trade_id "
        f"LEFT JOIN events AS e ON f.theta = e.theta "
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


def row_group_mismatches(fit_groups: list[int], summaries_groups: list[int]) -> int:
    """Row groups que difieren en posición, por filas, o que sobran de un lado."""
    return sum(a != b for a, b in zip_longest(fit_groups, summaries_groups))


def as_dict(cursor: duckdb.DuckDBPyConnection) -> dict:
    return dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=True))


def measure(
    con: duckdb.DuckDBPyConnection,
    theta: str,
    fit_groups: list[int],
    summaries_groups: list[int],
) -> dict:
    """Todas las cuentas de la partición; ninguna fila sale de DuckDB."""
    stats = as_dict(con.execute(stats_query()))
    joined = as_dict(con.execute(join_query(theta)))
    fit_rows, fit_dups = key_counts(con, "fit")
    summaries_rows, summaries_dups = key_counts(con, "summaries")
    _events_rows, events_dups = key_counts(con, "events")
    invariants = {
        "filas_fit_igual_summaries": abs(fit_rows - summaries_rows),
        "union_1a1": (
            joined.pop("sin_summaries")
            + anti_count(con, "summaries", "fit")
            + fit_dups
            + summaries_dups
        ),
        "union_events_1a1": (
            joined.pop("sin_events") + anti_count(con, "events", "fit") + events_dups
        ),
        "row_groups_iguales": row_group_mismatches(fit_groups, summaries_groups),
    }
    informative = {}
    for name, count in joined.items():
        (informative if name.startswith("info_") else invariants)[name] = count
    invariants["doubles_finitos"] = sum(stats[f"{c}__nofinito"] for c in COLUMNS)
    return {
        "stats": stats,
        "invariants": invariants,
        "informative": {k.removeprefix("info_"): n for k, n in informative.items()},
    }


def mismo_t(result: dict) -> int:
    return sum(
        count
        for name, count in result["informative"].items()
        if name.startswith("mismo_t_")
    )


def csv_rows(result: dict) -> list[dict]:
    stats = result["stats"]
    rows = []
    for c in COLUMNS:
        row = {
            "seccion": "columna",
            "columna": c,
            "filas": stats["filas"],
            "nulos": stats["filas"] - stats[f"{c}__n"],
            "min": stats[f"{c}__min"],
            "mediana": stats[f"{c}__med"],
            "max": stats[f"{c}__max"],
        }
        if c in ACCELS:
            row.update({name: stats[f"{c}__{name}"] for name, _ in QUANTILES})
        rows.append(row)
    for name, count in result["invariants"].items():
        rows.append({"seccion": "invariante", "columna": name, "violaciones": count})
    for name, count in result["informative"].items():
        rows.append({"seccion": "informativo", "columna": name, "conteo": count})
    for c in ACCELS:
        count, valued = stats[f"{c}__mayor_2"], stats[f"{c}__n"]
        rows.append(
            {
                "seccion": "informativo",
                "columna": f"abs_{c}_mayor_2",
                "conteo": count,
                "fraccion": count / valued if valued else None,
            }
        )
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
    month, theta, version, wall, verdict, rows=None, violations=None, same_t=None
) -> str:
    def cell(value):
        return "-" if value is None else value

    return (
        f"fit check: mes={month} theta={theta} state_version={cell(version)} "
        f"filas={cell(rows)} violaciones={cell(violations)} mismo_t={cell(same_t)} "
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
    name = f"l3_fit_check_{ns.month}_{theta}.csv"

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
        "fit": partition_file(l3_root, key, theta, ns.month, "fit.parquet"),
        "summaries": partition_file(l3_root, key, theta, ns.month, "summaries.parquet"),
        "events": partition_file(l2_root, key, theta, ns.month, "events.parquet"),
    }
    log(
        f"fit check: θ={theta} mes={ns.month} fit={paths['fit']} "
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
        fit_groups, fit_meta = read_footer(paths["fit"])
        summaries_groups, _ = read_footer(paths["summaries"])
    except (OSError, pa.ArrowException) as exc:
        return fail(f"no se pudo leer el pie: {type(exc).__name__}: {exc}")
    version = fit_meta.get("state_version")

    con = duckdb.connect()
    try:
        for table, dataset in datasets.items():
            con.register(table, dataset)
        result = measure(con, theta, fit_groups, summaries_groups)
    except (duckdb.Error, OSError, pa.ArrowException) as exc:
        return fail(f"{type(exc).__name__}: {exc}", version)
    finally:
        con.close()
    wall = time.perf_counter() - started

    stats, invariants = result["stats"], result["invariants"]
    log(f"{'invariante':<44} {'violaciones':>12}")
    for invariant, count in invariants.items():
        log(f"{invariant:<44} {count:>12}")
    log(f"{'informativo (no suma a violaciones)':<44} {'conteo':>12}")
    for name_, count in result["informative"].items():
        log(f"{name_:<44} {count:>12}")
    for c in ACCELS:
        log(f"{f'abs_{c}_mayor_2':<44} {stats[f'{c}__mayor_2']:>12}")
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
            version,
            wall,
            verdict,
            stats["filas"],
            violations,
            mismo_t(result),
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
