"""Sonda de `path.parquet` (familia T de L3) sobre un θ y un mes del lago (ITSC-395).

Abre el `path.parquet` y el `summaries.parquet` de L3 de una sola partición (un θ y un mes) y
deja un CSV con:
  - por cada una de las 13 columnas de datos de T: filas, nulos, mínimo, mediana y máximo;
  - por cada invariante: cuántas filas lo violan (0 es lo esperado).

Invariantes (TRD-L3 §7.2.1; las desigualdades con tolerancia 1e-9 por el redondeo de DOUBLE):
  - esqueleto: filas de `path` = filas de `summaries`; unión 1:1 por
    `(theta, confirm_agg_trade_id)` (sin huérfanas ni claves repetidas); mismos row groups y
    mismas filas por row group según los pies de los dos archivos;
  - `c_realized_var ≥ 0`, `os_realized_var ≥ 0`; `c_tortuosity ≥ 1`, `os_tortuosity ≥ 1`;
    `|c_sign_autocorr| ≤ 1`, `|os_sign_autocorr| ≤ 1`;
  - `0 ≤ c_max_gap ≤ c_duration` y `0 ≤ os_max_gap ≤ os_duration`, con las duraciones de
    `summaries`; `0 ≤ *_idle_fraction ≤ 1`; `c_idle_fraction = 0` donde `silence_us > c_duration`;
    `0 ≤ c_t25 ≤ c_t50 ≤ c_t75 ≤ 1`;
  - la regla de nulos, exacta, una invariante por columna: `os_realized_var` y `os_max_gap` ⇔
    `os_ticks = 0`; `c_sign_autocorr` ⇔ `c_ticks < 2` o todos los ticks del mismo lado
    (`c_quantity_buy = 0` o `c_quantity_sell = 0`); `os_sign_autocorr` igual con `os_*`;
    `c_idle_fraction` y `c_t25` a `c_t75` ⇔ `c_duration = 0`; `os_idle_fraction` ⇔ `os_ticks = 0`
    o `os_duration = 0`; `c_realized_var`, `c_tortuosity` y `c_max_gap` nunca nulas;
  - `os_tortuosity` es nula con `os_ticks = 0` y, además, con `p_E = p_C`. `summaries` no trae
    `confirm_price` ni `extreme_price`, así que la sonda no puede decidir `p_E = p_C`; usa la
    condición necesaria que sí ve: con `p_E = p_C`, `os_price_max = c_price_max` (upturn) u
    `os_price_min = c_price_min` (downturn). Una nula con `os_ticks > 0` que no cumple ninguna
    es violación; una no nula con `os_ticks = 0` también. La comprobación es deliberadamente
    laxa (sin la dirección, una igualdad del lado contrario la deja pasar) para no dar falsos
    positivos; un valor `inf` o `NaN` donde `p_E = p_C` lo atrapa «sin ±inf ni NaN»;
  - sin `±inf` ni `NaN` en las columnas DOUBLE.

Args:
  --theta 0.00098052  --month YYYY-MM          (obligatorios; un θ y un mes por corrida)
  --summaries-root gs://<proyecto>-dc-descriptors/l3   (la raíz de L3, donde viven los dos
                                                        archivos; por defecto, la del proyecto
                                                        según OPS_RESULTS_URI; también una ruta
                                                        local con la misma disposición hive)
  --provider binance --market spot --asset BTCUSDT
  --out <ruta o gs://…/archivo.csv>   (por defecto, `l3_path_check_<mes>_<θ>.csv` bajo
                                       OPS_RESULTS_URI o, en local, en el directorio actual;
                                       una ruta que termina en `/` es un directorio)

`state_version` y `silence_us` salen de los metadatos de `path.parquet`, leídos con PyArrow. No
importa ninguna capa: una importación cambiaría `uv.lock`.

Abre los dos archivos por su ruta completa y los registra en DuckDB como datasets de PyArrow de
un solo archivo: nunca un dataset sobre un prefijo, que pide el objeto `l3` sin barra y la
condición IAM por prefijo de `ops-script` lo niega con 403 (runbook de operación de L3, «La
trampa del 403 por prefijo»). Todas las cuentas son consultas de agregación en DuckDB: ninguna
fila pasa por Python. El pico es O(partición): las medianas retienen los valores de las 13
columnas.

Salida: la tabla de invariantes, la ruta del CSV y, como última línea,
`path check: mes=… theta=… state_version=… silence_us=… filas=… violaciones=… pared_s=…
veredicto=ok|violaciones|fallo`.

Códigos: 0 sin violaciones; 1 con violaciones, sin alguno de los dos archivos o si DuckDB o la
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
DOUBLES = (
    "c_realized_var",
    "os_realized_var",
    "c_tortuosity",
    "os_tortuosity",
    "c_sign_autocorr",
    "os_sign_autocorr",
    "c_idle_fraction",
    "os_idle_fraction",
    "c_t25",
    "c_t50",
    "c_t75",
)
# Las 13 columnas de datos de T, en el orden del esquema.
COLUMNS = (
    "c_realized_var",
    "os_realized_var",
    "c_tortuosity",
    "os_tortuosity",
    "c_sign_autocorr",
    "os_sign_autocorr",
    "c_max_gap",
    "os_max_gap",
    "c_idle_fraction",
    "os_idle_fraction",
    "c_t25",
    "c_t50",
    "c_t75",
)
CSV_HEADER = [
    "seccion",
    "columna",
    "filas",
    "nulos",
    "min",
    "mediana",
    "max",
    "violaciones",
]
BOTH = "p.confirm_agg_trade_id IS NOT NULL AND s.confirm_agg_trade_id IS NOT NULL"


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
    """Una pasada sobre `path` sola: estadísticas por columna y valores no finitos."""
    items = ["count(*) AS filas"]
    for c in COLUMNS:
        items += [
            f"count({c}) AS {c}__n",
            f"min({c}) AS {c}__min",
            f"median({c}) AS {c}__med",
            f"max({c}) AS {c}__max",
        ]
    for c in DOUBLES:
        items.append(f"count(*) FILTER (WHERE NOT isfinite({c})) AS {c}__nofinito")
    return f"SELECT {', '.join(items)} FROM path"


def v(condition: str) -> str:
    """Cuenta las filas con pareja que cumplen `condition` (la violación)."""
    return f"count(*) FILTER (WHERE ({BOTH}) AND ({condition}))"


def null_rule(column: str, expected: str) -> str:
    """Violaciones de la regla de nulos: nula donde no debe o no nula donde debe serlo."""
    return v(f"(p.{column} IS NULL) IS DISTINCT FROM ({expected})")


def join_query(silence_us: int) -> str:
    """Todas las invariantes que cruzan `path` con `summaries`, en una pasada."""
    items = {
        "sin_path": "count(*) FILTER (WHERE p.confirm_agg_trade_id IS NULL)",
        "sin_summaries": "count(*) FILTER (WHERE s.confirm_agg_trade_id IS NULL)",
        "c_realized_var_no_negativo": v(f"p.c_realized_var < -{TOL}"),
        "os_realized_var_no_negativo": v(f"p.os_realized_var < -{TOL}"),
        "c_tortuosity_al_menos_1": v(f"p.c_tortuosity < 1 - {TOL}"),
        "os_tortuosity_al_menos_1": v(f"p.os_tortuosity < 1 - {TOL}"),
        "c_sign_autocorr_acotada": v(f"abs(p.c_sign_autocorr) > 1 + {TOL}"),
        "os_sign_autocorr_acotada": v(f"abs(p.os_sign_autocorr) > 1 + {TOL}"),
        "c_max_gap_acotado": v("p.c_max_gap < 0 OR p.c_max_gap > s.c_duration"),
        "os_max_gap_acotado": v(
            "p.os_max_gap < 0 OR p.os_max_gap > coalesce(s.os_duration, -1)"
        ),
        "c_idle_fraction_en_0_1": v(
            f"p.c_idle_fraction < -{TOL} OR p.c_idle_fraction > 1 + {TOL}"
        ),
        "os_idle_fraction_en_0_1": v(
            f"p.os_idle_fraction < -{TOL} OR p.os_idle_fraction > 1 + {TOL}"
        ),
        "c_idle_fraction_cero_si_silencio_mayor": v(
            f"{silence_us} > s.c_duration AND p.c_idle_fraction <> 0"
        ),
        "c_t25_t50_t75_ordenados": v(
            f"p.c_t25 < -{TOL} OR p.c_t25 > p.c_t50 + {TOL} "
            f"OR p.c_t50 > p.c_t75 + {TOL} OR p.c_t75 > 1 + {TOL}"
        ),
        "nulos_c_realized_var": null_rule("c_realized_var", "false"),
        "nulos_os_realized_var": null_rule("os_realized_var", "s.os_ticks = 0"),
        "nulos_c_tortuosity": null_rule("c_tortuosity", "false"),
        # p_E = p_C no se ve en summaries: la condición necesaria se explica en el docstring.
        "nulos_os_tortuosity": v(
            "(s.os_ticks = 0 AND p.os_tortuosity IS NOT NULL) OR "
            "(s.os_ticks > 0 AND p.os_tortuosity IS NULL AND "
            "NOT (s.os_price_max = s.c_price_max OR s.os_price_min = s.c_price_min))"
        ),
        "nulos_c_sign_autocorr": null_rule(
            "c_sign_autocorr",
            "s.c_ticks < 2 OR s.c_quantity_buy = 0 OR s.c_quantity_sell = 0",
        ),
        "nulos_os_sign_autocorr": null_rule(
            "os_sign_autocorr",
            "s.os_ticks < 2 OR s.os_quantity_buy = 0 OR s.os_quantity_sell = 0",
        ),
        "nulos_c_max_gap": null_rule("c_max_gap", "false"),
        "nulos_os_max_gap": null_rule("os_max_gap", "s.os_ticks = 0"),
        "nulos_c_idle_fraction": null_rule("c_idle_fraction", "s.c_duration = 0"),
        "nulos_os_idle_fraction": null_rule(
            "os_idle_fraction", "s.os_ticks = 0 OR coalesce(s.os_duration, 0) = 0"
        ),
        "nulos_c_t25": null_rule("c_t25", "s.c_duration = 0"),
        "nulos_c_t50": null_rule("c_t50", "s.c_duration = 0"),
        "nulos_c_t75": null_rule("c_t75", "s.c_duration = 0"),
    }
    select = ", ".join(f"{expr} AS {name}" for name, expr in items.items())
    return (
        f"SELECT {select} FROM path AS p FULL JOIN summaries AS s "
        "ON p.theta = s.theta AND p.confirm_agg_trade_id = s.confirm_agg_trade_id"
    )


def key_counts(con: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int]:
    """Filas de la tabla y filas que sobran por claves repetidas."""
    rows, distinct = con.execute(
        f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT theta, "
        f"confirm_agg_trade_id FROM {table})) FROM {table}"
    ).fetchone()
    return rows, rows - distinct


def row_group_mismatches(path_groups: list[int], summaries_groups: list[int]) -> int:
    """Row groups que difieren en posición, por filas, o que sobran de un lado."""
    return sum(a != b for a, b in zip_longest(path_groups, summaries_groups))


def as_dict(cursor: duckdb.DuckDBPyConnection) -> dict:
    return dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=True))


def measure(
    con: duckdb.DuckDBPyConnection,
    silence_us: int,
    path_groups: list[int],
    summaries_groups: list[int],
) -> dict:
    """Todas las cuentas de la partición; ninguna fila sale de DuckDB."""
    stats = as_dict(con.execute(stats_query()))
    joined = as_dict(con.execute(join_query(silence_us)))
    path_rows, path_dups = key_counts(con, "path")
    summaries_rows, summaries_dups = key_counts(con, "summaries")
    invariants = {
        "filas_path_igual_summaries": abs(path_rows - summaries_rows),
        "union_1a1": (
            joined.pop("sin_path")
            + joined.pop("sin_summaries")
            + path_dups
            + summaries_dups
        ),
        "row_groups_iguales": row_group_mismatches(path_groups, summaries_groups),
        **joined,
        "doubles_finitos": sum(stats[f"{c}__nofinito"] for c in DOUBLES),
    }
    return {"stats": stats, "invariants": invariants}


def csv_rows(result: dict) -> list[dict]:
    stats = result["stats"]
    rows = []
    for c in COLUMNS:
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
    month, theta, version, silence, wall, verdict, rows=None, violations=None
) -> str:
    def cell(value):
        return "-" if value is None else value

    return (
        f"path check: mes={month} theta={theta} state_version={cell(version)} "
        f"silence_us={cell(silence)} filas={cell(rows)} "
        f"violaciones={cell(violations)} pared_s={wall:.1f} veredicto={verdict}"
    )


def run(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    if re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", ns.month) is None:
        parser.error(f"--month {ns.month!r} inválido: se espera YYYY-MM")
    try:
        theta = parse_theta(ns.theta)
    except ValueError as exc:
        parser.error(str(exc))
    key = (ns.provider, ns.market, ns.asset)
    name = f"l3_path_check_{ns.month}_{theta}.csv"

    name_project = project()
    if not ns.summaries_root and name_project is None:
        parser.error(
            "sin --summaries-root y sin OPS_RESULTS_URI: no sé de qué proyecto es"
        )
    root = ns.summaries_root or f"gs://{name_project}-dc-descriptors/l3"
    out = ns.out or f"{os.environ.get('OPS_RESULTS_URI', '')}{name}"
    if out.endswith("/"):
        out += name
    try:
        check_out(out)
    except ValueError as exc:
        parser.error(str(exc))

    started = time.perf_counter()

    def fail(message: str, version=None, silence=None) -> int:
        log(f"FALLO {message}")
        log(
            final_line(
                ns.month,
                theta,
                version,
                silence,
                time.perf_counter() - started,
                "fallo",
            )
        )
        return 1

    paths = {
        "path": partition_file(root, key, theta, ns.month, "path.parquet"),
        "summaries": partition_file(root, key, theta, ns.month, "summaries.parquet"),
    }
    log(
        f"path check: θ={theta} mes={ns.month} path={paths['path']} "
        f"summaries={paths['summaries']}"
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
        path_groups, path_meta = read_footer(paths["path"])
        summaries_groups, _ = read_footer(paths["summaries"])
    except (OSError, pa.ArrowException) as exc:
        return fail(f"no se pudo leer el pie: {type(exc).__name__}: {exc}")
    version = path_meta.get("state_version")
    try:
        silence_us = int(path_meta["silence_us"])
    except KeyError, ValueError:
        return fail(
            f"{paths['path']} no trae silence_us entero en los metadatos "
            f"({path_meta.get('silence_us')!r})",
            version,
        )

    con = duckdb.connect()
    try:
        for table, dataset in datasets.items():
            con.register(table, dataset)
        result = measure(con, silence_us, path_groups, summaries_groups)
    except (duckdb.Error, OSError, pa.ArrowException) as exc:
        return fail(f"{type(exc).__name__}: {exc}", version, silence_us)
    finally:
        con.close()
    wall = time.perf_counter() - started

    stats, invariants = result["stats"], result["invariants"]
    log(f"{'invariante':<44} {'violaciones':>12}")
    for invariant, count in invariants.items():
        log(f"{invariant:<44} {count:>12}")
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
            silence_us,
            wall,
            verdict,
            stats["filas"],
            violations,
        )
    )
    return 1 if violations else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theta", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--summaries-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--out")
    return run(parser, parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
