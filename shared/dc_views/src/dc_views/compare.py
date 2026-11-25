"""Vista `compare` de L3: overshoot contra confirmación, una columna por medida con pareja.

La vista no lista medidas: las descubre. Para cada relación entregada lee sus columnas
con `DESCRIBE`, `phase_pairs` separa las que tienen `c_X` y `os_X`, y `COMPARE_OPS`
dice si la comparación es una razón o una diferencia. Agregar un indicador con pareja a
una familia solo exige etiquetarlo en `COMPARE_OPS`; la vista no cambia.
"""

from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import duckdb
import pyarrow as pa

from dc_views.derived import load_derived, open_relation

COMPARE_VERSION = "1.0.0"

__all__ = [
    "COMPARE_OPS",
    "COMPARE_VERSION",
    "PhasePairs",
    "load_compare",
    "phase_pairs",
]

_C, _OS = "c_", "os_"
_KEYS = ("theta", "confirm_agg_trade_id")
# Criterio: `ratio` para magnitudes que nunca son negativas y no tienen techo; `diff` para
# las que tienen signo o viven entre 0 y 1.
_RATIO, _DIFF = "ratio", "diff"


def _ops(ratio: Iterable[str] = (), diff: Iterable[str] = ()) -> MappingProxyType:
    return MappingProxyType(
        {**dict.fromkeys(ratio, _RATIO), **dict.fromkeys(diff, _DIFF)}
    )


# Por relación y por medida (el nombre sin el prefijo `c_` u `os_`).
COMPARE_OPS = MappingProxyType(
    {
        "summaries": _ops(
            ratio=(
                "ticks",
                "duration",
                "quantity_buy",
                "quantity_sell",
                "price_max",
                "price_min",
                "vwap",
            )
        ),
        "derived": _ops(
            ratio=("tick_rate", "quantity_rate", "mean_trade", "range", "speed"),
            diff=("flow_imbalance", "vwap_distance"),
        ),
        "path": _ops(
            ratio=("realized_var", "tortuosity", "max_gap"),
            diff=("sign_autocorr", "idle_fraction"),
        ),
        "fit": _ops(diff=("slope", "accel", "jump_share")),
        "spectrum": _ops(
            diff=(
                *(f"fourier_band_{i}" for i in range(1, 8)),
                *(f"haar_level_{i}" for i in range(1, 8)),
                "energy_loc",
            )
        ),
    }
)


class PhasePairs(NamedTuple):
    """`paired`: medidas con `c_X` y `os_X`. `single`: columnas con una sola de las dos."""

    paired: tuple[str, ...]
    single: tuple[str, ...]


def phase_pairs(names: Iterable[str]) -> PhasePairs:
    """Separa las columnas de UNA relación en parejas `c_X`/`os_X` y columnas sin pareja.

    Las columnas sin prefijo `c_` ni `os_` (claves, `ev_*`, banderas, `year`, `month`) no
    son de ninguna fase y no aparecen. Nunca cruza relaciones: llama una vez por relación.
    El orden es el de aparición.
    """
    names = list(names)
    present = set(names)
    paired: list[str] = []
    single: list[str] = []
    for name in names:
        for prefix, other in ((_C, _OS), (_OS, _C)):
            if not name.startswith(prefix):
                continue
            measure = name.removeprefix(prefix)
            if other + measure in present:
                if prefix == _C:
                    paired.append(measure)
            else:
                single.append(name)
            break
    return PhasePairs(tuple(paired), tuple(single))


def _expression(alias: str, measure: str, op: str) -> str:
    c = f'{alias}."{_C}{measure}"::DOUBLE'
    os = f'{alias}."{_OS}{measure}"::DOUBLE'
    value = f"{os} / NULLIF({c}, 0)" if op == _RATIO else f"{os} - {c}"
    return f'{value} AS "cmp_{measure}"'


def load_compare(
    con: duckdb.DuckDBPyConnection,
    summaries: pa.Table | str | Path,
    events: pa.Table | str | Path,
    path: pa.Table | str | Path | None = None,
    fit: pa.Table | str | Path | None = None,
    spectrum: pa.Table | str | Path | None = None,
) -> str:
    """Crea `derived` y la vista `compare`; devuelve `COMPARE_VERSION`.

    Una fila por evento de `summaries` (`LEFT JOIN` por `(theta, confirm_agg_trade_id)`):
    las claves y una columna `cmp_<medida>` DOUBLE por pareja de cada relación entregada.
    `ratio` es `os_X / c_X` (nulo si `c_X` es 0); `diff`, `os_X - c_X`. Una familia no
    entregada no aporta columnas. Falla con `ValueError`, nombrando la medida, si una
    pareja no tiene etiqueta, una etiqueta no tiene pareja o una medida está en dos
    relaciones. Los argumentos son los de `load_derived` (tabla, archivo o raíz hive).
    """
    load_derived(con, summaries, events, path)
    delivered = {"summaries": "s", "derived": "d"}
    if path is not None:
        delivered["path"] = "p"
    for name, source, alias in (("fit", fit, "f"), ("spectrum", spectrum, "x")):
        if source is not None:
            con.register(name, open_relation(name, source))
            delivered[name] = alias

    owner: dict[str, str] = {}
    expressions: list[str] = []
    for relation, alias in delivered.items():
        names = [row[0] for row in con.execute(f"DESCRIBE {relation}").fetchall()]
        measures = phase_pairs(names).paired
        labels = COMPARE_OPS[relation]
        for measure in measures:
            if measure in owner:
                raise ValueError(
                    f"la medida {measure!r} está en {owner[measure]} y en {relation}"
                )
            owner[measure] = relation
            if measure not in labels:
                raise ValueError(
                    f"la medida {measure!r} de {relation} tiene pareja y no tiene etiqueta "
                    "en COMPARE_OPS"
                )
            expressions.append(_expression(alias, measure, labels[measure]))
        for measure in labels:
            if measure not in measures:
                raise ValueError(
                    f"la medida {measure!r} de {relation} tiene etiqueta en COMPARE_OPS "
                    "y no tiene pareja c_/os_"
                )

    joins = "".join(
        f" LEFT JOIN {relation} AS {alias} ON "
        + " AND ".join(f"s.{k} = {alias}.{k}" for k in _KEYS)
        for relation, alias in delivered.items()
        if relation != "summaries"
    )
    columns = ", ".join([*(f"s.{k}" for k in _KEYS), *expressions])
    con.execute(
        f"CREATE OR REPLACE VIEW compare AS SELECT {columns} FROM summaries AS s{joins}"
    )
    return COMPARE_VERSION
