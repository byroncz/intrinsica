"""Pasos 7 y 8 del núcleo: orden por (transact_time, agg_trade_id) e integridad de agg_trade_id."""

from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc
from dq import Severity, Status

from l1_ingest.checks import CheckResult

DETAILS_MAX = 10


def ensure_order(table: pa.Table) -> tuple[pa.Table, CheckResult]:
    """Reordena solo si la tabla no cumple el orden (ADR-L1-01).

    Si ya está ordenada devuelve la misma tabla, sin copiar. El desempate por
    agg_trade_id cuenta: mismo transact_time con ids invertidos es desorden.
    """
    t = table.column("transact_time").combine_chunks()
    i = table.column("agg_trade_id").combine_chunks()
    t_prev, t_next = t.slice(0, max(len(t) - 1, 0)), t.slice(1)
    i_prev, i_next = i.slice(0, max(len(i) - 1, 0)), i.slice(1)
    decreases = pc.or_(
        pc.less(t_next, t_prev),
        pc.and_(pc.equal(t_next, t_prev), pc.less(i_next, i_prev)),
    )
    n_decreases = pc.sum(decreases).as_py() or 0

    if n_decreases == 0:
        check = CheckResult("reorder_applied", Severity.INFO, Status.PASS, 0.0)
        return table, check

    order = pc.sort_indices(
        table,
        sort_keys=[("transact_time", "ascending"), ("agg_trade_id", "ascending")],
    )
    check = CheckResult(
        "reorder_applied", Severity.WARNING, Status.CORRECTED, float(n_decreases)
    )
    return table.take(order), check


class AggIdConflict(ValueError):
    """Dos filas con el mismo agg_trade_id difieren en alguna columna; lleva el hallazgo."""

    def __init__(self, message: str, check: CheckResult) -> None:
        super().__init__(message)
        self.check = check


def conflict_error(first: dict, second: dict) -> AggIdConflict:
    """Fail-closed: L1 no puede saber cuál de las dos filas es la verdadera."""
    rows = [_jsonable(first), _jsonable(second)]
    check = CheckResult(
        "aggid_duplicate",
        Severity.ERROR,
        Status.FAIL,
        1.0,
        {"agg_trade_id": first["agg_trade_id"], "rows": rows},
    )
    return AggIdConflict(
        f"agg_trade_id {first['agg_trade_id']} repetido con contenido distinto: {rows}",
        check,
    )


def _jsonable(row: dict) -> dict:
    # Los hallazgos van a JSON y Decimal no es serializable; str conserva la exactitud.
    return {k: str(v) if isinstance(v, Decimal) else v for k, v in row.items()}


def aggid_results(
    n_gaps: int, gaps: list[list[int]], n_removed: int, duplicates: list[int]
) -> list[CheckResult]:
    """Hallazgos de huecos y duplicados; `gaps` y `duplicates` pueden venir recortados.

    Los huecos son del proveedor y solo se registran (fail). Los duplicados
    idénticos ya se eliminaron: `n_removed` son las filas eliminadas (corrected).
    """
    gap = CheckResult("aggid_gap", Severity.INFO, Status.PASS, 0.0)
    if n_gaps:
        gap = CheckResult(
            "aggid_gap",
            Severity.WARNING,
            Status.FAIL,
            float(n_gaps),
            {"gaps": gaps[:DETAILS_MAX]},
        )
    dup = CheckResult("aggid_duplicate", Severity.INFO, Status.PASS, 0.0)
    if n_removed:
        dup = CheckResult(
            "aggid_duplicate",
            Severity.WARNING,
            Status.CORRECTED,
            float(n_removed),
            {"ids": duplicates[:DETAILS_MAX]},
        )
    return [gap, dup]


def dedupe_agg_trade_id(table: pa.Table) -> tuple[pa.Table, list[CheckResult]]:
    """Huecos y duplicados de agg_trade_id (§9.2), sobre la tabla completa.

    Dos filas con el mismo id e igual contenido se reducen a la primera de la
    tabla. Si difieren en alguna columna lanza `AggIdConflict`. Huecos y
    repetidos se evalúan sobre los ids ordenados (orden estable), sin depender
    del orden de la tabla. Un hueco es el rango de ids ausentes [desde, hasta],
    ambos inclusive.

    Precondición: la tabla viene de `ensure_order`. Ahí dos filas idénticas
    (mismo id y mismo transact_time) quedan contiguas, así que el filtro solo
    compara cada id con el de la fila anterior; si la máscara no elimina
    exactamente las filas repetidas, falla en vez de publicar ids repetidos.
    """
    ids = table.column("agg_trade_id").combine_chunks()
    order = pc.sort_indices(ids)
    ids = pc.take(ids, order)

    prev, nxt = ids.slice(0, max(len(ids) - 1, 0)), ids.slice(1)
    is_gap = pc.greater(pc.subtract(nxt, prev), 1)
    gap_from = pc.add(pc.filter(prev, is_gap), 1).to_pylist()
    gap_to = pc.subtract(pc.filter(nxt, is_gap), 1).to_pylist()

    repeat = pc.equal(nxt, prev)
    n_removed = pc.sum(repeat).as_py() or 0
    repeated: list[int] = []
    if n_removed:
        kept = pc.filter(order.slice(0, len(order) - 1), repeat)  # fila anterior del id
        removed = pc.filter(order.slice(1), repeat)
        _raise_if_divergent(table, kept, removed)
        repeated = pc.unique(pc.filter(nxt, repeat)).slice(0, DETAILS_MAX).to_pylist()
        # Máscara de un byte por fila en el orden de la tabla, sin arreglo de
        # posiciones: con la precondición, la repetida sigue a la que se queda.
        ids = table.column("agg_trade_id").combine_chunks()
        keep = pa.concat_arrays(
            [pa.array([True]), pc.not_equal(ids.slice(1), ids.slice(0, len(ids) - 1))]
        )
        if pc.sum(pc.invert(keep)).as_py() != n_removed:
            raise ValueError("dedupe_agg_trade_id exige una tabla de ensure_order")
        table = table.filter(keep)

    checks = aggid_results(
        len(gap_from),
        [list(g) for g in zip(gap_from, gap_to)],
        n_removed,
        repeated,
    )
    return table, checks


def _raise_if_divergent(table: pa.Table, kept: pa.Array, removed: pa.Array) -> None:
    """Compara cada fila repetida con la anterior de su id en todas las columnas."""
    same = None
    for name in table.schema.names:
        column = table.column(name)
        equal = pc.equal(column.take(removed), column.take(kept))
        same = equal if same is None else pc.and_(same, equal)
    if pc.all(same).as_py():
        return
    k = pc.index(same, False).as_py()
    first = table.slice(kept[k].as_py(), 1).to_pylist()[0]
    second = table.slice(removed[k].as_py(), 1).to_pylist()[0]
    raise conflict_error(first, second)
