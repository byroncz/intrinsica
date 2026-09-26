"""Pasos 7 y 8 del núcleo: orden por (transact_time, agg_trade_id) e integridad de agg_trade_id."""

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


def aggid_results(
    n_gaps: int, gaps: list[list[int]], n_duplicates: int, duplicates: list[int]
) -> list[CheckResult]:
    """Hallazgos de huecos y duplicados; `gaps` y `duplicates` pueden venir recortados."""

    def result(check_type: str, n: int, details: dict) -> CheckResult:
        if n == 0:
            return CheckResult(check_type, Severity.INFO, Status.PASS, 0.0)
        return CheckResult(check_type, Severity.WARNING, Status.FAIL, float(n), details)

    return [
        result("aggid_gap", n_gaps, {"gaps": gaps[:DETAILS_MAX]}),
        result("aggid_duplicate", n_duplicates, {"ids": duplicates[:DETAILS_MAX]}),
    ]


def check_agg_trade_id(table: pa.Table) -> list[CheckResult]:
    """Huecos y duplicados de agg_trade_id (log-and-continue, §9.2).

    Evalúa sobre los ids ordenados, así que no depende del orden de la tabla.
    Un hueco es el rango de ids ausentes [desde, hasta], ambos inclusive.
    """
    ids = table.column("agg_trade_id").combine_chunks()
    ids = pc.take(ids, pc.sort_indices(ids))

    counts = pc.value_counts(ids)
    repeated = pc.filter(counts.field("values"), pc.greater(counts.field("counts"), 1))

    prev, nxt = ids.slice(0, max(len(ids) - 1, 0)), ids.slice(1)
    is_gap = pc.greater(pc.subtract(nxt, prev), 1)
    gap_from = pc.add(pc.filter(prev, is_gap), 1).to_pylist()
    gap_to = pc.subtract(pc.filter(nxt, is_gap), 1).to_pylist()

    return aggid_results(
        len(gap_from),
        [list(g) for g in zip(gap_from, gap_to)],
        len(repeated),
        repeated.to_pylist(),
    )
