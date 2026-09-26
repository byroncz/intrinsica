"""Pasos 6 a 9 del núcleo por lotes: la RAM de una unidad es O(lote), no O(mes).

Cada lote se conforma, se verifica de forma incremental, se escribe al Parquet
y se suelta antes de pedir el siguiente. Los chequeos dan el mismo resultado
que `conform`, `ensure_order` y `check_agg_trade_id` sobre la tabla completa
mientras los datos lleguen en orden (lo normal en Binance). Si no llegan, se
lanza `NotStreamable` y el llamador usa la ruta materializada.
"""

from collections.abc import Iterable

import pyarrow as pa
import pyarrow.compute as pc
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.conform import classify_unit, conform_batch, unit_check, unit_error
from l1_ingest.integrity import DETAILS_MAX, aggid_results
from l1_ingest.write import ContentHasher, PartitionWriter


class NotStreamable(Exception):
    """Los datos no llegan ordenados por (transact_time, agg_trade_id) y con ids crecientes."""


def _extend(carry: pa.Array | None, column: pa.Array) -> pa.Array:
    return column if carry is None else pa.concat_arrays([carry, column])


def _last(column: pa.Array) -> pa.Array:
    # Copia de un valor: un slice retendría el buffer del lote entero.
    return pa.array([column[-1].as_py()], pa.int64())


class _SequenceCheck:
    """Orden y huecos/duplicados de agg_trade_id, lote a lote con el borde del anterior."""

    def __init__(self) -> None:
        self._time: pa.Array | None = None
        self._id: pa.Array | None = None
        self._last_was_repeat = False
        self.n_gaps = 0
        self.gaps: list[list[int]] = []
        self.n_duplicates = 0
        self.duplicates: list[int] = []

    def feed(self, batch: pa.RecordBatch) -> None:
        time = _extend(self._time, batch.column("transact_time"))
        ids = _extend(self._id, batch.column("agg_trade_id"))
        self._time, self._id = _last(time), _last(ids)
        if len(ids) < 2:
            return

        t_prev, t_next = time.slice(0, len(time) - 1), time.slice(1)
        i_prev, i_next = ids.slice(0, len(ids) - 1), ids.slice(1)
        step = pc.subtract(i_next, i_prev)
        disorder = pc.or_(
            pc.less(t_next, t_prev),
            pc.and_(pc.equal(t_next, t_prev), pc.less(i_next, i_prev)),
        )
        # Ids que retroceden: los chequeos de ids exigen ordenarlos primero.
        if pc.any(disorder).as_py() or pc.any(pc.less(step, 0)).as_py():
            raise NotStreamable

        is_gap = pc.greater(step, 1)
        self.n_gaps += pc.sum(is_gap).as_py() or 0
        if len(self.gaps) < DETAILS_MAX:
            room = DETAILS_MAX - len(self.gaps)
            gap_from = pc.add(pc.filter(i_prev, is_gap).slice(0, room), 1).to_pylist()
            gap_to = pc.subtract(
                pc.filter(i_next, is_gap).slice(0, room), 1
            ).to_pylist()
            self.gaps += [list(g) for g in zip(gap_from, gap_to)]

        # Un id repetido k veces deja k-1 pasos en cero y cuenta una sola vez.
        repeat = pc.equal(step, 0)
        before = pa.concat_arrays([pa.array([self._last_was_repeat]), repeat[:-1]])
        first_repeat = pc.and_(repeat, pc.invert(before))
        self.n_duplicates += pc.sum(first_repeat).as_py() or 0
        if len(self.duplicates) < DETAILS_MAX:
            room = DETAILS_MAX - len(self.duplicates)
            self.duplicates += (
                pc.filter(i_next, first_repeat).slice(0, room).to_pylist()
            )
        self._last_was_repeat = repeat[-1].as_py()

    def results(self) -> list[CheckResult]:
        return [
            CheckResult("reorder_applied", Severity.INFO, Status.PASS, 0.0),
            *aggid_results(self.n_gaps, self.gaps, self.n_duplicates, self.duplicates),
        ]


def stream_partition(
    batches: Iterable[pa.RecordBatch], path: str
) -> tuple[list[CheckResult], str]:
    """Conforma, verifica y escribe los lotes RAW en `path`; devuelve sus hallazgos y el hash.

    Los hallazgos vienen en el orden del pipeline: unidad de tiempo, reorden,
    huecos y duplicados. Si la unidad temporal no es válida lanza
    `TimestampUnitError` con el mín y máx de toda la unidad, sin dejar archivo.
    """
    hasher = ContentHasher()
    sequence = _SequenceCheck()
    lo = hi = unit = None
    invalid = False

    with PartitionWriter(path) as writer:
        for batch in batches:
            if batch.num_rows == 0:
                continue
            bounds = pc.min_max(batch.column("transact_time"))
            blo, bhi = bounds["min"].as_py(), bounds["max"].as_py()
            lo = blo if lo is None else min(lo, blo)
            hi = bhi if hi is None else max(hi, bhi)
            if invalid:
                continue  # solo se sigue calculando el rango para el hallazgo
            batch_unit = classify_unit(blo, bhi)
            unit = unit or batch_unit
            if batch_unit is None or batch_unit != unit:
                invalid = True
                continue
            out = conform_batch(batch, unit)
            del batch
            sequence.feed(out)
            writer.write_batch(out)
            hasher.update(out)
        if invalid or unit is None:
            raise unit_error(lo, hi)
        writer.commit()

    return [unit_check(unit, lo, hi), *sequence.results()], hasher.hexdigest()
