"""Un resumidor mínimo sobre el motor nativo, solo para probarlo contra el de Python.

Es lo que el orquestador (ITSC-358 es solo el motor) hará con más cuidado: entrega al motor
las fronteras de L2 de cada θ, lee los row groups del lector, los empuja y escribe lo que
sella. Aquí el lago es chico, así que se materializan todos los row groups y se puede
empujar en cualquier orden.
"""

import random
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from decimal import Decimal

import dc_pyo3
import pyarrow as pa
from dc_frames import FamilyWriter, family_path, l2_path
from dc_frames.lake import L1, EventGroup, Month, event_groups
from l3_dc_descriptors import native
from l3_dc_descriptors.event import EVENT_COLUMNS
from l3_dc_descriptors.reconcile import L1Month, reconcile
from l3_dc_descriptors.schema import FAMILY, STATE_VERSION, SUMMARIES_SCHEMA
from l3_dc_descriptors.summarize import DEFAULT_KEY, ThetaResult, _gate_theta


def summarize_native(
    l1_root,
    l2_root,
    l3_root,
    month: Month,
    thetas: Sequence[str],
    *,
    series_start: Month,
    threads: int = 4,
    order: str | Sequence[int] = "forward",
    seed: int = 0,
    lot: int | None = None,
) -> dict[Decimal, ThetaResult]:
    """`summarize_month` por el motor nativo. `order` es el orden de `push`: `forward`,
    `reversed`, `swapped` (dos vecinos), `shuffled` o una lista de índices. Sin `lot`, las
    fronteras de cada θ (sus row groups de L2) se entregan todas antes del primer `push`;
    con `lot`, en tramos de `lot` eventos, justo antes del row group que las necesita
    (`BoundaryFeed`).
    """
    key = DEFAULT_KEY
    values = [Decimal(t) for t in thetas]
    l1_root, l2_root, l3_root = str(l1_root), str(l2_root), str(l3_root)
    with ExitStack() as stack:
        writers = [
            stack.enter_context(
                FamilyWriter(
                    family_path(l3_root, key, text, month, FAMILY),
                    SUMMARIES_SCHEMA,
                    STATE_VERSION,
                    l2_path(l2_root, key, text, month),
                )
            )
            for text in thetas
        ]
        l1 = stack.enter_context(closing(L1Month(l1_root, key, month)))
        events = [
            list(event_groups(l2_root, key, text, month, month, EVENT_COLUMNS))
            for text in thetas
        ]
        # Todos los row groups desde el que contiene la primera referencia: el motor decide
        # solo qué eventos toca cada uno, sin el filtro del lector.
        reference = min(g[0].bounds[0, 0] for g in events if g)
        groups = list(
            L1(l1_root, key).row_groups(month, month, int(reference), lambda *_: True)
        )
        engine = native.new_engine(values, [w.row_groups for w in writers], threads)
        stack.callback(engine.close)
        feeds = [
            native.BoundaryFeed(engine, position, _lots(batches, lot))
            for position, batches in enumerate(events)
        ]

        def drain() -> None:
            while (sealed := engine.pop_sealed()) is not None:
                position, _, batch = native.to_batch(values[sealed[0]], sealed)
                writers[position].write_row_group(batch)

        try:
            if lot is None:
                for feed in feeds:
                    feed.cover(int(groups[-1].ids[-1]))
            for k in _order(order, len(groups), seed):
                rg = groups[k]
                for feed in feeds:
                    feed.cover(int(rg.ids[-1]))
                native.push(engine, k, rg.ids, rg.columns, rg.month < month)
                drain()
            stats = engine.finish()
            drain()
        except dc_pyo3.DescriptorError as error:
            raise native.to_summarize_error(error, thetas, month) from error

        results = {}
        for theta, text, writer, (rows, closed, before, first, last) in zip(
            values, thetas, writers, stats, strict=True
        ):
            carry, previous = _gate_theta(l2_root, key, month, series_start, text)
            reconciliation = reconcile(
                l1,
                theta=text,
                month=month,
                closed=closed,
                before_month=before,
                last_extreme=last,
                first_reference=first,
                previous_direction=previous,
                carry=carry,
            )
            results[theta] = ThetaResult(
                writer.path, rows, writer.commit(), reconciliation
            )
        return results


def _lots(groups: list[EventGroup], lot: int | None) -> Iterator[pa.RecordBatch]:
    """Los row groups de L2 de un θ tal cual, o cortados en tramos de `lot` eventos."""
    for group in groups:
        if lot is None:
            yield group.batch
            continue
        for start in range(0, group.batch.num_rows, lot):
            yield group.batch.slice(start, lot)


def _order(order: str | Sequence[int], n: int, seed: int) -> list[int]:
    ids = list(range(n))
    rng = random.Random(seed)
    if not isinstance(order, str):
        return list(order)
    match order:
        case "forward":
            return ids
        case "reversed":
            return ids[::-1]
        case "swapped":
            at = rng.randrange(n - 1)
            ids[at], ids[at + 1] = ids[at + 1], ids[at]
            return ids
        case "shuffled":
            rng.shuffle(ids)
            return ids
    raise ValueError(order)
