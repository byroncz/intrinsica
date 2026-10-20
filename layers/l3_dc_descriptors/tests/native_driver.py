"""Un resumidor mínimo sobre el motor nativo, solo para probarlo contra el de Python.

Es lo que el orquestador (ITSC-358 es solo el motor) hará con más cuidado: lee los row
groups del lector, los empuja al motor y escribe lo que sella. Aquí el lago es chico, así
que se materializan todos los row groups y se puede empujar en cualquier orden.
"""

import random
from collections.abc import Sequence
from contextlib import ExitStack, closing
from decimal import Decimal

import dc_pyo3
from dc_frames import FamilyWriter, family_path, l2_path, read_row_groups
from dc_frames.lake import Month
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
) -> dict[Decimal, ThetaResult]:
    """`summarize_month` por el motor nativo. `order` es el orden de `push`: `forward`,
    `reversed`, `swapped` (dos vecinos), `shuffled` o una lista de índices.
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
        groups = list(
            read_row_groups(
                values, l1_root, l2_root, month, month, *key, columns=EVENT_COLUMNS
            )
        )
        feeder = native.Feeder(values)
        pieces = [feeder.pieces(rg) for rg in groups]
        engine = native.new_engine(values, [w.row_groups for w in writers], threads)
        stack.callback(engine.close)

        def drain() -> None:
            while (sealed := engine.pop_sealed()) is not None:
                position, _, batch = native.to_batch(values[sealed[0]], sealed)
                writers[position].write_row_group(batch)

        try:
            for k in _order(order, len(groups), seed):
                native.push(engine, k, groups[k], pieces[k])
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
