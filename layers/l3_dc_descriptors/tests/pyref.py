"""La referencia de Python del resumidor: agregación por row group, en serie (ADR-L3-14).

Es el `summarize_month` de antes de ITSC-363, sin hilos ni lector anticipado: el lector de
`dc_frames` entrega cada row group de L1 con las posiciones de las fronteras de todos los
θ, `rowgroup.Prefixes` calcula lo común a los θ y, por θ, `ThetaAggregator` agrega los
eventos del row group con NumPy y completa el evento abierto con su acumulador parcial.
El motor nativo (`l3_dc_descriptors.summarize`) debe dar las mismas tablas, el mismo
`content_hash` y la misma conciliación; estas pruebas se lo exigen. No es parte de la
imagen: vive en `tests/`.
"""

from collections.abc import Iterable
from contextlib import ExitStack, closing
from decimal import Decimal
from pathlib import Path

from dc_frames import (
    FamilySkeletonMismatch,
    FamilyWriter,
    FrameBoundaryError,
    RowGroupFrames,
    ThetaEvents,
    family_path,
    l2_path,
    read_row_groups,
)
from dc_frames.lake import Month
from l3_dc_descriptors.errors import SkeletonMismatch, month_text
from l3_dc_descriptors.event import EVENT_COLUMNS
from l3_dc_descriptors.l2 import CarryOver
from l3_dc_descriptors.reconcile import L1Month, reconcile
from l3_dc_descriptors.schema import (
    COLUMN,
    FAMILY,
    STATE_VERSION,
    SUMMARIES_SCHEMA,
    RowGroupBuffer,
)
from l3_dc_descriptors.summarize import (
    DEFAULT_KEY,
    ThetaResult,
    _gate_theta,
    _inconsistent,
    _theta,
)
from rowgroup import Closed, Prefixes, ThetaAggregator

C_TICKS, OS_TICKS = COLUMN["c_ticks"], COLUMN["os_ticks"]


class Lane:
    """El avance de un θ en la pasada: su agregador, su escritor, su row group de salida
    y sus cuentas.
    """

    def __init__(
        self,
        theta: Decimal,
        text: str,
        month: Month,
        writer: FamilyWriter,
        carry: CarryOver,
        previous_direction: int,
    ) -> None:
        self.text = text
        self.month = month
        self.writer = writer
        self.carry = carry
        self.previous_direction = previous_direction
        self.aggregator = ThetaAggregator(text, month)
        self._theta = int(theta.scaleb(8))
        self._sizes = iter(writer.row_groups)
        self._rows: RowGroupBuffer | None = None
        self.rows = 0
        self.closed_ticks = 0
        self.before_month = 0
        self.first_reference: int | None = None
        self.last_extreme: int | None = None

    def _mismatch(self, detail: str) -> SkeletonMismatch:
        return SkeletonMismatch(self.text, self.month, FAMILY, detail)

    def add(self, agg: Prefixes, events: ThetaEvents) -> None:
        closed = self.aggregator.feed(agg, events)
        if closed is None:
            return
        self._write(closed)
        self.rows += len(closed.rows)
        self.closed_ticks += int(
            closed.rows[:, C_TICKS].sum() + closed.rows[:, OS_TICKS].sum()
        )
        self.before_month += closed.before_month
        if self.first_reference is None:
            self.first_reference = closed.first_reference
        self.last_extreme = closed.last_extreme

    def _write(self, closed: Closed) -> None:
        """Pasa las filas al row group de salida; lo escribe cada vez que se llena."""
        done = 0
        while done < len(closed.rows):
            if self._rows is None:
                size = next(self._sizes, None)
                if size is None:
                    raise self._mismatch(
                        f"el esqueleto tiene {len(self.writer.row_groups)} row groups y "
                        f"se escribe otro ({self.writer.path})"
                    )
                self._rows = RowGroupBuffer(size)
            take = min(self._rows.size - self._rows.fill, len(closed.rows) - done)
            self._rows.extend(closed.rows[done : done + take])
            done += take
            if self._rows.fill == self._rows.size:
                batch = self._rows.to_batch(self._theta)
                try:
                    self.writer.write_row_group(batch)
                except FamilySkeletonMismatch as error:
                    raise self._mismatch(error.detail) from error
                self._rows = None

    def commit(self) -> str:
        try:
            return self.writer.commit()
        except FamilySkeletonMismatch as error:
            raise self._mismatch(error.detail) from error


def consume(rg: RowGroupFrames, lanes: dict[Decimal, Lane]) -> None:
    """Agrega un row group en todos los θ que lo tocan."""
    agg = Prefixes(rg.frame)
    for events in rg.thetas:
        lanes[events.theta].add(agg, events)


def summarize_month(
    l1_root: str | Path,
    l2_root: str | Path,
    l3_root: str | Path,
    month: Month,
    thetas: Iterable[Decimal | str],
    *,
    series_start: Month,
    key: tuple[str, str, str] = DEFAULT_KEY,
) -> dict[Decimal, ThetaResult]:
    """`summarize_month` por la agregación de Python, un row group y un θ a la vez."""
    l1_root, l2_root, l3_root = str(l1_root), str(l2_root), str(l3_root)
    values = [_theta(t) for t in thetas]
    if not values or len(set(values)) != len(values):
        raise ValueError("thetas debe traer al menos un θ y sin repetidos")
    if month < series_start:
        raise ValueError(
            f"el mes {month_text(month)} es anterior al inicio de la serie "
            f"{month_text(series_start)}"
        )
    texts = {theta: f"{theta:.8f}" for theta in values}
    gated = [
        _gate_theta(l2_root, key, month, series_start, text) for text in texts.values()
    ]
    carries = dict(zip(texts, gated, strict=True))

    with ExitStack() as stack:
        lanes: dict[Decimal, Lane] = {}
        for theta, text in texts.items():
            writer = FamilyWriter(
                family_path(l3_root, key, text, month, FAMILY),
                SUMMARIES_SCHEMA,
                STATE_VERSION,
                l2_path(l2_root, key, text, month),
            )
            stack.enter_context(writer)
            lanes[theta] = Lane(theta, text, month, writer, *carries[theta])
        l1 = stack.enter_context(closing(L1Month(l1_root, key, month)))
        row_groups = stack.enter_context(
            closing(
                read_row_groups(
                    values,
                    l1_root,
                    l2_root,
                    month,
                    month,
                    *key,
                    columns=EVENT_COLUMNS,
                )
            )
        )
        try:
            for rg in row_groups:
                consume(rg, lanes)
                del rg  # el siguiente row group no convive con este
        except FrameBoundaryError as error:
            raise _inconsistent(error, month) from error

        reconciled = {
            theta: reconcile(
                l1,
                theta=lane.text,
                month=month,
                closed=lane.closed_ticks,
                before_month=lane.before_month,
                last_extreme=lane.last_extreme,
                first_reference=lane.first_reference,
                previous_direction=lane.previous_direction,
                carry=lane.carry,
            )
            for theta, lane in lanes.items()
        }
        results = {}
        for theta, lane in lanes.items():
            digest = lane.commit()
            results[theta] = ThetaResult(
                lane.writer.path, lane.rows, digest, reconciled[theta]
            )
    return results
