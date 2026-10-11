"""El resumidor: de L1 y L2 de un mes a `summaries.parquet` de cada θ (TRD-L3 §8.1).

Una pasada por mes y por lista de θ, sin estado entre meses (ADR-L3-12). Los eventos
llegan del lector en modo por chunks y con fan-out: un row group de L1 en vuelo más el
anterior, compartido; los ticks de cada fase se agregan y se sueltan antes de pedir el
siguiente evento. Cada θ arma su row group de salida (el mismo número de filas que el
de `events.parquet`) en un buffer por columna y lo entrega al escritor de familias.
"""

from collections.abc import Iterable
from contextlib import ExitStack, closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dc_frames import (
    EventFrames,
    FamilySkeletonMismatch,
    FamilyWriter,
    FrameBoundaryError,
    family_path,
    l2_path,
    read_frames,
)
from dc_frames.lake import Month, month_of, month_ordinal

from l3_dc_descriptors.errors import L2StateInconsistent, SkeletonMismatch, month_text
from l3_dc_descriptors.event import EVENT_COLUMNS, summarize_event
from l3_dc_descriptors.l2 import EVENTS, CarryOver, read_carry_over, require
from l3_dc_descriptors.reconcile import L1Month, Reconciliation, reconcile
from l3_dc_descriptors.schema import (
    COLUMN,
    FAMILY,
    STATE_VERSION,
    SUMMARIES_SCHEMA,
    RowGroupBuffer,
)

DEFAULT_KEY = ("binance", "spot", "BTCUSDT")
C_TICKS, OS_TICKS = COLUMN["c_ticks"], COLUMN["os_ticks"]
_THETA_SCALE = Decimal("1e-8")


@dataclass(frozen=True, slots=True)
class ThetaResult:
    """Lo que la unidad dejó de un θ: su archivo, sus filas, su hash y su conciliación."""

    path: str
    rows: int
    content_hash: str
    reconciliation: Reconciliation


class _Lane:
    """El avance de un θ en la pasada: su escritor, su row group de salida y sus cuentas."""

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

    def add(self, ev: EventFrames) -> None:
        event = summarize_event(ev, self.text, self.month)
        if self._rows is None:
            size = next(self._sizes, None)
            if size is None:
                raise self._mismatch(
                    f"el esqueleto tiene {len(self.writer.row_groups)} row groups y "
                    f"se escribe otro ({self.writer.path})"
                )
            self._rows = RowGroupBuffer(size)
        self._rows.append(event.values)
        self.rows += 1
        self.closed_ticks += event.values[C_TICKS] + event.values[OS_TICKS]
        self.before_month += ev.ticks_before_month
        if self.first_reference is None:
            self.first_reference = event.reference
        self.last_extreme = event.extreme
        if self._rows.fill == self._rows.size:
            batch = self._rows.to_batch(self._theta)
            try:
                self.writer.write_row_group(batch)
            except FamilySkeletonMismatch as error:
                raise self._mismatch(error.detail) from error
            self._rows = None

    def commit(self) -> str:
        """Publica el archivo del θ y devuelve su `content_hash`."""
        try:
            return self.writer.commit()
        except FamilySkeletonMismatch as error:
            raise self._mismatch(error.detail) from error


def _theta(value: Decimal | str) -> Decimal:
    theta = Decimal(value).quantize(_THETA_SCALE)
    if theta != Decimal(value):
        raise ValueError(f"θ {value} no cabe en 8 decimales, que es la partición de L2")
    return theta


def _inconsistent(error: FrameBoundaryError, month: Month) -> L2StateInconsistent:
    """El `FrameBoundaryError` del lector como `l2_state_inconsistent` (§9.2).

    El θ, el campo y los dos valores salen de los atributos del error; el mensaje
    completo viaja en `details["message"]`.
    """
    result = L2StateInconsistent.mismatch(
        error.theta, month, error.field, error.got, error.wanted
    )
    result.details["message"] = str(error)
    return result


def summarize_month(
    l1_root: str | Path,
    l2_root: str | Path,
    l3_root: str | Path,
    month: Month,
    thetas: Iterable[Decimal | str],
    *,
    series_start: Month,
    provider: str = DEFAULT_KEY[0],
    market: str = DEFAULT_KEY[1],
    asset: str = DEFAULT_KEY[2],
) -> dict[Decimal, ThetaResult]:
    """Escribe `summaries.parquet` del mes para cada θ y devuelve lo que dejó de cada uno.

    Los eventos son los que cierran en `month` (la partición de L2); sus ticks pueden
    estar en meses anteriores de L1, que el lector relee. `series_start` es el primer mes
    de la serie: el único cuyo mes anterior no tiene carry-over de L2.

    Atómico por archivo y secuencial por θ: la unidad comprueba la entrada de L2, resume
    todos los θ y concilia los ticks de cada uno antes de publicar el primero; un error
    de entrada, de resumen o de conciliación (todos son `SummarizeError` o `FramesError`)
    no deja ningún archivo y borra los temporales. Si la publicación misma falla en un θ
    (por ejemplo, el `move` a GCS), los θ ya publicados quedan: re-ejecutar el mes los
    reescribe con el mismo contenido.
    """
    l1_root, l2_root, l3_root = str(l1_root), str(l2_root), str(l3_root)
    key = (provider, market, asset)
    values = [_theta(t) for t in thetas]
    if not values or len(set(values)) != len(values):
        raise ValueError("thetas debe traer al menos un θ y sin repetidos")
    if month < series_start:
        raise ValueError(
            f"el mes {month_text(month)} es anterior al inicio de la serie "
            f"{month_text(series_start)}"
        )
    texts = {theta: f"{theta:.8f}" for theta in values}

    # Compuerta de entrada (§9.2), antes de abrir ningún escritor.
    carries: dict[Decimal, tuple[CarryOver, int]] = {}
    for theta, text in texts.items():
        require(l2_path(l2_root, key, text, month), text, month, EVENTS)
        carry = read_carry_over(l2_root, key, text, month)
        if month == series_start:
            previous = 0
        else:
            before = month_of(month_ordinal(month) - 1)
            previous = read_carry_over(l2_root, key, text, before).direction
        carries[theta] = (carry, previous)

    with ExitStack() as stack:
        lanes: dict[Decimal, _Lane] = {}
        for theta, text in texts.items():
            writer = FamilyWriter(
                family_path(l3_root, key, text, month, FAMILY),
                SUMMARIES_SCHEMA,
                STATE_VERSION,
                l2_path(l2_root, key, text, month),
            )
            stack.enter_context(writer)
            lanes[theta] = _Lane(theta, text, month, writer, *carries[theta])
        l1 = stack.enter_context(closing(L1Month(l1_root, key, month)))
        try:
            for ev in read_frames(
                values,
                l1_root,
                l2_root,
                month,
                month,
                *key,
                chunks=True,
                columns=EVENT_COLUMNS,
            ):
                lanes[ev.theta].add(ev)
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
