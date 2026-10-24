"""El adaptador del motor nativo de agregación (`dc_pyo3.Engine`, ITSC-358).

El motor (crate `dc_descriptors`) agrega por row group y θ en hilos nativos y calcula él
mismo dónde cae cada frontera de L2 en cada row group; esta capa de Python solo lo
orquesta: entrega por lotes las columnas de L2 de los eventos de cada θ, por delante de los
row groups que las tocan y no más (`push_boundaries`, `BoundaryFeed`), traduce cada row
group del lector a los buffers de Arrow que `Engine.push` espera, envuelve
en Arrow los row groups de salida que el motor sella y convierte sus errores en las
excepciones tipadas de `errors` y del lector. No agrega nada: la aritmética y los
invariantes de §7.2 viven en Rust (ADR-L3-16).

Lo orquesta `summarize.summarize_month` (ITSC-363). La referencia con la que se compara
es la agregación de Python por row group, que vive en `tests/` (`rowgroup.py`, `pyref.py`).
"""

from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any, NamedTuple

import dc_pyo3
import numpy as np
import pyarrow as pa
from dc_frames import FramesInputError
from dc_frames.lake import BOUNDARY_COLUMNS, Month

from l3_dc_descriptors.errors import (
    L2StateInconsistent,
    PhaseValueError,
    SkeletonMismatch,
)
from l3_dc_descriptors.schema import FAMILY, SUMMARIES_SCHEMA
from l3_dc_descriptors.timing import Retained

# Las 18 columnas de datos como las comprueba el motor: (nombre, bytes por valor, nulable).
ENGINE_SCHEMA = tuple(
    (field.name, 16 if pa.types.is_decimal(field.type) else 8, field.nullable)
    for field in list(SUMMARIES_SCHEMA)[1:]
)

_NAMES = ("referencia", "confirmación", "extremo")

# Los (buffer, offset) de una columna: el `memoryview` de bytes y el `offset` del `Array`.
type Region = tuple[memoryview, int]


class EventBatch(NamedTuple):
    """Un lote de filas de `events.parquet` de L2 de un θ (`Engine.push_boundaries`)."""

    rows: int
    reference: Region
    confirm: Region
    confirm_time: Region
    extreme: Region
    direction: Region
    confirm_price: Region
    extreme_price: Region


def _bytes(array: pa.Array) -> Region:
    """Los bytes del buffer de datos de un array de Arrow y su offset, sin copia."""
    buffer = array.buffers()[1]
    return memoryview(buffer if buffer is not None else b"").cast("B"), array.offset


def event_batch(batch: pa.RecordBatch) -> EventBatch:
    """Las siete columnas de L2 de un lote de eventos, sin copiar."""
    return EventBatch(
        batch.num_rows,
        _bytes(batch.column("reference_agg_trade_id")),
        _bytes(batch.column("confirm_agg_trade_id")),
        _bytes(batch.column("confirm_time")),
        _bytes(batch.column("extreme_agg_trade_id")),
        _bytes(batch.column("direction")),
        _bytes(batch.column("confirm_price")),
        _bytes(batch.column("extreme_price")),
    )


def push_boundaries(
    engine: dc_pyo3.Engine,
    position: int,
    batch: pa.RecordBatch,
    retained: Retained | None = None,
) -> None:
    """Entrega al motor un lote de eventos del θ en `position`, detrás de los anteriores.

    `batch` son filas consecutivas de `events.parquet` (un row group de `EventGroup.batch`
    o un tramo). El motor retiene sus buffers de Arrow (no se copian) hasta que sus eventos
    cierran, así que el llamador no necesita conservarlos. Con `retained`, anota el lote
    para medir cuántos bytes de fronteras tiene el motor en mano.
    """
    entry = event_batch(batch)
    engine.push_boundaries(position, entry)
    if retained is not None:
        retained.hold(batch.nbytes, entry.reference[0].obj)


class BoundaryFeed:
    """Las fronteras de un θ por delante de los row groups de L1, y no más (ADR-L3-16).

    `batches` son los lotes de L2 del θ en orden. Antes de empujar un row group cuyo último
    id es `high`, `cover(high)` entrega lotes hasta que el último entregado empieza después
    de `high` (ningún evento que falte toca el row group) o se acaban: lo que `Engine.push`
    exige. Los lotes se leen a medida que hacen falta y no se conservan aquí.
    """

    def __init__(
        self,
        engine: dc_pyo3.Engine,
        position: int,
        batches: Iterable[pa.RecordBatch],
        retained: Retained | None = None,
    ) -> None:
        self._engine = engine
        self._position = position
        self._retained = retained
        self._batches = iter(batches)
        self._last: int | None = None  # referencia del último evento entregado
        self._references = np.empty(0, "<i8")  # las del último lote entregado
        self._high = -1  # último id del row group que `cover` cubrió
        self.position = position
        self.delivered = 0  # eventos entregados al motor: no baja

    def cover(self, high: int) -> None:
        """Entrega lo que falta para un row group cuyo último id es `high`."""
        self._high = high
        while self._last is None or self._last <= high:
            batch = next(self._batches, None)
            if batch is None:
                return
            if batch.num_rows == 0:
                continue
            push_boundaries(self._engine, self._position, batch, self._retained)
            self.delivered += batch.num_rows
            # Sin copia (la columna no tiene nulos): solo la de este lote, ya ordenada.
            self._references = batch.column("reference_agg_trade_id").to_numpy(
                zero_copy_only=False
            )
            self._last = int(self._references[-1])

    @property
    def ahead(self) -> int:
        """Eventos entregados cuya referencia está después del último id cubierto: `cover`
        los trajo por delante y ningún tick empujado los toca aún.

        Solo el último lote puede traer alguno: `cover` sigue entregando mientras el último
        lote termina en un id ≤ `high`, así que todos los anteriores quedaron atrás.
        """
        return len(self._references) - int(
            np.searchsorted(self._references, self._high, side="right")
        )


def push(
    engine: dc_pyo3.Engine,
    row_group_id: int,
    ids: np.ndarray,
    columns: Sequence[pa.Array],
    before_month: bool,
) -> None:
    """Empuja un row group de L1 al motor, que retiene sus buffers hasta terminarlo.

    `ids` son los `agg_trade_id` de los ticks y `columns` las cuatro columnas de una trama
    en el orden de `Frame` (`transact_time`, `price`, `quantity`, `is_buyer_maker`), como
    las entrega el lector (`RowGroup.ids` y `RowGroup.columns`, o `RowGroupFrames.ids` y
    `RowGroupFrames.frame`). `before_month` dice que el row group es de un mes anterior al
    de los eventos. `row_group_id` es el ordinal del row group entre los que entrega el
    lector (0, 1, 2, ..., sin huecos). El row group no se copia: el llamador puede soltar
    sus arreglos en cuanto vuelve, y `engine.inflight()` dice cuántos siguen anclados. Las
    posiciones de las fronteras no se calculan aquí: el motor las resuelve con `ids`.
    """
    time, price, quantity, sold = columns
    engine.push(
        row_group_id,
        len(time),
        (memoryview(ids).cast("B"), 0),
        _bytes(time),
        _bytes(price),
        _bytes(quantity),
        _bytes(sold),
        before_month,
    )


def new_engine(
    thetas: Sequence[Decimal], row_groups: Sequence[Sequence[int]], threads: int | None
) -> dc_pyo3.Engine:
    """Un motor para `thetas` (los `Decimal` de la unidad) con el esqueleto de cada uno."""
    return dc_pyo3.Engine(
        [int(theta.scaleb(8)) for theta in thetas],
        ENGINE_SCHEMA,
        [list(sizes) for sizes in row_groups],
        threads,
    )


def to_batch(theta: Decimal, sealed: tuple) -> tuple[int, int, pa.RecordBatch]:
    """Un row group de salida sellado como `RecordBatch`, sin copiar los buffers del motor.

    Devuelve `(posición del θ, índice del row group de salida, batch)`.
    """
    position, index, rows, columns, validity, null_count = sealed
    constant = np.zeros((rows, 2), "<i8")
    constant[:, 0] = int(theta.scaleb(8))
    arrays = [
        pa.Array.from_buffers(
            SUMMARIES_SCHEMA.field("theta").type,
            rows,
            [None, pa.py_buffer(constant)],
        )
    ]
    fields = list(SUMMARIES_SCHEMA)[1:]
    for column, field in zip(columns, fields, strict=True):
        nulls = null_count if field.nullable else 0
        mask = (
            pa.py_buffer(validity) if field.nullable and validity is not None else None
        )
        arrays.append(
            pa.Array.from_buffers(
                field.type, rows, [mask, pa.py_buffer(column)], null_count=nulls
            )
        )
    return position, index, pa.RecordBatch.from_arrays(arrays, schema=SUMMARIES_SCHEMA)


def to_summarize_error(
    error: Exception, texts: Sequence[str], month: Month
) -> Exception:
    """El `DescriptorError` del motor como la excepción tipada que habría lanzado el
    camino por row group de Python (mismo `check_type` y mismos `details`). Cualquier
    otro error vuelve tal cual.
    """
    if not isinstance(error, dc_pyo3.DescriptorError):
        return error
    kind: str = error.kind
    theta = texts[error.theta] if hasattr(error, "theta") else ""
    event: Any = getattr(error, "confirm_agg_trade_id", None)
    match kind:
        case "direction":
            return L2StateInconsistent.mismatch(
                theta, month, "direction", error.got, "1 o -1"
            )
        case "invariant":
            return L2StateInconsistent.mismatch(
                theta, month, error.field, error.got, error.wanted
            )
        case "quantity_too_large":
            return PhaseValueError(
                theta,
                event,
                f"la cantidad de la fase {error.phase} no cabe en DECIMAL(18,8)",
            )
        case "no_quantity":
            return PhaseValueError(
                theta, event, f"la fase {error.phase} no tiene cantidad"
            )
        case "notional_overflow":
            return PhaseValueError(
                theta,
                event,
                f"la suma de price·quantity de la fase {error.phase} no cabe en 128 bits",
            )
        case "boundary":
            r, c, e = error.bounds
            name = _NAMES[BOUNDARY_COLUMNS.index(error.field)]
            return _boundary(
                month,
                f"θ {Decimal(theta)}: el id de {name} {error.got} del evento "
                f"(referencia {r}, confirmación {c}, extremo {e}) "
                f"no es un tick de L1: cae en un hueco del proveedor o L1 y L2 no "
                f"cuadran (row group de L1 con ids {error.low}..{error.high})",
                theta=theta,
                field=error.field,
                got=error.got,
                wanted="un tick de L1",
            )
        case "confirm_time":
            r, c, e = error.bounds
            return _boundary(
                month,
                f"θ {Decimal(theta)}: el tick de confirmación {c} (evento con "
                f"referencia {r}) tiene transact_time {error.got} y L2 "
                f"dice confirm_time {error.wanted}: L1 y L2 no cuadran",
                theta=theta,
                field="confirm_time",
                got=error.got,
                wanted=error.wanted,
            )
        case "ticks" | "input":
            return FramesInputError(str(error))
        case "skeleton":
            return SkeletonMismatch(
                theta,
                month,
                FAMILY,
                f"el esqueleto tiene {error.rows} filas y llega el evento {error.event}",
            )
        case "incomplete":
            # Las casillas que quedan vacías al terminar L1 son eventos sin cerrar: el
            # extremo de alguno no existe como tick (el hallazgo del lector de antes).
            result = L2StateInconsistent.mismatch(
                theta, month, "extreme_agg_trade_id", "ausente", "un tick de L1"
            )
            result.details["message"] = (
                "L1 terminó antes de cerrar todos los eventos; el extremo no existe "
                f"como tick. θ {theta}: faltan {error.missing} de las "
                f"{error.expected} filas del esqueleto"
            )
            return result
    return error


def _boundary(
    month: Month, message: str, *, theta: str, field: str, got: Any, wanted: Any
) -> L2StateInconsistent:
    """Lo que el resumidor de Python deja del `FrameBoundaryError` del lector: un
    `l2_state_inconsistent` (§9.2) con el mensaje completo en `details["message"]`.
    """
    result = L2StateInconsistent.mismatch(theta, month, field, str(got), str(wanted))
    result.details["message"] = message
    return result
