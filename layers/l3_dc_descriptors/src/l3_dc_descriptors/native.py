"""El adaptador del motor nativo de agregación (`dc_pyo3.Engine`, ITSC-358).

El motor (crate `dc_descriptors`) agrega por row group y θ en hilos nativos; esta capa de
Python solo lo orquesta: traduce un `RowGroupFrames` del lector a los buffers de Arrow y
a las columnas de L2 que `Engine.push` espera, envuelve en Arrow los row groups de salida
que el motor sella y convierte sus errores en las excepciones tipadas de `errors`. No
agrega nada: la aritmética y los invariantes de §7.2 viven en Rust (ADR-L3-16).

Todavía no está en el flujo de `summarize_month`: lo conecta el orquestador (ITSC-358 es
solo el motor). `rowgroup.py` sigue siendo la referencia con la que se compara.
"""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, NamedTuple

import dc_pyo3
import numpy as np
import pyarrow as pa
from dc_frames import FramesInputError, RowGroupFrames, ThetaEvents
from dc_frames.lake import Month

from l3_dc_descriptors.errors import (
    L2StateInconsistent,
    PhaseValueError,
    SkeletonMismatch,
)
from l3_dc_descriptors.schema import FAMILY, SUMMARIES_SCHEMA

# Las 18 columnas de datos como las comprueba el motor: (nombre, bytes por valor, nulable).
ENGINE_SCHEMA = tuple(
    (field.name, 16 if pa.types.is_decimal(field.type) else 8, field.nullable)
    for field in list(SUMMARIES_SCHEMA)[1:]
)


class Piece(NamedTuple):
    """Las columnas de L2 de los eventos de un θ en un row group (`Engine.push`)."""

    theta: int  # posición del θ en la lista del motor
    first_event: int  # índice de la primera fila en `events.parquet` de L2
    positions: np.ndarray  # (n, 3) int64
    starts_before: np.ndarray  # n bytes 0/1
    ends_after: np.ndarray
    ticks_before_month: np.ndarray
    reference: np.ndarray
    confirm: np.ndarray
    confirm_time: np.ndarray
    extreme: np.ndarray
    direction: np.ndarray  # int8
    confirm_price: tuple[memoryview, int]  # (bytes del buffer, offset) decimal128
    extreme_price: tuple[memoryview, int]


def _bytes(array: pa.Array) -> tuple[memoryview, int]:
    """Los bytes del buffer de datos de un array de Arrow y su offset, sin copia."""
    buffer = array.buffers()[1]
    return memoryview(buffer if buffer is not None else b"").cast("B"), array.offset


def _column(events: pa.RecordBatch, name: str) -> np.ndarray:
    return events.column(name).to_numpy(zero_copy_only=True)


class Feeder:
    """Arma lo que `Engine.push` recibe, llevando la cuenta de eventos de cada θ.

    El índice de un evento en `events.parquet` es su casilla en el motor. El lector
    entrega los eventos de cada θ en orden y repite el evento abierto al pasar de un row
    group al siguiente (`starts_before`), así que el primer evento de una entrega es el
    siguiente a los ya vistos, o el último visto si continúa. Hay que llamar a `pieces`
    en el orden del lector, un row group tras otro.
    """

    def __init__(self, thetas: Sequence[Decimal]) -> None:
        self.index = {theta: i for i, theta in enumerate(thetas)}
        self._seen = [0] * len(self.index)

    def piece(self, theta: ThetaEvents) -> Piece:
        i = self.index[theta.theta]
        events = theta.events
        resumed = int(theta.starts_before[0])
        first = self._seen[i] - resumed
        self._seen[i] += len(theta) - resumed
        return Piece(
            i,
            first,
            np.ascontiguousarray(theta.positions, np.int64),
            theta.starts_before.view(np.uint8),
            theta.ends_after.view(np.uint8),
            np.ascontiguousarray(theta.ticks_before_month, np.int64),
            _column(events, "reference_agg_trade_id"),
            _column(events, "confirm_agg_trade_id"),
            _column(events, "confirm_time"),
            _column(events, "extreme_agg_trade_id"),
            _column(events, "direction"),
            _bytes(events.column("confirm_price")),
            _bytes(events.column("extreme_price")),
        )

    def pieces(self, rg: RowGroupFrames) -> list[Piece]:
        return [self.piece(theta) for theta in rg.thetas]


def push(
    engine: dc_pyo3.Engine, row_group_id: int, rg: RowGroupFrames, pieces: list[Piece]
) -> None:
    """Empuja un row group al motor, que retiene sus buffers hasta terminarlo.

    `row_group_id` es el ordinal del row group entre los que entrega el lector (0, 1, 2,
    ..., sin huecos). El row group no se copia: el llamador puede soltar `rg` en cuanto
    vuelve, y `engine.inflight()` dice cuántos siguen anclados.
    """
    frame = rg.frame
    engine.push(
        row_group_id,
        len(frame),
        _bytes(frame.transact_time),
        _bytes(frame.price),
        _bytes(frame.quantity),
        _bytes(frame.is_buyer_maker),
        pieces,
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
            return SkeletonMismatch(
                theta,
                month,
                FAMILY,
                f"faltan {error.missing} de las {error.expected} filas del esqueleto",
            )
    return error
