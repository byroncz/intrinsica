"""Tipos públicos del lector de tramas (TRD-L3 §7.4)."""

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pyarrow as pa


class FramesError(Exception):
    """Base de los errores del lector."""


class FramesInputError(FramesError):
    """Una entrada falta o no cumple su contrato: archivo, esquema u orden."""


class FrameBoundaryError(FramesError):
    """L1 y L2 no cuadran: una frontera de un evento no existe como tick en L1.

    El lector no corre el límite al tick vecino: falla (fail-closed). `theta` (con 8
    decimales), `field` (la columna de L2 que no cuadra), `got` y `wanted` describen el
    fallo sin tener que leer el mensaje.
    """

    def __init__(
        self, message: str, *, theta: str, field: str, got: object, wanted: object
    ) -> None:
        super().__init__(message)
        self.theta = theta
        self.field = field
        self.got = str(got)
        self.wanted = str(wanted)


class FamilySkeletonMismatch(FramesError):
    """Un archivo de familia no repite filas, orden o row groups del esqueleto de la partición.

    Es el hallazgo `family_skeleton_mismatch` (TRD-L3 §7.3): el escritor aborta y no
    publica nada. `detail` dice qué difiere.
    """

    check_type = "family_skeleton_mismatch"

    def __init__(self, message: str, detail: str) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Frame:
    """Los ticks de una fase: cuatro arrays de Arrow del mismo largo y en orden de la serie.

    Los tipos son los de L1: `transact_time` INT64 (µs UTC), `price` y `quantity`
    `decimal128(18,8)` e `is_buyer_maker` BOOL. Dentro de un row group son slices sin
    copia de los buffers de L1; si se conservan, anclan ese row group en memoria.
    """

    transact_time: pa.Array
    price: pa.Array
    quantity: pa.Array
    is_buyer_maker: pa.Array

    def __len__(self) -> int:
        return len(self.transact_time)


@dataclass(frozen=True, slots=True)
class EventFrames:
    """Un evento cerrado: su fila de L2 y las tramas de sus dos fases.

    `event` es la fila de `events.parquet` sin transformar (un `RecordBatch` de una
    fila). `confirmation` son los ticks de `(R, C]` y `overshoot` los de `(C, E]`;
    el overshoot es vacío (largo 0) si `E = C`.

    En **modo por chunks** (`read_frames(chunks=True)`) cada fase es un iterador de
    `Frame`, uno por row group con ticks de esa fase, en orden de la serie; el
    overshoot vacío es un iterador sin elementos. Cada iterador se recorre una sola
    vez y se consume antes de pedir el siguiente evento: después, ancla los row groups
    de sus trozos. Los dos últimos trozos son slices del row group en curso y del
    anterior; los que preceden se releen de L1 al avanzar, de uno en uno.

    `ticks_before_month` son los ticks de `(R, E]` que están en meses de L1 anteriores
    al de la partición de L2 que cierra el evento; 0 si el evento nació en ese mes.

    `over_budget` marca un evento que supera `max_event_ticks` (modo por evento): sus
    dos tramas vienen vacías.
    """

    theta: Decimal
    event: pa.RecordBatch
    confirmation: Frame | Iterator[Frame]
    overshoot: Frame | Iterator[Frame]
    ticks_before_month: int = 0
    over_budget: bool = False


@dataclass(frozen=True, slots=True)
class ThetaEvents:
    """Los eventos de un θ que tocan un row group de L1, como arreglos y sin ticks.

    Un evento toca el row group desde el que contiene su `reference_agg_trade_id` hasta
    el que contiene su `extreme_agg_trade_id`: aparece en cada uno de ellos, en orden de
    `confirm_agg_trade_id`. Todos los arreglos tienen una fila por evento.

    `events` son las filas de L2 (las siete columnas de TRD-L3 §7.1, o las pedidas en
    `columns`). `positions` es de forma `(n, 3)` con las posiciones `R`, `C` y `E` en el
    row group: `searchsorted(agg_trade_id, id, "right")`, es decir, cuántos ticks del
    row group tienen id ≤ la frontera. La confirmación es `[R, C)` y el overshoot
    `[C, E)`, medio abiertos; las fases consecutivas de un θ comparten corte
    (`E_i = R_{i+1}`). Una frontera fuera del row group queda en 0 (anterior) o en su
    largo (posterior).

    `starts_before` dice que el evento empieza antes del row group (`R` < primer id) y
    `ends_after` que termina después (`E` > último id): un evento con ambas banderas
    falsas cabe entero; con `starts_before` falso y `ends_after` verdadero, abre aquí; y
    con `ends_after` falso, cierra aquí. `ticks_before_month` son los ticks de `(R, E]`
    en row groups de un mes anterior al de la partición de L2 que cierra el evento. Solo
    vale donde `ends_after` es falso (el evento cierra en este row group); en los demás
    es 0.
    """

    theta: Decimal
    events: pa.RecordBatch
    positions: np.ndarray
    starts_before: np.ndarray
    ends_after: np.ndarray
    ticks_before_month: np.ndarray

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class RowGroupFrames:
    """Un row group de L1 con los eventos de todos los θ que lo tocan (ADR-L3-14).

    `month` es el mes del archivo de L1 del que sale y `before_start` dice que es
    anterior al `start` del pase: lo recorre solo para completar eventos que cruzan.
    `frame` son los cuatro arrays del row group, vistas sin copia del buffer de Arrow
    con los tipos de L1; sus posiciones son las de `ThetaEvents.positions`. `thetas`
    trae solo los θ con algún evento que toca el row group, en el orden pedido.

    El lector no retiene el row group: el consumidor debe soltar el elemento (y todo lo
    que sea vista de él) antes de pedir el siguiente, o conviven dos.
    """

    month: tuple[int, int]
    before_start: bool
    frame: Frame
    thetas: tuple[ThetaEvents, ...]
