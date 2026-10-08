"""Tipos públicos del lector de tramas (TRD-L3 §7.4)."""

from dataclasses import dataclass
from decimal import Decimal

import pyarrow as pa


class FramesError(Exception):
    """Base de los errores del lector."""


class FramesInputError(FramesError):
    """Una entrada falta o no cumple su contrato: archivo, esquema u orden."""


class FrameBoundaryError(FramesError):
    """L1 y L2 no cuadran: una frontera de un evento no existe como tick en L1.

    El lector no corre el límite al tick vecino: falla (fail-closed).
    """


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
    """

    theta: Decimal
    event: pa.RecordBatch
    confirmation: Frame
    overshoot: Frame
