"""Bindings PyO3 sobre dc_core (fan-out de N θ y carry-over) y dc_descriptors (motor de L3)."""

from collections.abc import Sequence
from typing import Protocol, Self

SCALE: int
PRICE_LIMIT: int
STATE_VERSION: str

# (price, time, agg_trade_id): precio entero en escala SCALE, tiempo en µs UTC.
type Point = tuple[int, int, int]

class Event:
    @property
    def direction(self) -> int: ...
    @property
    def reference(self) -> Point: ...
    @property
    def confirm(self) -> Point: ...
    @property
    def extreme(self) -> Point: ...

class EventColumns:
    """Los eventos de un θ como columnas, sin un `Event` por evento.

    `buffers()` son 10 `bytes` en el orden de `events.parquet` (`reference_*`,
    `confirm_*`, `extreme_*`: precio, tiempo, `agg_trade_id`; luego
    `direction`), en el layout de Arrow: precio `decimal128` (16 B
    little-endian), tiempo e id `int64`, `direction` `int8`. `theta` no va.
    """

    def buffers(self) -> list[bytes]: ...
    def __len__(self) -> int: ...

class CarryOver:
    def __new__(
        cls,
        theta: int,
        state_version: str,
        direction: int,
        ext_high: Point,
        ext_low: Point,
        pending: tuple[Point, Point] | None = None,
    ) -> Self: ...
    @property
    def theta(self) -> int: ...
    @property
    def state_version(self) -> str: ...
    @property
    def direction(self) -> int: ...
    @property
    def ext_high(self) -> Point: ...
    @property
    def ext_low(self) -> Point: ...
    @property
    def pending(self) -> tuple[Point, Point] | None: ...
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes) -> CarryOver: ...

class FanOut:
    def __new__(cls, thetas: Sequence[int], threads: int | None = None) -> Self: ...
    @staticmethod
    def from_carry_over(
        thetas: Sequence[int],
        carry: Sequence[CarryOver],
        threads: int | None = None,
    ) -> FanOut: ...
    @property
    def thetas(self) -> list[int]: ...
    def feed_batch(
        self,
        prices: memoryview,
        times: memoryview,
        ids: memoryview,
    ) -> list[list[Event]]: ...
    def feed_batch_columns(
        self,
        prices: memoryview,
        times: memoryview,
        ids: memoryview,
    ) -> list[EventColumns]: ...
    def finish(self) -> list[Event | None]: ...
    def finish_columns(self) -> list[EventColumns]: ...
    def discarded(self) -> list[int]: ...
    def carry_overs(self) -> list[CarryOver]: ...
    def __len__(self) -> int: ...

class DescriptorError(ValueError):
    """Un fallo del motor de L3. `kind` dice cuál: `input`, `ticks`, `internal`,
    `skeleton`, `incomplete`, `direction`, `invariant`, `quantity_too_large`,
    `no_quantity` o `notional_overflow`. Según el caso trae `theta` (posición),
    `event`, `confirm_agg_trade_id`, `phase`, `field`, `got`, `wanted`, `rows`,
    `missing` o `expected`.
    """

    kind: str

class SealedColumn:
    """Bytes de una columna de salida en el layout de Arrow, de solo lectura, con
    protocolo de buffer (`pyarrow.py_buffer`, `memoryview`). Poseen la memoria del motor."""

    def __len__(self) -> int: ...

class Piece(Protocol):
    """Las columnas de L2 de los eventos de un θ en un row group (cualquier objeto con
    estos atributos). `positions` es `(n, 3)` `int64`; los flags, `n` bytes 0/1;
    `direction`, `int8`; los precios, `(bytes del buffer decimal128, offset)`."""

    theta: int
    first_event: int
    positions: object
    starts_before: object
    ends_after: object
    ticks_before_month: object
    reference: object
    confirm: object
    confirm_time: object
    extreme: object
    direction: object
    confirm_price: tuple[memoryview, int]
    extreme_price: tuple[memoryview, int]

# (rows, closed_ticks, before_month, first_reference, last_extreme)
type Stats = tuple[int, int, int, int | None, int | None]
type Sealed = tuple[int, int, int, list[SealedColumn], SealedColumn | None, int]

class Engine:
    def __new__(
        cls,
        thetas: Sequence[int],
        schema: Sequence[tuple[str, int, bool]],
        row_groups: Sequence[Sequence[int]],
        threads: int | None = None,
    ) -> Self: ...
    def push(
        self,
        row_group_id: int,
        rows: int,
        time: tuple[memoryview, int],
        price: tuple[memoryview, int],
        quantity: tuple[memoryview, int],
        sold: tuple[memoryview, int],
        pieces: Sequence[Piece],
    ) -> None: ...
    def pop_sealed(self) -> Sealed | None: ...
    def inflight(self) -> int: ...
    def busy(self) -> float: ...
    def finish(self) -> list[Stats]: ...
    def close(self) -> None: ...
