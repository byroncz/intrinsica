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
    `skeleton`, `incomplete`, `boundary`, `confirm_time`, `direction`, `invariant`,
    `quantity_too_large`, `no_quantity` o `notional_overflow`. Según el caso trae `theta`
    (posición), `event`, `confirm_agg_trade_id`, `phase`, `field`, `got`, `wanted`, `rows`,
    `missing` o `expected`. `boundary` (una frontera de un evento que no es un tick de L1)
    trae además `reason` (`not_a_tick` o `unseen`), `bounds` (`R`, `C`, `E`), `low` y
    `high` (primer y último id del row group); `field` es la columna de L2 de la frontera.
    `confirm_time` trae `bounds`, `got` (el `transact_time` del tick `C`) y `wanted`.
    """

    kind: str

class SealedColumn:
    """Bytes de una columna de salida en el layout de Arrow, de solo lectura, con
    protocolo de buffer (`pyarrow.py_buffer`, `memoryview`). Poseen la memoria del motor."""

    def __len__(self) -> int: ...

class EventBatch(Protocol):
    """Un row group de `events.parquet` de L2 de un θ (cualquier objeto con estos
    atributos): `rows` y, por columna, `(bytes del buffer, offset)` en el layout de Arrow.
    `reference`, `confirm`, `confirm_time` y `extreme` son `int64`; `direction`, `int8`;
    los precios, `decimal128`."""

    rows: int
    reference: tuple[memoryview, int]
    confirm: tuple[memoryview, int]
    confirm_time: tuple[memoryview, int]
    extreme: tuple[memoryview, int]
    direction: tuple[memoryview, int]
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
    def set_boundaries(self, theta: int, batches: Sequence[EventBatch]) -> None: ...
    def push(
        self,
        row_group_id: int,
        rows: int,
        ids: tuple[memoryview, int],
        time: tuple[memoryview, int],
        price: tuple[memoryview, int],
        quantity: tuple[memoryview, int],
        sold: tuple[memoryview, int],
        before_month: bool,
    ) -> None: ...
    def locate(
        self,
        theta: int,
        rows: int,
        ids: tuple[memoryview, int],
        time: tuple[memoryview, int],
        prev_high: int | None = None,
    ) -> list[tuple[int, tuple[int, int, int], bool, bool]]: ...
    def pop_sealed(self) -> Sealed | None: ...
    def inflight(self) -> int: ...
    def busy(self) -> float: ...
    def finish(self) -> list[Stats]: ...
    def close(self) -> None: ...
