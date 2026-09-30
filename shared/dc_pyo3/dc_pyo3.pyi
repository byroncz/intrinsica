"""Bindings PyO3 sobre dc_core: fan-out de N θ y carry-over."""

from collections.abc import Sequence
from typing import Self

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
