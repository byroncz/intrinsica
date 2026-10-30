"""Lo que el motor sella, hasta el disco (ADR-L3-16): un recolector y un pool de escritores.

El motor entrega sus row groups de salida (`Engine.pop_sealed`) en cuanto se llenan y cada
θ sale en orden de row group. Un hilo recolector (`collect`) los recoge y los entrega a
`SealedWriters`, que los escribe en un pool de `writers` hilos. Un `FamilyWriter` exige los
row groups de su archivo en orden (`j`, `j + 1`) y no admite dos hilos a la vez, así que
cada θ tiene una cola con su «siguiente esperado»: un row group que llega antes que su
anterior espera en la cola, y a lo más un escritor atiende a un θ a la vez. Las colas están
acotadas por los row groups en vuelo del motor, no por el mes.
"""

import threading
import time
from collections.abc import Sequence
from concurrent.futures import Executor
from decimal import Decimal
from typing import Self

import dc_pyo3
from dc_frames import FamilySkeletonMismatch, FamilyWriter
from dc_frames.lake import Month

from l3_dc_descriptors import native
from l3_dc_descriptors.errors import SkeletonMismatch
from l3_dc_descriptors.schema import FAMILY
from l3_dc_descriptors.timing import BusyClock

# Cada cuánto mira el recolector si el motor selló algo y el orquestador si hay lugar. El
# motor no avisa; con row groups de ≈ 1 s de cómputo, 2 ms cuestan menos de 0,1 % de un core.
POLL_S = 0.002


class _Queue:
    """Los row groups sellados de un θ que aún no se escribieron."""

    __slots__ = ("next", "running", "waiting")

    def __init__(self) -> None:
        self.waiting: dict[int, tuple] = {}
        self.next = 0
        self.running = False


class SealedWriters:
    """Escribe los row groups sellados: cada θ en orden, `pool` hilos a la vez.

    `put` no bloquea. `pending` es cuántos row groups están sin escribir (en cola o en
    escritura): el orquestador espera a que baje de un tope antes de empujar más al motor,
    así que lo sellado que se acumula en RAM está acotado. El primer error de un
    escritor se guarda (`error`) y detiene los demás: lo que quede en las colas se
    descarta. `clock` mide la pared con algún escritor ocupado.
    """

    def __init__(
        self,
        pool: Executor,
        writers: Sequence[FamilyWriter],
        thetas: Sequence[Decimal],
        texts: Sequence[str],
        month: Month,
        clock: BusyClock,
        family: str = FAMILY,
    ) -> None:
        self._family = family
        self._pool = pool
        self._writers = writers
        self._thetas = thetas
        self._texts = texts
        self._month = month
        self._clock = clock
        self._queues = [_Queue() for _ in writers]
        self._cond = threading.Condition()
        self._pending = 0
        self._error: BaseException | None = None

    @property
    def pending(self) -> int:
        with self._cond:
            return self._pending

    @property
    def error(self) -> BaseException | None:
        with self._cond:
            return self._error

    def put(self, sealed: tuple) -> None:
        """Encola un row group sellado `(θ, índice, …)` y despierta a un escritor si hace falta."""
        position, index = sealed[0], sealed[1]
        with self._cond:
            if self._error is not None:
                return
            queue = self._queues[position]
            if index < queue.next or index in queue.waiting:
                self._fail(
                    SkeletonMismatch(
                        self._texts[position],
                        self._month,
                        FAMILY,
                        f"el motor entregó dos veces el row group {index}",
                    )
                )
                return
            queue.waiting[index] = sealed
            self._pending += 1
            if not queue.running and queue.next in queue.waiting:
                queue.running = True
                self._pool.submit(self._drain, position)

    def _fail(self, error: BaseException) -> None:
        """Guarda el primer error y descarta lo que espera (con `_cond`)."""
        if self._error is None:
            self._error = error
        for queue in self._queues:
            self._pending -= len(queue.waiting)
            queue.waiting.clear()
        self._cond.notify_all()

    def _drain(self, position: int) -> None:
        queue = self._queues[position]
        while True:
            with self._cond:
                sealed = queue.waiting.pop(queue.next, None)
                if sealed is None or self._error is not None:
                    queue.running = False
                    self._cond.notify_all()
                    return
            try:
                self._write(position, sealed)
            except BaseException as error:  # noqa: BLE001 - sale en `error`
                with self._cond:
                    self._pending -= 1
                    queue.running = False
                    self._fail(error)
                return
            del sealed
            with self._cond:
                queue.next += 1
                self._pending -= 1
                self._cond.notify_all()

    def _write(self, position: int, sealed: tuple) -> None:
        try:
            with self._clock.busy():
                _, _, batch = native.to_batch(
                    self._thetas[position], sealed, self._family
                )
                self._writers[position].write_row_group(batch)
        except FamilySkeletonMismatch as error:
            raise SkeletonMismatch(
                self._texts[position], self._month, self._family, error.detail
            ) from error

    def join(self) -> None:
        """Espera a que se escriba todo lo entregado; relanza el primer error."""
        with self._cond:
            while self._pending > 0 and self._error is None:
                self._cond.wait()
            while any(queue.running for queue in self._queues):
                self._cond.wait()
            if self._error is not None:
                raise self._error

    def stop(self) -> None:
        """Descarta lo que espera y espera a los escritores que ya están escribiendo."""
        with self._cond:
            self._fail(self._error or _Stopped())
            while any(queue.running for queue in self._queues):
                self._cond.wait()


class _Stopped(Exception):
    """Marca interna de `stop()`: no sale de este módulo."""


class Collector:
    """Un hilo que recoge del motor lo que sella y se lo pasa a `SealedWriters`.

    Mientras el motor sigue empujando, sondea `pop_sealed` cada `POLL_S`; al `stop()` hace
    una última pasada (el motor ya terminó, así que todo lo sellado está ahí) y termina.
    Un error del motor (`DescriptorError`) o de `put` se guarda en `error` y termina el
    hilo; el orquestador lo ve en su siguiente vuelta.
    """

    def __init__(self, engine: dc_pyo3.Engine, sink: SealedWriters) -> None:
        self._engine = engine
        self._sink = sink
        self._stopping = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="l3-collector", daemon=True
        )

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        """Hace la última pasada y espera al hilo. Idempotente."""
        self._stopping.set()
        self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                last = (
                    self._stopping.is_set()
                )  # antes de vaciar: la pasada final cuenta
                while (sealed := self._engine.pop_sealed()) is not None:
                    self._sink.put(sealed)
                    del sealed
                if last:
                    return
                time.sleep(POLL_S)
        except BaseException as error:  # noqa: BLE001 - lo relanza el orquestador
            self.error = error
