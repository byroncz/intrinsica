"""Escritura de los eventos de los θ del mes en paralelo, con RAM acotada.

Cada θ escribe su propio Parquet, así que los escritores codifican y hashean a
la vez en un pool de hilos (pyarrow y `hashlib` sueltan el GIL). Dos reglas:

- **Un θ, un hilo a la vez y en orden.** Un escritor no admite dos tareas a la
  vez y sus filas deben salir en el orden en que cerraron. Cada θ tiene su cola
  y a lo más una tarea que la vacía (patrón actor); las colas de θ distintos
  corren en paralelo.
- **Tope de tramos en vuelo.** El hilo principal no pasa al tramo `k` mientras
  el `k - MAX_CHUNKS_IN_FLIGHT` no se haya escrito entero. Los eventos de un
  tramo son ~6 MiB (los 50 θ), así que el pico es O(tramo), no O(mes), aunque
  un θ lento se atrase.
"""

import threading
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import dc_pyo3

from l2_dc_events.events import EventWriter

# Tramos de `FEED_TICKS` ticks cuyos eventos pueden estar sin escribir a la vez:
# uno se escribe mientras el fan-out calcula el siguiente. Medido en 2020-01
# (14 M ticks, 10 cores): con 1 tramo, ~10 s y ~238 MiB; con 2, ~8 s y ~250; con
# 3, ~7,1 s y ~265; con 4, ~6,7 s y ~273. Se toma 2: 3 veces más rápido que la
# base (24,9 s) sin subir su RSS pico (261 MiB), que es lo que pide RNF-L2-01.
MAX_CHUNKS_IN_FLIGHT = 2


class _Chunk:
    """Los bloques de un tramo que faltan por escribir."""

    def __init__(self, pending: int) -> None:
        self._pending = pending
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.error: Exception | None = None
        if not pending:
            self._done.set()

    def finish_one(self, error: Exception | None) -> None:
        with self._lock:
            if error is not None and self.error is None:
                self.error = error
            self._pending -= 1
            if not self._pending:
                self._done.set()

    def wait(self) -> None:
        self._done.wait()
        if self.error is not None:
            raise self.error


class ParallelWriters:
    """Los `EventWriter` de los θ, alimentados en paralelo desde un solo hilo.

    `submit` recibe los bloques de un tramo (uno por θ, en el orden de los
    escritores) y vuelve en cuanto los encola, salvo que ya haya
    `MAX_CHUNKS_IN_FLIGHT` sin escribir. `wait` espera a todos. Un error de
    cualquier escritor se relanza en el siguiente `submit` o `wait`, y desde
    entonces los escritores dejan de escribir.
    """

    def __init__(
        self, pool: ThreadPoolExecutor, writers: Sequence[EventWriter]
    ) -> None:
        self._pool = pool
        self._writers = writers
        self._lock = threading.Lock()
        self._queues: list[deque[tuple[dc_pyo3.EventColumns, _Chunk]]] = [
            deque() for _ in writers
        ]
        self._running = [False] * len(writers)
        self._failed = False
        self._inflight: deque[_Chunk] = deque()

    def submit(self, blocks: Sequence[dc_pyo3.EventColumns]) -> None:
        if len(blocks) != len(self._writers):
            raise ValueError(f"{len(blocks)} bloques para {len(self._writers)} θ")
        while len(self._inflight) >= MAX_CHUNKS_IN_FLIGHT:
            self._inflight.popleft().wait()
        chunk = _Chunk(sum(1 for block in blocks if len(block)))
        self._inflight.append(chunk)
        for i, block in enumerate(blocks):
            if not len(block):
                continue
            with self._lock:
                self._queues[i].append((block, chunk))
                start = not self._running[i]
                self._running[i] = True
            if start:
                self._pool.submit(self._drain, i)

    def wait(self) -> None:
        while self._inflight:
            self._inflight.popleft().wait()

    def _drain(self, i: int) -> None:
        """Escribe, en orden, todo lo que haya en la cola del θ `i`."""
        writer = self._writers[i]
        while True:
            with self._lock:
                if not self._queues[i]:
                    self._running[i] = False
                    return
                block, chunk = self._queues[i].popleft()
            error = None
            try:
                if not self._failed:
                    writer.add(block)
            except Exception as exc:  # noqa: BLE001 - se relanza en el hilo principal
                error = exc
                self._failed = True
            # Soltar los buffers antes de esperar la siguiente tarea.
            del block
            chunk.finish_one(error)
