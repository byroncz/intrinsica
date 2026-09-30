"""La escritura de los θ en paralelo: orden por θ, errores y tope de tramos."""

import random
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import dc_pyo3
import pytest
from l2_dc_events import parallel
from l2_dc_events.events import to_batch
from l2_dc_events.parallel import ParallelWriters


class Block:
    """Un bloque con solo lo que `ParallelWriters` mira: su largo."""

    def __init__(self, tag, n=1):
        self.tag, self.n = tag, n

    def __len__(self):
        return self.n


class Recorder:
    """Un escritor que anota qué recibe y en qué hilo."""

    def __init__(self, delay=0.0, fail_on=None, gate=None):
        self.added, self.threads = [], set()
        self.delay, self.fail_on, self.gate = delay, fail_on, gate
        self._busy = threading.Lock()

    def add(self, block):
        # Dos tareas a la vez sobre el mismo escritor romperían el orden.
        assert self._busy.acquire(blocking=False), "dos hilos en el mismo escritor"
        try:
            if self.gate is not None:
                self.gate.wait(timeout=10)
            time.sleep(self.delay * random.random())
            if block.tag == self.fail_on:
                raise RuntimeError(f"falla en {block.tag}")
            self.added.append(block.tag)
            self.threads.add(threading.get_ident())
        finally:
            self._busy.release()


@pytest.fixture
def pool():
    with ThreadPoolExecutor(max_workers=4) as pool:
        yield pool


def test_each_theta_gets_its_blocks_in_order_and_theta_run_in_parallel(pool):
    writers = [Recorder(delay=0.002) for _ in range(12)]
    parallel_writers = ParallelWriters(pool, writers)
    for tag in range(40):
        # El θ 3 nunca emite; los demás, en desorden de tamaño.
        blocks = [Block(tag, 0 if i == 3 else 1) for i in range(len(writers))]
        parallel_writers.submit(blocks)
    parallel_writers.wait()
    for i, writer in enumerate(writers):
        assert writer.added == ([] if i == 3 else list(range(40)))
    assert len(set().union(*(w.threads for w in writers))) > 1


def test_an_error_is_raised_in_the_caller_and_stops_further_writes(pool):
    writers = [Recorder(), Recorder(fail_on=2), Recorder()]
    parallel_writers = ParallelWriters(pool, writers)
    with pytest.raises(RuntimeError, match="falla en 2"):
        for tag in range(50):
            parallel_writers.submit([Block(tag)] * 3)
        parallel_writers.wait()
    assert writers[1].added == [0, 1]


def test_at_most_max_chunks_are_in_flight(pool, monkeypatch):
    monkeypatch.setattr(parallel, "MAX_CHUNKS_IN_FLIGHT", 2)
    gate = threading.Event()
    writers = [Recorder(gate=gate)]
    parallel_writers = ParallelWriters(pool, writers)
    submitted = []

    def feed():
        for tag in range(4):
            parallel_writers.submit([Block(tag)])
            submitted.append(tag)

    thread = threading.Thread(target=feed)
    thread.start()
    time.sleep(0.3)
    # Con el escritor detenido, el tercer tramo no entra: quedan dos en vuelo.
    assert submitted == [0, 1]
    gate.set()
    thread.join(timeout=10)
    parallel_writers.wait()
    assert submitted == [0, 1, 2, 3]
    assert writers[0].added == [0, 1, 2, 3]


def test_blocks_must_match_the_writers(pool):
    with pytest.raises(ValueError, match="bloques"):
        ParallelWriters(pool, [Recorder()]).submit([])


def test_to_batch_wraps_the_binding_buffers_without_changing_the_values():
    fanout = dc_pyo3.FanOut([10_000_000])
    scale = dc_pyo3.SCALE
    prices = (100, 111, 120, 107, 100)
    times = memoryview(array("q", range(1, 6)))
    price_buffer = memoryview(
        b"".join((p * scale).to_bytes(16, "little") for p in prices)
    )
    (block,) = fanout.feed_batch_columns(price_buffer, times, times)
    batch = to_batch(block, 10_000_000)
    assert batch.num_rows == 1
    (row,) = batch.to_pylist()
    assert row["direction"] == 1
    assert (row["reference_price"], row["confirm_price"], row["extreme_price"]) == (
        100,
        111,
        120,
    )
    assert (row["reference_time"], row["confirm_time"], row["extreme_time"]) == (
        1,
        2,
        3,
    )
    assert str(row["theta"]) == "0.10000000"
