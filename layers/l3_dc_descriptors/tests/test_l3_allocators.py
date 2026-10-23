"""El umbral de `mmap` de glibc es `MALLOC_MMAP_THRESHOLD_` si es válido y 32 MiB si no
(ITSC-366, ITSC-367); `unit_timing` lo registra (prueba en test_l3_timing).
"""

import logging
import os
import time

import pytest
from l3_dc_descriptors import memory
from l3_dc_descriptors.memory import DEFAULT_MMAP_THRESHOLD, MAX_MMAP_THRESHOLD

ENV = "MALLOC_MMAP_THRESHOLD_"
TRIM_ENV = "MALLOC_TRIM_THRESHOLD_"


class FakeLibc:
    """Un `ctypes.CDLL(None)` que anota lo que recibe `mallopt`."""

    def __init__(self, accepts: bool = True) -> None:
        self.calls: list[tuple[int, int]] = []
        self.accepts = accepts

    def mallopt(self, option: int, value: int) -> int:
        self.calls.append((option, value))
        return int(self.accepts)


def forget_env(mp: pytest.MonkeyPatch) -> None:
    """Quita la variable y deja anotado que, al deshacer, vuelva a estar ausente.

    `delenv` sobre una variable que no existe no anota nada: lo que `tune_allocators`
    escribe después se filtraría al resto de la sesión. `setenv` primero sí lo anota.
    """
    mp.setenv(ENV, "")
    mp.delenv(ENV)


@pytest.fixture
def libc(monkeypatch):
    fake = FakeLibc()
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: fake)
    forget_env(monkeypatch)
    return fake


def test_forgetting_the_variable_restores_its_absence(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    mp = pytest.MonkeyPatch()
    forget_env(mp)
    os.environ[ENV] = "33554432"  # lo que deja `tune_allocators`
    mp.undo()
    assert ENV not in os.environ


def test_without_the_variable_the_threshold_is_32_mib(libc):
    memory.tune_allocators()
    assert libc.calls == [(memory._M_MMAP_THRESHOLD, 33554432)]
    assert DEFAULT_MMAP_THRESHOLD == 33554432 == MAX_MMAP_THRESHOLD
    assert memory.mmap_threshold() == 33554432


@pytest.mark.parametrize("value", [0, 16384, 65536, 33554432])
def test_a_valid_variable_is_what_mallopt_receives(libc, monkeypatch, value):
    monkeypatch.setenv(ENV, str(value))
    memory.tune_allocators()
    assert libc.calls == [(memory._M_MMAP_THRESHOLD, value)]
    assert memory.mmap_threshold() == value


@pytest.mark.parametrize(
    "text",
    ["", "mucho", "1.5", "-1", " 5", "1_000", str(MAX_MMAP_THRESHOLD + 1), "0x10000"],
)
def test_an_invalid_variable_warns_and_falls_back_to_the_default(
    libc, monkeypatch, caplog, text
):
    monkeypatch.setenv(ENV, text)
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        memory.tune_allocators()
    assert libc.calls == [(memory._M_MMAP_THRESHOLD, DEFAULT_MMAP_THRESHOLD)]
    assert memory.mmap_threshold() == DEFAULT_MMAP_THRESHOLD
    assert any(ENV in record.getMessage() for record in caplog.records)


def test_a_rejected_threshold_stays_dynamic(monkeypatch, caplog):
    fake = FakeLibc(accepts=False)
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: fake)
    monkeypatch.setenv(ENV, "65536")
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        memory.tune_allocators()
    assert memory.mmap_threshold() is None
    assert any("rechazó" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, None),
        ("1073741824", 1073741824),
        ("0", 0),
        ("", None),
        ("1G", None),
        ("-1", None),
    ],
)
def test_trim_threshold_is_what_the_environment_says_or_none(
    monkeypatch, text, expected
):
    monkeypatch.delenv(TRIM_ENV, raising=False)
    if text is not None:
        monkeypatch.setenv(TRIM_ENV, text)
    assert memory.trim_threshold() == expected


def test_release_to_kernel_trims_every_arena_and_reports_the_rss(monkeypatch):
    calls: list[int] = []

    class Libc:
        def malloc_trim(self, pad: int) -> int:
            calls.append(pad)
            return 1

    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: Libc())
    before, after = memory.release_to_kernel()
    assert calls == [0]
    assert before is not None and before > 0
    assert after is not None and after > 0


def test_release_to_kernel_without_malloc_trim_warns_and_goes_on(monkeypatch, caplog):
    class Libc:  # un libc sin malloc_trim: `getattr` lanza AttributeError
        pass

    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: Libc())
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        before, after = memory.release_to_kernel()
    assert before is not None and after is not None
    assert any("malloc_trim" in record.getMessage() for record in caplog.records)


def test_rss_bytes_is_the_current_rss_not_the_peak():
    rss = memory.rss_bytes()
    assert rss is not None and rss > 1 << 20


# ITSC-375: el muestreador de memoria y sus muestras.


@pytest.mark.parametrize(
    ("text", "expected"),
    [(None, None), ("", None), (" ", None), ("0.05", 0.05), ("1", 1.0), (" 2 ", 2.0)],
)
def test_sample_period_is_the_environment_value_or_none(text, expected):
    env = {} if text is None else {"L3_MEMORY_SAMPLE_S": text}
    assert memory.sample_period(env) == expected


@pytest.mark.parametrize("text", ["0", "-1", "nan", "inf", "mucho", "1s"])
def test_a_bad_sample_period_raises_with_the_variable_name(text):
    with pytest.raises(ValueError, match="L3_MEMORY_SAMPLE_S"):
        memory.sample_period({"L3_MEMORY_SAMPLE_S": text})


def test_a_sample_has_the_glibc_accounting_and_the_arrow_bytes():
    sample = memory.take_sample(time.perf_counter(), lambda: 3)
    assert sample.rss is not None and sample.rss > 0
    # Las arenas piden al sistema lo que está en uso más lo libre que retienen.
    assert sample.arena == sample.uordblks + sample.fordblks
    assert sample.hblkhd is not None and sample.keepcost is not None
    assert sample.engine_inflight == 3 and sample.elapsed_s >= 0
    details = sample.details()
    assert list(details) == [
        "rss_mib",
        "arena_mib",
        "hblkhd_mib",
        "uordblks_mib",
        "fordblks_mib",
        "keepcost_mib",
        "arrow_mib",
        "engine_inflight",
        "elapsed_s",
    ]
    assert details["rss_mib"] == round(sample.rss / (1 << 20), 1)


def test_a_sample_counts_the_arrow_buffers_alive():
    import pyarrow as pa

    before = memory.take_sample(time.perf_counter())
    buffer = pa.allocate_buffer(64 << 20)
    during = memory.take_sample(time.perf_counter())
    assert during.arrow - before.arrow >= buffer.size
    del buffer


def test_a_sample_without_mallinfo2_has_no_glibc_fields_and_does_not_fail(
    monkeypatch, caplog
):
    memory._mallinfo2_function.cache_clear()

    class Libc:  # un libc sin mallinfo2: `getattr` lanza AttributeError
        pass

    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: Libc())
    try:
        sample = memory.take_sample(time.perf_counter())
        with caplog.at_level(logging.WARNING, logger=memory.logger.name):
            sampler = memory.MemorySampler(0.01)
            sampler.start()
            sampler.stop()
    finally:
        memory._mallinfo2_function.cache_clear()
    assert sample.rss is not None and sample.arrow >= 0
    glibc = ("arena", "hblkhd", "uordblks", "fordblks", "keepcost")
    assert all(getattr(sample, name) is None for name in glibc)
    assert sample.details()["arena_mib"] is None
    assert sampler.peak is not None and sampler.peak.arena is None
    assert any("mallinfo2" in record.getMessage() for record in caplog.records)


def test_a_sample_survives_an_engine_that_already_closed():
    def closed() -> int:
        raise RuntimeError("el motor está cerrado")

    assert memory.take_sample(time.perf_counter(), closed).engine_inflight is None


def test_the_sampler_keeps_the_sample_of_the_highest_rss_and_stops_with_the_unit():
    import pyarrow as pa

    sampler = memory.MemorySampler(0.01)
    sampler.inflight = lambda: 7
    sampler.start()
    held = [pa.allocate_buffer(32 << 20) for _ in range(4)]
    for buffer in held:  # toca las páginas para que cuenten en el RSS
        buffer.to_pybytes()
    time.sleep(0.15)
    sampler.stop()
    ticks = sampler.ticks
    assert not sampler._thread.is_alive()
    assert ticks >= 2
    assert sampler.peak is not None and sampler.peak.engine_inflight == 7
    assert sampler.peak.arrow >= 4 * (32 << 20)
    time.sleep(0.05)
    assert sampler.ticks == ticks  # parado: no sigue muestreando
    del held
