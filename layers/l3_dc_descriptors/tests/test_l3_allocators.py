"""El umbral de `mmap` de glibc es `MALLOC_MMAP_THRESHOLD_` si es válido y 16 KiB si no
(ITSC-366); `unit_timing` lo registra (prueba en test_l3_timing).
"""

import logging

import pytest
from l3_dc_descriptors import memory
from l3_dc_descriptors.memory import DEFAULT_MMAP_THRESHOLD, MAX_MMAP_THRESHOLD

ENV = "MALLOC_MMAP_THRESHOLD_"


class FakeLibc:
    """Un `ctypes.CDLL(None)` que anota lo que recibe `mallopt`."""

    def __init__(self, accepts: bool = True) -> None:
        self.calls: list[tuple[int, int]] = []
        self.accepts = accepts

    def mallopt(self, option: int, value: int) -> int:
        self.calls.append((option, value))
        return int(self.accepts)


@pytest.fixture
def libc(monkeypatch):
    fake = FakeLibc()
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: fake)
    monkeypatch.delenv(
        ENV, raising=False
    )  # que el fixture restaure lo que tune_allocators deje
    return fake


def test_without_the_variable_the_threshold_is_16_kib(libc, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    memory.tune_allocators()
    assert libc.calls == [(memory._M_MMAP_THRESHOLD, 16384)]
    assert DEFAULT_MMAP_THRESHOLD == 16384
    assert memory.mmap_threshold() == 16384


@pytest.mark.parametrize("value", [0, 65536, 33554432])
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
    assert libc.calls == [(memory._M_MMAP_THRESHOLD, 16384)]
    assert memory.mmap_threshold() == 16384
    assert any(ENV in record.getMessage() for record in caplog.records)


def test_a_rejected_threshold_stays_dynamic(monkeypatch, caplog):
    fake = FakeLibc(accepts=False)
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: fake)
    monkeypatch.setenv(ENV, "65536")
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        memory.tune_allocators()
    assert memory.mmap_threshold() is None
    assert any("rechazó" in record.getMessage() for record in caplog.records)
