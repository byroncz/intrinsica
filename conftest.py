"""Fixtures de toda la suite."""

import os

import pytest

pytest_plugins = ["pytester"]

MMAP_ENV = "MALLOC_MMAP_THRESHOLD_"


@pytest.fixture(autouse=True)
def restore_mmap_threshold(monkeypatch):
    """Deja `MALLOC_MMAP_THRESHOLD_` como estaba al terminar cada prueba (ITSC-366).

    `l3_dc_descriptors.cli.main` llama a `tune_allocators`, que escribe la variable en
    `os.environ`. Sin esto se filtra al resto de la sesión y los subprocesos de otras
    pruebas (la medición de RSS de `test_l3_memory.py`) la heredan: su glibc la lee al
    arrancar y mide con otro asignador según el orden de la suite. `delenv` sobre una
    variable ausente no anota nada, por eso se hace `setenv` antes.
    """
    before = os.environ.get(MMAP_ENV)
    if before is None:
        monkeypatch.setenv(MMAP_ENV, "")
        monkeypatch.delenv(MMAP_ENV)
    else:
        monkeypatch.setenv(MMAP_ENV, before)
