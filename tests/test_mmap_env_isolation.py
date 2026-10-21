"""Ninguna prueba deja `MALLOC_MMAP_THRESHOLD_` distinta a como la encontró (ITSC-366)."""

import os
from pathlib import Path

MMAP_ENV = "MALLOC_MMAP_THRESHOLD_"
ROOT = Path(__file__).resolve().parents[1]


def test_a_variable_written_by_a_test_does_not_reach_the_next(pytester):
    pytester.makeconftest((ROOT / "conftest.py").read_text())
    pytester.makepyfile(
        f"""
        import os

        BEFORE = {os.environ.get(MMAP_ENV)!r}

        def test_a_writes():
            os.environ["{MMAP_ENV}"] = "16384"  # lo que deja `tune_allocators`

        def test_b_sees_the_original():
            assert os.environ.get("{MMAP_ENV}") == BEFORE
        """
    )
    pytester.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=2)
