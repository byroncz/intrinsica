"""Apertura de Parquet: `pre_buffer` y el pool de E/S de Arrow (`DC_FRAMES_IO_THREADS`)."""

import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import lake


def io_threads_after_import(**env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k != lake.IO_THREADS_ENV}
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import dc_frames, pyarrow as pa; print(pa.io_thread_count())",
        ],
        capture_output=True,
        text=True,
        env={**base, **env},
        check=False,
    )


def test_the_io_pool_is_8_threads_by_default():
    done = io_threads_after_import()
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(lake.DEFAULT_IO_THREADS) == "8"


def test_the_env_var_sets_the_io_pool_at_import():
    done = io_threads_after_import(DC_FRAMES_IO_THREADS="3")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "3"


@pytest.mark.parametrize("text", ["0", "-2", "ocho", ""])
def test_a_bad_value_fails_loudly(monkeypatch, text):
    monkeypatch.setenv(lake.IO_THREADS_ENV, text)
    with pytest.raises(ValueError, match="DC_FRAMES_IO_THREADS"):
        lake.io_threads()


def test_open_parquet_passes_pre_buffer_on_request_and_the_footer_to_arrow(
    tmp_path, monkeypatch
):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), path, row_group_size=2)
    seen: dict = {}
    real = lake.pq.ParquetFile

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(lake.pq, "ParquetFile", spy)
    first = lake.open_parquet(str(path), "x")
    assert seen["pre_buffer"] is False and seen["metadata"] is None
    lake.open_parquet(str(path), "x", pre_buffer=True).close()
    assert seen["pre_buffer"] is True
    again = lake.open_parquet(str(path), "x", first.metadata)
    assert seen["metadata"] is first.metadata
    assert again.read_row_group(1).column("a").to_pylist() == [3]
    first.close()
    again.close()
