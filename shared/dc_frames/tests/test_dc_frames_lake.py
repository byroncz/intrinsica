"""Apertura de Parquet: `pre_buffer` y el pool de E/S de Arrow (`DC_FRAMES_IO_THREADS`)."""

import os
import pathlib
import subprocess
import sys
from contextlib import closing

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import FramesInputError, lake


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


# Archivos abiertos una vez y leídos por tramos (ITSC-363)


@pytest.fixture
def events_path(tmp_path):
    from frames_lake import KEY, MONTH, build_lake, theta_text

    _, l2_root, _, events = build_lake(tmp_path, l2_row_group=7)
    theta = min(events)
    path = lake.l2_path(str(l2_root), KEY, theta_text(theta), MONTH)
    return pathlib.Path(path), events[theta]


EVENT_COLUMNS = ("direction", "confirm_price", "extreme_price")


def test_an_event_file_opens_once_and_reads_by_lots_of_row_groups(
    events_path, monkeypatch
):
    path, rows = events_path
    opened = []
    real = lake.resolve_fs

    def counting(target):
        opened.append(target)
        return real(target)

    monkeypatch.setattr(lake, "resolve_fs", counting)
    with lake.EventFile(str(path), "L2", EVENT_COLUMNS) as file:
        assert len(opened) == 1
        assert file.columns[:4] == [
            "reference_agg_trade_id",
            "confirm_agg_trade_id",
            "extreme_agg_trade_id",
            "confirm_time",
        ]
        assert file.columns[4:] == list(EVENT_COLUMNS)
        sizes = [rows for rows in file.rows if rows]
        assert sum(sizes) == len(rows) and len(sizes) > 3
        one_each = file.lots(1)
        assert [len(lot) for lot in one_each] == [1] * len(sizes)
        everything = file.lots(1 << 40)
        assert len(everything) == 1 and everything[0] == list(range(len(sizes)))
        # Un tramo se cierra al llegar a los bytes pedidos: cada uno (salvo el último) los
        # alcanza y sin su último row group no los alcanzaba.
        wanted_bytes = 2 * file.compressed([0])
        two = file.lots(wanted_bytes)
        assert [i for lot in two for i in lot] == list(range(len(sizes)))
        for lot in two[:-1]:
            assert file.compressed(lot) >= wanted_bytes > file.compressed(lot[:-1])
        # Cada tramo se lee con un lector propio y el archivo no se vuelve a abrir.
        got = pa.Table.from_batches([b for lot in two for b in file.batches(lot, 5)])
        assert opened == [str(path)]
    wanted = pq.read_table(path, columns=file.columns)
    assert got.equals(wanted)


def test_batches_have_their_own_buffers_and_at_most_batch_rows_rows(events_path):
    path, _ = events_path
    with lake.EventFile(str(path), "L2") as file:
        batches = list(file.batches(file.lots(1 << 40)[0], 4))
    assert all(b.num_rows <= 4 for b in batches)
    first, second = batches[0].column(0).buffers()[1], batches[1].column(0).buffers()[1]
    assert first.address != second.address
    assert sum(b.num_rows for b in batches) == sum(file.rows)


def test_an_event_file_gives_the_id_ranges_from_the_footer(events_path):
    path, rows = events_path
    with lake.EventFile(str(path), "L2") as file:
        assert file.id_range("reference_agg_trade_id") == (
            min(int(r["reference_agg_trade_id"]) for r in rows),
            max(int(r["reference_agg_trade_id"]) for r in rows),
        )
        assert file.id_range("extreme_agg_trade_id")[1] == max(
            int(r["extreme_agg_trade_id"]) for r in rows
        )


def test_an_event_file_without_statistics_has_no_id_range(tmp_path):
    table = pa.table(
        {
            name: pa.array([1, 2, 3], pa.int64())
            for name in (*lake.BOUNDARY_COLUMNS, lake.CONFIRM_TIME)
        }
    )
    path = tmp_path / "events.parquet"
    pq.write_table(table, path, write_statistics=False)
    with lake.EventFile(str(path), "L2") as file:
        assert file.id_range("reference_agg_trade_id") is None
    pq.write_table(table.slice(0, 0), path)
    with lake.EventFile(str(path), "L2") as file:
        assert file.id_range("reference_agg_trade_id") is None
        assert file.lots(1) == []


def test_an_event_file_that_is_missing_or_lacks_a_column_fails_and_closes(tmp_path):
    with pytest.raises(FramesInputError, match="falta L2 del θ"):
        lake.EventFile(str(tmp_path / "no-existe.parquet"), "L2 del θ 1")
    table = pa.table({"reference_agg_trade_id": pa.array([1], pa.int64())})
    path = tmp_path / "events.parquet"
    pq.write_table(table, path)
    with pytest.raises(FramesInputError, match="confirm_agg_trade_id"):
        lake.EventFile(str(path), "L2")
    full = pa.table(
        {
            name: pa.array([1], pa.int64())
            for name in (*lake.BOUNDARY_COLUMNS, lake.CONFIRM_TIME)
        }
    )
    pq.write_table(full, path)
    with pytest.raises(FramesInputError, match="direction"):
        lake.EventFile(str(path), "L2", ("direction",))
    wrong = pa.table(
        {name: pa.array([1], pa.int32()) for name in lake.BOUNDARY_COLUMNS}
        | {lake.CONFIRM_TIME: pa.array([1], pa.int64())}
    )
    pq.write_table(wrong, path)
    with pytest.raises(FramesInputError, match="int32"):
        lake.EventFile(str(path), "L2")


def test_readers_of_one_source_can_read_at_the_same_time(events_path):
    import threading

    path, _ = events_path
    with lake.ParquetSource(str(path), "L2") as source:
        groups = source.metadata.num_row_groups
        assert groups > 3
        results: dict[int, list] = {}
        barrier = threading.Barrier(groups)

        def read(i: int) -> None:
            with closing(source.reader()) as reader:
                barrier.wait()
                results[i] = (
                    reader.read_row_group(i).column("confirm_agg_trade_id").to_pylist()
                )

        threads = [threading.Thread(target=read, args=(i,)) for i in range(groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    whole = pq.read_table(path).column("confirm_agg_trade_id").to_pylist()
    assert [v for i in range(groups) for v in results[i]] == whole
