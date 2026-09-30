import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import table_of
from l2_dc_events.landing import (
    COLUMNS,
    LandingError,
    consolidated_path,
    open_consolidated,
    read_batches,
    ticks_of,
)
from l2_dc_events.timing import Phases


def ticks(n, start=1):
    return [(100_000_000 * (start + i), 1_000 + i, start + i) for i in range(n)]


def test_consolidated_path_is_the_hive_layout_of_l1():
    assert consolidated_path("/l/", "binance", "spot", "BTCUSDT", 2017, 8) == (
        "/l/provider=binance/market=spot/asset=BTCUSDT/year=2017/month=08/"
        "consolidated.parquet"
    )


def test_reads_one_row_group_per_batch_and_only_the_three_columns(write_month):
    path = write_month(ticks(25), row_group_size=10)
    parquet = open_consolidated(str(path))
    assert parquet.num_row_groups == 3
    batches = list(read_batches(parquet))
    assert [b.num_rows for b in batches] == [10, 10, 5]
    assert all(
        b.schema.names == ["price", "transact_time", "agg_trade_id"] for b in batches
    )


def test_ticks_are_zero_copy_views_of_the_batch_columns(write_month):
    path = write_month(ticks(8), row_group_size=8)
    (batch,) = read_batches(open_consolidated(str(path)))
    view = ticks_of(batch)
    assert len(view) == 8
    assert view.times.tolist() == [1_000 + i for i in range(8)]
    assert view.agg_trade_ids.tolist() == list(range(1, 9))
    prices = [
        int.from_bytes(bytes(view.prices[i * 16 : (i + 1) * 16]), "little")
        for i in range(8)
    ]
    assert prices == [100_000_000 * (i + 1) for i in range(8)]


def test_ticks_of_a_sliced_batch_respects_the_offset():
    batch = table_of(ticks(10)).to_batches()[0].slice(3, 4)
    view = ticks_of(batch.select(["price", "transact_time", "agg_trade_id"]))
    assert view.agg_trade_ids.tolist() == [4, 5, 6, 7]
    assert len(view.prices) == 4 * 16


def test_nulls_are_rejected():
    column = pa.array([1, None], pa.int64())
    batch = pa.record_batch(
        [
            pa.array([1, 2], pa.decimal128(18, 8)),
            column,
            pa.array([1, 2], pa.int64()),
        ],
        names=["price", "transact_time", "agg_trade_id"],
    )
    with pytest.raises(LandingError, match="nulos"):
        ticks_of(batch)


def test_only_provisionals_fails_clearly(write_month):
    path = write_month(ticks(3), row_group_size=3, name="provisional-day=05.parquet")
    with pytest.raises(LandingError, match="solo hay 1 provisionales.*ADR-L2-09"):
        open_consolidated(str(path.with_name("consolidated.parquet")))


def test_missing_month_fails_clearly(tmp_path):
    with pytest.raises(LandingError, match="no ha publicado el mes"):
        open_consolidated(str(tmp_path / "nada" / "consolidated.parquet"))


def test_wrong_price_type_and_missing_columns_fail(tmp_path):
    bad = tmp_path / "consolidated.parquet"
    pq.write_table(
        pa.table({"price": [1.0], "transact_time": [1], "agg_trade_id": [1]}), bad
    )
    with pytest.raises(LandingError, match="price es double"):
        open_consolidated(str(bad))
    pq.write_table(pa.table({"price": pa.array([1], pa.decimal128(18, 8))}), bad)
    with pytest.raises(LandingError, match="falta la columna 'transact_time'"):
        open_consolidated(str(bad))


def test_the_whole_table_is_never_materialized(write_month):
    """La memoria de Arrow no crece con el mes: es la de un row group en vuelo."""
    groups, rows = 10, 50_000
    path = write_month(ticks(groups * rows), row_group_size=rows)
    parquet = open_consolidated(str(path))
    assert parquet.num_row_groups == groups

    one_group = rows * (16 + 8 + 8)  # price decimal128 + tiempo + id
    baseline = pa.total_allocated_bytes()
    seen = []
    for batch in read_batches(parquet):
        seen.append((batch.num_rows, pa.total_allocated_bytes() - baseline))
        del batch  # el consumidor suelta el lote antes de pedir el siguiente

    assert [n for n, _ in seen] == [rows] * groups
    # El lector de Parquet retiene buffers propios (~1,65 row groups medidos),
    # pero la memoria es la misma en el primer row group que en el último:
    # retener el anterior la subiría un row group por lote, y el mes entero
    # serían 10 row groups (25 MB).
    used = [in_flight for _, in_flight in seen]
    assert max(used) - min(used) < one_group / 2
    assert max(used) < groups * one_group / 3


def test_a_uri_root_explains_a_month_with_only_provisionals(write_month):
    """`file://` reproduce a `gs://`: el sistema de archivos rechaza URIs."""
    path = write_month(ticks(3), row_group_size=3, name="provisional-day=05.parquet")
    uri = f"file://{path.with_name('consolidated.parquet')}"
    with pytest.raises(LandingError, match="solo hay 1 provisionales.*ADR-L2-09"):
        open_consolidated(uri)


def test_a_uri_root_explains_a_missing_month(tmp_path):
    with pytest.raises(LandingError, match="no ha publicado el mes"):
        open_consolidated(f"file://{tmp_path}/nada/consolidated.parquet")


def _open_fds_of(path):
    """Descriptores abiertos sobre `path` (Linux); salta la prueba sin /proc."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        pytest.skip("sin /proc/self/fd")
    return [fd for fd in fd_dir.iterdir() if fd.exists() and fd.resolve() == path]


def test_closing_the_parquet_file_releases_the_landing_file(write_month):
    """`close()` suelta el archivo aunque el `ParquetFile` siga referenciado.

    Sin depender del recolector: es lo que evita dejar abierta la conexión a GCS.
    """
    path = write_month(ticks(3), row_group_size=3).resolve()
    parquet = open_consolidated(str(path))
    assert len(_open_fds_of(path)) == 1
    parquet.close()
    assert _open_fds_of(path) == []


def test_a_file_that_breaks_the_contract_is_not_left_open(tmp_path):
    path = (tmp_path / "consolidated.parquet").resolve()
    pq.write_table(pa.table({"price": [1]}), path)
    with pytest.raises(LandingError, match="falta la columna"):
        open_consolidated(str(path))
    assert _open_fds_of(path) == []


def test_read_batches_counts_row_groups_and_the_bytes_of_the_three_columns(write_month):
    path = write_month(ticks(25), row_group_size=10)
    parquet = open_consolidated(str(path))
    phases = Phases()
    assert sum(b.num_rows for b in read_batches(parquet, phases)) == 25
    assert phases.row_groups == 3
    columns = [
        column
        for index in range(parquet.metadata.num_row_groups)
        for column in map(
            parquet.metadata.row_group(index).column,
            range(parquet.metadata.num_columns),
        )
    ]
    three = sum(c.total_compressed_size for c in columns if c.path_in_schema in COLUMNS)
    everything = sum(c.total_compressed_size for c in columns)
    assert phases.bytes_in == three < everything


def test_waiting_is_read_time_and_the_cpu_of_the_thread_is_decode_time(
    write_month, monkeypatch
):
    parquet = open_consolidated(str(write_month(ticks(25), row_group_size=10)))
    real = parquet.read_row_group

    def slow(*args, **kwargs):  # espera de I/O: pared sin CPU
        time.sleep(0.05)
        return real(*args, **kwargs)

    monkeypatch.setattr(parquet, "read_row_group", slow)
    phases = Phases()
    list(read_batches(parquet, phases))
    assert phases.read_s >= 0.14
    assert phases.decode_s < phases.read_s / 3


def test_read_batches_without_phases_reads_the_same(write_month):
    path = write_month(ticks(25), row_group_size=10)
    plain = [b.to_pylist() for b in read_batches(open_consolidated(str(path)))]
    timed = [
        b.to_pylist() for b in read_batches(open_consolidated(str(path)), Phases())
    ]
    assert plain == timed
