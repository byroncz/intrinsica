"""ITSC-290, bloque 2: la codificación física de la salida no cambia su contenido."""

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest
from l2_dc_events.carry import read_carry_over
from l2_dc_events.pipeline import RunContext, Unit, process_unit
from l2_dc_events.schema import EVENTS_SCHEMA
from l2_dc_events.write import CARRY_OVER, EVENTS, partition_path
from pyutils import content_hash

AUG = Unit(2017, 8)


@pytest.fixture
def ctx(write_month, tmp_path):
    return RunContext(
        mode="backfill",
        run_id="test",
        image_version="0.1.0+test",
        series_start=(2017, 8),
        landing_root=write_month.landing,
        events_root=tmp_path / "events",
        dq_root=tmp_path / "dq",
    )


@pytest.fixture
def month(fixture_ticks, write_month, ctx):
    write_month(fixture_ticks, row_group_size=1_000)
    result = process_unit(AUG, ctx)
    return ctx, result


def _text(values):
    """Los motores devuelven decimales como `Decimal` o texto: se comparan en texto."""
    return [None if v is None else str(v) for v in values]


def _path(ctx, theta, filename):
    return partition_path(
        ctx.events_root, "binance", "spot", "BTCUSDT", theta, 2017, 8, filename
    )


def _output_files(ctx, result):
    from l2_dc_events.thetas import load_thetas

    for theta, n in zip(load_thetas(), result.events_per_theta, strict=True):
        yield theta, n, _path(ctx, theta, EVENTS), _path(ctx, theta, CARRY_OVER)


def test_output_has_no_dictionary_and_delta_on_the_integer_columns(month):
    ctx, result = month
    checked = 0
    for _, n, events, carry in _output_files(ctx, result):
        for path in (events, carry):
            meta = pq.ParquetFile(path).metadata
            for g in range(meta.num_row_groups):
                for c in range(meta.num_columns):
                    column = meta.row_group(g).column(c)
                    assert not column.has_dictionary_page, (path, column.path_in_schema)
                    if column.physical_type in ("INT32", "INT64") and n:
                        assert "DELTA_BINARY_PACKED" in column.encodings
                        checked += 1
                    if column.physical_type == "FIXED_LEN_BYTE_ARRAY":
                        assert "DELTA_BINARY_PACKED" not in column.encodings
    assert checked


def test_content_hash_is_the_same_as_the_hash_of_the_values_read_back(month):
    """El hash de la corrida es el del contenido: coincide con el de la tabla leída."""
    ctx, result = month
    for (theta, n, events, carry), events_hash, carry_hash in zip(
        _output_files(ctx, result),
        result.events_hashes,
        result.carry_over_hashes,
        strict=True,
    ):
        assert content_hash(pq.read_table(events)) == events_hash, theta
        assert content_hash(pq.read_table(carry)) == carry_hash, theta
        assert pq.read_table(events).num_rows == n


def test_pyarrow_duckdb_and_polars_read_the_same_values(month):
    ctx, result = month
    busiest = max(_output_files(ctx, result), key=lambda f: f[1])
    _, n, events, carry = busiest
    assert n > 100

    assert pq.read_table(events).schema.equals(EVENTS_SCHEMA)
    for path, columns in ((events, EVENTS_SCHEMA.names), (carry, None)):
        arrow = pq.read_table(path)
        duck = duckdb.sql(
            f"SELECT * FROM read_parquet('{path}', hive_partitioning = false)"
        ).to_arrow_table()
        pol = pl.read_parquet(path, hive_partitioning=False)
        assert duck.num_rows == pol.height == arrow.num_rows
        for name in columns or arrow.column_names:
            values = _text(arrow.column(name).to_pylist())
            assert _text(duck.column(name).to_pylist()) == values, name
            assert _text(pol.get_column(name).to_list()) == values, name


def test_carry_over_written_with_the_new_encoding_still_loads(month):
    ctx, result = month
    for theta, _, _, carry in _output_files(ctx, result):
        assert read_carry_over(carry, theta).theta == theta
