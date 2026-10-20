"""El motor nativo de agregación contra el resumidor de Python y contra el camino por
evento (ITSC-358, ADR-L3-16).

Tres implementaciones deben dar la misma tabla y el mismo `content_hash`: el resumidor de
Python por row group (`rowgroup.py`, la referencia), el camino por evento (`by_event`, el
oráculo) y el motor de Rust, con 1 y 4 hilos y en cualquier orden de `push`. Los row
groups de L1 son de 50 ticks para que los eventos crucen row groups y las fronteras caigan
en sus bordes.
"""

import by_event
import dc_pyo3
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors import (
    SUMMARIES_SCHEMA,
    L2StateInconsistent,
    summarize_month,
)
from l3_lake import M1, build_lake, fixture_ticks, month_cuts
from native_driver import summarize_native
from pyutils import content_hash

ROW_GROUP = 50


@pytest.fixture(scope="module")
def edge_lake(tmp_path_factory):
    ticks = fixture_ticks(seed=11, drop=0.12)
    lake = build_lake(
        tmp_path_factory.mktemp("native"),
        ticks,
        month_cuts(ticks, 2300),
        l1_row_group=ROW_GROUP,
    )
    reference = {
        month: summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            threads=1,
            prefetch=0,
        )
        for month in lake.months
    }
    return lake, reference


def native(lake, tmp_path, month, **kwargs):
    return summarize_native(
        lake.l1, lake.l2, tmp_path, month, lake.theta_texts, series_start=M1, **kwargs
    )


def assert_same(got, wanted):
    assert got.keys() == wanted.keys()
    for theta, result in got.items():
        other = wanted[theta]
        assert result.rows == other.rows
        assert result.content_hash == other.content_hash
        table = pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
        assert table.equals(pq.read_table(other.path, schema=SUMMARIES_SCHEMA))
        assert content_hash(table) == result.content_hash
        # Lo que alimenta la conciliación de §9.1 sale igual: incluye la primera
        # referencia y el último extremo, que decide el motor.
        assert result.reconciliation == other.reconciliation


@pytest.mark.parametrize("threads", [1, 4])
def test_same_tables_hashes_and_reconciliation_as_the_python_summarizer(
    edge_lake, tmp_path, threads
):
    lake, reference = edge_lake
    compared = 0
    for month in lake.months:
        got = native(lake, tmp_path / f"t{threads}", month, threads=threads)
        assert_same(got, reference[month])
        compared += sum(r.rows for r in got.values())
    assert compared == sum(len(rows) for rows in lake.events.values())


def test_passes_the_event_by_event_oracle(edge_lake, tmp_path):
    lake, _ = edge_lake
    for month in lake.months:
        oracle = by_event.summarize_month_by_event(
            lake.l1, lake.l2, tmp_path / "oracle", month, lake.theta_texts
        )
        got = native(lake, tmp_path / "native", month)
        for theta, result in got.items():
            wanted = oracle[theta]
            assert result.rows == wanted["rows"]
            assert result.content_hash == wanted["content_hash"]
            assert result.reconciliation.closed == wanted["closed_ticks"]
            assert result.reconciliation.before_month == wanted["before_month"]
            table = pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
            assert table.equals(pq.read_table(wanted["path"], schema=SUMMARIES_SCHEMA))


@pytest.mark.parametrize("order", ["reversed", "swapped", "shuffled"])
@pytest.mark.parametrize("threads", [1, 4])
def test_the_push_order_does_not_change_a_bit(edge_lake, tmp_path, order, threads):
    """Barajar el orden de `push` no cambia el resultado (el motor admite cualquier
    orden; el orquestador solo adelanta unos pocos row groups).
    """
    lake, reference = edge_lake
    month = lake.months[-1]
    for seed in range(3):
        got = native(
            lake,
            tmp_path / f"{order}{threads}{seed}",
            month,
            threads=threads,
            order=order,
            seed=seed,
        )
        assert_same(got, reference[month])


def tamper(lake, theta: int, month, column: str, row: int, value) -> None:
    """Cambia una fila de `events.parquet` de L2 conservando sus row groups."""
    import pyarrow as pa
    from dc_frames import l2_path
    from l3_dc_descriptors.summarize import DEFAULT_KEY

    text = f"{theta / 10**8:.8f}"
    path = l2_path(str(lake.l2), DEFAULT_KEY, text, month)
    table = pq.read_table(path)
    sizes = pq.ParquetFile(path).metadata.row_group(0).num_rows
    values = table.column(column).to_pylist()
    values[row] = value
    column_index = table.schema.get_field_index(column)
    table = table.set_column(
        column_index,
        table.schema.field(column),
        pa.array(values, table.schema.field(column).type),
    )
    pq.write_table(table, path, row_group_size=sizes)


@pytest.mark.parametrize(
    ("column", "value", "field"),
    [
        ("direction", 0, "direction"),
        ("confirm_price", None, "confirm_price"),
    ],
)
def test_a_broken_event_raises_the_same_error_as_the_python_summarizer(
    tmp_path, column, value, field
):
    from decimal import Decimal

    ticks = fixture_ticks(seed=3, drop=0.1)
    lake = build_lake(
        tmp_path / "lake", ticks, month_cuts(ticks, 2400), l1_row_group=ROW_GROUP
    )
    month = lake.months[-1]
    theta = 250_000
    value = Decimal("1e9") if column == "confirm_price" else value
    tamper(lake, theta, month, column, 3, value)
    texts = [f"{theta / 10**8:.8f}"]
    with pytest.raises(L2StateInconsistent) as wanted:
        summarize_month(
            lake.l1,
            lake.l2,
            tmp_path / "py",
            month,
            texts,
            series_start=M1,
            threads=1,
            prefetch=0,
        )
    with pytest.raises(L2StateInconsistent) as got:
        summarize_native(
            lake.l1, lake.l2, tmp_path / "rs", month, texts, series_start=M1, threads=4
        )
    assert got.value.check_type == wanted.value.check_type == "l2_state_inconsistent"
    assert got.value.details == wanted.value.details
    assert wanted.value.details["field"] == field
    assert isinstance(got.value.__cause__, dc_pyo3.DescriptorError)


def test_the_engine_checks_its_output_schema_and_can_be_closed():
    from l3_dc_descriptors import native

    wrong = [*native.ENGINE_SCHEMA[:-1], ("confirm_instant_price_worst", 8, False)]
    with pytest.raises(ValueError, match="esquema de salida"):
        dc_pyo3.Engine([100_000], wrong, [[1]])
    engine = dc_pyo3.Engine([100_000], native.ENGINE_SCHEMA, [[1]], 2)
    assert (engine.inflight(), engine.busy()) == (0, 0.0)
    engine.close()
    engine.close()  # idempotente
    with pytest.raises(ValueError, match="cerrado"):
        engine.inflight()


def pushed_engine(lake, threads=2):
    """Un motor con todos los row groups del primer mes y el primer θ ya empujados.

    Devuelve `(engine, ref)`: `ref` es una referencia débil al `memoryview` del precio
    del primer row group, para observar cuándo el motor lo suelta.
    """
    import weakref
    from decimal import Decimal

    from dc_frames import l2_path, read_row_groups
    from l3_dc_descriptors import native
    from l3_dc_descriptors.event import EVENT_COLUMNS
    from l3_dc_descriptors.summarize import DEFAULT_KEY

    month, text = lake.months[0], lake.theta_texts[0]
    values = [Decimal(text)]
    parquet = pq.ParquetFile(l2_path(str(lake.l2), DEFAULT_KEY, text, month))
    sizes = [
        parquet.metadata.row_group(i).num_rows
        for i in range(parquet.metadata.num_row_groups)
    ]
    engine = native.new_engine(values, [sizes], threads)
    feeder = native.Feeder(values)
    watched = None
    groups = read_row_groups(
        values,
        str(lake.l1),
        str(lake.l2),
        month,
        month,
        *DEFAULT_KEY,
        columns=EVENT_COLUMNS,
    )
    for k, rg in enumerate(groups):
        pieces = feeder.pieces(rg)
        if k == 0:
            watched = native._bytes(rg.frame.price)[0]
            watched_args = (
                k,
                len(rg.frame),
                native._bytes(rg.frame.transact_time),
                (watched, 0),
                native._bytes(rg.frame.quantity),
                native._bytes(rg.frame.is_buyer_maker),
                pieces,
            )
            engine.push(*watched_args)
            del watched_args
        else:
            native.push(engine, k, rg, pieces)
    return engine, weakref.ref(watched)


def test_sealed_columns_are_read_only_buffers_that_arrow_wraps_without_copying(
    edge_lake,
):
    import numpy as np
    import pyarrow as pa

    lake, _ = edge_lake
    engine, _ = pushed_engine(lake)
    engine.finish()
    position, index, rows, columns, validity, null_count = engine.pop_sealed()
    assert (position, index) == (0, 0)
    ticks = memoryview(columns[1])
    assert ticks.readonly and len(ticks) == 8 * rows
    with pytest.raises(TypeError):
        ticks[0] = 1
    # Arrow y NumPy ven el mismo bloque: ninguna copia.
    assert (
        pa.py_buffer(columns[1]).address
        == np.frombuffer(columns[1], np.int64).ctypes.data
    )
    assert (validity is None) == (null_count == 0)
    engine.close()


def test_the_engine_releases_a_row_group_when_its_items_are_done(edge_lake):
    """`inflight()` baja a 0 y el motor suelta los buffers del row group: nadie más los
    ancla (el `memoryview` que se le pasó muere).
    """
    import time

    lake, _ = edge_lake
    engine, watched = pushed_engine(lake)
    engine.finish()
    deadline = time.monotonic() + 5
    while watched() is not None and time.monotonic() < deadline:
        engine.inflight()  # cada llamada procesa los decref que dejaron los hilos
        time.sleep(0.01)
    assert watched() is None, "el motor sigue anclando el row group"
    assert engine.inflight() == 0
    engine.close()
