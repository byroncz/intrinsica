"""El motor nativo de agregación contra el resumidor de Python y contra el camino por
evento (ITSC-358, ADR-L3-16).

Tres implementaciones deben dar la misma tabla y el mismo `content_hash`: el resumidor de
Python por row group (`rowgroup.py`, la referencia), el camino por evento (`by_event`, el
oráculo) y el motor de Rust, con 1 y 4 hilos y en cualquier orden de `push`. Los row
groups de L1 son de 50 ticks para que los eventos crucen row groups y las fronteras caigan
en sus bordes.
"""

from decimal import Decimal

import by_event
import dc_pyo3
import pyarrow.parquet as pq
import pytest
from dc_frames import l2_path
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


@pytest.mark.parametrize("lot", [1, 3])
@pytest.mark.parametrize("order", ["forward", "swapped"])
def test_boundaries_in_small_lots_interleaved_with_the_pushes_change_nothing(
    edge_lake, tmp_path, lot, order
):
    """Las fronteras en lotes de pocos eventos, entregadas justo antes del row group que
    las necesita y no más (`BoundaryFeed`), dan la misma tabla, el mismo hash y la misma
    conciliación que todas de una vez (ADR-L3-16).
    """
    lake, reference = edge_lake
    for month in lake.months:
        got = native(
            lake,
            tmp_path / f"{order}{lot}",
            month,
            threads=4,
            order=order,
            lot=lot,
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


def frame_columns(rg):
    frame = rg.frame
    return frame.transact_time, frame.price, frame.quantity, frame.is_buyer_maker


def pushed_engine(lake, threads=2):
    """Un motor con todos los row groups del primer mes y el primer θ ya empujados.

    Devuelve `(engine, ref, lot)`: `ref` es una referencia débil al `memoryview` del precio
    del primer row group y `lot` otra al de `reference_agg_trade_id` del primer lote de
    fronteras, para observar cuándo el motor los suelta.
    """
    import weakref
    from decimal import Decimal

    from dc_frames import l2_path, read_row_groups
    from dc_frames.lake import event_groups
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
    lots = [
        native.event_batch(group.batch)
        for group in event_groups(
            str(lake.l2), DEFAULT_KEY, text, month, month, EVENT_COLUMNS
        )
    ]
    lot = weakref.ref(lots[0].reference[0])
    for batch in lots:
        engine.push_boundaries(0, batch)
    del lots, batch
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
        if k == 0:
            watched = native._bytes(rg.frame.price)[0]
            watched_args = (
                k,
                len(rg.frame),
                (memoryview(rg.ids).cast("B"), 0),
                native._bytes(rg.frame.transact_time),
                (watched, 0),
                native._bytes(rg.frame.quantity),
                native._bytes(rg.frame.is_buyer_maker),
                rg.before_start,
            )
            engine.push(*watched_args)
            del watched_args
        else:
            native.push(engine, k, rg.ids, frame_columns(rg), rg.before_start)
    return engine, weakref.ref(watched), lot


def test_sealed_columns_are_read_only_buffers_that_arrow_wraps_without_copying(
    edge_lake,
):
    import numpy as np
    import pyarrow as pa

    lake, _ = edge_lake
    engine, _, _ = pushed_engine(lake)
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
    engine, watched, _ = pushed_engine(lake)
    engine.finish()
    deadline = time.monotonic() + 5
    while watched() is not None and time.monotonic() < deadline:
        engine.inflight()  # cada llamada procesa los decref que dejaron los hilos
        time.sleep(0.01)
    assert watched() is None, "el motor sigue anclando el row group"
    assert engine.inflight() == 0
    engine.close()


def test_the_engine_releases_a_lot_of_boundaries_when_its_events_closed(edge_lake):
    """Con el motor vivo, un lote de fronteras cuyos eventos cerraron ya no está anclado
    (su `memoryview` muere) y `needed` llega al total del θ: el motor no retiene las
    fronteras del mes (ADR-L3-16).
    """
    import time

    lake, _ = edge_lake
    engine, _, lot = pushed_engine(lake)
    ((rows, *_),) = engine.finish()
    deadline = time.monotonic() + 5
    while lot() is not None and time.monotonic() < deadline:
        engine.inflight()  # cada llamada procesa los decref que dejaron los hilos
        time.sleep(0.01)
    assert lot() is None, "el motor sigue anclando el lote de fronteras"
    assert engine.needed(0) == rows > 0
    engine.close()


def test_a_row_group_before_its_boundaries_is_a_contract_error(edge_lake):
    """`push` no espera fronteras que no llegaron: falla con `kind` `input`."""
    from decimal import Decimal

    from dc_frames import read_row_groups
    from l3_dc_descriptors import native
    from l3_dc_descriptors.event import EVENT_COLUMNS
    from l3_dc_descriptors.summarize import DEFAULT_KEY

    lake, _ = edge_lake
    month, text = lake.months[0], lake.theta_texts[0]
    engine = native.new_engine([Decimal(text)], [[1]], 1)
    rg = next(
        iter(
            read_row_groups(
                [Decimal(text)],
                str(lake.l1),
                str(lake.l2),
                month,
                month,
                *DEFAULT_KEY,
                columns=EVENT_COLUMNS,
            )
        )
    )
    with pytest.raises(
        dc_pyo3.DescriptorError, match="antes de las fronteras"
    ) as error:
        native.push(engine, 0, rg.ids, frame_columns(rg), rg.before_start)
    assert error.value.kind == "input"
    engine.close()


def test_the_engine_locates_the_boundaries_like_the_reader_row_group_by_row_group(
    edge_lake,
):
    """Las posiciones que el motor calcula con los ids son las que `read_row_groups`
    entrega: mismas filas, mismos cortes `(R, C, E)`, mismas banderas, y la misma cuenta
    de ticks anteriores al mes.
    """
    import numpy as np
    from dc_frames import read_row_groups
    from dc_frames.lake import event_groups
    from l3_dc_descriptors import native
    from l3_dc_descriptors.event import EVENT_COLUMNS
    from l3_dc_descriptors.summarize import DEFAULT_KEY

    lake, _ = edge_lake
    values = [Decimal(text) for text in lake.theta_texts]
    compared = crossing = 0
    for month in lake.months:
        groups = list(
            read_row_groups(
                values,
                str(lake.l1),
                str(lake.l2),
                month,
                month,
                *DEFAULT_KEY,
                columns=EVENT_COLUMNS,
            )
        )
        skeletons = [
            [
                pq.ParquetFile(l2_path(str(lake.l2), DEFAULT_KEY, text, month))
                .metadata.row_group(i)
                .num_rows
                for i in range(
                    pq.ParquetFile(
                        l2_path(str(lake.l2), DEFAULT_KEY, text, month)
                    ).metadata.num_row_groups
                )
            ]
            for text in lake.theta_texts
        ]
        engine = native.new_engine(values, skeletons, 2)
        for position, text in enumerate(lake.theta_texts):
            for group in event_groups(
                str(lake.l2), DEFAULT_KEY, text, month, month, EVENT_COLUMNS
            ):
                native.push_boundaries(engine, position, group.batch)
        seen = [0] * len(values)
        before = [0] * len(values)
        for k, rg in enumerate(groups):
            previous = int(groups[k - 1].ids[-1]) if k else None
            by_theta = {t.theta: t for t in rg.thetas}
            for position, theta in enumerate(values):
                found = engine.locate(
                    position,
                    len(rg.frame),
                    (memoryview(rg.ids).cast("B"), 0),
                    native._bytes(rg.frame.transact_time),
                    previous,
                )
                expected = by_theta.get(theta)
                if expected is None:
                    assert found == [], (month, k, theta)
                    continue
                resumed = int(expected.starts_before[0])
                first = seen[position] - resumed
                seen[position] += len(expected) - resumed
                before[position] += int(expected.ticks_before_month.sum())
                assert [at[0] for at in found] == list(
                    range(first, first + len(expected))
                ), (month, k, theta)
                assert np.array_equal(
                    np.array([at[1] for at in found], np.int64), expected.positions
                ), (month, k, theta)
                assert [at[2] for at in found] == expected.starts_before.tolist()
                assert [at[3] for at in found] == expected.ends_after.tolist()
                compared += len(found)
                crossing += int(
                    expected.starts_before.sum() + expected.ends_after.sum()
                )
            native.push(engine, k, rg.ids, frame_columns(rg), rg.before_start)
        stats = engine.finish()
        assert [s[2] for s in stats] == before
        engine.close()
    assert compared > 200 and crossing > 20, (compared, crossing)


@pytest.mark.parametrize("kind", ["not_a_tick", "confirm_time"])
def test_a_boundary_that_does_not_match_l1_raises_the_error_of_the_reader(
    tmp_path, kind
):
    """El motor falla con las mismas comprobaciones del lector y la misma excepción que el
    resumidor de Python: `l2_state_inconsistent`, con los mismos `details` y mensaje.
    """
    ticks = fixture_ticks(seed=3, drop=0.1)
    lake = build_lake(
        tmp_path / "lake", ticks, month_cuts(ticks, 2400), l1_row_group=ROW_GROUP
    )
    month = lake.months[-1]
    theta = 250_000
    present = {tick.id for tick in lake.all_ticks()}
    rows = lake.events[(theta, month)]
    if kind == "confirm_time":
        row = 3
        column, value = "confirm_time", rows[row]["confirm_time"] + 1
    else:
        # Un evento con un id sin tick entre su referencia y su confirmación.
        row, value = next(
            (i, gap)
            for i, event in enumerate(rows)
            for gap in range(
                event["reference_agg_trade_id"] + 1, event["confirm_agg_trade_id"]
            )
            if gap not in present
        )
        column = "confirm_agg_trade_id"
    tamper(lake, theta, month, column, row, value)
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
    assert isinstance(got.value.__cause__, dc_pyo3.DescriptorError)
    assert got.value.check_type == wanted.value.check_type
    assert got.value.details == wanted.value.details
    assert "message" in got.value.details
