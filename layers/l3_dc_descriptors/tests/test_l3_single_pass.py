"""Lo que la unidad deja de hacer dos veces (ITSC-365, ADR-L3-16): escritor sin relectura del
esqueleto, linaje desde los bytes ya leídos, compuerta y apertura en una pasada y L1 leído una
vez. Cada prueba fija una garantía que se conserva y la compara con la referencia de antes
(`reference_gate`): el mismo valor, sin el viaje de más.
"""

import threading
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import FramesInputError
from dc_frames import family as family_module
from dc_frames import lake as frames_lake
from dc_frames.lake import L1, EventFile, l1_path
from l3_dc_descriptors import SkeletonMismatch, boundaries, native, summarize_month
from l3_dc_descriptors import summarize as orchestrator
from l3_dc_descriptors.boundaries import L2Events
from l3_dc_descriptors.reconcile import L1Month
from l3_dc_descriptors.summarize import DEFAULT_KEY, DEFAULT_WRITERS, _engine_threads
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts
from reference_gate import file_hash, lineage

SMALL = 100_000
PARTITION = "provider=binance/market=spot/asset=BTCUSDT"


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    ticks = fixture_ticks(seed=5, drop=0.05)
    return build_lake(
        tmp_path_factory.mktemp("lake"),
        ticks,
        month_cuts(ticks, 1500, 3000),
        l1_row_group=150,
    )


def run(lake, month=M2, thetas=None, l3=None, **options):
    return summarize_month(
        lake.l1,
        lake.l2,
        l3 or lake.l3,
        month,
        thetas or lake.theta_texts,
        series_start=M1,
        **options,
    )


def theta_text(theta: int) -> str:
    return f"{Decimal(theta).scaleb(-8):.8f}"


# --- Linaje desde los bytes ya leídos ------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("small_lots", [False, True])
def test_the_lineage_is_what_rereading_the_files_gives(
    lake, monkeypatch, threads, small_lots
):
    """El hash de `events` y de `carry_over` de cada θ es el de `file_hash`, que relee el
    archivo y es el que L2 publica; no cambia con los lotes ni con el pool.
    """
    if small_lots:
        monkeypatch.setattr(boundaries, "L2_LOT_BYTES", 1)
        monkeypatch.setattr(boundaries, "L2_BATCH_ROWS", 3)
    for month in lake.months:
        results = run(lake, month, threads=threads)
        for text, result in zip(lake.theta_texts, results.values(), strict=True):
            events, carry = lineage(str(lake.l2), DEFAULT_KEY, text, month)
            assert result.l2_events_content_hash == events
            assert result.l2_carry_over_content_hash == carry


def test_the_lineage_of_a_month_without_events_is_the_hash_of_an_empty_file(lake):
    """Un mes cuyo L2 no trae eventos también tiene hash: el del archivo vacío."""
    empty = [m for m in lake.months if not any(lake.events[t, m] for t in lake.thetas)]
    for month in empty:
        for text, result in zip(
            lake.theta_texts, run(lake, month).values(), strict=True
        ):
            assert (
                result.l2_events_content_hash
                == lineage(str(lake.l2), DEFAULT_KEY, text, month)[0]
            )


def test_every_l2_batch_is_read_even_if_the_engine_never_asked_for_it(lake):
    """El hash cubre el archivo entero aunque el motor no pida el final de las fronteras."""
    results = run(lake, M2, engine_threads=1, readers=1, inflight=1, writers=1)
    for text, result in zip(lake.theta_texts, results.values(), strict=True):
        path = next(lake.l2.rglob(f"theta={text}/year=2017/month=09/events.parquet"))
        assert result.l2_events_content_hash == file_hash(str(path))


def test_l2_events_hash_matches_file_hash_for_each_column_order(lake):
    """`L2Events.tap` hashea en el orden del archivo aunque el lote traiga las fronteras
    primero.
    """
    path = next(lake.l2.rglob("theta=0.00100000/year=2017/month=09/events.parquet"))
    with EventFile(str(path), "L2", all_columns=True) as file:
        events = L2Events(file, Decimal("0.001"))
        for lot in file.lots(1 << 20):
            for batch in file.batches(lot, 4, 1 << 20):
                assert events.tap(batch).schema.names == list(boundaries.EVENT_COLUMNS)
        assert file.schema.names != file.columns  # el lote sale en otro orden
        assert events.hexdigest() == file_hash(str(path))


# --- Escritor sin relectura del esqueleto --------------------------------------------------


def test_the_writer_never_opens_the_skeleton_nor_rereads_events(lake, monkeypatch):
    opened_skeletons = []
    real = family_module.ParquetSource

    def spy(*args):
        opened_skeletons.append(args)
        return real(*args)

    event_files = []
    init = EventFile.__init__

    def spy_file(self, path, *args, **kwargs):
        event_files.append(path)
        init(self, path, *args, **kwargs)

    reads = []
    read = pq.ParquetFile.read_row_group

    def spy_read(self, i, columns=None, **kwargs):
        reads.append(self.metadata.num_columns)
        return read(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(family_module, "ParquetSource", spy)
    monkeypatch.setattr(EventFile, "__init__", spy_file)
    run(lake, M2)
    assert opened_skeletons == []
    # Un `EventFile` por θ y ninguno más: el archivo se abre una vez.
    assert sorted(event_files) == sorted(set(event_files))
    assert len(event_files) == len(lake.thetas)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_read)
    run(lake, M2)
    # Los `read_row_group` que quedan son los del lector de L1 (8 columnas de un archivo
    # de 8), nunca el de 2 claves de `events.parquet`.
    assert reads and set(reads) == {
        frames_lake.pq.ParquetFile(
            l1_path(lake.l1, DEFAULT_KEY, M2)
        ).metadata.num_columns
    }


def alter_first_sealed(monkeypatch, name, change):
    """El primer row group que el motor sella para el segundo θ sale con `name` cambiada:
    `change(valores)` devuelve la columna nueva (un `pa.array`).
    """
    sealed = native.to_batch

    def altered(theta, item):
        position, index, batch = sealed(theta, item)
        if position == 1 and index == 0:
            at = batch.schema.get_field_index(name)
            column = change(batch.column(at))
            batch = batch.set_column(at, batch.schema.field(at), column)
        return position, index, batch

    monkeypatch.setattr(native, "to_batch", altered)


def shift_first_id(column):
    ids = column.to_numpy().copy()
    ids[0] -= 1
    return pa.array(ids)


def null_second_id(column):
    ids = column.to_pylist()
    ids[1] = None
    return pa.array(ids, column.type)


def other_theta(column):
    return pa.array([Decimal("0.00000007")] * len(column), column.type)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize(
    ("name", "change", "detail"),
    [
        ("confirm_agg_trade_id", shift_first_id, "confirm_agg_trade_id no es"),
        ("confirm_agg_trade_id", null_second_id, "fila 1: confirm_agg_trade_id no es"),
        ("theta", other_theta, "fila 0: theta no es"),
    ],
    ids=["id", "null id", "theta"],
)
def test_a_misaligned_row_group_is_a_family_skeleton_mismatch(
    lake, monkeypatch, tmp_path, threads, name, change, detail
):
    """Una clave distinta de la que L3 leyó en L2, en un row group sellado, aborta con el
    mismo `family_skeleton_mismatch` que daba la relectura del esqueleto, también si el
    id es nulo y también si lo que cambió es el `theta`.
    """
    alter_first_sealed(monkeypatch, name, change)
    with pytest.raises(SkeletonMismatch) as error:
        run(lake, M2, l3=tmp_path, threads=threads)
    assert error.value.check_type == "family_skeleton_mismatch"
    assert error.value.details["theta"] == lake.theta_texts[1]
    assert detail in error.value.details["detail"]
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_a_theta_that_is_not_the_partitions_is_a_family_skeleton_mismatch(
    tmp_path_factory,
):
    ticks = fixture_ticks(seed=5, drop=0.05)
    lake = build_lake(
        tmp_path_factory.mktemp("theta"), ticks, month_cuts(ticks, 1500, 3000)
    )
    path = next(
        lake.l2.rglob(
            f"theta={theta_text(lake.thetas[0])}/year=2017/month=09/events.parquet"
        )
    )
    table = pq.read_table(path)
    other = pa.array(
        [Decimal("0.00000007")] * table.num_rows, table.schema.field("theta").type
    )
    size = pq.ParquetFile(path).metadata.row_group(0).num_rows
    pq.write_table(
        table.set_column(table.schema.get_field_index("theta"), "theta", other),
        path,
        row_group_size=size,
    )
    with pytest.raises(SkeletonMismatch) as error:
        run(lake, M2)
    assert error.value.details["theta"] == lake.theta_texts[0]
    assert "theta no es el del esqueleto" in error.value.details["detail"]


def test_l2_events_keeps_one_digest_per_row_group_and_pops_it(lake):
    path = next(lake.l2.rglob("theta=0.00100000/year=2017/month=09/events.parquet"))
    with EventFile(str(path), "L2", all_columns=True) as file:
        events = L2Events(file, Decimal("0.001"))
        groups = pq.ParquetFile(path)
        sealed = [
            groups.read_row_group(i, columns=["theta", "confirm_agg_trade_id"])
            .combine_chunks()
            .to_batches()[0]
            for i in range(groups.num_row_groups)
        ]
        # Antes de leer nada, el esqueleto no tiene huellas: no puede aprobar.
        assert "aún no leyó" in events.differing_key(0, 0, sealed[0])
        for lot in file.lots(1 << 20):
            for batch in file.batches(lot, 2, 1 << 20):
                events.tap(batch)
        assert events.row_groups == file.rows
        first = 0
        for n, batch in enumerate(sealed):
            assert events.differing_key(n, first, batch) is None
            first += batch.num_rows
        # La huella se entrega una vez: no queda un dict que crezca con el mes.
        assert not events._done


# --- Compuerta y apertura en una pasada; L1 leído una vez ----------------------------------


def test_the_l1_footer_of_the_month_is_read_once_and_shared(lake, monkeypatch):
    path = l1_path(lake.l1, DEFAULT_KEY, M2)
    without_footer = []
    real = frames_lake.open_parquet

    def spy(p, what, metadata=None, pre_buffer=False):
        if p == path and metadata is None:
            without_footer.append(p)
        return real(p, what, metadata, pre_buffer)

    sources = []
    init = frames_lake.ParquetSource.__init__

    def spy_source(self, p, what):
        sources.append(p)
        init(self, p, what)

    monkeypatch.setattr(frames_lake, "open_parquet", spy)
    monkeypatch.setattr(frames_lake.ParquetSource, "__init__", spy_source)
    run(lake, M2)
    # Ni el lector ni la conciliación piden el pie del mes por su cuenta: `L1Month` lo lee
    # (un `ParquetSource`) y se lo pasa al lector.
    assert without_footer == []
    assert sources.count(path) == 1


def test_the_reader_with_a_shared_footer_gives_the_same_row_groups(lake):
    key = DEFAULT_KEY
    first = lake.ticks[M2][0].id
    reference = L1(lake.l1, key).row_groups(M2, M2, first, lambda *_: True)
    month = L1Month(str(lake.l1), key, M2)
    try:
        shared = L1(lake.l1, key, {M2: month.footer}).row_groups(
            M2, M2, first, lambda *_: True
        )
        for left, right in zip(reference, shared, strict=True):
            assert left.month == right.month and (left.ids == right.ids).all()
    finally:
        month.close()


def test_the_reconciliation_counts_from_the_row_groups_the_reader_delivered(lake):
    key = DEFAULT_KEY
    first = lake.ticks[M2][0].id
    month = L1Month(str(lake.l1), key, M2)
    fresh = L1Month(str(lake.l1), key, M2)
    try:
        ids = [
            int(rg.ids[len(rg.ids) // 2])
            for rg in L1(lake.l1, key).row_groups(M2, M2, first, lambda *_: True)
            if rg.month == M2
        ]
        assert len(ids) > 3
        month.watch(ids)
        reads = []
        real = month._parquet.reader
        month._parquet.reader = lambda: (reads.append(1), real())[1]
        for rg in L1(lake.l1, key).row_groups(M2, M2, first, lambda *_: True):
            if rg.month == M2:
                month.observe(rg.ids)
        expected = [fresh.count_up_to(i) for i in ids]
        assert [month.count_up_to(i) for i in ids] == expected
        assert reads == []  # no decodificó ni un row group
        # Un id que nadie anotó se cuenta decodificando, con el mismo resultado.
        other = ids[1] + 1
        assert month.count_up_to(other) == fresh.count_up_to(other)
        assert reads  # y esta sí leyó el row group
    finally:
        month.close()
        fresh.close()


def test_a_unit_reconciles_without_decoding_l1_again(lake, monkeypatch):
    """Con la unidad completa, `count_up_to` solo decodifica row groups que el lector no
    entregó (ninguno en este lago: el lector pasa por todos).
    """
    reads = []
    real = frames_lake.ParquetSource.reader

    def spy(self):
        reads.append(self.path)
        return real(self)

    monkeypatch.setattr(frames_lake.ParquetSource, "reader", spy)
    run(lake, M2)
    month_path = l1_path(lake.l1, DEFAULT_KEY, M2)
    assert [p for p in reads if p == month_path] == []


# --- Entrada en dos tramos: el motor no espera a los escritores ----------------------------

THREAD_PREFIXES = ("l3-theta", "l3-write", "l3-collector")


def test_the_engine_gets_its_first_row_group_before_the_writers_are_open(
    lake, monkeypatch, tmp_path
):
    """Un escritor que no termina de abrirse hasta que el motor recibe un row group de L1 no
    detiene la unidad: los escritores son el segundo tramo de la entrada.
    """
    pushed = threading.Event()
    push = native.push

    def spying_push(*args):
        pushed.set()
        return push(*args)

    open_writer = orchestrator._open_writer

    def late(l3_root, key, month, text, skeleton):
        if text == lake.theta_texts[0]:
            assert pushed.wait(10), (
                "el motor no recibió nada con el escritor abriéndose"
            )
        return open_writer(l3_root, key, month, text, skeleton)

    monkeypatch.setattr(native, "push", spying_push)
    monkeypatch.setattr(orchestrator, "_open_writer", late)
    results = run(lake, M2, l3=tmp_path, threads=4)
    assert pushed.is_set() and len(results) == len(lake.thetas)


@pytest.mark.parametrize("threads", [1, 4])
def test_a_writer_that_cannot_open_aborts_the_unit_without_publishing(
    lake, monkeypatch, tmp_path, threads
):
    open_writer = orchestrator._open_writer

    def broken(l3_root, key, month, text, skeleton):
        if text == lake.theta_texts[1]:
            raise FramesInputError("no se pudo abrir el escritor")
        return open_writer(l3_root, key, month, text, skeleton)

    monkeypatch.setattr(orchestrator, "_open_writer", broken)
    with pytest.raises(FramesInputError, match="no se pudo abrir el escritor"):
        run(lake, M2, l3=tmp_path, threads=threads)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert [
        t.name for t in threading.enumerate() if t.name.startswith(THREAD_PREFIXES)
    ] == []


@pytest.mark.parametrize("threads", [1, 4])
def test_a_stream_that_cannot_start_closes_the_file_it_opened(
    lake, monkeypatch, tmp_path, threads
):
    opened, closed = [], []
    init, close = EventFile.__init__, EventFile.close

    def spy_init(self, path, *args, **kwargs):
        opened.append(path)
        init(self, path, *args, **kwargs)

    def spy_close(self):
        closed.append(self.path)
        close(self)

    def refuse(*args, **kwargs):
        raise RuntimeError("el pool se cierra")

    monkeypatch.setattr(EventFile, "__init__", spy_init)
    monkeypatch.setattr(EventFile, "close", spy_close)
    monkeypatch.setattr(orchestrator, "BatchStream", refuse)
    with pytest.raises(RuntimeError, match="el pool se cierra"):
        run(lake, M2, l3=tmp_path, threads=threads)
    assert opened and sorted(closed) == sorted(opened)


# --- Valores por defecto -------------------------------------------------------------------


def test_the_default_writers_and_the_engine_threads():
    assert DEFAULT_WRITERS == 12
    assert _engine_threads(3) == 3
    assert _engine_threads(None) >= 1
