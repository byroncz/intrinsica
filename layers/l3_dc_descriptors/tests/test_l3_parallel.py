"""El orquestador sobre el motor nativo (ITSC-363, ADR-L3-16): lo que escribe no depende de
los hilos ni de los topes.

Estas pruebas no miden velocidad. Corren `summarize_month` con 1, 2 y 4 hilos del motor, 1 y
2 lectores de L1, 1 y 4 row groups en vuelo y 1 y 4 escritores, con las fronteras de L2 en un
solo lote o en lotes de pocos eventos, y exigen exactamente la misma tabla, el mismo
`content_hash` y la misma conciliación que la referencia de Python (`pyref`, en serie). Un
lago con row groups de L1 repartidos a mano fuerza un evento que cruza tres row groups, uno
cuya referencia es el último tick de un row group y uno que cubre varios row groups enteros.
"""

import itertools
import threading
import time
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pyref
import pytest
from dc_frames import FamilySkeletonMismatch, FamilyWriter
from dc_frames.lake import EventFile, l1_path
from l3_dc_descriptors import (
    SUMMARIES_SCHEMA,
    L2InputMissing,
    SkeletonMismatch,
    boundaries,
    outputs,
    summarize,
    summarize_month,
)
from l3_dc_descriptors.timing import BusyClock, Phases, ReadMeter
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts

ENGINE_THREADS = (1, 2, 4)
READERS = (1, 2)
INFLIGHT = (1, 4)
WRITERS = (1, 4)
KNOBS = list(itertools.product(ENGINE_THREADS, READERS, INFLIGHT, WRITERS))
THREAD_PREFIXES = ("l3-theta", "l3-write", "l3-collector", "dc-frames-read")


@pytest.fixture(scope="module")
def ticks():
    return fixture_ticks(seed=5, drop=0.05)


@pytest.fixture(scope="module")
def lake(tmp_path_factory, ticks):
    # Row groups chicos: cada uno lo tocan los 5 θ, y hay eventos que cruzan row groups.
    return build_lake(
        tmp_path_factory.mktemp("lake"),
        ticks,
        month_cuts(ticks, 1500, 3000),
        l1_row_group=150,
    )


def l3_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith(THREAD_PREFIXES)]


def run_all(lake, runner=summarize_month, **options) -> dict:
    """Los resultados y las tablas de todos los meses, leídas antes de la siguiente corrida
    (que reescribe los mismos archivos).
    """
    results, tables = {}, {}
    for month in lake.months:
        results[month] = runner(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            **options,
        )
        for theta, result in results[month].items():
            tables[month, theta] = pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
    return {"results": results, "tables": tables}


@pytest.fixture(scope="module")
def reference(lake):
    return run_all(lake, pyref.summarize_month)


def assert_same(got: dict, expected: dict) -> None:
    assert got["results"].keys() == expected["results"].keys()
    for month, per_theta in expected["results"].items():
        for theta, result in per_theta.items():
            other = got["results"][month][theta]
            assert other.content_hash == result.content_hash
            assert other.rows == result.rows
            assert other.reconciliation == result.reconciliation
    left, right = expected["tables"], got["tables"]
    assert left.keys() == right.keys()
    assert all(left[key].equals(right[key]) for key in left)
    assert sum(t.num_rows for t in left.values()) > 0


@pytest.fixture
def tiny_l2(monkeypatch):
    """Tramos de L2 de un row group y lotes de 3 eventos: muchos lotes por θ, que el motor
    suelta a medida que cierran sus eventos.
    """
    monkeypatch.setattr(boundaries, "L2_LOT_BYTES", 1)
    monkeypatch.setattr(boundaries, "L2_BATCH_ROWS", 3)


@pytest.mark.parametrize(("engine_threads", "readers", "inflight", "writers"), KNOBS)
def test_the_knobs_give_the_same_table_and_hash(
    lake, reference, engine_threads, readers, inflight, writers
):
    got = run_all(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
    )
    assert_same(got, reference)
    assert l3_threads() == []


@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight", "writers"),
    [(1, 1, 1, 1), (2, 2, 4, 1), (4, 1, 4, 4), (4, 2, 1, 4)],
)
def test_the_boundaries_in_small_lots_change_nothing(
    lake, reference, tiny_l2, engine_threads, readers, inflight, writers
):
    got = run_all(
        lake,
        threads=4,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
        writers=writers,
    )
    assert_same(got, reference)


@pytest.mark.parametrize("threads", [1, 2, 50])
def test_the_io_pool_changes_nothing(lake, reference, threads):
    assert_same(run_all(lake, threads=threads), reference)
    assert l3_threads() == []


def test_the_defaults_give_the_same_result_too(lake, reference):
    assert_same(run_all(lake), reference)


@pytest.mark.parametrize(
    "option", ["threads", "readers", "inflight", "writers", "engine_threads"]
)
def test_it_rejects_a_knob_out_of_range(lake, option):
    with pytest.raises(ValueError, match=option):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            **{option: 0},
        )


def test_it_never_uses_more_io_threads_than_thetas():
    assert summarize._workers(8, 5) == 5
    assert summarize._workers(3, 50) == 3
    assert 1 <= summarize._workers(None, 50) <= 50


def test_the_phases_report_how_it_ran(lake):
    phases = Phases()
    summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M2,
        lake.theta_texts,
        series_start=M1,
        phases=phases,
        threads=3,
        engine_threads=2,
        readers=1,
        inflight=3,
        writers=2,
    )
    assert (phases.threads, phases.engine_threads, phases.readers) == (3, 2, 1)
    assert (phases.inflight, phases.writers) == (3, 2)
    assert 1 <= phases.inflight_max <= 3
    assert phases.items > 0 and phases.items % len(lake.thetas) == 0
    assert 0 < phases.engine_busy <= 1
    assert phases.aggregate_s > 0 and phases.setup_s > 0 and phases.publish_s > 0
    assert phases.read.bytes > 0 and phases.read.clock.seconds > 0
    assert phases.write.seconds > 0


# Row groups de L1 repartidos a mano


def relayout(lake, month, sizes: list[int]) -> None:
    """Reescribe `consolidated.parquet` del mes con row groups de esos tamaños (el último,
    lo que quede). L2 solo habla de ids, así que no cambia.
    """
    path = l1_path(str(lake.l1), ("binance", "spot", "BTCUSDT"), month)
    table = pq.read_table(path)
    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    start = 0
    for size in [*sizes, table.num_rows]:
        stop = min(start + size, table.num_rows)
        if stop > start:
            writer.write_table(table.slice(start, stop - start), row_group_size=size)
        start = stop
    writer.close()


def position(ticks, tick_id: int) -> int:
    return next(i for i, t in enumerate(ticks) if t.id == tick_id)


def hand_made_cuts(lake, month) -> list[int]:
    """Cortes de row group tales que, en algún θ, haya un evento que cruza tres row groups
    o más y otro cuya referencia es el último tick de un row group.
    """
    ticks = lake.ticks[month]
    present = {t.id for t in ticks}
    events = [
        e
        for theta in lake.thetas
        for e in lake.events[theta, month]
        if {e["reference_agg_trade_id"], e["extreme_agg_trade_id"]} <= present
    ]
    long = max(
        events, key=lambda e: e["extreme_agg_trade_id"] - e["reference_agg_trade_id"]
    )
    start = position(ticks, long["reference_agg_trade_id"])
    stop = position(ticks, long["extreme_agg_trade_id"])
    assert stop - start >= 30, "el lago no trae un evento largo"
    # La referencia del evento largo es el último tick de su row group; el evento abarca
    # varios row groups de 10 ticks, y los demás row groups son de 25.
    cuts = []
    done = 0
    while done < start + 1:
        step = min(25, start + 1 - done)
        cuts.append(step)
        done += step
    cuts += [10] * ((stop - start) // 10 + 1)
    return cuts


def crossing(ticks, cuts, events) -> tuple[int, int]:
    """Cuántos eventos cruzan tres row groups o más, y cuántos tienen su referencia como
    último tick de un row group.
    """
    edges = list(itertools.accumulate(cuts))
    group = {}
    for g, (a, b) in enumerate(itertools.pairwise([0, *edges, len(ticks)])):
        for i in range(a, b):
            group[ticks[i].id] = g
    inside = [
        e
        for e in events
        if {e["reference_agg_trade_id"], e["extreme_agg_trade_id"]} <= group.keys()
    ]
    wide = sum(
        group[e["extreme_agg_trade_id"]] - group[e["reference_agg_trade_id"]] >= 2
        for e in inside
    )
    last = {ticks[edge - 1].id for edge in edges}
    on_edge = sum(e["reference_agg_trade_id"] in last for e in events)
    return wide, on_edge


@pytest.mark.parametrize(
    ("engine_threads", "readers", "inflight"), [(1, 1, 1), (4, 2, 4), (2, 2, 2)]
)
def test_events_over_three_row_groups_and_on_a_row_group_edge(
    tmp_path, ticks, engine_threads, readers, inflight
):
    lake = build_lake(tmp_path / "lake", ticks, month_cuts(ticks, 1500, 3000))
    month = lake.months[1]
    cuts = hand_made_cuts(lake, month)
    relayout(lake, month, cuts)
    events = [e for theta in lake.thetas for e in lake.events[theta, month]]
    wide, on_edge = crossing(lake.ticks[month], cuts, events)
    assert wide > 0 and on_edge > 0
    wanted = pyref.summarize_month(
        lake.l1, lake.l2, tmp_path / "py", month, lake.theta_texts, series_start=M1
    )
    got = summarize_month(
        lake.l1,
        lake.l2,
        tmp_path / "rs",
        month,
        lake.theta_texts,
        series_start=M1,
        engine_threads=engine_threads,
        readers=readers,
        inflight=inflight,
    )
    for theta, result in wanted.items():
        assert got[theta].content_hash == result.content_hash
        assert got[theta].rows == result.rows
        assert got[theta].reconciliation == result.reconciliation
        assert pq.read_table(got[theta].path, schema=SUMMARIES_SCHEMA).equals(
            pq.read_table(result.path, schema=SUMMARIES_SCHEMA)
        )


# Errores


def test_a_failing_writer_aborts_the_unit_without_publishing_and_without_threads(
    lake, tmp_path, monkeypatch
):
    real = FamilyWriter.write_row_group
    fail = set(lake.theta_texts[1::2])

    def failing(self, batch):
        theta = f"{batch.column('theta')[0].as_py():.8f}"
        if theta in fail:
            raise FamilySkeletonMismatch("mensaje", f"falla {theta}")
        real(self, batch)

    monkeypatch.setattr(FamilyWriter, "write_row_group", failing)
    for kwargs in ({"threads": 1, "writers": 1}, {"threads": 4, "writers": 4}):
        with pytest.raises(SkeletonMismatch) as caught:
            summarize_month(
                lake.l1,
                lake.l2,
                tmp_path,
                M1,
                lake.theta_texts,
                series_start=M1,
                **kwargs,
            )
        assert caught.value.details["theta"] in fail
        assert l3_threads() == []
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_a_failing_commit_is_the_skeleton_error_of_its_theta(
    lake, tmp_path, monkeypatch
):
    def refuse(self):
        raise FamilySkeletonMismatch("mensaje", "faltan row groups")

    monkeypatch.setattr(FamilyWriter, "commit", refuse)
    with pytest.raises(SkeletonMismatch) as caught:
        summarize_month(
            lake.l1, lake.l2, tmp_path, M1, lake.theta_texts, series_start=M1, threads=4
        )
    assert caught.value.details["theta"] == lake.theta_texts[0]
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]
    assert l3_threads() == []


def test_a_missing_events_file_is_the_gate_error_of_the_first_theta(
    lake, tmp_path, ticks
):
    base = build_lake(tmp_path / "gate", ticks, month_cuts(ticks, 1500, 3000))
    thetas = base.theta_texts
    for text in thetas[:2]:
        events = (
            base.l2
            / f"provider=binance/market=spot/asset=BTCUSDT/theta={text}/year=2017/month=09"
            / "events.parquet"
        )
        events.unlink()
    messages = set()
    for threads in (1, 4):
        with pytest.raises(L2InputMissing) as caught:
            summarize_month(
                base.l1, base.l2, base.l3, M2, thetas, series_start=M1, threads=threads
            )
        messages.add(str(caught.value))
        assert caught.value.details["theta"] == thetas[0]
        assert [p for p in base.l3.rglob("*") if p.is_file()] == []
        assert l3_threads() == []
    assert len(messages) == 1


# Contrapresión: a lo más K row groups en el motor y pocos sellados sin escribir


def test_inflight_never_passes_the_cap(lake):
    for inflight in (1, 2, 3):
        phases = Phases()
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            M1,
            lake.theta_texts,
            series_start=M1,
            phases=phases,
            inflight=inflight,
            readers=2,
        )
        assert 1 <= phases.inflight_max <= inflight


def test_slow_writers_hold_back_the_pushes(lake, monkeypatch, tmp_path):
    """Con escritores lentos, lo sellado sin escribir se queda cerca del tope y la unidad
    igual termina con el mismo resultado; sin tope se acumula todo lo que el motor sella.
    """
    seen: list[int] = []
    real = outputs.SealedWriters.put

    def spying(self, sealed):
        real(self, sealed)
        seen.append(self.pending)

    slow = FamilyWriter.write_row_group

    def sleepy(self, batch):
        time.sleep(0.004)
        slow(self, batch)

    monkeypatch.setattr(outputs.SealedWriters, "put", spying)
    monkeypatch.setattr(FamilyWriter, "write_row_group", sleepy)
    peaks, hashes = {}, {}
    for cap in (1, 10**6):
        seen.clear()
        monkeypatch.setattr(summarize, "PENDING_PER_WRITER", cap)
        results = summarize_month(
            lake.l1,
            lake.l2,
            tmp_path / str(cap),
            M2,
            lake.theta_texts,
            series_start=M1,
            writers=1,
            inflight=2,
            readers=1,
        )
        peaks[cap] = max(seen)
        hashes[cap] = {t: r.content_hash for t, r in results.items()}
    wanted = pyref.summarize_month(
        lake.l1, lake.l2, tmp_path / "py", M2, lake.theta_texts, series_start=M1
    )
    assert hashes[1] == hashes[10**6] == {t: r.content_hash for t, r in wanted.items()}
    assert peaks[1] < peaks[10**6]


# Escritores: cola por θ con «siguiente esperado»


class FakeWriter:
    """Un escritor que anota qué row groups recibe, de qué hilo y si hubo dos a la vez."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.order: list[int] = []
        self.active = 0
        self.overlap = False
        self.lock = threading.Lock()

    def write_row_group(self, batch: pa.RecordBatch) -> None:
        with self.lock:
            self.active += 1
            self.overlap = self.overlap or self.active > 1
        time.sleep(self.delay)
        with self.lock:
            self.order.append(batch.column("confirm_agg_trade_id")[0].as_py())
            self.active -= 1


class Wrapped:
    """Bytes con protocolo de buffer, como `SealedColumn`."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __buffer__(self, flags):
        return memoryview(self.data).__buffer__(flags)

    def __len__(self) -> int:
        return len(self.data)


def fake_sealed(position: int, index: int):
    """Un row group sellado de 1 fila en el formato de `Engine.pop_sealed`: `(θ, índice,
    filas, columnas, validez, nulos)`, con el índice como `confirm_agg_trade_id`.
    """
    columns = []
    for field in list(SUMMARIES_SCHEMA)[1:]:
        width = 16 if pa.types.is_decimal(field.type) else 8
        value = index if field.name == "confirm_agg_trade_id" else 0
        columns.append(Wrapped(value.to_bytes(width, "little", signed=True)))
    return (position, index, 1, columns, None, 0)


def test_each_theta_writes_its_row_groups_in_order_and_never_two_at_a_time():
    from concurrent.futures import ThreadPoolExecutor

    fakes = [FakeWriter(0.002) for _ in range(3)]
    clock = BusyClock()
    thetas = [Decimal("0.001"), Decimal("0.002"), Decimal("0.003")]
    with ThreadPoolExecutor(4) as pool:
        sink = outputs.SealedWriters(pool, fakes, thetas, ["a", "b", "c"], M1, clock)
        # Llegan al revés y revueltos entre los θ: 4 antes que 3, 2 antes que 1…
        order = [(0, 3), (1, 1), (0, 2), (2, 0), (1, 0), (0, 1), (2, 1), (0, 0), (1, 2)]
        for position, index in order:
            sink.put(fake_sealed(position, index))
        sink.join()
    assert sink.pending == 0 and sink.error is None
    assert [f.order for f in fakes] == [[0, 1, 2, 3], [0, 1, 2], [0, 1]]
    assert not any(f.overlap for f in fakes)
    assert clock.seconds > 0


def test_a_row_group_that_waits_for_its_predecessor_is_not_written_yet():
    from concurrent.futures import ThreadPoolExecutor

    fake = FakeWriter()
    with ThreadPoolExecutor(2) as pool:
        sink = outputs.SealedWriters(
            pool, [fake], [Decimal("0.001")], ["a"], M1, BusyClock()
        )
        sink.put(fake_sealed(0, 1))
        time.sleep(0.05)
        assert fake.order == [] and sink.pending == 1
        sink.put(fake_sealed(0, 0))
        sink.join()
    assert fake.order == [0, 1]


def test_a_writer_error_is_kept_and_stops_the_rest():
    from concurrent.futures import ThreadPoolExecutor

    class Failing(FakeWriter):
        def write_row_group(self, batch):
            raise FamilySkeletonMismatch("mensaje", "falla")

    with ThreadPoolExecutor(2) as pool:
        sink = outputs.SealedWriters(
            pool,
            [Failing(), FakeWriter()],
            [Decimal("0.001"), Decimal("0.002")],
            ["a", "b"],
            M1,
            BusyClock(),
        )
        sink.put(fake_sealed(0, 0))
        sink.put(fake_sealed(1, 0))
        with pytest.raises(SkeletonMismatch):
            sink.join()
        assert isinstance(sink.error, SkeletonMismatch)
        sink.put(fake_sealed(1, 1))  # tras el error no se encola nada
        assert sink.pending == 0
        sink.wait_below(1)  # con un error no espera


def test_the_same_row_group_twice_is_an_error():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as pool:
        sink = outputs.SealedWriters(
            pool, [FakeWriter()], [Decimal("0.001")], ["a"], M1, BusyClock()
        )
        sink.put(fake_sealed(0, 1))
        sink.put(fake_sealed(0, 1))
        with pytest.raises(SkeletonMismatch, match="dos veces"):
            sink.join()


# Fronteras de L2 por delante


@pytest.fixture(scope="module")
def events_file(lake):
    text = lake.theta_texts[0]
    path = (
        lake.l2
        / f"provider=binance/market=spot/asset=BTCUSDT/theta={text}/year=2017/month=09"
        / "events.parquet"
    )
    return str(path)


def read_all(path: str, **options):
    with EventFile(path, "x", ("direction", "confirm_price", "extreme_price")) as file:
        stream = boundaries.BatchStream(file, **options)
        try:
            return list(stream)
        finally:
            stream.close()


def test_the_stream_gives_the_events_in_order_with_and_without_a_pool(events_file):
    wanted = pq.read_table(events_file).column("confirm_agg_trade_id").to_pylist()
    from concurrent.futures import ThreadPoolExecutor

    for pool_size in (0, 1, 3):
        meter = ReadMeter()
        pool = ThreadPoolExecutor(pool_size) if pool_size else None
        try:
            batches = read_all(
                events_file, pool=pool, meter=meter, lot_bytes=1, batch_rows=4
            )
        finally:
            if pool:
                pool.shutdown()
        got = [v for b in batches for v in b.column("confirm_agg_trade_id").to_pylist()]
        assert got == wanted
        assert all(b.num_rows <= 4 for b in batches)
        assert meter.bytes > 0 and meter.clock.seconds > 0


def test_the_stream_stays_a_few_batches_ahead(events_file):
    from concurrent.futures import ThreadPoolExecutor

    with (
        ThreadPoolExecutor(2) as pool,
        EventFile(events_file, "x") as file,
    ):
        stream = boundaries.BatchStream(
            file, pool, ReadMeter(), lot_bytes=1, batch_rows=2, ahead=2
        )
        first = next(stream)
        time.sleep(0.1)
        assert first.num_rows == 2
        with stream._cond:
            assert len(stream._ready) <= 2
        stream.close()
        assert l3_threads() == []
        assert not stream._ready


def test_a_read_error_comes_out_at_the_batch_that_would_have_brought_it(events_file):
    from concurrent.futures import ThreadPoolExecutor

    class Broken:
        def __init__(self, file):
            self.file = file

        def lots(self, lot_bytes):
            yield from self.file.lots(lot_bytes)

        def compressed(self, lot):
            return self.file.compressed(lot)

        def batches(self, lot, rows):
            yield from itertools.islice(self.file.batches(lot, rows), 1)
            raise OSError("se cortó la lectura")

    with ThreadPoolExecutor(2) as pool, EventFile(events_file, "x") as file:
        stream = boundaries.BatchStream(
            Broken(file), pool, ReadMeter(), lot_bytes=1, batch_rows=2
        )
        got = 0
        with pytest.raises(OSError, match="se cortó"):
            while True:
                next(stream)
                got += 1
        assert got >= 1
        stream.close()


def test_a_busy_clock_unites_the_intervals_of_concurrent_operations():
    clock = BusyClock()
    done = threading.Event()

    def work():
        with clock.busy():
            done.wait(1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    done.set()
    for t in threads:
        t.join()
    assert 0.09 < clock.seconds < 0.4  # no 4 × 0,1 s: la pared, no la suma
