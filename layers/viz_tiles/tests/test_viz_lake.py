import base64
import hashlib
from datetime import date
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lake_fixture import EVENTS, MONTH, build_lake, events_path, read_events
from viz_tiles import lake
from viz_tiles.lake import (
    EventsIndex,
    InputError,
    MonthEvents,
    find_extreme,
    input_hash,
    object_stat,
    read_carry_over,
    read_month_events,
)

DAY = date(2017, 8, 18)
DEC = pa.decimal128(18, 8)
EMPTY_TYPES = {"direction": pa.int8(), "confirm_price": DEC}


def test_crc32c_of_a_local_file_matches_the_standard_vector(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"123456789")
    assert object_stat(str(path)) == (9, "e3069283")


def test_gcs_stat_comes_from_the_object_metadata(monkeypatch):
    class Blob:
        size = 123
        crc32c = base64.b64encode(bytes.fromhex("e3069283")).decode()

    class Bucket:
        def __init__(self, name):
            self.name = name

        def get_blob(self, name):
            assert (self.name, name) == ("bucket", "l1/a/consolidated.parquet")
            return Blob()

    class Client:
        def bucket(self, name):
            return Bucket(name)

    monkeypatch.setattr(lake, "_gcs_client", lambda: Client())
    assert object_stat("gs://bucket/l1/a/consolidated.parquet") == (123, "e3069283")


def test_gcs_stat_of_a_missing_object_raises(monkeypatch):
    class Bucket:
        def get_blob(self, name):
            return None

    class Client:
        def bucket(self, name):
            return Bucket()

    monkeypatch.setattr(lake, "_gcs_client", lambda: Client())
    with pytest.raises(FileNotFoundError):
        object_stat("gs://bucket/x")


def test_input_hash_is_the_sha256_of_the_canonical_text():
    stats = {"b/y.parquet": (20, "0000000b"), "a/x.parquet": (10, "0000000a")}
    text = "1.0.0\n2017-08-18\na/x.parquet\t10\t0000000a\nb/y.parquet\t20\t0000000b\n"
    assert input_hash("1.0.0", DAY, stats) == hashlib.sha256(text.encode()).hexdigest()
    # No depende del orden en que llegan los archivos.
    assert input_hash("1.0.0", DAY, dict(reversed(stats.items()))) == input_hash(
        "1.0.0", DAY, stats
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: ("1.0.1", DAY, s),
        lambda s: ("1.0.0", date(2017, 8, 19), s),
        lambda s: ("1.0.0", DAY, {**s, "a": (2, "00000001")}),
        lambda s: ("1.0.0", DAY, {**s, "a": (1, "00000002")}),
        lambda s: ("1.0.0", DAY, {**s, "b": (1, "00000001")}),
    ],
)
def test_input_hash_changes_with_any_input(change):
    base = {"a": (1, "00000001")}
    assert input_hash(*change(base)) != input_hash("1.0.0", DAY, base)


def test_events_index_lists_months_and_thetas(tmp_path):
    roots, _, _ = build_lake(tmp_path)
    index = EventsIndex.list(roots.events, "binance", "spot", "BTCUSDT")
    assert index.thetas(MONTH) == [f"0.{t:08d}" for t in sorted(read_events())]
    assert index.thetas((2017, 9)) == []
    assert index.has(MONTH, "0.00100000", EVENTS)
    assert not index.has((2017, 9), "0.00100000", EVENTS)
    assert (
        EventsIndex.list(tmp_path / "nada", "binance", "spot", "BTCUSDT").thetas(MONTH)
        == []
    )


def test_read_month_events_reads_each_row_group_once_and_keeps_every_event(
    tmp_path, monkeypatch
):
    """Una sola lectura del mes por θ: de ahí salen los eventos de cada día."""
    roots, _, _ = build_lake(tmp_path)
    theta = 100_000
    path = str(events_path(roots, theta, MONTH, EVENTS))
    rows = read_events()[theta][:-1]
    groups = pq.ParquetFile(path).num_row_groups
    assert groups > 10  # row groups de 3 filas

    reads = []
    original = pq.ParquetFile.read_row_group

    def counting(self, i, *args, **kwargs):
        reads.append(i)
        return original(self, i, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counting)
    events = read_month_events(path)
    assert sorted(reads) == list(range(groups))
    assert len(events) == len(rows)
    assert events.reference_id.tolist() == [
        int(r["reference_agg_trade_id"]) for r in rows
    ]
    assert events.confirm_id.tolist() == [int(r["confirm_agg_trade_id"]) for r in rows]
    assert events.extreme_id.tolist() == [int(r["extreme_agg_trade_id"]) for r in rows]
    for name in ("reference", "confirm", "extreme"):
        assert getattr(events, f"{name}_time").tolist() == [
            int(r[f"{name}_time"]) for r in rows
        ]
    assert events.direction.tolist() == [int(r["direction"]) for r in rows]
    # El precio de la confirmación llega en enteros de 10⁻⁸, sin pasar por `float`.
    assert events.confirm_price.tolist() == [
        int(Decimal(r["confirm_price"]).scaleb(8)) for r in rows
    ]


def test_touching_keeps_the_events_that_reach_the_ticks_of_a_day():
    events = MonthEvents(
        reference_id=np.array([10, 30, 55, 90], np.int64),
        confirm_id=np.array([20, 40, 60, 100], np.int64),
        extreme_id=np.array([30, 55, 90, 120], np.int64),
        reference_time=np.arange(4, dtype=np.int64),
        confirm_time=np.arange(4, dtype=np.int64),
        extreme_time=np.arange(4, dtype=np.int64),
        direction=np.array([1, -1, 1, -1], np.int8),
        confirm_price=np.arange(4, dtype=np.int64),
    )
    # Un evento toca [lo, hi] si su referencia es < hi y su extremo >= lo.
    assert events.touching(31, 60).reference_id.tolist() == [30, 55]
    assert events.touching(55, 56).reference_id.tolist() == [30, 55]
    assert events.touching(120, 130).reference_id.tolist() == [90]
    assert len(events.touching(200, 300)) == 0
    assert len(events.touching(1, 10)) == 0


def test_month_events_are_sorted_by_reference_and_a_duplicate_is_rejected(tmp_path):
    def write(refs):
        path = tmp_path / "events.parquet"
        n = len(refs)
        pq.write_table(
            pa.table(
                {
                    "reference_agg_trade_id": pa.array(refs, pa.int64()),
                    "confirm_agg_trade_id": pa.array([r + 3 for r in refs], pa.int64()),
                    "extreme_agg_trade_id": pa.array([r + 5 for r in refs], pa.int64()),
                    "reference_time": pa.array(range(n), pa.int64()),
                    "confirm_time": pa.array(range(n), pa.int64()),
                    "extreme_time": pa.array(range(n), pa.int64()),
                    "direction": pa.array([1] * n, pa.int8()),
                    "confirm_price": pa.array([Decimal(1)] * n, DEC),
                }
            ),
            path,
            row_group_size=2,
        )
        return str(path)

    assert read_month_events(write([30, 10, 20])).reference_id.tolist() == [10, 20, 30]
    with pytest.raises(ValueError, match="misma referencia"):
        read_month_events(write([10, 20, 20]))


def test_month_events_of_an_empty_file_are_empty(tmp_path):
    path = tmp_path / "events.parquet"
    pq.write_table(
        pa.table(
            {
                name: pa.array([], EMPTY_TYPES.get(name, pa.int64()))
                for name in lake.EVENT_COLUMNS
            }
        ),
        path,
    )
    assert len(read_month_events(str(path))) == 0


def test_find_extreme_returns_the_closed_event(tmp_path):
    roots, _, _ = build_lake(tmp_path)
    path = str(events_path(roots, 100_000, MONTH, EVENTS))
    row = read_events()[100_000][5]
    assert find_extreme(path, int(row["reference_agg_trade_id"])) == (
        int(row["extreme_agg_trade_id"]),
        int(row["extreme_time"]),
    )
    assert find_extreme(path, 1) is None


def test_unreadable_carry_over_is_an_input_error(tmp_path):
    path = tmp_path / "carry_over.parquet"
    path.write_bytes(b"no es parquet")
    with pytest.raises(InputError) as raised:
        read_carry_over(str(path), "0.00100000")
    assert raised.value.what == "carry_over"
    assert raised.value.details["theta"] == "0.00100000"
