import base64
import hashlib
from datetime import date

import pyarrow.parquet as pq
import pytest
from lake_fixture import EVENTS, MONTH, build_lake, events_path, read_events
from viz_tiles import lake
from viz_tiles.lake import (
    EventsIndex,
    InputError,
    find_extreme,
    input_hash,
    object_stat,
    read_carry_over,
)
from viz_tiles.lake import (
    read_events as read_event_rows,
)

DAY = date(2017, 8, 18)


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


def test_read_events_reads_only_row_groups_that_can_match(tmp_path, monkeypatch):
    roots, _, _ = build_lake(tmp_path)
    theta = 100_000
    path = str(events_path(roots, theta, MONTH, EVENTS))
    rows = read_events()[theta][:-1]
    groups = pq.ParquetFile(path).num_row_groups
    assert (
        groups > 10
    )  # row groups de 3 filas: el salto por estadísticas tiene qué saltar

    lo = int(rows[100]["reference_agg_trade_id"]) + 1
    hi = int(rows[110]["extreme_agg_trade_id"])
    reads = []
    original = pq.ParquetFile.read_row_group

    def counting(self, i, *args, **kwargs):
        reads.append(i)
        return original(self, i, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counting)
    table = read_event_rows(path, lo, hi)
    assert len(reads) < groups // 3
    expected = [
        r
        for r in rows
        if int(r["reference_agg_trade_id"]) < hi
        and int(r["extreme_agg_trade_id"]) >= lo
    ]
    assert table["reference_agg_trade_id"].to_pylist() == [
        int(r["reference_agg_trade_id"]) for r in expected
    ]
    assert table.num_rows >= 10


def test_read_events_without_matches_is_an_empty_table(tmp_path):
    roots, _, _ = build_lake(tmp_path)
    path = str(events_path(roots, 100_000, MONTH, EVENTS))
    table = read_event_rows(path, 1, 2)
    assert table.num_rows == 0
    assert table.column_names == [
        "reference_agg_trade_id",
        "confirm_agg_trade_id",
        "extreme_agg_trade_id",
        "direction",
    ]


def test_find_extreme_returns_the_closed_event(tmp_path):
    roots, _, _ = build_lake(tmp_path)
    path = str(events_path(roots, 100_000, MONTH, EVENTS))
    row = read_events()[100_000][5]
    assert find_extreme(path, int(row["reference_agg_trade_id"])) == int(
        row["extreme_agg_trade_id"]
    )
    assert find_extreme(path, 1) is None


def test_unreadable_carry_over_is_an_input_error(tmp_path):
    path = tmp_path / "carry_over.parquet"
    path.write_bytes(b"no es parquet")
    with pytest.raises(InputError) as raised:
        read_carry_over(str(path), "0.00100000")
    assert raised.value.what == "carry_over"
    assert raised.value.details["theta"] == "0.00100000"
