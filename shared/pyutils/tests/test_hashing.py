import pyarrow as pa
import pytest
from pyutils import ContentHasher, content_hash

RAW = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("flag", pa.bool_(), nullable=False),
    ]
)
MIXED = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("maybe", pa.int64(), nullable=True),
    ]
)


def _hash(schema, batches):
    hasher = ContentHasher(schema)
    for batch in batches:
        hasher.update(batch)
    return hasher.hexdigest()


def test_ignores_chunking():
    table = pa.table({"id": list(range(6)), "flag": [True, False] * 3}, schema=RAW)
    chunked = pa.concat_tables([table.slice(0, 2), table.slice(2)])
    assert chunked.column(0).num_chunks == 2
    assert content_hash(chunked) == content_hash(table)


def test_changes_with_a_value_and_with_the_schema():
    base = pa.table({"id": [1, 2], "flag": [True, False]}, schema=RAW)
    other = pa.table({"id": [1, 3], "flag": [True, False]}, schema=RAW)
    renamed = base.rename_columns(["id", "other"])
    assert content_hash(base) != content_hash(other)
    assert content_hash(base) != content_hash(renamed)


def test_ignores_schema_metadata():
    table = pa.table({"id": [1], "flag": [True]}, schema=RAW)
    with_meta = table.replace_schema_metadata({b"x": b"y"})
    assert content_hash(with_meta) == content_hash(table)


def test_ignores_bits_outside_a_slice():
    big = pa.table({"id": list(range(6)), "flag": [True] * 6}, schema=RAW)
    small = pa.table({"id": [1, 2], "flag": [True, True]}, schema=RAW)
    assert content_hash(big.slice(1, 2)) == content_hash(small)


def test_text_and_nullable_columns_are_hashed_by_value_whatever_the_chunking():
    rows = [
        {"id": 1, "name": "a", "maybe": None},
        {"id": 2, "name": "bc", "maybe": 5},
        {"id": 3, "name": "d", "maybe": None},
    ]

    def batch(items):
        return pa.RecordBatch.from_pylist(items, schema=MIXED)

    whole = _hash(MIXED, [batch(rows)])
    assert whole == _hash(MIXED, [batch(rows[:1]), batch(rows[1:])])
    assert whole != _hash(MIXED, [batch(rows[:2])])
    changed = [rows[0], {**rows[1], "maybe": None}, rows[2]]
    assert whole != _hash(MIXED, [batch(changed)])


def test_a_null_in_a_non_nullable_column_is_rejected():
    batch = pa.record_batch({"id": pa.array([1, None]), "flag": [True, True]})
    with pytest.raises(ValueError, match="nulos"):
        ContentHasher(RAW).update(batch)
