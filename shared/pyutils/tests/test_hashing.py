import hashlib
import struct
from itertools import pairwise

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


FLOATS = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("maybe", pa.float64(), nullable=True),
    ]
)
SPECIAL = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.5e-300, 1e300]


def _floats(n=40, nulls=True):
    """Filas con valores de todo tipo; con `nulls`, cada tercera fila de `maybe` es nula."""
    price = [SPECIAL[i % len(SPECIAL)] if i % 5 == 0 else i * 0.1 for i in range(n)]
    maybe = [None if nulls and i % 3 == 0 else p / 7 for i, p in enumerate(price)]
    return pa.table(
        {"id": list(range(n)), "price": price, "maybe": maybe}, schema=FLOATS
    )


def _reference(schema, table):
    """El digest que documenta `ContentHasher`, armado a mano con hashlib."""
    columns = []
    for field in schema:
        digest = hashlib.sha256()
        for value in table.column(field.name).to_pylist():
            if field.type == pa.int64():
                digest.update(struct.pack("<q", value))
            elif value is None:
                digest.update(bytes(9))
            elif field.nullable:
                digest.update(b"\x01" + struct.pack("<d", value))
            else:
                digest.update(struct.pack("<d", value))
        columns.append(digest.digest())
    final = hashlib.sha256(schema.serialize().to_pybytes())
    for column in columns:
        final.update(column)
    return final.hexdigest()


@pytest.mark.parametrize("nulls", [False, True])
def test_float64_matches_the_documented_bytes(nulls):
    table = _floats(nulls=nulls)
    assert content_hash(table) == _reference(FLOATS, table)


@pytest.mark.parametrize("nulls", [False, True])
def test_float64_ignores_chunking_and_slice_offsets(nulls):
    table = _floats(nulls=nulls)
    whole = _hash(FLOATS, table.to_batches())
    assert whole == _reference(FLOATS, table)
    for cuts in ([1], [7, 8], [3, 4, 5, 6, 33], list(range(1, 40))):
        bounds = [0, *cuts, table.num_rows]
        pieces = [table.slice(a, b - a) for a, b in pairwise(bounds)]
        batches = [batch for piece in pieces for batch in piece.to_batches()]
        assert _hash(FLOATS, batches) == whole
    # Un batch con offset: el hash es el de las filas que ve, no el del buffer entero.
    tail = table.slice(11, 17)
    assert content_hash(tail) == _reference(FLOATS, tail)
    # Un batch vacío no cambia nada.
    empty = pa.RecordBatch.from_pylist([], schema=FLOATS)
    assert _hash(FLOATS, [empty, *table.to_batches(), empty]) == whole


def _with(table, name, index, value):
    values = table.column(name).to_pylist()
    values[index] = value
    return table.set_column(
        table.schema.get_field_index(name),
        table.schema.field(name),
        pa.array(values, pa.float64()),
    )


@pytest.mark.parametrize("name", ["price", "maybe"])
def test_float64_tells_one_bit_apart(name):
    table = _floats()
    value = table.column(name)[1].as_py()
    flipped = struct.unpack("<d", (struct.pack("<d", value)[:-1] + b"\x7e"))[0]
    nudged = value + abs(value) * 2**-52
    digests = {
        content_hash(table),
        content_hash(_with(table, name, 1, flipped)),
        content_hash(_with(table, name, 1, nudged)),
    }
    assert len(digests) == 3
    assert content_hash(_with(table, name, 2, 0.0)) != content_hash(
        _with(table, name, 2, -0.0)
    )


def test_float64_tells_a_null_from_a_value():
    table = _floats()
    base = content_hash(table)
    # Un nulo no es 0.0 ni NaN, y un valor no es un nulo.
    assert content_hash(_with(table, "maybe", 3, 0.0)) != base
    assert content_hash(_with(table, "maybe", 3, float("nan"))) != base
    assert content_hash(_with(table, "maybe", 4, None)) != base
    assert content_hash(_with(table, "maybe", 3, 0.0)) != content_hash(
        _with(table, "maybe", 3, -0.0)
    )


def test_float64_nan_and_infinities_are_hashed_by_their_bits():
    quiet = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
    payload = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]
    table = _floats(nulls=False)
    assert content_hash(_with(table, "price", 1, quiet)) == content_hash(
        _with(table, "price", 1, quiet)
    )
    assert content_hash(_with(table, "price", 1, quiet)) != content_hash(
        _with(table, "price", 1, payload)
    )


def test_float64_does_not_change_the_other_types():
    ints = pa.schema([pa.field("n", pa.int64(), nullable=False)])
    table = pa.table({"n": [1, 2, 3]}, schema=ints)
    assert content_hash(table) == _reference(ints, table)


def test_a_null_in_a_non_nullable_float_is_rejected():
    schema = pa.schema([pa.field("price", pa.float64(), nullable=False)])
    batch = pa.record_batch({"price": pa.array([1.0, None])})
    with pytest.raises(ValueError, match="nulos"):
        ContentHasher(schema).update(batch)
