"""El hasher vectorizado da el mismo digest que el hasher valor a valor de antes.

`ReferenceHasher` es `ContentHasher` tal como era antes de ITSC-359: las columnas
nulables se hashean celda a celda con `repr` de Python. Se conserva aquí como
referencia y no se debe tocar: el digest ya está publicado en L1, L2 y L3.
"""

import hashlib
import random
import time
from decimal import Decimal
from itertools import pairwise

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pyutils import ContentHasher, content_hash


def _raw_bytes(column: pa.Array) -> memoryview:
    if pa.types.is_boolean(column.type):
        column = pc.cast(column, pa.uint8())
    if column.null_count:
        raise ValueError("una columna no nulable trae nulos: no se puede hashear")
    width = column.type.bit_width // 8
    start = column.offset * width
    return memoryview(column.buffers()[1])[start : start + len(column) * width]


def _is_raw(field: pa.Field) -> bool:
    kind = field.type
    fixed = (
        pa.types.is_integer(kind)
        or pa.types.is_decimal(kind)
        or pa.types.is_boolean(kind)
    )
    return fixed and not field.nullable


class ReferenceHasher:
    def __init__(self, schema: pa.Schema) -> None:
        self._schema = schema.remove_metadata()
        self._columns = [hashlib.sha256() for _ in self._schema]
        self._by_value = [not _is_raw(field) for field in self._schema]

    def update(self, batch: pa.RecordBatch) -> None:
        for digest, by_value, column in zip(
            self._columns, self._by_value, batch.columns, strict=True
        ):
            if by_value:
                for value in column.to_pylist():
                    digest.update(repr(value).encode() + b"\0")
            else:
                digest.update(_raw_bytes(column))

    def hexdigest(self) -> str:
        digest = hashlib.sha256(self._schema.serialize().to_pybytes())
        for column in self._columns:
            digest.update(column.digest())
        return digest.hexdigest()


DECIMAL = pa.decimal128(18, 8)
THETA = pa.decimal128(9, 8)
TYPES = [
    DECIMAL,
    THETA,
    pa.decimal128(38, 10),
    pa.decimal128(9, 0),
    pa.decimal128(38, 0),
    pa.decimal128(9, 6),
    pa.decimal128(9, -2),
    pa.decimal128(2, 8),
    pa.decimal128(8, 8),
    pa.decimal256(50, 2),
    pa.decimal256(60, 10),
    pa.int64(),
    pa.bool_(),
]
NULL_SHARES = [0.0, 0.3, 1.0]


def _unscaled(rng: random.Random, precision: int) -> int:
    top = 10**precision - 1
    kind = rng.random()
    if kind < 0.1:
        return 0
    if kind < 0.3:
        return rng.randint(-99, 99)  # exponente ajustado menor que −6 con escala 8
    if kind < 0.4:
        edges = [top, -top, 100, -100, 99, 1, -1, 10, 1000000]
        return rng.choice([edge for edge in edges if abs(edge) <= top])
    digits = rng.randint(1, precision)
    return rng.randint(-(10**digits - 1), 10**digits - 1)


def _value(rng: random.Random, kind: pa.DataType):
    if pa.types.is_boolean(kind):
        return rng.random() < 0.5
    if pa.types.is_integer(kind):
        if rng.random() < 0.1:
            return rng.choice([0, -(2**63), 2**63 - 1, -1])
        return rng.randint(-(10 ** rng.randint(1, 18)), 10 ** rng.randint(1, 18))
    unscaled = _unscaled(rng, kind.precision)
    return Decimal(f"{unscaled}E{-kind.scale}")


def _column(rng: random.Random, kind: pa.DataType, size: int, null_share: float):
    values = [
        None if rng.random() < null_share else _value(rng, kind) for _ in range(size)
    ]
    return pa.array(values, kind)


def _batch(rng, kinds, nullable, size, null_share):
    schema = pa.schema(
        [pa.field(f"c{i}", kind, nullable=nullable) for i, kind in enumerate(kinds)]
    )
    # Una columna no nulable no admite nulos: el reparto de nulos solo vale si nulable.
    share = null_share if nullable else 0.0
    columns = [_column(rng, kind, size, share) for kind in kinds]
    return schema, pa.RecordBatch.from_arrays(columns, schema=schema)


def _digest(hasher_class, schema, batches):
    hasher = hasher_class(schema)
    for batch in batches:
        hasher.update(batch)
    return hasher.hexdigest()


@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("null_share", NULL_SHARES)
@pytest.mark.parametrize("size", [0, 1, 7, 1000])
def test_same_digest_as_the_value_by_value_hasher(size, null_share, nullable):
    rng = random.Random(f"{size}-{null_share}-{nullable}")
    for _ in range(5):
        schema, batch = _batch(rng, TYPES, nullable, size, null_share)
        assert _digest(ContentHasher, schema, [batch]) == _digest(
            ReferenceHasher, schema, [batch]
        )


def test_same_digest_whatever_the_slicing_of_the_batch():
    rng = random.Random("slices")
    schema, batch = _batch(rng, TYPES, True, 600, 0.3)
    expected = _digest(ReferenceHasher, schema, [batch])
    cuts = [0, 1, 250, 251, 600]
    parts = [batch.slice(a, b - a) for a, b in pairwise(cuts)]
    assert _digest(ContentHasher, schema, parts) == expected
    assert _digest(ReferenceHasher, schema, parts) == expected


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "0.00000010",
        "0.00000001",
        "-0.00000001",
        "0.00000099",
        "0.00000100",
        "0.0000010",
        "-0.00001234",
        "1",
        "-1.5",
        "9999999999.99999999",
        "-9999999999.99999999",
    ],
)
def test_decimal_edge_cases(text):
    schema = pa.schema([pa.field("d", DECIMAL, nullable=True)])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([Decimal(text), None, Decimal(text)], DECIMAL)], schema=schema
    )
    assert _digest(ContentHasher, schema, [batch]) == _digest(
        ReferenceHasher, schema, [batch]
    )


def test_content_hash_of_a_table_matches_the_reference():
    rng = random.Random("table")
    schema, batch = _batch(rng, TYPES, True, 300, 0.3)
    table = pa.Table.from_batches([batch, batch.slice(10)])
    assert content_hash(table) == _digest(ReferenceHasher, schema, table.to_batches())


def test_a_known_digest_stays_pinned():
    """Un digest fijo, calculado con el hasher anterior: si cambia, se rompió la
    comparabilidad con lo ya publicado en L1, L2 y L3."""
    schema = pa.schema(
        [
            pa.field("price", DECIMAL, nullable=True),
            pa.field("n", pa.int64(), nullable=True),
            pa.field("ok", pa.bool_(), nullable=True),
            pa.field("qty", DECIMAL, nullable=False),
            pa.field("id", pa.int64(), nullable=False),
        ]
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(
                [Decimal(0), Decimal("0.0000001"), None, Decimal("-12.5")], DECIMAL
            ),
            pa.array([1, None, -3, 0], pa.int64()),
            pa.array([True, False, None, True], pa.bool_()),
            pa.array(
                [Decimal("1.5"), Decimal(0), Decimal("-0.00000001"), Decimal(7)],
                DECIMAL,
            ),
            pa.array([4, 3, 2, 1], pa.int64()),
        ],
        schema=schema,
    )
    digest = _digest(ContentHasher, schema, [batch])
    assert digest == _digest(ReferenceHasher, schema, [batch])
    assert digest == PINNED_DIGEST


PINNED_DIGEST = "2be5aae4624eef73ad39135ee68d474846858d1742c2184e065647c13b77c46b"


PERF_COLUMNS = ["os_price_max", "os_price_min", "os_duration", "os_vwap"]


def test_vectorized_hasher_is_much_faster():
    """Cuatro columnas nulables como las de `summaries.parquet`, 32 768 filas, 30 % de nulos."""
    rng = random.Random("perf")
    size = 32_768
    kinds = [DECIMAL, DECIMAL, pa.int64(), DECIMAL]
    schema = pa.schema(
        [pa.field(name, kind) for name, kind in zip(PERF_COLUMNS, kinds, strict=True)]
    )

    def realistic(kind):
        # Precios y duraciones como los de L3: ni ceros ni valores minúsculos.
        values = [
            None if rng.random() < 0.3 else rng.randint(10**6, 10**17)
            for _ in range(size)
        ]
        if pa.types.is_decimal(kind):
            values = [None if v is None else Decimal(f"{v}E-8") for v in values]
        return pa.array(values, kind)

    batch = pa.RecordBatch.from_arrays([realistic(k) for k in kinds], schema=schema)
    assert _digest(ContentHasher, schema, [batch]) == _digest(
        ReferenceHasher, schema, [batch]
    )

    def best(hasher_class, repeats):
        times = []
        for _ in range(repeats):
            hasher = hasher_class(schema)
            start = time.perf_counter()
            hasher.update(batch)
            times.append(time.perf_counter() - start)
        return min(times)

    before = best(ReferenceHasher, 3)
    after = best(ContentHasher, 5)
    print(f"update de 32 768 filas: antes {before:.4f} s, ahora {after:.4f} s")
    # La meta de ITSC-359 era 10×; se miden ≈ 8× (el `cast` decimal→texto de Arrow es
    # el 54 % de lo que queda). La guarda es 5×: atrapa una regresión al bucle en Python.
    assert after < before / 5
