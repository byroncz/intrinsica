import os
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyutils import PartitionWriter, content_hash, resolve_fs

SCHEMA = pa.schema(
    [
        pa.field("t", pa.int64(), nullable=False),
        pa.field("v", pa.int32(), nullable=False),
    ]
)


def _table(n: int = 5) -> pa.Table:
    return pa.table({"t": list(range(n)), "v": [1] * n}, schema=SCHEMA)


class _FsWithoutTmpDelete:
    """Como GCS tras `move`: borrar lo que no existe lanza un OSError genérico."""

    def __init__(self, fs):
        self._fs = fs
        self.deleted = []

    def __getattr__(self, name):
        return getattr(self._fs, name)

    def delete_file(self, path):
        self.deleted.append(path)
        raise OSError("object does not exist")


def test_writes_with_the_schema_and_physical_properties_it_is_given(tmp_path):
    path = str(tmp_path / "a" / "x.parquet")
    with PartitionWriter(path, SCHEMA, [("t", "ascending")]) as writer:
        writer.write_table(_table())
        assert writer.commit() == path

    pf = pq.ParquetFile(path)
    assert pf.schema_arrow.equals(SCHEMA)
    row_group = pf.metadata.row_group(0)
    assert [(s.column_index, s.descending) for s in row_group.sorting_columns] == [
        (0, False)
    ]
    for c in range(row_group.num_columns):
        col = row_group.column(c)
        assert col.compression == "ZSTD"
        assert col.statistics.has_min_max


def test_without_sort_order_no_sorting_columns_are_written(tmp_path):
    path = str(tmp_path / "x.parquet")
    with PartitionWriter(path, SCHEMA) as writer:
        writer.write_table(_table())
        writer.commit()
    assert not pq.ParquetFile(path).metadata.row_group(0).sorting_columns


def test_each_write_is_its_own_row_group_by_default(tmp_path):
    path = str(tmp_path / "x.parquet")
    with PartitionWriter(path, SCHEMA) as writer:
        writer.write_batch(_table(7).to_batches()[0])
        writer.write_table(_table(3))
        writer.commit()
    meta = pq.ParquetFile(path).metadata
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == [7, 3]


def test_row_group_size_splits_a_larger_write(tmp_path):
    path = str(tmp_path / "x.parquet")
    with PartitionWriter(path, SCHEMA, row_group_size=4) as writer:
        writer.write_table(_table(10))
        writer.commit()
    meta = pq.ParquetFile(path).metadata
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == [
        4,
        4,
        2,
    ]


def test_rejects_other_schema(tmp_path):
    with PartitionWriter(str(tmp_path / "x.parquet"), SCHEMA) as writer:
        with pytest.raises(ValueError, match="esquema"):
            writer.write_table(pa.table({"a": [1]}))
        with pytest.raises(ValueError, match="esquema"):
            writer.write_batch(pa.record_batch({"a": [1]}))


def test_commit_exitoso_no_intenta_borrar_el_temporal(tmp_path):
    path = str(tmp_path / "c.parquet")
    with PartitionWriter(path, SCHEMA) as writer:
        writer._fs = fake = _FsWithoutTmpDelete(writer._fs)
        writer.write_table(_table(3))
        writer.commit()
    assert fake.deleted == []
    assert pq.read_table(path).num_rows == 3


def test_error_de_borrado_no_oculta_la_excepcion_original(tmp_path):
    path = str(tmp_path / "c.parquet")
    with (
        pytest.raises(RuntimeError, match="original"),
        PartitionWriter(path, SCHEMA) as writer,
    ):
        writer._fs = _FsWithoutTmpDelete(writer._fs)
        raise RuntimeError("original")


def test_abort_keeps_the_previous_file_and_leaves_no_temporary(tmp_path):
    path = str(tmp_path / "c.parquet")
    with PartitionWriter(path, SCHEMA) as writer:
        writer.write_table(_table(3))
        writer.commit()
    with pytest.raises(RuntimeError, match="boom"), PartitionWriter(path, SCHEMA) as w:
        w.write_table(_table(5))
        raise RuntimeError("boom")
    assert pq.read_table(path).num_rows == 3
    assert [p.name for p in tmp_path.iterdir()] == ["c.parquet"]


def test_overwrite_leaves_a_single_file(tmp_path):
    path = str(tmp_path / "c.parquet")
    for n in (3, 5):
        with PartitionWriter(path, SCHEMA) as writer:
            writer.write_table(_table(n))
            writer.commit()
    assert pq.read_table(path).num_rows == 5
    assert [p.name for p in tmp_path.iterdir()] == ["c.parquet"]


def test_resolve_fs_local_paths_come_back_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, resolved = resolve_fs("rel/x.parquet")
    assert resolved == str(tmp_path / "rel" / "x.parquet")
    _, from_path = resolve_fs(tmp_path)
    assert from_path == str(tmp_path)


def _compact_table(n: int = 5_000) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("price", pa.decimal128(18, 8), nullable=False),
            pa.field("time", pa.int64(), nullable=False),
            pa.field("direction", pa.int8(), nullable=False),
            pa.field("month", pa.int32(), nullable=True),
            pa.field("name", pa.string(), nullable=False),
        ]
    )
    return pa.table(
        {
            "price": pa.array(
                [
                    Decimal(3_000_000_000_000 + i % 977 * 1_234_567).scaleb(-8)
                    for i in range(n)
                ],
                pa.decimal128(18, 8),
            ),
            "time": [1_600_000_000_000_000 + i * 37 for i in range(n)],
            "direction": [1 if i % 3 else -1 for i in range(n)],
            "month": [None if i % 5 == 0 else i % 12 + 1 for i in range(n)],
            "name": ["BTCUSDT"] * n,
        },
        schema=schema,
    )


def test_compact_encoding_has_no_dictionary_and_delta_on_integers(tmp_path):
    table = _compact_table()
    path = str(tmp_path / "compact.parquet")
    with PartitionWriter(path, table.schema, compact_encoding=True) as writer:
        writer.write_table(table)
        writer.commit()

    column = {
        c.path_in_schema: c
        for c in map(pq.ParquetFile(path).metadata.row_group(0).column, range(5))
    }
    for name in ("time", "direction", "month"):
        assert "DELTA_BINARY_PACKED" in column[name].encodings, name
    for name in ("price", "name"):
        assert "DELTA_BINARY_PACKED" not in column[name].encodings, name
    for name, c in column.items():
        assert not c.has_dictionary_page, name
        assert "PLAIN_DICTIONARY" not in c.encodings, name
        assert "RLE_DICTIONARY" not in c.encodings, name
        assert c.compression == "ZSTD"
    assert pq.read_table(path).equals(table)


def test_compact_encoding_keeps_the_logical_content_and_shrinks_the_file(tmp_path):
    table = _compact_table(50_000)
    paths = {}
    for compact in (False, True):
        paths[compact] = str(tmp_path / f"{compact}.parquet")
        with PartitionWriter(
            paths[compact], table.schema, compact_encoding=compact
        ) as writer:
            writer.write_table(table)
            writer.commit()

    assert pq.read_table(paths[True]).equals(pq.read_table(paths[False]))
    assert content_hash(pq.read_table(paths[True])) == content_hash(
        pq.read_table(paths[False])
    )
    assert os.path.getsize(paths[True]) < os.path.getsize(paths[False])


def test_commit_times_closing_and_moving_the_file(tmp_path):
    with PartitionWriter(str(tmp_path / "t.parquet"), SCHEMA) as writer:
        assert (writer.close_s, writer.move_s) == (0.0, 0.0)
        writer.write_table(_table(3))
        writer.commit()
    assert writer.close_s > 0
    assert writer.move_s > 0
