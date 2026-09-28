from decimal import Decimal

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from l1_ingest.schema import OUTPUT_SCHEMA
from l1_ingest.write import (
    CONSOLIDATED,
    PartitionWriter,
    content_hash,
    day_filename,
    partition_path,
    write_partition,
)


def _table(n: int = 5) -> pa.Table:
    return pa.table(
        {
            "agg_trade_id": list(range(n)),
            "price": [Decimal("100.12345678")] * n,
            "quantity": [Decimal("0.5")] * n,
            "first_trade_id": list(range(n)),
            "last_trade_id": list(range(n)),
            "transact_time": [1_700_000_000_000_000 + i for i in range(n)],
            "is_buyer_maker": [i % 2 == 0 for i in range(n)],
            "is_best_match": [True] * n,
        },
        schema=OUTPUT_SCHEMA,
    )


def test_partition_path(tmp_path):
    path = partition_path(tmp_path, "binance", "spot", "BTCUSDT", 2024, 3, CONSOLIDATED)
    assert path == (
        f"{tmp_path}/provider=binance/market=spot/asset=BTCUSDT"
        "/year=2024/month=03/consolidated.parquet"
    )


def test_partition_path_gcs_root():
    path = partition_path("gs://b/l1/", "binance", "spot", "BTCUSDT", 2024, 12, "x")
    assert path == (
        "gs://b/l1/provider=binance/market=spot/asset=BTCUSDT/year=2024/month=12/x"
    )


def test_day_filename():
    assert day_filename(3) == "provisional-day=03.parquet"
    assert day_filename(21) == "provisional-day=21.parquet"


def test_write_partition_physical_properties(tmp_path):
    path = partition_path(tmp_path, "binance", "spot", "BTCUSDT", 2024, 3, "c.parquet")
    assert write_partition(_table(), path) == path

    pf = pq.ParquetFile(path)
    assert pf.schema_arrow.equals(OUTPUT_SCHEMA)
    meta = pf.metadata
    assert meta.num_rows == 5
    for rg in range(meta.num_row_groups):
        row_group = meta.row_group(rg)
        assert [(s.column_index, s.descending) for s in row_group.sorting_columns] == [
            (5, False),
            (0, False),
        ]
        for c in range(row_group.num_columns):
            col = row_group.column(c)
            assert col.compression == "ZSTD"
            assert col.is_stats_set
            assert col.statistics.has_min_max


def test_write_partition_rejects_other_schema(tmp_path):
    with pytest.raises(ValueError):
        write_partition(pa.table({"a": [1]}), str(tmp_path / "x.parquet"))


def test_overwrite_leaves_single_file(tmp_path):
    path = partition_path(tmp_path, "binance", "spot", "BTCUSDT", 2024, 3, "c.parquet")
    write_partition(_table(3), path)
    write_partition(_table(5), path)
    assert pq.read_table(path).num_rows == 5
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["c.parquet"]


def test_content_hash_ignores_chunking():
    table = _table(6)
    chunked = pa.concat_tables([table.slice(0, 2), table.slice(2)])
    assert chunked.column(0).num_chunks == 2
    assert content_hash(chunked) == content_hash(table)


def test_content_hash_changes_with_a_row():
    table = _table()
    ids = table.column("agg_trade_id").to_pylist()
    ids[2] += 100
    other = table.set_column(0, OUTPUT_SCHEMA.field(0), pa.array(ids, pa.int64()))
    assert content_hash(other) != content_hash(table)


def test_content_hash_survives_roundtrip(tmp_path):
    table = _table()
    path = str(tmp_path / "c.parquet")
    write_partition(table, path)
    assert content_hash(pq.read_table(path, schema=OUTPUT_SCHEMA)) == content_hash(
        table
    )


def test_content_hash_ignores_bits_outside_a_slice(tmp_path):
    flags = [True, False, True, True, True, False, True, False, True, True]
    big = _table(10).set_column(6, OUTPUT_SCHEMA.field(6), pa.array(flags, pa.bool_()))
    sliced = big.slice(0, 3)
    fresh = _table(3).set_column(
        6, OUTPUT_SCHEMA.field(6), pa.array(flags[:3], pa.bool_())
    )
    path = str(tmp_path / "c.parquet")
    write_partition(sliced, path)
    roundtrip = pq.read_table(path, schema=OUTPUT_SCHEMA)
    assert content_hash(sliced) == content_hash(fresh) == content_hash(roundtrip)


def test_write_partition_rejects_unsorted_table(tmp_path):
    table = _table(3).take([2, 0, 1])
    with pytest.raises(ValueError, match="ordenada"):
        write_partition(table, str(tmp_path / "c.parquet"))
    assert list(tmp_path.iterdir()) == []


def test_write_partition_rejects_unsorted_agg_trade_id_on_equal_time(tmp_path):
    table = _table(3).set_column(5, OUTPUT_SCHEMA.field(5), pa.array([1, 1, 2]))
    with pytest.raises(ValueError, match="ordenada"):
        write_partition(table.take([1, 0, 2]), str(tmp_path / "c.parquet"))


def test_content_hash_ignores_schema_metadata():
    table = _table()
    assert content_hash(table.replace_schema_metadata({b"x": b"y"})) == content_hash(
        table
    )


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


def test_commit_exitoso_no_intenta_borrar_el_temporal(tmp_path):
    path = str(tmp_path / "c.parquet")
    with PartitionWriter(path) as writer:
        writer._fs = fake = _FsWithoutTmpDelete(writer._fs)
        writer.write_table(_table(3))
        writer.commit()
    assert fake.deleted == []
    assert pq.read_table(path).num_rows == 3


def test_error_de_borrado_no_oculta_la_excepcion_original(tmp_path):
    path = str(tmp_path / "c.parquet")
    with (
        pytest.raises(RuntimeError, match="original"),
        PartitionWriter(path) as writer,
    ):
        writer._fs = _FsWithoutTmpDelete(writer._fs)
        raise RuntimeError("original")


def test_abort_local_deja_el_destino_intacto_y_sin_temporal(tmp_path):
    path = str(tmp_path / "c.parquet")
    write_partition(_table(3), path)
    with pytest.raises(RuntimeError, match="boom"), PartitionWriter(path) as writer:
        writer.write_table(_table(5))
        raise RuntimeError("boom")
    assert pq.read_table(path).num_rows == 3
    assert [p.name for p in tmp_path.iterdir()] == ["c.parquet"]


class _FakeGcsSink:
    """Buffer en memoria; el objeto solo "sube" a `objects` si se cierra.

    Como el `GcsFileSystem` real: mientras el stream no cierra con éxito,
    `write_partition` puede llenarlo de bytes sin que nada quede subido.
    """

    def __init__(self, fs, path):
        self._fs = fs
        self._path = path
        self._buffer = bytearray()
        self.closed = False

    def write(self, data):
        self._buffer.extend(data)
        return len(data)

    def tell(self):
        return len(self._buffer)

    def writable(self):
        return True

    def seekable(self):
        return False

    def readable(self):
        return False

    def close(self):
        self._fs.objects[self._path] = bytes(self._buffer)
        self.closed = True


class _FakeGcsFileSystem:
    """Simula lo justo de GCS: `open_output_stream` no crea nada hasta cerrar."""

    def __init__(self, existing: dict[str, bytes] | None = None):
        self.objects = dict(existing or {})
        self.opened: list[str] = []

    def get_file_info(self, path):
        found = path in self.objects
        return pafs.FileInfo(
            path, pafs.FileType.File if found else pafs.FileType.NotFound
        )

    def open_output_stream(self, path, **kwargs):
        self.opened.append(path)
        return _FakeGcsSink(self, path)

    def delete_file(self, path):
        if path not in self.objects:
            raise OSError(f"no existe: {path}")
        del self.objects[path]


def _as_gcs(tmp_path, fake_fs) -> PartitionWriter:
    """`PartitionWriter` sobre `fake_fs`, como si `path` fuera `gs://...`."""
    writer = PartitionWriter(str(tmp_path / "c.parquet"))
    writer._fs = fake_fs
    writer._is_gcs = True
    return writer


def test_gcs_commit_escribe_directo_sobre_target_sin_tmp(tmp_path):
    fake_fs = _FakeGcsFileSystem()
    writer = _as_gcs(tmp_path, fake_fs)
    with writer as w:
        w.write_table(_table(3))
        w.commit()
    assert fake_fs.opened == [writer._target]
    assert list(fake_fs.objects) == [writer._target]
    assert fake_fs.objects[writer._target]


def test_gcs_abort_no_crea_ningun_objeto(tmp_path):
    fake_fs = _FakeGcsFileSystem()
    writer = _as_gcs(tmp_path, fake_fs)
    with pytest.raises(RuntimeError, match="boom"), writer as w:
        w.write_table(_table(3))
        raise RuntimeError("boom")
    assert fake_fs.objects == {}


def test_gcs_abort_no_toca_una_version_previa(tmp_path):
    fake_fs = _FakeGcsFileSystem()
    writer = _as_gcs(tmp_path, fake_fs)
    fake_fs.objects[writer._target] = b"contenido previo"
    with pytest.raises(RuntimeError, match="boom"), writer as w:
        w.write_table(_table(3))
        raise RuntimeError("boom")
    assert fake_fs.objects[writer._target] == b"contenido previo"
