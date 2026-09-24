import pyarrow.fs as pafs
import pyarrow.parquet as pq
from dq.emit import _resolve, _write
from test_emit import make


def test_gs_root_resolves_without_network():
    fs, base = _resolve("gs://bucket-x/prefijo")
    assert isinstance(fs, pafs.GcsFileSystem)
    assert base == "bucket-x/prefijo"


def test_write_is_backend_agnostic(tmp_path):
    fs = pafs.SubTreeFileSystem(str(tmp_path), pafs.LocalFileSystem())
    (path,) = _write([make(), make()], fs, "bucket-x/prefijo")
    assert path.startswith("bucket-x/prefijo/detected_date=2026-09-24/run-1-")
    assert pq.read_table(path, filesystem=fs).num_rows == 2
    metadata = pq.ParquetFile(path, filesystem=fs).metadata
    assert metadata.row_group(0).column(0).compression == "ZSTD"
