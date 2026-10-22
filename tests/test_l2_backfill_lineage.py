"""`l2_backfill_lineage.py` (ITSC-368) contra un lago que L2 escribió de verdad.

L2 publica `content_hash.json` por sí mismo; para probar el backfill se borran y se comprueba
que el script devuelve exactamente lo que L2 había publicado, sin recalcular lo que ya tiene
un hallazgo y recalculando lo que no.
"""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from pyutils import lineage
from test_ops_scripts import run_l2, write_landing

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "layers/ops_tools/scripts/l2_backfill_lineage.py"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
PARTITIONS = 150  # 50 θ × 3 meses


def load():
    spec = importlib.util.spec_from_file_location("l2_backfill_lineage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def published(lake: Path) -> dict[Path, dict]:
    return {
        path: lineage.read(str(path))
        for path in sorted((lake / "events").rglob(lineage.LINEAGE))
    }


def args(lake: Path, *extra: str) -> list[str]:
    return [
        "--events-root",
        str(lake / "events"),
        "--dq-root",
        str(lake / "dq"),
        *extra,
    ]


@pytest.fixture
def lake(tmp_path):
    write_landing(tmp_path / "landing")
    run_l2(tmp_path, "--from", "2017-08", "--to", "2017-10")
    return tmp_path


@pytest.fixture
def legacy(lake):
    """El lago como estaba antes de ITSC-368: lo que L2 publicó, y sin `content_hash.json`."""
    before = published(lake)
    assert len(before) == PARTITIONS
    for path in before:
        path.unlink()
    return lake, before


def last_line(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_the_backfill_restores_what_l2_published(legacy, capsys):
    lake, before = legacy
    assert load().main(args(lake)) == 0
    after = published(lake)
    assert after.keys() == before.keys()
    for path, expected in before.items():
        found = after[path]
        for key in ("events_content_hash", "carry_over_content_hash", "hasher_version"):
            assert found[key] == expected[key]
    assert last_line(capsys) == (
        f"particiones: {PARTITIONS}; escritas: {PARTITIONS} (hallazgo: {PARTITIONS}, "
        "recalculadas: 0); ya existían: 0; sin hallazgo: 0; incompletas: 0; "
        "muestra: 50; discrepancias: 0; errores: 0"
    )


def test_the_sample_is_at_least_five_percent_and_has_the_first_month(legacy, capsys):
    """Con 150 particiones el 5 % son 8, pero el primer mes entero (50) es obligatorio."""
    lake, _ = legacy
    script = load()
    verified = []
    real = script.partition_hashes
    script.partition_hashes = lambda root, parts, key: (
        verified.append(key),
        real(root, parts, key),
    )[1]
    assert script.main(args(lake)) == 0
    months = {(year, month) for _, year, month in verified}
    assert months == {(2017, 8)}
    assert len(verified) == 50

    script.partition_hashes = real
    sample = script.choose_sample(
        [(f"0.{n:08d}", 2020, 5) for n in range(1, 101)],
        forced={(2021, 1)},
        disagree={("0.00000007", 2020, 5)},
        fraction=0.05,
        seed=0,
    )
    assert len(sample) == 5 and ("0.00000007", 2020, 5) in sample
    assert sample == script.choose_sample(
        [(f"0.{n:08d}", 2020, 5) for n in range(1, 101)],
        {(2021, 1)},
        {("0.00000007", 2020, 5)},
        0.05,
        0,
    )


def test_a_rerun_creates_nothing_and_never_overwrites(legacy, capsys):
    lake, _ = legacy
    script = load()
    assert script.main(args(lake)) == 0
    first = {p: p.read_bytes() for p in published(lake)}
    assert script.main(args(lake)) == 0
    assert {p: p.read_bytes() for p in published(lake)} == first
    assert last_line(capsys) == (
        f"particiones: {PARTITIONS}; escritas: 0 (hallazgo: 0, recalculadas: 0); "
        f"ya existían: {PARTITIONS}; sin hallazgo: 0; incompletas: 0; muestra: 0; "
        "discrepancias: 0; errores: 0"
    )


def test_a_partition_without_finding_is_recomputed_not_left_without_lineage(
    legacy, tmp_path_factory, capsys
):
    lake, before = legacy
    empty = tmp_path_factory.mktemp("sin-hallazgos")
    argv = ["--events-root", str(lake / "events"), "--dq-root", str(empty)]
    assert load().main(argv) == 0
    after = published(lake)
    assert after.keys() == before.keys()
    for path, expected in before.items():
        assert after[path]["events_content_hash"] == expected["events_content_hash"]
        assert (
            after[path]["carry_over_content_hash"]
            == expected["carry_over_content_hash"]
        )
    assert last_line(capsys) == (
        f"particiones: {PARTITIONS}; escritas: {PARTITIONS} (hallazgo: 0, recalculadas: "
        f"{PARTITIONS}); ya existían: 0; sin hallazgo: {PARTITIONS}; incompletas: 0; "
        "muestra: 0; discrepancias: 0; errores: 0"
    )


def test_a_finding_that_does_not_describe_the_file_aborts_without_writing(
    legacy, capsys
):
    lake, _ = legacy
    path = next(
        (lake / "events").rglob("theta=0.00010000/year=2017/month=08/events.parquet")
    )
    table = pq.read_table(path)
    index = table.schema.get_field_index("direction")
    flipped = pa.array([-d for d in table.column("direction").to_pylist()], pa.int8())
    pq.write_table(table.set_column(index, "direction", flipped), path)

    assert load().main(args(lake)) == 1
    out = capsys.readouterr().out
    assert "DISCREPANCIA θ=0.00010000 2017-08" in out
    assert "FALLO muestra_difiere" in out
    assert not published(lake)


def test_dry_run_verifies_but_writes_nothing(legacy, capsys):
    lake, _ = legacy
    assert load().main(args(lake, "--dry-run")) == 0
    assert not published(lake)
    out = capsys.readouterr().out
    assert f"--dry-run: se escribirían {PARTITIONS} content_hash.json" in out


def test_a_partition_with_one_parquet_is_reported_and_skipped(legacy, capsys):
    lake, _ = legacy
    broken = next(
        (lake / "events").rglob(
            "theta=0.00010000/year=2017/month=09/carry_over.parquet"
        )
    )
    broken.unlink()
    assert load().main(args(lake)) == 0
    out = capsys.readouterr().out
    assert "INCOMPLETA θ=0.00010000 2017-09: ['events.parquet']" in out
    assert len(published(lake)) == PARTITIONS - 1
    assert out.strip().splitlines()[-1].count("incompletas: 1") == 1


def test_nothing_to_cover_is_a_failure(tmp_path, capsys):
    (tmp_path / "events").mkdir()
    (tmp_path / "dq").mkdir()
    assert load().main(args(tmp_path)) == 1
    assert "FALLO sin_particiones" in capsys.readouterr().out


class IamLikeHandler(pafs.FileSystemHandler):
    """Un disco local que responde como GCS con la condición IAM de ops-script (ITSC-370).

    Preguntar por una ruta que no es un archivo existente ni termina en `/` (en GCS, el objeto
    `l2` sin barra) da PermissionError, no "no existe"; el listado con selector pasa.
    """

    def __init__(self):
        self.local = pafs.LocalFileSystem()

    def get_type_name(self):
        return "iam-like"

    def __eq__(self, other):
        return isinstance(other, IamLikeHandler)

    def __ne__(self, other):
        return not self == other

    def get_file_info(self, paths):
        infos = []
        for path in paths:
            info = self.local.get_file_info(path)
            if info.type != pafs.FileType.File and not path.endswith("/"):
                raise PermissionError(f"403 storage.objects.get denegado para {path}")
            infos.append(info)
        return infos

    def get_file_info_selector(self, selector):
        return self.local.get_file_info(selector)

    def create_dir(self, path, recursive):
        self.local.create_dir(path, recursive=recursive)

    def delete_dir(self, path):
        self.local.delete_dir(path)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        self.local.delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

    def delete_root_dir_contents(self):
        raise NotImplementedError

    def delete_file(self, path):
        self.local.delete_file(path)

    def move(self, src, dest):
        self.local.move(src, dest)

    def copy_file(self, src, dest):
        self.local.copy_file(src, dest)

    def open_input_stream(self, path):
        return self.local.open_input_stream(path)

    def open_input_file(self, path):
        return self.local.open_input_file(path)

    def open_output_stream(self, path, metadata):
        return self.local.open_output_stream(path, metadata=metadata)

    def open_append_stream(self, path, metadata):
        return self.local.open_append_stream(path, metadata=metadata)

    def normalize_path(self, path):
        return path


def test_the_findings_are_read_without_asking_for_the_prefix_without_slash(
    legacy, monkeypatch, capsys
):
    """El prefijo de hallazgos no puede consultarse como objeto: ds.dataset(prefijo) lo hacía."""
    lake, _ = legacy
    script = load()
    real = script.resolve_fs
    monkeypatch.setattr(
        script,
        "resolve_fs",
        lambda root: (pafs.PyFileSystem(IamLikeHandler()), real(root)[1]),
    )
    with pytest.raises(PermissionError):
        pafs.PyFileSystem(IamLikeHandler()).get_file_info(str(lake / "dq"))

    assert script.main(args(lake, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith(f"particiones: {PARTITIONS};")
    assert not published(lake)
