import functools

import pytest
from l1_ingest import pipeline
from l1_ingest.download import ChecksumError, fetch
from l1_ingest.pipeline import RunContext, Unit, process_unit

CHECKS = {
    "checksum_fail",
    "header_detected",
    "timestamp_unit_corrected",
    "reorder_applied",
    "aggid_gap",
    "aggid_duplicate",
}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(
        pipeline, "fetch", functools.partial(fetch, sleep=lambda s: None)
    )


def _ctx(tmp_path, base, mode="backfill"):
    return RunContext(
        mode=mode,
        run_id="run-1",
        image_version="0.1.0+test",
        landing_root=tmp_path / "landing",
        dq_root=tmp_path / "dq",
        manifest_root=tmp_path / "manifest",
        source_base_url=base,
    )


def _files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_day_is_provisional_with_six_findings(tmp_path, publish_zip):
    publish, base = publish_zip
    publish(day=5)
    result = process_unit(Unit(2024, 3, 5), _ctx(tmp_path, base, "daily"))

    assert result.path.endswith(
        "provider=binance/market=spot/asset=BTCUSDT/year=2024/month=03/"
        "provisional-day=05.parquet"
    )
    assert {f.check_type for f in result.findings} == CHECKS
    assert len(result.findings) == 6
    assert {f.stage for f in result.findings} == {"provisional"}
    assert len(result.content_hash) == 64


def test_month_is_canonical(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    result = process_unit(Unit(2024, 3), _ctx(tmp_path, base))

    assert result.path.endswith("year=2024/month=03/consolidated.parquet")
    assert {f.stage for f in result.findings} == {"canonical"}
    assert {f.mode for f in result.findings} == {"backfill"}


def test_only_partition_findings_and_manifest_touch_disk(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    process_unit(Unit(2024, 3), _ctx(tmp_path, base))

    assert {p.suffix for p in _files(tmp_path / "landing")} == {".parquet"}
    assert len(_files(tmp_path / "landing")) == 1
    assert len(_files(tmp_path / "dq")) == 1
    assert len(_files(tmp_path / "manifest")) == 1


def test_persistent_bad_checksum_emits_checksum_fail(tmp_path, publish_zip):
    publish, base = publish_zip
    publish(checksum="0" * 64)
    with pytest.raises(ChecksumError):
        process_unit(Unit(2024, 3), _ctx(tmp_path, base))

    from dq.reader import current_findings

    rows = current_findings(tmp_path / "dq").to_pylist()
    assert [(r["check_type"], r["severity"], r["status"]) for r in rows] == [
        ("checksum_fail", "error", "fail")
    ]
    assert not (tmp_path / "landing").exists()


def test_two_runs_same_hash_and_append_only(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    first = process_unit(Unit(2024, 3), _ctx(tmp_path, base))
    second = process_unit(Unit(2024, 3), _ctx(tmp_path, base))

    assert first.content_hash == second.content_hash
    assert len(_files(tmp_path / "landing")) == 1
    assert len(_files(tmp_path / "dq")) == 2
    assert len(_files(tmp_path / "manifest")) == 2
