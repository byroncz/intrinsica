import functools
import hashlib

import pytest
from l1_ingest import download, pipeline
from l1_ingest.download import ChecksumError, fetch
from l1_ingest.manifest import last_sha256
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
    monkeypatch.setattr(
        pipeline,
        "fetch_checksum",
        functools.partial(download.fetch_checksum, sleep=lambda s: None),
    )


def _ctx(tmp_path, base, mode="backfill", force=False):
    return RunContext(
        mode=mode,
        run_id="run-1",
        image_version="0.1.0+test",
        landing_root=tmp_path / "landing",
        dq_root=tmp_path / "dq",
        manifest_root=tmp_path / "manifest",
        source_base_url=base,
        force=force,
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
    second = process_unit(Unit(2024, 3), _ctx(tmp_path, base, force=True))

    assert first.content_hash == second.content_hash
    assert len(_files(tmp_path / "landing")) == 1
    assert len(_files(tmp_path / "dq")) == 2
    assert len(_files(tmp_path / "manifest")) == 2


@pytest.fixture
def requested(monkeypatch):
    urls = []
    real = download._get

    def spy(url):
        urls.append(url)
        return real(url)

    monkeypatch.setattr(download, "_get", spy)
    return urls


@pytest.mark.parametrize("day", [None, 6])
def test_matching_checksum_skips_without_downloading_zip(
    tmp_path, publish_zip, requested, caplog, day
):
    publish, base = publish_zip
    publish(day=day)
    mode = "backfill" if day is None else "daily"
    unit = Unit(2024, 3, day)
    process_unit(unit, _ctx(tmp_path, base, mode))
    before = [(p, p.stat().st_mtime_ns) for p in _files(tmp_path)]
    requested.clear()

    with caplog.at_level("INFO"):
        result = process_unit(unit, _ctx(tmp_path, base, mode))

    assert result.skipped
    assert [u for u in requested if not u.endswith(".CHECKSUM")] == []
    assert len(requested) == 1
    assert "salto unidad=" in caplog.text and "sha256=" in caplog.text
    assert [(p, p.stat().st_mtime_ns) for p in _files(tmp_path)] == before


def test_drift_reprocesses_and_emits_checksum_drift(tmp_path, publish_zip):
    publish, base = publish_zip
    publish()
    unit = Unit(2024, 3)
    process_unit(unit, _ctx(tmp_path, base))
    url = f"{base}/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-03.zip"
    old = last_sha256(tmp_path / "manifest", "binance", "spot", "BTCUSDT", 2024, 3, url)
    # Binance republica el .CHECKSUM con otro hash; el ZIP sigue siendo válido.
    zip_path = tmp_path / "data/spot/monthly/aggTrades/BTCUSDT" / url.rsplit("/", 1)[1]
    data = zip_path.read_bytes() + b"\0"
    zip_path.write_bytes(data)
    new = hashlib.sha256(data).hexdigest()
    zip_path.with_name(zip_path.name + ".CHECKSUM").write_text(f"{new}  x.zip\n")

    result = process_unit(unit, _ctx(tmp_path, base))

    assert not result.skipped
    drift = [f for f in result.findings if f.check_type == "checksum_drift"]
    assert len(drift) == 1
    assert (drift[0].severity, drift[0].status) == ("warning", "corrected")
    assert drift[0].details == {"previous_sha256": old, "new_sha256": new}
    assert len(result.findings) == 7
    assert (
        last_sha256(tmp_path / "manifest", "binance", "spot", "BTCUSDT", 2024, 3, url)
        == new
    )


def test_missing_output_processes_even_with_manifest_rows(tmp_path, publish_zip):
    publish, base = publish_zip
    publish(day=6)
    unit = Unit(2024, 3, 6)
    first = process_unit(unit, _ctx(tmp_path, base, "daily"))
    next((tmp_path / "landing").rglob("provisional-day=06.parquet")).unlink()

    second = process_unit(unit, _ctx(tmp_path, base, "daily"))

    assert not second.skipped
    assert second.content_hash == first.content_hash
    assert not [f for f in second.findings if f.check_type == "checksum_drift"]
    assert len(_files(tmp_path / "manifest")) == 2


def test_force_reprocesses_when_checksum_matches(tmp_path, publish_zip, requested):
    publish, base = publish_zip
    publish()
    process_unit(Unit(2024, 3), _ctx(tmp_path, base))
    requested.clear()

    result = process_unit(Unit(2024, 3), _ctx(tmp_path, base, force=True))

    assert not result.skipped
    assert any(u.endswith(".zip") for u in requested)
    assert not [f for f in result.findings if f.check_type == "checksum_drift"]
