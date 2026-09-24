from datetime import UTC, datetime

from dq import Finding, emit_findings
from dq.reader import current_findings

DAY1 = int(datetime(2026, 9, 24, 12, tzinfo=UTC).timestamp() * 1_000_000)
DAY2 = int(datetime(2026, 9, 25, 12, tzinfo=UTC).timestamp() * 1_000_000)


def make(**overrides) -> Finding:
    values = {
        "detected_at": DAY1,
        "layer": "l1",
        "mode": "batch",
        "check_type": "gap",
        "severity": "warning",
        "stage": "provisional",
        "status": "fail",
        "provider": "binance",
        "market": "spot",
        "asset": "BTCUSDT",
        "year": 2026,
        "month": 9,
        "details": {"missing": 3},
        "run_id": "run-1",
        "image_version": "0.1.0",
    }
    return Finding(**(values | overrides))


def test_provisional_to_canonical(tmp_path):
    provisional = make()
    canonical = make(
        finding_id=provisional.finding_id,
        detected_at=DAY2,
        stage="canonical",
        status="pass",
        run_id="run-2",
    )
    other = make()
    files = emit_findings([provisional, other], tmp_path)
    files += emit_findings([canonical], tmp_path)

    table = current_findings(tmp_path)

    rows = {r["finding_id"]: r for r in table.to_pylist()}
    assert len(rows) == table.num_rows == 2
    assert rows[provisional.finding_id]["stage"] == "canonical"
    assert rows[other.finding_id]["stage"] == "provisional"
    assert len(files) == 2
    assert len(list(tmp_path.glob("detected_date=*/*.parquet"))) == 2
