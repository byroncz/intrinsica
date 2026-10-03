import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / ".github/scripts/findings-summary.sh"


def finding(**overrides) -> dict:
    return {
        "finding_id": "f1",
        "layer": "l1",
        "mode": "backfill",
        "check_type": "source_delayed",
        "severity": "error",
        "stage": "canonical",
        "status": "fail",
        "provider": "binance",
        "market": "spot",
        "asset": "BTCUSDT",
        "year": 2026,
        "month": 9,
        "metric_value": 3.0,
        "details": {"reason": "3 días de retraso sobre el calendario publicado"},
        "run_id": "r1",
    } | overrides


def summary(*lines: str, **env) -> list[str]:
    out = subprocess.run(
        [SCRIPT],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", **env},
    )
    return out.stdout.splitlines()


def test_line_with_cloud_run_prefix_becomes_a_row():
    # Cloud Run entrega la línea con el prefijo de fecha del formato de logging.
    line = "2026-10-03 06:21:00,123 " + json.dumps(finding())
    rows = summary(line)
    assert "| Check | Severidad | Unidad | Valor | Detalle |" in rows
    assert (
        "| source_delayed | error | l1 BTCUSDT 2026-09 | 3.0 | "
        "3 días de retraso sobre el calendario publicado |"
    ) in rows


def test_no_findings_says_so():
    rows = summary("2026-10-03 06:21:00 inicio unidad=x", "sonda: unit=x {no json}")
    assert rows[-1] == "sin hallazgos"
    assert not any(r.startswith("| Check") for r in rows)


def test_unit_uses_the_day_when_the_finding_carries_it():
    rows = summary(
        json.dumps(
            finding(
                severity="warning",
                details={"unit": "binance/spot/BTCUSDT/2026-10-03", "k": 1},
            )
        )
    )
    assert any("l1 binance/spot/BTCUSDT/2026-10-03" in r for r in rows)


def test_details_without_reason_and_null_metric():
    rows = summary(
        json.dumps(
            finding(
                check_type="aggid_gap",
                layer="l2",
                severity="info",
                metric_value=None,
                details={"gaps": 2},
            )
        )
    )
    assert '| aggid_gap | info | l2 BTCUSDT 2026-09 | - | {"gaps":2} |' in rows


def test_pipe_in_details_does_not_break_the_table():
    rows = summary(json.dumps(finding(details={"reason": "a | b"})))
    assert any(r.endswith("a \\| b |") for r in rows)


def test_errors_first_and_retries_are_not_duplicated():
    info = finding(finding_id="i", severity="info", check_type="checksum_fail")
    error = finding(finding_id="e")
    rows = summary(json.dumps(info), json.dumps(error), json.dumps(error))
    table = [
        r for r in rows if r.startswith("| ") and "Check" not in r and "---" not in r
    ]
    assert [r.split(" | ")[0] for r in table] == ["| source_delayed", "| checksum_fail"]
    assert "2 hallazgos: 1 error, 0 warning, 1 info." in rows


def test_max_rows_truncates_with_a_note():
    lines = [json.dumps(finding(finding_id=str(n))) for n in range(5)]
    rows = summary(*lines, MAX_ROWS="2")
    assert len([r for r in rows if r.startswith("| source_delayed")]) == 2
    assert "Se muestran 2 de 5 filas." in rows


def test_run_job_workflow_appends_the_table_to_the_summary():
    workflow = (SCRIPT.parents[1] / "workflows/run-job.yml").read_text()
    assert (
        '.github/scripts/findings-summary.sh <<< "$lines" >> "$GITHUB_STEP_SUMMARY"'
        in workflow
    )
