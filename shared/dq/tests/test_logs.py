import json
import logging
import sys

import pytest
from dq import Finding, configure_logging, emit_findings
from dq.logs import CloudRunFormatter


def make(**overrides) -> Finding:
    values = {
        "layer": "l1",
        "mode": "monthly-close",
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
        "details": {"reason": "3 días de retraso"},
        "run_id": "run-1",
        "image_version": "0.1.0",
    }
    return Finding(**(values | overrides))


@pytest.fixture
def lines(tmp_path):
    """Líneas JSON que `CloudRunFormatter` escribiría para lo que emita `dq`."""
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger = logging.getLogger("dq")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield lambda *findings: (
            emit_findings(list(findings), tmp_path),
            [json.loads(CloudRunFormatter().format(r)) for r in records],
        )[1]
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize(
    ("severity", "expected"),
    [("info", "INFO"), ("warning", "WARNING"), ("error", "ERROR")],
)
def test_finding_line_promotes_severity_and_flattens_fields(lines, severity, expected):
    finding = make(severity=severity)
    (entry,) = lines(finding)
    assert entry["severity"] == expected
    assert entry["finding_id"] == finding.finding_id
    assert entry["check_type"] == "source_delayed"
    assert entry["details"] == {"reason": "3 días de retraso"}
    # `message` conserva la línea JSON de siempre, con la severidad del hallazgo.
    assert json.loads(entry["message"])["severity"] == severity


def test_plain_record_has_no_finding_id():
    record = logging.LogRecord("l1", logging.ERROR, "", 0, "falló %s", ("x",), None)
    assert json.loads(CloudRunFormatter().format(record)) == {
        "severity": "ERROR",
        "message": "falló x",
    }


def test_exception_stays_in_one_line():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "l1", logging.ERROR, "", 0, "fin", (), sys.exc_info()
        )
    line = CloudRunFormatter().format(record)
    assert "\n" not in line
    assert "ValueError: boom" in json.loads(line)["message"]


@pytest.fixture
def clean_root(monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.WARNING)
    return root


def _stream_handler(root):
    # pytest agrega sus propios handlers a la raíz: se busca el de basicConfig.
    (handler,) = [h for h in root.handlers if type(h) is logging.StreamHandler]
    return handler


def test_configure_logging_json_inside_cloud_run(clean_root):
    clean_root.handlers.clear()  # pytest ya puso los suyos en la raíz
    configure_logging({"CLOUD_RUN_JOB": "l1-daily"})
    handler = _stream_handler(clean_root)
    assert isinstance(handler.formatter, CloudRunFormatter)
    assert handler.stream is sys.stdout


def test_configure_logging_text_outside_cloud_run(clean_root):
    clean_root.handlers.clear()  # pytest ya puso los suyos en la raíz
    configure_logging({})
    handler = _stream_handler(clean_root)
    assert not isinstance(handler.formatter, CloudRunFormatter)
