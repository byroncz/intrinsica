"""Cada excepción tipada del resumidor se traduce a su hallazgo con los `details` que trae."""

import pytest
from dq import Severity, Stage, Status
from l3_dc_descriptors import (
    L2InputMissing,
    L2StateInconsistent,
    PhaseQuantityMismatch,
    PhaseTicksMismatch,
    PhaseValueError,
    SkeletonMismatch,
    findings,
)
from l3_dc_descriptors.context import RunContext, Unit

CTX = RunContext(
    mode="monthly",
    run_id="run",
    image_version="0.4.0+test",
    series_start=(2017, 8),
    landing_root="l1",
    events_root="l2",
    summaries_root="l3",
    dq_root="dq",
)
UNIT = Unit(2017, 9)
THETA = "0.00100000"
MONTH = (2017, 9)

ERRORS = [
    L2InputMissing(THETA, MONTH, "events.parquet", "l2/x"),
    L2StateInconsistent.mismatch(THETA, MONTH, "confirm_time", 1, 2),
    L2StateInconsistent.no_pending(THETA, MONTH, 1, False),
    PhaseQuantityMismatch(THETA, 7, "confirmation", "1.00000000", "2.00000000"),
    PhaseTicksMismatch(THETA, MONTH, 10, 5, 1, 3, 1),
    SkeletonMismatch(THETA, MONTH, "summaries", "otro número de filas"),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
def test_each_error_becomes_its_finding(error):
    finding = findings.from_error(CTX, UNIT, error)
    assert finding.check_type == error.check_type
    assert finding.details == error.details
    assert (finding.severity, finding.status) == (Severity.ERROR, Status.FAIL)
    assert finding.stage == Stage.CANONICAL
    assert (finding.layer, finding.mode) == ("l3", "monthly")
    assert (finding.provider, finding.market, finding.asset) == (
        "binance",
        "spot",
        "BTCUSDT",
    )
    assert (finding.year, finding.month) == (2017, 9)
    assert (finding.run_id, finding.image_version) == ("run", "0.4.0+test")


def test_a_value_error_has_no_check_type_of_its_own():
    assert findings.from_error(CTX, UNIT, PhaseValueError(THETA, 7, "no cabe")) is None


def test_event_over_budget_is_a_warning_with_the_budget_and_the_family():
    spectrum = RunContext(
        mode="monthly",
        run_id="run",
        image_version="0.4.0+test",
        series_start=(2017, 8),
        landing_root="l1",
        events_root="l2",
        summaries_root="l3",
        dq_root="dq",
        family="spectrum",
    )
    finding = findings.event_over_budget(
        spectrum,
        UNIT,
        theta=THETA,
        confirm_agg_trade_id=7,
        ticks=20_000_001,
        budget=20_000_000,
    )
    assert finding.check_type == "event_over_budget"
    assert (finding.severity, finding.status) == (Severity.WARNING, Status.FAIL)
    assert finding.details == {
        "theta": THETA,
        "confirm_agg_trade_id": 7,
        "ticks": 20_000_001,
        "budget": 20_000_000,
        "family": "spectrum",
    }
    assert (finding.layer, finding.stage) == ("l3", Stage.CANONICAL)


def test_every_check_type_of_the_catalog_has_a_builder():
    # TRD-L3 §9.3 menos `event_over_budget`, que solo aplica a `spectrum` y tiene su prueba.
    built = [
        findings.l2_input_missing(CTX, UNIT, theta=THETA, file="events.parquet"),
        findings.input_missing(CTX, UNIT, "input_missing", {}),
        findings.input_missing(CTX, UNIT, "input_provisional_only", {}),
        findings.theta_catalog_invalid(CTX, UNIT, source="x", problems=[]),
        findings.theta_behind_frontier(
            CTX, UNIT, theta=THETA, frontier=None, reason="l3_frontier"
        ),
        findings.theta_config_drift(
            CTX, UNIT, theta=THETA, months=1, last_month="2017-08"
        ),
        findings.summaries_summary(
            CTX,
            UNIT,
            theta=THETA,
            rows=1,
            content_hash="a",
            l2_events_content_hash="b",
            l2_carry_over_content_hash="c",
            unassigned_ticks=0,
        ),
    ]
    types = {f.check_type for f in built} | {e.check_type for e in ERRORS}
    assert types == {
        "l2_input_missing",
        "l2_state_inconsistent",
        "phase_quantity_mismatch",
        "family_skeleton_mismatch",
        "phase_ticks_mismatch",
        "input_missing",
        "input_provisional_only",
        "theta_catalog_invalid",
        "theta_behind_frontier",
        "theta_config_drift",
        "summaries_summary",
    }
