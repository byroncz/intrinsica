"""Aritmética exacta y fail-closed de las fases (TRD-L3 §7.2, §13.3)."""

from dataclasses import replace
from decimal import Decimal

import oracle
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors import (
    SUMMARIES_SCHEMA,
    PhaseQuantityMismatch,
    PhaseValueError,
    phase,
    summarize_month,
)
from l3_dc_descriptors.phase import divide_half_even
from l3_lake import M1, build_lake, fixture_ticks

THETAS = (250_000, 1_000_000)


def run(lake):
    return summarize_month(
        lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1
    )


def files(lake):
    return [p for p in lake.l3.rglob("*") if p.is_file()]


def test_notional_that_overflows_int64_is_exact(tmp_path):
    # Precios de ~10⁵ USD y cantidades de hasta 5 000 BTC: cada tick aporta ~10²⁴ a
    # Σ price·quantity (escala 10⁻¹⁶), contra el tope de INT64 de ~9,2·10¹⁸.
    ticks = fixture_ticks(seed=5, price_factor=25, quantity=5_000)
    assert max(t.price * t.quantity for t in ticks) * 10**16 > 2**63
    lake = build_lake(tmp_path, ticks, thetas=THETAS)
    results = run(lake)
    all_ticks = lake.all_ticks()
    for theta, result in results.items():
        table = pq.read_table(result.path, schema=SUMMARIES_SCHEMA).to_pylist()
        events = lake.events[int(theta.scaleb(8)), M1]
        assert len(table) == len(events) > 0
        for row, event in zip(table, events, strict=True):
            assert row == oracle.from_ticks(event | {"theta": theta}, all_ticks)
            assert row["c_price_min"] <= row["c_vwap"] <= row["c_price_max"]


def test_forced_quantity_difference_aborts(tmp_path, monkeypatch):
    lake = build_lake(tmp_path, fixture_ticks(), thetas=THETAS)
    real = phase._side_sum
    monkeypatch.setattr(phase, "_side_sum", lambda q, mask: real(q, mask) + 1)
    with pytest.raises(PhaseQuantityMismatch) as error:
        run(lake)
    assert error.value.check_type == "phase_quantity_mismatch"
    details = error.value.details
    assert set(details) == {"theta", "confirm_agg_trade_id", "phase", "expected", "got"}
    assert details["phase"] == "confirmación"
    # buy + sell se pasa de la suma de quantity en 2 unidades de 10⁻⁸ (una por lado).
    assert Decimal(details["got"]) - Decimal(details["expected"]) == Decimal("2e-8")
    assert files(lake) == []


def test_quantity_that_does_not_fit_decimal_18_8_aborts(tmp_path):
    ticks = [
        replace(t, quantity=Decimal(9_000_000_000), sold=False)
        for t in fixture_ticks(seed=2)
    ]
    lake = build_lake(tmp_path, ticks, thetas=THETAS)
    with pytest.raises(PhaseValueError, match="no cabe en DECIMAL"):
        run(lake)
    assert files(lake) == []


@pytest.mark.parametrize(
    ("numerator", "denominator", "wanted"),
    [
        (5, 2, 2),  # 2,5 → 2: el par
        (7, 2, 4),  # 3,5 → 4: el par
        (-5, 2, -2),
        (1, 3, 0),
        (2, 3, 1),
        (10, 5, 2),
        (123_456_789_050, 100, 1_234_567_890),  # …890,5 → …890
        (123_456_789_150, 100, 1_234_567_892),  # …891,5 → …892
    ],
)
def test_vwap_rounds_half_even(numerator, denominator, wanted):
    assert divide_half_even(numerator, denominator) == wanted
