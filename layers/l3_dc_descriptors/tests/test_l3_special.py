"""Aritmética exacta y fail-closed de las fases (TRD-L3 §7.2, §13.3)."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import oracle
import pyarrow.parquet as pq
import pyref
import pytest
import rowgroup
from dc_frames import FramesInputError
from l3_dc_descriptors import (
    SUMMARIES_SCHEMA,
    PhaseQuantityMismatch,
    PhaseValueError,
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
    # La guarda es de la referencia de Python; en el motor `buy + sell` es la suma de
    # `quantity` por construcción (ADR-L3-16).
    lake = build_lake(tmp_path, fixture_ticks(), thetas=THETAS)
    real = rowgroup.Prefixes.__init__

    def skewed(self, frame):
        real(self, frame)
        self.buy += np.arange(len(self.buy))  # cada tick suma 10⁻⁸ de más a `buy`

    monkeypatch.setattr(rowgroup.Prefixes, "__init__", skewed)
    with pytest.raises(PhaseQuantityMismatch) as error:
        pyref.summarize_month(
            lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1
        )
    assert error.value.check_type == "phase_quantity_mismatch"
    details = error.value.details
    assert set(details) == {"theta", "confirm_agg_trade_id", "phase", "expected", "got"}
    assert details["phase"] == "confirmación"
    # buy + sell se pasa de la suma de quantity en 10⁻⁸ por cada tick de la fase.
    theta = Decimal(details["theta"])
    event = next(
        e
        for e in lake.events[int(theta.scaleb(8)), M1]
        if e["confirm_agg_trade_id"] == details["confirm_agg_trade_id"]
    )
    wanted = oracle.from_ticks(event | {"theta": theta}, lake.all_ticks())
    excess = Decimal(details["got"]) - Decimal(details["expected"])
    assert excess == wanted["c_ticks"] * Decimal("1e-8")
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


def negative_lake(tmp_path, index: int):
    ticks = fixture_ticks(seed=5)
    ticks[index] = replace(ticks[index], quantity=Decimal(-1))
    return build_lake(tmp_path, ticks, thetas=THETAS)


def test_a_negative_quantity_in_a_phase_aborts_with_both_summarizers(tmp_path):
    lake = negative_lake(tmp_path, len(fixture_ticks(seed=5)) // 2)
    for summarizer in (pyref.summarize_month, summarize_month):
        with pytest.raises(FramesInputError, match="negativos"):
            summarizer(
                lake.l1,
                lake.l2,
                tmp_path / summarizer.__module__,
                M1,
                lake.theta_texts,
                series_start=M1,
            )
    assert files(lake) == []


def test_a_negative_quantity_no_phase_touches_is_the_one_difference_with_the_reference(
    tmp_path,
):
    """Costo asumido (3) de ADR-L3-16: el motor valida solo los ticks que alguna fase toca.
    El primer tick del mes no es de ninguna fase (las fases son `(R, C]` y `(C, E]`): la
    referencia de Python rechazaba el row group entero, el motor lo acepta y la conciliación
    (§9.1) lo cuenta entre los ticks sin dueño, igual que sin el valor negativo.
    """
    lake = negative_lake(tmp_path, 0)
    with pytest.raises(FramesInputError, match="negativos"):
        pyref.summarize_month(
            lake.l1, lake.l2, tmp_path / "py", M1, lake.theta_texts, series_start=M1
        )
    got = summarize_month(
        lake.l1, lake.l2, tmp_path / "rs", M1, lake.theta_texts, series_start=M1
    )
    clean = build_lake(tmp_path / "clean", fixture_ticks(seed=5), thetas=THETAS)
    wanted = pyref.summarize_month(
        clean.l1, clean.l2, tmp_path / "ok", M1, clean.theta_texts, series_start=M1
    )
    for theta, result in got.items():
        assert result.reconciliation == wanted[theta].reconciliation
        assert result.reconciliation.unassigned > 0
        assert result.content_hash == wanted[theta].content_hash


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
