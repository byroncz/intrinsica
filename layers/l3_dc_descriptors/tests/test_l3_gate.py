"""Una entrada de L2 faltante o incoherente aborta sin escribir (TRD-L3 §9.2, §13.6)."""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from dc_frames.lake import l1_path
from frames_lake import write_carry_over
from l3_dc_descriptors import (
    L2InputMissing,
    L2StateInconsistent,
    PhaseTicksMismatch,
    summarize_month,
)
from l3_dc_descriptors.summarize import DEFAULT_KEY
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts

SMALL, BIG = 100_000, 2_000_000
PARTITION = "provider=binance/market=spot/asset=BTCUSDT"


@pytest.fixture
def lake(tmp_path):
    ticks = fixture_ticks(seed=3, drop=0.1)
    return build_lake(tmp_path, ticks, month_cuts(ticks, 2400))


def theta_text(theta: int) -> str:
    return f"{Decimal(theta).scaleb(-8):.8f}"


def l2_file(lake, theta: int, month, name: str):
    return (
        lake.l2
        / f"{PARTITION}/theta={theta_text(theta)}/year={month[0]}/month={month[1]:02d}"
        / name
    )


def attempt(lake, month=M2, thetas=(SMALL, BIG), **options):
    """Corre la unidad y devuelve el error; no debe haber dejado ningún archivo."""
    with pytest.raises(Exception) as error:
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            [theta_text(t) for t in thetas],
            series_start=M1,
            **options,
        )
    assert [p for p in lake.l3.rglob("*") if p.is_file()] == []
    return error.value


def rewrite(path, change):
    """Reescribe un Parquet con la tabla que devuelve `change(tabla)`, mismo row group."""
    size = pq.ParquetFile(path).metadata.row_group(0).num_rows
    pq.write_table(change(pq.read_table(path)), path, row_group_size=size)


def test_missing_events_names_the_file(lake):
    l2_file(lake, BIG, M2, "events.parquet").unlink()
    error = attempt(lake)
    assert isinstance(error, L2InputMissing)
    assert error.check_type == "l2_input_missing"
    assert error.details == {
        "theta": theta_text(BIG),
        "month": "2017-09",
        "file": "events.parquet",
    }


def test_missing_carry_over(lake):
    l2_file(lake, SMALL, M2, "carry_over.parquet").unlink()
    error = attempt(lake)
    assert isinstance(error, L2InputMissing)
    assert error.details["file"] == "carry_over.parquet"


@pytest.mark.parametrize("threads", [1, 4])
def test_the_gate_error_comes_before_the_l1_error(lake, threads):
    # Con L1 ilegible y un carry-over ausente sale el `l2_input_missing` de la compuerta,
    # no el error de L1 (la compuerta se espera antes que el pie de L1).
    Path(l1_path(lake.l1, DEFAULT_KEY, M2)).write_bytes(b"no es parquet")
    l2_file(lake, SMALL, M2, "carry_over.parquet").unlink()
    error = attempt(lake, threads=threads)
    assert isinstance(error, L2InputMissing)
    assert error.details["file"] == "carry_over.parquet"


def test_missing_carry_over_of_the_previous_month(lake):
    # Fuera del primer mes de la serie, la dirección del mes anterior es entrada.
    l2_file(lake, SMALL, M1, "carry_over.parquet").unlink()
    error = attempt(lake)
    assert isinstance(error, L2InputMissing)
    assert error.details["month"] == "2017-08"


def test_month_before_the_series_start(lake):
    with pytest.raises(ValueError, match="anterior al inicio"):
        summarize_month(lake.l1, lake.l2, lake.l3, M1, ["0.00100000"], series_start=M2)


def test_direction_without_pending_event(lake):
    write_carry_over(lake.l2, SMALL, M2, 1, has_pending=False)
    error = attempt(lake)
    assert isinstance(error, L2StateInconsistent)
    assert error.check_type == "l2_state_inconsistent"
    assert error.details == {
        "theta": theta_text(SMALL),
        "month": "2017-09",
        "direction": 1,
        "has_pending_event": False,
    }


def test_boundary_id_that_is_not_a_tick(lake):
    row = lake.events[SMALL, M2][len(lake.events[SMALL, M2]) // 2]
    missing = row["confirm_agg_trade_id"]
    path = lake.l1 / f"{PARTITION}/year=2017/month=09/consolidated.parquet"
    rewrite(path, lambda t: t.filter(pc.not_equal(t["agg_trade_id"], missing)))
    error = attempt(lake)
    assert isinstance(error, L2StateInconsistent)
    assert error.details["field"] == "confirm_agg_trade_id"
    assert error.details["got"] == str(missing)
    assert error.details["wanted"] == "un tick de L1"
    assert error.details["month"] == "2017-09"
    assert error.details["theta"] in {theta_text(SMALL), theta_text(BIG)}


def test_confirm_time_that_is_not_the_tick_time(lake):
    path = l2_file(lake, SMALL, M2, "events.parquet")
    rewrite(
        path,
        lambda t: t.set_column(
            t.schema.get_field_index("confirm_time"),
            "confirm_time",
            pc.add(t["confirm_time"], 1),
        ),
    )
    error = attempt(lake, thetas=(SMALL,))
    assert isinstance(error, L2StateInconsistent)
    assert error.details["field"] == "confirm_time"
    assert error.details["theta"] == theta_text(SMALL)


def test_l1_that_ends_before_the_extreme(lake):
    last = lake.events[BIG, M2][-1]["extreme_agg_trade_id"]
    path = lake.l1 / f"{PARTITION}/year=2017/month=09/consolidated.parquet"
    rewrite(path, lambda t: t.filter(pc.less(t["agg_trade_id"], last)))
    error = attempt(lake, thetas=(BIG,))
    assert isinstance(error, L2StateInconsistent)
    assert error.details["field"] == "extreme_agg_trade_id"
    assert error.details["theta"] == theta_text(BIG)
    assert "L1 terminó" in error.details["message"]


def test_row_that_contradicts_its_ticks(lake):
    rows = lake.events[SMALL, M1]
    index = next(
        i
        for i, r in enumerate(rows)
        if r["extreme_agg_trade_id"] > r["confirm_agg_trade_id"]
    )
    up = rows[index]["direction"] == 1

    def raise_extreme(table):
        prices = table["extreme_price"].to_pylist()
        prices[index] += Decimal("0.01") if up else Decimal("-0.01")
        return table.set_column(
            table.schema.get_field_index("extreme_price"),
            "extreme_price",
            pa.array(prices, table.schema.field("extreme_price").type),
        )

    rewrite(l2_file(lake, SMALL, M1, "events.parquet"), raise_extreme)
    with pytest.raises(L2StateInconsistent) as error:
        summarize_month(
            lake.l1, lake.l2, lake.l3, M1, [theta_text(SMALL)], series_start=M1
        )
    assert error.value.details["field"] == ("os_price_max" if up else "os_price_min")
    assert error.value.details["theta"] == theta_text(SMALL)
    assert [p for p in lake.l3.rglob("*") if p.is_file()] == []


def test_ticks_that_do_not_add_up(lake):
    # Sin una fila, los ticks de ese evento no son de ningún evento cerrado ni de la cola.
    path = l2_file(lake, SMALL, M1, "events.parquet")
    size = len(lake.events[SMALL, M1]) // 2
    rewrite(path, lambda t: pa.concat_tables([t.slice(0, size), t.slice(size + 1)]))
    with pytest.raises(PhaseTicksMismatch) as error:
        summarize_month(
            lake.l1, lake.l2, lake.l3, M1, [theta_text(SMALL)], series_start=M1
        )
    details = error.value.details
    assert error.value.check_type == "phase_ticks_mismatch"
    assert set(details) == {
        "theta",
        "month",
        "N_M",
        "sigma_closed",
        "P_M",
        "tail_M",
        "U_M",
    }
    assert (
        details["N_M"]
        != details["sigma_closed"] - details["P_M"] + details["tail_M"] + details["U_M"]
    )
    assert [p for p in lake.l3.rglob("*") if p.is_file()] == []


def test_a_failing_theta_leaves_the_others_unpublished(lake):
    l2_file(lake, BIG, M2, "carry_over.parquet").unlink()
    assert isinstance(attempt(lake, thetas=(SMALL, BIG)), L2InputMissing)
