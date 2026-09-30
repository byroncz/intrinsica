"""La salida de L2: `events.parquet`, `carry_over.parquet` y los hallazgos de DQ."""

import dataclasses
import json
from decimal import Decimal
from pathlib import Path

import dc_pyo3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l2_dc_events.carry import CarryOverError, Coordinates, read_carry_over, to_batch
from l2_dc_events.landing import LandingError, open_consolidated, read_batches, ticks_of
from l2_dc_events.pipeline import RunContext, Unit, process_unit
from l2_dc_events.schema import CARRY_OVER_SCHEMA, EVENTS_SCHEMA
from l2_dc_events.thetas import load_thetas
from l2_dc_events.write import (
    CARRY_OVER,
    EVENTS,
    format_theta,
    partition_path,
)
from pyutils import ContentHasher, PartitionWriter

THETAS = load_thetas()
# Un θ del config con eventos en el fixture de un día (≈ 1,6 %).
THETA = THETAS[40]
AUG = Unit(2017, 8)
SEP = Unit(2017, 9)


@pytest.fixture
def ctx(write_month, tmp_path):
    return RunContext(
        mode="backfill",
        run_id="test",
        image_version="0.1.0+test",
        series_start=(2017, 8),
        landing_root=write_month.landing,
        events_root=tmp_path / "events",
        dq_root=tmp_path / "dq",
    )


def path_of(ctx, unit, theta, filename=EVENTS):
    return partition_path(
        ctx.events_root,
        unit.provider,
        unit.market,
        unit.asset,
        theta,
        unit.year,
        unit.month,
        filename,
    )


def event_tuples(ctx, unit, theta):
    """Los eventos de la partición como `(direction, reference, confirm, extreme)`."""
    rows = pq.read_table(path_of(ctx, unit, theta)).to_pylist()

    def point(row, name):
        return (
            int(row[f"{name}_price"].scaleb(8)),
            row[f"{name}_time"],
            row[f"{name}_agg_trade_id"],
        )

    return [
        (
            r["direction"],
            point(r, "reference"),
            point(r, "confirm"),
            point(r, "extreme"),
        )
        for r in rows
    ]


def direct_events(path, thetas):
    """Los eventos cerrados de un mes de punta a punta, sin pasar por la capa."""
    fanout = dc_pyo3.FanOut(thetas)
    out = [[] for _ in thetas]
    with open_consolidated(str(path)) as parquet:
        for batch in read_batches(parquet):
            ticks = ticks_of(batch)
            closed = fanout.feed_batch(ticks.prices, ticks.times, ticks.agg_trade_ids)
            for acc, new in zip(out, closed, strict=True):
                acc.extend(new)
            del batch, ticks
    for acc, last in zip(out, fanout.finish(), strict=True):
        if last is not None:
            acc.append(last)
    return [
        [(e.direction, e.reference, e.confirm, e.extreme) for e in acc] for acc in out
    ]


def json_details(row):
    return json.loads(row["details"])


def findings_of(ctx):
    table = pq.read_table(ctx.dq_root, partitioning="hive")
    return table.to_pylist()


def split_index(ticks):
    """Un índice cerca de la mitad donde cambia `transact_time`."""
    i = len(ticks) // 2
    while ticks[i][1] == ticks[i - 1][1]:
        i += 1
    return i


def test_the_events_partition_follows_the_contract(fixture_ticks, write_month, ctx):
    path = write_month(fixture_ticks, row_group_size=1_000)
    result = process_unit(AUG, ctx)

    theta = THETA
    partition = Path(path_of(ctx, AUG, theta))
    assert partition.relative_to(ctx.events_root).parts == (
        "provider=binance",
        "market=spot",
        "asset=BTCUSDT",
        "theta=0.01596764",
        "year=2017",
        "month=08",
        "events.parquet",
    )
    assert pq.read_schema(partition).equals(EVENTS_SCHEMA)

    index = THETAS.index(theta)
    assert event_tuples(ctx, AUG, theta) == direct_events(path, THETAS)[index]
    assert result.events_per_theta[index] == pq.read_metadata(partition).num_rows

    column = pq.read_metadata(partition).row_group(0).column(0)
    assert column.compression == "ZSTD"
    assert column.is_stats_set
    sorting = pq.read_metadata(partition).row_group(0).sorting_columns
    assert [s.column_index for s in sorting] == [
        EVENTS_SCHEMA.get_field_index("confirm_time")
    ]


def test_events_carry_the_partition_theta_and_are_ordered_by_confirmation(
    fixture_ticks, write_month, ctx
):
    write_month(fixture_ticks, row_group_size=1_000)
    process_unit(AUG, ctx)
    for theta in (THETAS[0], THETAS[25], THETAS[-1]):
        table = pq.read_table(path_of(ctx, AUG, theta))
        assert set(table["theta"].to_pylist()) <= {Decimal(theta).scaleb(-8)}
        times = table["confirm_time"].to_pylist()
        assert times == sorted(times)
        for row in table.to_pylist():
            # Invariante del extremo (§7.2): nunca antes de la confirmación.
            assert row["extreme_agg_trade_id"] >= row["confirm_agg_trade_id"]
            assert row["extreme_time"] >= row["confirm_time"]


def test_rerunning_the_month_gives_the_same_content_hash(
    fixture_ticks, write_month, ctx
):
    write_month(fixture_ticks, row_group_size=1_000)
    first = process_unit(AUG, ctx)
    second = process_unit(AUG, dataclasses.replace(ctx, run_id="otra"))
    assert first.events_hashes == second.events_hashes
    assert first.carry_over_hashes == second.carry_over_hashes
    assert len(set(first.events_hashes)) > 1  # cada θ tiene su contenido
    # Ningún temporal huérfano: solo los 100 archivos publicados.
    files = [p.name for p in Path(ctx.events_root).rglob("*") if p.is_file()]
    assert sorted(set(files)) == [CARRY_OVER, EVENTS]
    assert len(files) == 2 * len(THETAS)


def test_the_hash_does_not_depend_on_the_row_group_size(
    fixture_ticks, write_month, ctx, tmp_path
):
    write_month(fixture_ticks, row_group_size=333)
    a = process_unit(AUG, ctx)
    write_month(fixture_ticks, row_group_size=4_735)
    b = process_unit(AUG, ctx)
    assert a.events_hashes == b.events_hashes
    assert a.carry_over_hashes == b.carry_over_hashes


def test_the_chain_of_two_months_equals_one_month(fixture_ticks, write_month, ctx):
    """El carry-over por Parquet no pierde nada en el borde: Ago + Sep = el mes entero."""
    whole = write_month(fixture_ticks, row_group_size=1_000, year=2017, month=7)
    expected = direct_events(whole, THETAS)

    cut = split_index(fixture_ticks)
    write_month(fixture_ticks[:cut], row_group_size=1_000, month=8)
    write_month(fixture_ticks[cut:], row_group_size=1_000, month=9)
    process_unit(AUG, ctx)
    process_unit(SEP, ctx)

    for index, theta in enumerate(THETAS):
        chained = event_tuples(ctx, AUG, theta) + event_tuples(ctx, SEP, theta)
        assert chained == expected[index], f"θ={theta}"


def test_the_pending_event_of_the_carry_over_is_written_in_the_next_month(
    fixture_ticks, write_month, ctx
):
    cut = split_index(fixture_ticks)
    write_month(fixture_ticks[:cut], row_group_size=1_000, month=8)
    write_month(fixture_ticks[cut:], row_group_size=1_000, month=9)
    process_unit(AUG, ctx)
    theta = THETA
    carry = read_carry_over(path_of(ctx, AUG, theta, CARRY_OVER), theta)
    assert carry.pending is not None
    reference, confirm = carry.pending

    process_unit(SEP, ctx)
    first = event_tuples(ctx, SEP, theta)[0]
    # ADR-L2-06: lo completa el mes que confirma el evento siguiente.
    assert (first[1], first[2]) == (reference, confirm)
    assert first[0] == carry.direction


def test_the_carry_over_row_matches_the_contract(fixture_ticks, write_month, ctx):
    write_month(fixture_ticks, row_group_size=1_000)
    process_unit(AUG, ctx)
    theta = THETA
    table = pq.read_table(path_of(ctx, AUG, theta, CARRY_OVER))
    assert table.schema.equals(CARRY_OVER_SCHEMA)
    (row,) = table.to_pylist()
    assert (row["provider"], row["market"], row["asset"]) == (
        "binance",
        "spot",
        "BTCUSDT",
    )
    assert (row["theta"], row["year"], row["month"]) == (
        Decimal(THETA).scaleb(-8),
        2017,
        8,
    )
    assert row["state_version"] == dc_pyo3.STATE_VERSION
    assert row["has_pending_event"] is True
    assert row["pending_confirm_agg_trade_id"] is not None


def test_a_carry_over_without_pending_event_writes_nulls(tmp_path):
    point = (100_000_000, 1_000, 1)
    carry = dc_pyo3.CarryOver(THETA, dc_pyo3.STATE_VERSION, 0, point, point)
    batch = to_batch(carry, Coordinates("binance", "spot", "BTCUSDT", 2017, 8))
    (row,) = batch.to_pylist()
    assert row["has_pending_event"] is False
    assert [
        v
        for k, v in row.items()
        if k.startswith("pending_") and k != "has_pending_event"
    ] == [None] * 6

    path = str(tmp_path / "carry_over.parquet")
    pq.write_table(pa.Table.from_batches([batch]), path)
    assert read_carry_over(path, THETA) == carry


def test_an_empty_theta_still_publishes_a_valid_partition(write_month, ctx):
    # Una serie monótona no revierte nunca: ningún θ cierra eventos.
    ticks = [(100_000_000 * (1 + i), 1_000 + i, 1 + i) for i in range(100)]
    write_month(ticks, row_group_size=50)
    result = process_unit(AUG, ctx)
    assert result.events_per_theta == [0] * len(THETAS)
    table = pq.read_table(path_of(ctx, AUG, THETAS[0]))
    assert table.num_rows == 0
    assert table.schema.equals(EVENTS_SCHEMA)


def test_a_failure_mid_unit_leaves_nothing_behind(fixture_ticks, write_month, ctx):
    write_month(fixture_ticks, row_group_size=1_000)

    class Exploding:
        thetas = THETAS[:3]

        def feed_batch_columns(self, prices, times, ids):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        process_unit(AUG, ctx, fanout=Exploding())
    assert not [p for p in Path(ctx.events_root).rglob("*") if p.is_file()]


def test_the_partition_writer_keeps_the_previous_file_if_it_aborts(tmp_path):
    path = str(tmp_path / "x" / "events.parquet")
    with PartitionWriter(path, EVENTS_SCHEMA) as writer:
        writer.write_batch(pa.RecordBatch.from_pylist([], schema=EVENTS_SCHEMA))
        writer.commit()
    before = Path(path).read_bytes()
    with pytest.raises(RuntimeError), PartitionWriter(path, EVENTS_SCHEMA):
        raise RuntimeError("aborta")
    assert Path(path).read_bytes() == before
    assert [p.name for p in Path(path).parent.iterdir()] == ["events.parquet"]


def test_content_hash_ignores_the_chunking_and_sees_the_values():
    def hash_of(batches, schema=EVENTS_SCHEMA):
        hasher = ContentHasher(schema)
        for batch in batches:
            hasher.update(batch)
        return hasher.hexdigest()

    row = {
        "reference_price": Decimal(1),
        "reference_time": 1,
        "reference_agg_trade_id": 1,
        "confirm_price": Decimal(2),
        "confirm_time": 2,
        "confirm_agg_trade_id": 2,
        "extreme_price": Decimal(3),
        "extreme_time": 3,
        "extreme_agg_trade_id": 3,
        "direction": 1,
        "theta": Decimal("0.02"),
    }
    other = {**row, "extreme_price": Decimal(4)}
    rows = [row, other, row]

    def batch(items):
        return pa.RecordBatch.from_pylist(items, schema=EVENTS_SCHEMA)

    whole = hash_of([batch(rows)])
    assert whole == hash_of([batch(rows[:1]), batch(rows[1:])])
    assert whole == hash_of([batch(rows[:2]), batch(rows[2:])])
    assert whole != hash_of([batch(rows[:2])])
    assert whole != hash_of([batch([row, row, row])])


@pytest.mark.parametrize(
    ("theta", "text"),
    [(10_000, "0.00010000"), (5_000_000, "0.05000000"), (1, "0.00000001")],
)
def test_format_theta_is_fixed_width(theta, text):
    assert format_theta(theta) == text


@pytest.mark.parametrize("theta", [0, -1, 100_000_000])
def test_format_theta_rejects_out_of_range(theta):
    with pytest.raises(ValueError):
        format_theta(theta)


# --- Hallazgos de DQ ------------------------------------------------------


def test_every_theta_gets_a_summary_finding(fixture_ticks, write_month, ctx):
    write_month(fixture_ticks, row_group_size=1_000)
    result = process_unit(AUG, ctx)
    rows = [r for r in findings_of(ctx) if r["check_type"] == "events_summary"]
    assert len(rows) == len(THETAS)
    by_theta = {json_details(r)["theta"]: r for r in rows}
    for index, theta in enumerate(THETAS):
        row = by_theta[theta]
        assert (row["layer"], row["mode"], row["stage"]) == (
            "l2",
            "backfill",
            "canonical",
        )
        assert (row["severity"], row["status"]) == ("info", "pass")
        assert (row["year"], row["month"]) == (2017, 8)
        assert row["metric_value"] == result.events_per_theta[index]
        details = json_details(row)
        assert details["events_content_hash"] == result.events_hashes[index]
        assert details["carry_over_content_hash"] == result.carry_over_hashes[index]
    # La guarda de §9.1 no dispara con el instante atómico.
    assert not [
        r for r in findings_of(ctx) if r["check_type"] == "dc_zero_tick_discarded"
    ]


def test_only_provisionals_emits_input_provisional_only(write_month, ctx):
    ticks = [(100_000_000, 1_000, 1), (200_000_000, 1_001, 2)]
    write_month(ticks, row_group_size=2, name="provisional-day=05.parquet")
    with pytest.raises(LandingError) as error:
        process_unit(AUG, ctx)
    assert error.value.check_type == "input_provisional_only"
    (row,) = findings_of(ctx)
    assert (row["check_type"], row["severity"], row["status"]) == (
        "input_provisional_only",
        "error",
        "fail",
    )
    assert row["layer"] == "l2"
    assert json_details(row)["provisionals"] == 1


def test_a_missing_month_emits_input_missing(ctx):
    with pytest.raises(LandingError):
        process_unit(AUG, ctx)
    (row,) = findings_of(ctx)
    assert row["check_type"] == "input_missing"
    assert "expected_path" in json_details(row)


def test_a_missing_carry_over_aborts_the_unit_and_emits_one_finding_per_theta(
    fixture_ticks, write_month, ctx
):
    write_month(fixture_ticks, row_group_size=1_000, month=9)
    with pytest.raises(CarryOverError) as error:
        process_unit(SEP, ctx)
    assert error.value.check_type == "carry_over_missing"
    rows = findings_of(ctx)
    assert len(rows) == len(THETAS)
    assert {r["check_type"] for r in rows} == {"carry_over_missing"}
    assert {(r["severity"], r["status"]) for r in rows} == {("error", "fail")}
    assert {json_details(r)["theta"] for r in rows} == set(THETAS)
    assert not list(Path(ctx.events_root).rglob("*.parquet"))


def test_an_incompatible_state_version_aborts_the_unit(fixture_ticks, write_month, ctx):
    write_month(fixture_ticks[:2_000], row_group_size=1_000, month=8)
    write_month(fixture_ticks[2_000:], row_group_size=1_000, month=9)
    process_unit(AUG, ctx)

    target = path_of(ctx, AUG, THETAS[7], CARRY_OVER)
    table = pq.read_table(target)
    index = table.schema.get_field_index("state_version")
    table = table.set_column(index, "state_version", pa.array(["9.9.9"]))
    pq.write_table(table, target)

    before = sorted(str(p) for p in Path(ctx.events_root).rglob("*.parquet"))
    with pytest.raises(CarryOverError) as error:
        process_unit(SEP, ctx)
    assert error.value.check_type == "carry_over_version_mismatch"
    mismatch = [
        r for r in findings_of(ctx) if r["check_type"] == "carry_over_version_mismatch"
    ]
    (row,) = mismatch
    assert json_details(row) | {"path": None} == {
        "theta": THETAS[7],
        "found": "9.9.9",
        "expected": dc_pyo3.STATE_VERSION,
        "path": None,
    }
    assert sorted(str(p) for p in Path(ctx.events_root).rglob("*.parquet")) == before


def _carry_table(tmp_path, rows: int) -> pa.Table:
    point = (100_000_000, 1_000, 1)
    carry = dc_pyo3.CarryOver(THETA, dc_pyo3.STATE_VERSION, 0, point, point)
    batch = to_batch(carry, Coordinates("binance", "spot", "BTCUSDT", 2017, 8))
    return pa.Table.from_batches([batch] * rows, schema=CARRY_OVER_SCHEMA)


@pytest.mark.parametrize("kind", ["empty_file", "truncated", "no_rows", "two_rows"])
def test_an_unreadable_carry_over_is_a_carry_over_error(tmp_path, kind):
    path = tmp_path / "carry_over.parquet"
    if kind == "empty_file":
        path.write_bytes(b"")
    elif kind == "truncated":
        pq.write_table(_carry_table(tmp_path, 1), path)
        path.write_bytes(path.read_bytes()[:-20])
    else:
        pq.write_table(_carry_table(tmp_path, 0 if kind == "no_rows" else 2), path)
    with pytest.raises(CarryOverError) as error:
        read_carry_over(str(path), THETA)
    assert error.value.check_type == "carry_over_version_mismatch"
    assert error.value.details["path"] == str(path)
    assert error.value.details["reason"]


def test_a_corrupt_carry_over_emits_a_finding_and_aborts_the_unit(
    fixture_ticks, write_month, ctx
):
    write_month(fixture_ticks[:2_000], row_group_size=1_000, month=8)
    write_month(fixture_ticks[2_000:], row_group_size=1_000, month=9)
    process_unit(AUG, ctx)
    Path(path_of(ctx, AUG, THETAS[7], CARRY_OVER)).write_bytes(b"")

    with pytest.raises(CarryOverError):
        process_unit(SEP, ctx)
    (row,) = [
        r for r in findings_of(ctx) if r["check_type"] == "carry_over_version_mismatch"
    ]
    assert json_details(row)["theta"] == THETAS[7]


def test_a_carry_over_of_another_theta_is_config_drift(fixture_ticks, write_month, ctx):
    write_month(fixture_ticks[:2_000], row_group_size=1_000, month=8)
    process_unit(AUG, ctx)
    # El carry-over de θ₃ copiado a la partición de θ₄: la columna no coincide.
    wrong = Path(path_of(ctx, AUG, THETAS[4], CARRY_OVER))
    wrong.write_bytes(Path(path_of(ctx, AUG, THETAS[3], CARRY_OVER)).read_bytes())
    with pytest.raises(CarryOverError) as error:
        read_carry_over(str(wrong), THETAS[4])
    assert error.value.check_type == "theta_config_drift"
    write_month(fixture_ticks[2_000:], row_group_size=1_000, month=9)
    with pytest.raises(CarryOverError):
        process_unit(SEP, ctx)
    (row,) = [r for r in findings_of(ctx) if r["check_type"] == "theta_config_drift"]
    assert (row["severity"], row["status"]) == ("warning", "fail")


def test_the_first_month_of_the_series_needs_no_carry_over(
    fixture_ticks, write_month, ctx
):
    write_month(fixture_ticks, row_group_size=1_000)
    process_unit(AUG, ctx)
    # Re-ejecutarlo sigue siendo el primero: arranca en frío otra vez.
    process_unit(AUG, ctx)


def test_content_hash_is_the_one_l2_has_always_published():
    # Fijado en ITSC-245, al extraer `ContentHasher` a `shared/pyutils`: el
    # hash de un mismo contenido no puede cambiar con la extracción.
    def batch(schema, row):
        return pa.RecordBatch.from_pylist(row, schema=schema)

    def value(field):
        kind = field.type
        if pa.types.is_decimal(kind):
            return Decimal(1)
        if pa.types.is_string(kind):
            return "x"
        return True if pa.types.is_boolean(kind) else 1

    event = {field.name: value(field) for field in EVENTS_SCHEMA}
    carry = {field.name: value(field) for field in CARRY_OVER_SCHEMA}
    carry["pending_reference_time"] = None
    expected = {
        "events": "b5de26248f5c6e5d20c71e9f5abaee016c05b8536d928229e8334772ed25e13a",
        "carry": "cc105c416b95de4d2cf40ab3771dc6dcbc0eb61f5c58f6e223431e8d192bd90f",
    }
    events = ContentHasher(EVENTS_SCHEMA)
    events.update(batch(EVENTS_SCHEMA, [event, event]))
    carry_over = ContentHasher(CARRY_OVER_SCHEMA)
    carry_over.update(batch(CARRY_OVER_SCHEMA, [carry]))
    assert events.hexdigest() == expected["events"]
    assert carry_over.hexdigest() == expected["carry"]
