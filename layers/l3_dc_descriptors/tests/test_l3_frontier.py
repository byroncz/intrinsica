"""La frontera por θ y familia (TRD-L3 §8.2): serie contigua desde el inicio y archivo válido."""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors import frontier, summarize_month
from l3_dc_descriptors.frontier import (
    frontiers,
    invalid_reason,
    l2_missing,
    list_l2,
    list_summaries,
    theta_text,
)
from l3_lake import KEY, M1, M2, M3, build_lake, fixture_ticks, month_cuts

TICKS = fixture_ticks()
SMALL, BIG = 100_000, 2_000_000
START = (2017, 8)
PARTITION = "provider=binance/market=spot/asset=BTCUSDT"


@pytest.fixture
def lake(tmp_path):
    lake = build_lake(tmp_path, TICKS, month_cuts(TICKS, 1500, 3000))
    for month in lake.months:
        summarize_month(
            lake.l1, lake.l2, lake.l3, month, lake.theta_texts, series_start=M1
        )
    return lake


def ordinal(month) -> int:
    return month[0] * 12 + month[1] - 1


def file_of(root: Path, theta: int, month, name: str) -> Path:
    return (
        root
        / PARTITION
        / f"theta={theta_text(theta)}/year={month[0]}/month={month[1]:02d}/{name}"
    )


def frontier_of(lake, theta: int = SMALL) -> int | None:
    listed = list_summaries(str(lake.l3), KEY)
    result = frontiers([theta], listed, START, str(lake.l3), str(lake.l2), KEY)
    return result[theta]


def test_a_complete_series_reaches_its_last_month(lake):
    assert frontier_of(lake) == ordinal(M3)


def test_a_theta_without_files_has_no_frontier(lake):
    assert frontiers([123_456], {}, START, str(lake.l3), str(lake.l2), KEY) == {
        123_456: None
    }


def test_a_hole_ends_the_series_before_it(lake):
    file_of(lake.l3, SMALL, M2, "summaries.parquet").unlink()
    assert frontier_of(lake) == ordinal(M1)
    assert frontier_of(lake, BIG) == ordinal(M3)


def test_a_series_that_does_not_start_at_the_series_start_has_no_frontier(lake):
    file_of(lake.l3, SMALL, M1, "summaries.parquet").unlink()
    assert frontier_of(lake) is None


def rewrite(path: Path, change) -> None:
    size = pq.ParquetFile(path).metadata.row_group(0).num_rows
    pq.write_table(change(pq.read_table(path)), path, row_group_size=size)


def with_state_version(table: pa.Table, version: str) -> pa.Table:
    return table.replace_schema_metadata({"state_version": version})


def other_theta(table: pa.Table) -> pa.Table:
    theta = pa.array([Decimal("0.00300000")] * table.num_rows, pa.decimal128(9, 8))
    return table.set_column(0, table.schema.field(0), theta)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (lambda t: with_state_version(t, "0.9.0"), "state_version"),
        (lambda t: t.drop_columns(["os_vwap"]), "esquema"),
        (other_theta, "no es del θ"),
    ],
    ids=["state_version", "schema", "theta"],
)
def test_an_invalid_file_at_the_frontier_moves_it_back(lake, damage, reason):
    path = file_of(lake.l3, SMALL, M3, "summaries.parquet")
    rewrite(path, damage)
    problem = invalid_reason(str(lake.l3), str(lake.l2), KEY, SMALL, M3)
    assert problem is not None and reason in problem
    assert frontier_of(lake) == ordinal(M2)


def test_other_row_groups_than_l2_are_invalid(lake):
    path = file_of(lake.l3, SMALL, M3, "summaries.parquet")
    table = pq.read_table(path)
    pq.write_table(table, path, row_group_size=table.num_rows)
    problem = invalid_reason(str(lake.l3), str(lake.l2), KEY, SMALL, M3)
    assert problem is not None and "row group" in problem
    assert frontier_of(lake) == ordinal(M2)


def test_an_unreadable_file_is_invalid(lake):
    file_of(lake.l3, SMALL, M3, "summaries.parquet").write_bytes(b"no soy parquet")
    problem = invalid_reason(str(lake.l3), str(lake.l2), KEY, SMALL, M3)
    assert problem is not None and "no se pudo leer" in problem
    assert frontier_of(lake) == ordinal(M2)


def test_the_frontier_never_passes_the_one_of_l2(lake):
    file_of(lake.l2, SMALL, M3, "events.parquet").unlink()
    assert frontier_of(lake) == ordinal(M2)
    assert frontier_of(lake, BIG) == ordinal(M3)


def test_every_invalid_month_leaves_no_frontier(lake):
    for month in lake.months:
        file_of(lake.l3, SMALL, month, "summaries.parquet").write_bytes(b"x")
    assert frontier_of(lake) is None


def test_only_the_frontier_file_of_each_theta_is_opened(lake, monkeypatch):
    calls = []
    real = frontier.invalid_reason

    def counting(*args):
        calls.append(args[3:])
        return real(*args)

    monkeypatch.setattr(frontier, "invalid_reason", counting)
    listed = list_summaries(str(lake.l3), KEY)
    result = frontiers(lake.thetas, listed, START, str(lake.l3), str(lake.l2), KEY)
    assert set(result.values()) == {ordinal(M3)}
    assert sorted(calls) == sorted((t, M3) for t in lake.thetas)


def test_the_listing_covers_every_theta_of_the_lake_not_only_the_catalog(lake):
    listed = list_summaries(str(lake.l3), KEY)
    assert set(listed) == set(lake.thetas)
    assert all(
        set(months) == {ordinal(m) for m in lake.months} for months in listed.values()
    )
    assert list_summaries(str(lake.l3) + "/no-existe", KEY) == {}


def test_l2_availability_comes_from_one_listing(lake):
    listed = list_l2(str(lake.l2), KEY)
    assert l2_missing(listed, SMALL, ordinal(M3)) == []
    file_of(lake.l2, SMALL, M3, "carry_over.parquet").unlink()
    file_of(lake.l2, BIG, M3, "events.parquet").unlink()
    listed = list_l2(str(lake.l2), KEY)
    assert l2_missing(listed, SMALL, ordinal(M3)) == ["carry_over.parquet"]
    assert l2_missing(listed, BIG, ordinal(M3)) == ["events.parquet"]
    assert l2_missing(listed, 7, ordinal(M3)) == [
        "events.parquet",
        "carry_over.parquet",
    ]
