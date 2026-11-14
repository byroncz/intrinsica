"""La frontera por θ y familia (TRD-L3 §8.2): serie contigua desde el inicio y archivo válido."""

from decimal import Decimal
from pathlib import Path

import dc_frames.lake as lake_module
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from cli_support import emitted, env_for, write_catalog
from l3_dc_descriptors import frontier, summarize_month
from l3_dc_descriptors.cli import main
from l3_dc_descriptors.frontier import (
    FooterReader,
    frontiers,
    invalid_reason,
    l2_missing,
    list_l2,
    list_summaries,
    scan_frontiers,
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
    meta = pq.ParquetFile(path).metadata
    # Un mes sin eventos del θ no tiene row groups: se reescribe sin tamaño.
    size = meta.row_group(0).num_rows if meta.num_row_groups else None
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


def test_every_month_of_the_series_is_verified_not_only_the_last_valid_one(
    lake, monkeypatch
):
    """Verificar todos los meses lee más pies (ITSC-419), nunca datos: cada (θ, mes) una vez."""
    calls = []
    real = frontier.invalid_reason

    def counting(*args):
        calls.append(args[3:5])
        return real(*args)

    monkeypatch.setattr(frontier, "invalid_reason", counting)
    listed = list_summaries(str(lake.l3), KEY)
    result = frontiers(lake.thetas, listed, START, str(lake.l3), str(lake.l2), KEY)
    assert set(result.values()) == {ordinal(M3)}
    assert sorted(calls) == sorted((t, m) for t in lake.thetas for m in lake.months)


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


# La familia T (`path.parquet`): su frontera sale de su propio archivo y nunca supera la de S.


@pytest.fixture
def path_lake(lake):
    for month in lake.months:
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            family="path",
        )
    return lake


def path_frontier_of(lake, theta: int = SMALL) -> int | None:
    summaries = list_summaries(str(lake.l3), KEY)
    listed = list_summaries(str(lake.l3), KEY, "path")
    result = frontiers(
        [theta],
        listed,
        START,
        str(lake.l3),
        str(lake.l2),
        KEY,
        "path",
        summaries,
    )
    return result[theta]


def test_the_listing_of_a_family_looks_for_its_own_file(path_lake):
    path = list_summaries(str(path_lake.l3), KEY, "path")
    assert set(path) == set(path_lake.thetas)
    file_of(path_lake.l3, SMALL, M2, "path.parquet").unlink()
    assert ordinal(M2) not in list_summaries(str(path_lake.l3), KEY, "path")[SMALL]
    assert ordinal(M2) in list_summaries(str(path_lake.l3), KEY)[SMALL]


def test_a_complete_path_series_reaches_its_last_month(path_lake):
    assert path_frontier_of(path_lake) == ordinal(M3)


def test_the_path_frontier_does_not_depend_on_having_the_path_file_for_s(lake):
    assert frontier_of(lake) == ordinal(M3)
    assert path_frontier_of(lake) is None


def test_a_hole_in_the_path_series_ends_it_before_the_hole(path_lake):
    file_of(path_lake.l3, SMALL, M2, "path.parquet").unlink()
    assert path_frontier_of(path_lake) == ordinal(M1)
    assert frontier_of(path_lake) == ordinal(M3)


def test_the_path_frontier_never_passes_the_one_of_summaries(path_lake):
    file_of(path_lake.l3, SMALL, M3, "summaries.parquet").unlink()
    assert path_frontier_of(path_lake) == ordinal(M2)
    assert path_frontier_of(path_lake, BIG) == ordinal(M3)
    # Un summaries.parquet inválido también baja el techo.
    file_of(path_lake.l3, BIG, M3, "summaries.parquet").write_bytes(b"x")
    assert path_frontier_of(path_lake, BIG) == ordinal(M2)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (lambda t: with_state_version(t, "0.9.0"), "state_version"),
        (lambda t: t.drop_columns(["os_tortuosity"]), "esquema"),
        (lambda t: t.drop_columns(["c_t75"]), "esquema"),
    ],
    ids=["state_version", "schema", "schema_other_column"],
)
def test_an_invalid_path_file_moves_the_path_frontier_back(path_lake, damage, reason):
    path = file_of(path_lake.l3, SMALL, M3, "path.parquet")
    rewrite(path, damage)
    problem = invalid_reason(
        str(path_lake.l3), str(path_lake.l2), KEY, SMALL, M3, "path"
    )
    assert problem is not None and reason in problem
    assert path_frontier_of(path_lake) == ordinal(M2)
    # La frontera de S no se entera.
    assert frontier_of(path_lake) == ordinal(M3)


def test_a_path_file_is_not_checked_against_the_summaries_contract(path_lake):
    # El mismo archivo es válido como `path` y no como `summaries`.
    assert (
        invalid_reason(str(path_lake.l3), str(path_lake.l2), KEY, SMALL, M3, "path")
        is None
    )
    summaries = file_of(path_lake.l3, SMALL, M3, "summaries.parquet")
    path = file_of(path_lake.l3, SMALL, M3, "path.parquet")
    path.write_bytes(summaries.read_bytes())
    problem = invalid_reason(
        str(path_lake.l3), str(path_lake.l2), KEY, SMALL, M3, "path"
    )
    assert problem is not None and "esquema" in problem


def test_other_row_groups_than_l2_are_invalid_for_path_too(path_lake):
    path = file_of(path_lake.l3, SMALL, M3, "path.parquet")
    table = pq.read_table(path)
    pq.write_table(table, path, row_group_size=table.num_rows)
    problem = invalid_reason(
        str(path_lake.l3), str(path_lake.l2), KEY, SMALL, M3, "path"
    )
    assert problem is not None and "row group" in problem
    assert path_frontier_of(path_lake) == ordinal(M2)


def test_a_family_other_than_summaries_needs_the_summaries_listing(path_lake):
    listed = list_summaries(str(path_lake.l3), KEY, "path")
    with pytest.raises(ValueError, match="summaries"):
        frontiers(
            [SMALL], listed, START, str(path_lake.l3), str(path_lake.l2), KEY, "path"
        )


# `monthly --family path` en la CLI.


def monthly_path(lake, tmp_path, *extra) -> int:
    catalog = write_catalog(tmp_path / "thetas.yaml", lake.thetas)
    argv = ["--mode", "monthly", "--series-start", "2017-08", "--family", "path"]
    return main([*argv, *extra], env_for(lake, tmp_path, catalog))


def test_monthly_path_names_the_path_frontier_as_the_reason(lake, tmp_path):
    # S llegó a 2017-10, T solo hizo 2017-08: T está en 2017-08 y el mes 2017-10 no es el siguiente.
    summarize_month(
        lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1, family="path"
    )
    before = {p: p.read_bytes() for p in lake.l3.rglob("summaries.parquet")}

    assert monthly_path(lake, tmp_path, "--from", "2017-10") == 1

    rows = emitted(tmp_path, "theta_behind_frontier")
    assert len(rows) == len(lake.thetas)
    assert {r["details"]["reason"] for r in rows} == {"l3_frontier"}
    assert {r["details"]["frontier"] for r in rows} == {"2017-08"}
    assert {r["details"]["family"] for r in rows} == {"path"}
    assert not file_of(lake.l3, SMALL, M3, "path.parquet").exists()
    assert before == {p: p.read_bytes() for p in lake.l3.rglob("summaries.parquet")}


def test_monthly_path_names_summaries_as_the_reason_when_the_month_has_none(
    lake, tmp_path
):
    summarize_month(
        lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1, family="path"
    )
    file_of(lake.l3, BIG, M2, "summaries.parquet").unlink()

    assert monthly_path(lake, tmp_path, "--from", "2017-09") == 0

    (row,) = emitted(tmp_path, "theta_behind_frontier")
    assert row["details"] == {
        "theta": theta_text(BIG),
        "frontier": "2017-08",
        "reason": "summaries_missing",
        "family": "path",
    }
    assert file_of(lake.l3, SMALL, M2, "path.parquet").exists()
    assert not file_of(lake.l3, BIG, M2, "path.parquet").exists()


# La familia A (`fit.parquet`): su frontera sale de su propio archivo y nunca supera la de S.


@pytest.fixture
def fit_lake(lake):
    for month in lake.months:
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            family="fit",
        )
    return lake


def fit_frontier_of(lake, theta: int = SMALL) -> int | None:
    summaries = list_summaries(str(lake.l3), KEY)
    listed = list_summaries(str(lake.l3), KEY, "fit")
    result = frontiers(
        [theta], listed, START, str(lake.l3), str(lake.l2), KEY, "fit", summaries
    )
    return result[theta]


def test_a_complete_fit_series_reaches_its_last_month(fit_lake):
    assert fit_frontier_of(fit_lake) == ordinal(M3)


def test_the_fit_frontier_does_not_depend_on_having_the_fit_file_for_s(lake):
    assert frontier_of(lake) == ordinal(M3)
    assert fit_frontier_of(lake) is None


def test_a_hole_in_the_fit_series_ends_it_before_the_hole(fit_lake):
    file_of(fit_lake.l3, SMALL, M2, "fit.parquet").unlink()
    assert fit_frontier_of(fit_lake) == ordinal(M1)
    assert frontier_of(fit_lake) == ordinal(M3)


def test_the_fit_frontier_never_passes_the_one_of_summaries(fit_lake):
    file_of(fit_lake.l3, SMALL, M3, "summaries.parquet").unlink()
    assert fit_frontier_of(fit_lake) == ordinal(M2)
    assert fit_frontier_of(fit_lake, BIG) == ordinal(M3)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (lambda t: with_state_version(t, "0.9.0"), "state_version"),
        (lambda t: t.drop_columns(["ev_jump_share"]), "esquema"),
        (lambda t: t.drop_columns(["c_slope"]), "esquema"),
    ],
    ids=["state_version", "schema", "schema_other_column"],
)
def test_an_invalid_fit_file_moves_the_fit_frontier_back(fit_lake, damage, reason):
    rewrite(file_of(fit_lake.l3, SMALL, M3, "fit.parquet"), damage)
    problem = invalid_reason(str(fit_lake.l3), str(fit_lake.l2), KEY, SMALL, M3, "fit")
    assert problem is not None and reason in problem
    assert fit_frontier_of(fit_lake) == ordinal(M2)
    # Las fronteras de S no se enteran.
    assert frontier_of(fit_lake) == ordinal(M3)


def test_a_summaries_copy_is_not_a_valid_fit_file(fit_lake):
    # El mismo contenido es válido como `summaries` y no como `fit`.
    summaries = file_of(fit_lake.l3, SMALL, M3, "summaries.parquet")
    fit = file_of(fit_lake.l3, SMALL, M3, "fit.parquet")
    fit.write_bytes(summaries.read_bytes())
    problem = invalid_reason(str(fit_lake.l3), str(fit_lake.l2), KEY, SMALL, M3, "fit")
    assert problem is not None and "esquema" in problem


def test_other_row_groups_than_l2_are_invalid_for_fit_too(fit_lake):
    fit = file_of(fit_lake.l3, SMALL, M3, "fit.parquet")
    table = pq.read_table(fit)
    pq.write_table(table, fit, row_group_size=table.num_rows)
    problem = invalid_reason(str(fit_lake.l3), str(fit_lake.l2), KEY, SMALL, M3, "fit")
    assert problem is not None and "row group" in problem
    assert fit_frontier_of(fit_lake) == ordinal(M2)


def monthly_fit(lake, tmp_path, *extra) -> int:
    catalog = write_catalog(tmp_path / "thetas.yaml", lake.thetas)
    argv = ["--mode", "monthly", "--series-start", "2017-08", "--family", "fit"]
    return main([*argv, *extra], env_for(lake, tmp_path, catalog))


def test_monthly_fit_names_the_fit_frontier_as_the_reason(lake, tmp_path):
    # S llegó a 2017-10, A solo hizo 2017-08: A está en 2017-08 y el mes 2017-10 no es el siguiente.
    summarize_month(
        lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1, family="fit"
    )
    others = ("summaries.parquet", "path.parquet")
    before = {p: p.read_bytes() for n in others for p in lake.l3.rglob(n)}

    assert monthly_fit(lake, tmp_path, "--from", "2017-10") == 1

    rows = emitted(tmp_path, "theta_behind_frontier")
    assert len(rows) == len(lake.thetas)
    assert {r["details"]["reason"] for r in rows} == {"l3_frontier"}
    assert {r["details"]["frontier"] for r in rows} == {"2017-08"}
    assert {r["details"]["family"] for r in rows} == {"fit"}
    assert not file_of(lake.l3, SMALL, M3, "fit.parquet").exists()
    assert before == {p: p.read_bytes() for n in others for p in lake.l3.rglob(n)}


def test_monthly_fit_names_summaries_as_the_reason_when_the_month_has_none(
    lake, tmp_path
):
    summarize_month(
        lake.l1, lake.l2, lake.l3, M1, lake.theta_texts, series_start=M1, family="fit"
    )
    file_of(lake.l3, BIG, M2, "summaries.parquet").unlink()

    assert monthly_fit(lake, tmp_path, "--from", "2017-09") == 0

    (row,) = emitted(tmp_path, "theta_behind_frontier")
    assert row["details"] == {
        "theta": theta_text(BIG),
        "frontier": "2017-08",
        "reason": "summaries_missing",
        "family": "fit",
    }
    assert file_of(lake.l3, SMALL, M2, "fit.parquet").exists()
    assert not file_of(lake.l3, BIG, M2, "fit.parquet").exists()


# La familia F (`spectrum.parquet`): su frontera sale de su propio archivo y nunca supera la de S.


@pytest.fixture
def spectrum_lake(lake):
    for month in lake.months:
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=M1,
            family="spectrum",
        )
    return lake


def spectrum_frontier_of(lake, theta: int = SMALL) -> int | None:
    summaries = list_summaries(str(lake.l3), KEY)
    listed = list_summaries(str(lake.l3), KEY, "spectrum")
    result = frontiers(
        [theta], listed, START, str(lake.l3), str(lake.l2), KEY, "spectrum", summaries
    )
    return result[theta]


def test_a_complete_spectrum_series_reaches_its_last_month(spectrum_lake):
    assert spectrum_frontier_of(spectrum_lake, BIG) == ordinal(M3)


def test_the_spectrum_frontier_does_not_depend_on_having_the_file_for_s(lake):
    assert frontier_of(lake) == ordinal(M3)
    assert spectrum_frontier_of(lake) is None


def test_a_hole_in_the_spectrum_series_ends_it_before_the_hole(spectrum_lake):
    file_of(spectrum_lake.l3, SMALL, M2, "spectrum.parquet").unlink()
    assert spectrum_frontier_of(spectrum_lake) == ordinal(M1)
    assert frontier_of(spectrum_lake) == ordinal(M3)


def test_the_spectrum_frontier_never_passes_the_one_of_summaries(spectrum_lake):
    file_of(spectrum_lake.l3, SMALL, M3, "summaries.parquet").unlink()
    assert spectrum_frontier_of(spectrum_lake) == ordinal(M2)
    assert spectrum_frontier_of(spectrum_lake, BIG) == ordinal(M3)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (lambda t: with_state_version(t, "0.9.0"), "state_version"),
        (lambda t: t.drop_columns(["os_haar_level_7"]), "esquema"),
        (lambda t: t.drop_columns(["c_fourier_band_1"]), "esquema"),
    ],
    ids=["state_version", "schema", "schema_other_column"],
)
def test_an_invalid_spectrum_file_moves_the_spectrum_frontier_back(
    spectrum_lake, damage, reason
):
    rewrite(file_of(spectrum_lake.l3, SMALL, M3, "spectrum.parquet"), damage)
    problem = invalid_reason(
        str(spectrum_lake.l3), str(spectrum_lake.l2), KEY, SMALL, M3, "spectrum"
    )
    assert problem is not None and reason in problem
    assert spectrum_frontier_of(spectrum_lake) == ordinal(M2)
    # Las fronteras de S no se enteran.
    assert frontier_of(spectrum_lake) == ordinal(M3)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        # Un `spectrum.parquet` 1.0.0 de verdad (la corrida de prueba de 2023-03): sus 28
        # columnas, sin `c_energy_loc` ni `os_energy_loc`, y su versión. Lo frena el esquema.
        (
            lambda t: with_state_version(
                t.drop_columns(["c_energy_loc", "os_energy_loc"]), "1.0.0"
            ),
            "esquema",
        ),
        # Las dos columnas de más con la versión vieja, o solo una de ellas: también atrás.
        (lambda t: with_state_version(t, "1.0.0"), "state_version"),
        (lambda t: t.drop_columns(["os_energy_loc"]), "esquema"),
        # Un `spectrum.parquet` 1.1.0 (presupuesto de 20 M, ITSC-417): mismo esquema de 30
        # columnas, otra versión. Cambiar el presupuesto es versión nueva (decisión 7): el
        # backfill lo reescribe para que sus eventos entre 20 M y 130 M dejen de ser nulos.
        (lambda t: with_state_version(t, "1.1.0"), "state_version"),
    ],
    ids=["file_1_0_0", "version_1_0_0", "one_location_missing", "version_1_1_0"],
)
def test_a_spectrum_file_of_an_older_version_is_behind_the_frontier(
    spectrum_lake, damage, reason
):
    """El backfill de F lo reescribe: la frontera de F queda antes de ese mes, la de S no."""
    rewrite(file_of(spectrum_lake.l3, SMALL, M3, "spectrum.parquet"), damage)
    problem = invalid_reason(
        str(spectrum_lake.l3), str(spectrum_lake.l2), KEY, SMALL, M3, "spectrum"
    )
    assert problem is not None and reason in problem
    assert spectrum_frontier_of(spectrum_lake) == ordinal(M2)
    assert frontier_of(spectrum_lake) == ordinal(M3)


def test_the_spectrum_frontier_does_not_look_at_the_parameters_in_the_footer(
    spectrum_lake,
):
    # Cambiar n_min, bands, window o budget_ticks es una state_version nueva (decisión 7): el
    # pie con otros parámetros y la misma versión sigue siendo una frontera válida.
    version = pq.ParquetFile(
        file_of(spectrum_lake.l3, SMALL, M3, "spectrum.parquet")
    ).schema_arrow.metadata[b"state_version"]

    def other_budget(table):
        return table.replace_schema_metadata(
            {"state_version": version.decode(), "budget_ticks": "1"}
        )

    rewrite(file_of(spectrum_lake.l3, SMALL, M3, "spectrum.parquet"), other_budget)
    assert spectrum_frontier_of(spectrum_lake) == ordinal(M3)


def test_a_summaries_copy_is_not_a_valid_spectrum_file(spectrum_lake):
    summaries = file_of(spectrum_lake.l3, SMALL, M3, "summaries.parquet")
    spectrum = file_of(spectrum_lake.l3, SMALL, M3, "spectrum.parquet")
    spectrum.write_bytes(summaries.read_bytes())
    problem = invalid_reason(
        str(spectrum_lake.l3), str(spectrum_lake.l2), KEY, SMALL, M3, "spectrum"
    )
    assert problem is not None and "esquema" in problem


def test_other_row_groups_than_l2_are_invalid_for_spectrum_too(spectrum_lake):
    spectrum = file_of(spectrum_lake.l3, SMALL, M3, "spectrum.parquet")
    table = pq.read_table(spectrum)
    pq.write_table(table, spectrum, row_group_size=table.num_rows)
    problem = invalid_reason(
        str(spectrum_lake.l3), str(spectrum_lake.l2), KEY, SMALL, M3, "spectrum"
    )
    assert problem is not None and "row group" in problem
    assert spectrum_frontier_of(spectrum_lake) == ordinal(M2)


def monthly_spectrum(lake, tmp_path, *extra) -> int:
    catalog = write_catalog(tmp_path / "thetas.yaml", lake.thetas)
    argv = ["--mode", "monthly", "--series-start", "2017-08", "--family", "spectrum"]
    return main([*argv, *extra], env_for(lake, tmp_path, catalog))


def test_monthly_spectrum_names_the_spectrum_frontier_as_the_reason(lake, tmp_path):
    # S llegó a 2017-10, F solo hizo 2017-08: F está en 2017-08 y el mes 2017-10 no es el siguiente.
    summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M1,
        lake.theta_texts,
        series_start=M1,
        family="spectrum",
    )
    others = ("summaries.parquet", "path.parquet", "fit.parquet")
    before = {p: p.read_bytes() for n in others for p in lake.l3.rglob(n)}

    assert monthly_spectrum(lake, tmp_path, "--from", "2017-10") == 1

    rows = emitted(tmp_path, "theta_behind_frontier")
    assert len(rows) == len(lake.thetas)
    assert {r["details"]["reason"] for r in rows} == {"l3_frontier"}
    assert {r["details"]["frontier"] for r in rows} == {"2017-08"}
    assert {r["details"]["family"] for r in rows} == {"spectrum"}
    assert not file_of(lake.l3, SMALL, M3, "spectrum.parquet").exists()
    assert before == {p: p.read_bytes() for n in others for p in lake.l3.rglob(n)}


def test_monthly_spectrum_names_summaries_as_the_reason_when_the_month_has_none(
    lake, tmp_path
):
    summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M1,
        lake.theta_texts,
        series_start=M1,
        family="spectrum",
    )
    file_of(lake.l3, BIG, M2, "summaries.parquet").unlink()

    assert monthly_spectrum(lake, tmp_path, "--from", "2017-09") == 0

    (row,) = emitted(tmp_path, "theta_behind_frontier")
    assert row["details"] == {
        "theta": theta_text(BIG),
        "frontier": "2017-08",
        "reason": "summaries_missing",
        "family": "spectrum",
    }
    assert file_of(lake.l3, SMALL, M2, "spectrum.parquet").exists()
    assert not file_of(lake.l3, BIG, M2, "spectrum.parquet").exists()


# El incidente de ITSC-419: un archivo válido en medio de archivos de la versión anterior.


def test_a_valid_file_after_an_invalid_one_does_not_hide_it(lake):
    """Con un inválido en el mes `m` la frontera es `m − 1`, aunque `m + 1` sea válido."""
    rewrite(
        file_of(lake.l3, SMALL, M2, "summaries.parquet"),
        lambda t: with_state_version(t, "0.9.0"),
    )
    assert frontier_of(lake) == ordinal(M1)
    assert frontier_of(lake, BIG) == ordinal(M3)


def test_an_invalid_first_month_leaves_no_frontier_even_with_valid_ones_after(lake):
    rewrite(
        file_of(lake.l3, SMALL, M1, "summaries.parquet"),
        lambda t: with_state_version(t, "0.9.0"),
    )
    assert frontier_of(lake) is None


def test_the_scan_reports_every_invalid_month_with_the_first_reason(lake):
    for month in (M1, M3):
        rewrite(
            file_of(lake.l3, SMALL, month, "summaries.parquet"),
            lambda t: with_state_version(t, "0.9.0"),
        )
    listed = list_summaries(str(lake.l3), KEY)
    found = scan_frontiers([SMALL, BIG], listed, START, str(lake.l3), str(lake.l2), KEY)
    assert found[SMALL].reached is None
    assert sorted(found[SMALL].invalid) == [ordinal(M1), ordinal(M3)]
    assert found[SMALL].first_invalid == ordinal(M1)
    assert "summaries.parquet: state_version 0.9.0" in found[SMALL].invalid[ordinal(M1)]
    assert found[SMALL].checked == 3
    assert found[BIG].reached == ordinal(M3) and not found[BIG].invalid


FAMILIES = ("summaries", "path", "fit", "spectrum")
OLD = "0.9.0"
INCIDENT_THETAS = (SMALL, BIG)


@pytest.fixture(params=FAMILIES)
def family(request) -> str:
    return request.param


@pytest.fixture
def incident(tmp_path, family):
    """Cinco meses de la familia, todos con la versión anterior salvo el tercero (2023-01 del
    incidente): un válido rodeado de archivos de otra versión.
    """
    cuts = month_cuts(TICKS, 800, 1600, 2400, 3200)
    lake = build_lake(tmp_path, TICKS, cuts, thetas=INCIDENT_THETAS)
    assert len(lake.months) == 5
    for name in dict.fromkeys(("summaries", family)):
        for month in lake.months:
            summarize_month(
                lake.l1,
                lake.l2,
                lake.l3,
                month,
                lake.theta_texts,
                series_start=M1,
                family=name,
            )
    for month in lake.months:
        if month != lake.months[2]:
            for theta in INCIDENT_THETAS:
                rewrite(
                    file_of(lake.l3, theta, month, f"{family}.parquet"),
                    lambda t: with_state_version(t, OLD),
                )
    return lake


def family_scan(lake, family: str, workers: int | None = None, reader=None):
    summaries = list_summaries(str(lake.l3), KEY)
    listed = list_summaries(str(lake.l3), KEY, family)
    return scan_frontiers(
        INCIDENT_THETAS,
        listed,
        START,
        str(lake.l3),
        str(lake.l2),
        KEY,
        family,
        summaries,
        workers,
        reader,
    )


def run_family(lake, tmp_path, family: str, *argv: str) -> int:
    catalog = write_catalog(tmp_path / "thetas.yaml", lake.thetas)
    base = ["--series-start", "2017-08", "--family", family]
    return main([*argv, *base], env_for(lake, tmp_path, catalog))


def test_the_incident_gives_no_frontier_not_the_third_month(incident, family):
    found = family_scan(incident, family)
    third, others = incident.months[2], incident.months[:2] + incident.months[3:]
    for theta in INCIDENT_THETAS:
        assert found[theta].reached is None
        assert sorted(found[theta].invalid) == sorted(ordinal(m) for m in others)
        assert ordinal(third) not in found[theta].invalid
        assert (
            f"{family}.parquet: state_version {OLD}"
            in (found[theta].invalid[ordinal(others[0])])
        )


def test_backfill_without_from_or_to_rewrites_the_five_months(
    incident, tmp_path, family, caplog
):
    caplog.set_level("INFO")
    assert run_family(incident, tmp_path, family, "--mode", "backfill") == 0
    # Los cinco quedan en la versión vigente, también el tercero (reescribirlo es inocuo).
    for theta in INCIDENT_THETAS:
        for month in incident.months:
            assert (
                invalid_reason(
                    str(incident.l3), str(incident.l2), KEY, theta, month, family
                )
                is None
            )
    assert set(family_scan(incident, family)[SMALL].invalid) == set()
    assert family_scan(incident, family)[SMALL].reached == ordinal(incident.months[-1])
    (row,) = emitted(tmp_path, "backfill_summary")
    assert row["details"] == {
        "family": family,
        "processed": 5,
        "skipped": 0,
        "months": 5,
        "first": "2017-08",
        "last": "2017-12",
    }
    assert row["severity"] == "info"
    assert (
        "backfill: 5 meses procesados, 0 saltados por estar al día, de 5 en el rango"
        in caplog.text
    )


def test_the_regression_leaves_one_warning_per_theta_not_one_per_month(
    incident, tmp_path, family
):
    assert run_family(incident, tmp_path, family, "--mode", "backfill") == 0
    rows = emitted(tmp_path, "theta_frontier_regressed")
    assert len(rows) == len(INCIDENT_THETAS)
    assert {r["severity"] for r in rows} == {"warning"}
    assert {r["details"]["theta"] for r in rows} == {
        theta_text(t) for t in INCIDENT_THETAS
    }
    for row in rows:
        details = row["details"]
        assert details["family"] == family
        assert details["first_invalid_month"] == "2017-08"
        assert details["invalid_months"] == 4
        assert details["frontier"] is None
        assert f"{family}.parquet: state_version {OLD}" in details["reason"]


def test_monthly_on_the_incident_series_leaves_the_theta_behind_and_writes_nothing(
    incident, tmp_path, family
):
    before = {p: p.read_bytes() for p in incident.l3.rglob("*.parquet")}
    assert (
        run_family(incident, tmp_path, family, "--mode", "monthly", "--from", "2017-12")
        == 1
    )
    rows = emitted(tmp_path, "theta_behind_frontier")
    assert len(rows) == len(INCIDENT_THETAS)
    assert {r["details"]["reason"] for r in rows} == {"l3_frontier"}
    assert {r["details"]["frontier"] for r in rows} == {None}
    assert {r["details"]["family"] for r in rows} == {family}
    assert before == {p: p.read_bytes() for p in incident.l3.rglob("*.parquet")}
    assert len(emitted(tmp_path, "theta_frontier_regressed")) == len(INCIDENT_THETAS)


def test_a_backfill_that_starts_after_the_invalid_month_exits_with_1(
    incident, tmp_path, family
):
    """El rango no repara el primer mes inválido: el cierre no puede pasar como éxito."""
    code = run_family(
        incident, tmp_path, family, "--mode", "backfill", "--from", "2017-10"
    )
    assert code == 1
    (row,) = emitted(tmp_path, "backfill_summary")
    assert row["details"]["processed"] == 3


def test_a_backfill_inside_the_valid_prefix_skips_and_counts_the_months(lake, tmp_path):
    rewrite(
        file_of(lake.l3, SMALL, M3, "summaries.parquet"),
        lambda t: with_state_version(t, OLD),
    )
    argv = ["--mode", "backfill", "--from", "2017-08", "--to", "2017-09"]
    assert run_family(lake, tmp_path, "summaries", *argv) == 0
    (row,) = emitted(tmp_path, "backfill_summary")
    assert (row["details"]["processed"], row["details"]["skipped"]) == (0, 2)
    # El inválido está fuera del rango: se avisa, pero el rango pedido llegó a su fin.
    (warning,) = emitted(tmp_path, "theta_frontier_regressed")
    assert warning["details"]["first_invalid_month"] == "2017-10"
    assert warning["details"]["frontier"] == "2017-09"


def test_a_family_backfill_over_an_invalid_summaries_month_exits_with_1(
    path_lake, tmp_path, caplog
):
    """La pasada de T no reescribe `summaries.parquet`: no puede dejar el cierre en 0."""
    rewrite(
        file_of(path_lake.l3, SMALL, M2, "summaries.parquet"),
        lambda t: with_state_version(t, OLD),
    )
    assert run_family(path_lake, tmp_path, "path", "--mode", "backfill") == 1
    assert "corre primero el backfill de summaries" in caplog.text
    # El inválido sigue ahí: la frontera de T no lo oculta.
    assert family_scan(path_lake, "path")[SMALL].ceiling_invalid.keys() == {ordinal(M2)}


def test_the_months_after_every_theta_left_the_range_are_not_counted_as_up_to_date(
    lake, tmp_path
):
    """Con L2 ausente para todos los θ desde M2, `N + K < M` delata los meses que no se miraron."""
    for theta in lake.thetas:
        for month in (M2, M3):
            file_of(lake.l2, theta, month, "events.parquet").unlink()
    assert run_family(lake, tmp_path, "summaries", "--mode", "backfill") == 1
    (row,) = emitted(tmp_path, "backfill_summary")
    details = row["details"]
    assert (details["processed"], details["skipped"], details["months"]) == (0, 1, 3)


def test_the_log_reports_how_many_footers_the_frontier_read(
    incident, tmp_path, family, caplog
):
    """Un archivo por mes y θ (el doble fuera de S) y los pies que se pidieron de verdad (ITSC-420).

    En S son dos por archivo (el suyo y el de `events.parquet`); fuera de S, el de
    `events.parquet` de un mes se lee una vez aunque lo pidan la familia y el techo de S: el
    de la familia, el de `summaries.parquet` y el de `events.parquet`, tres por (θ, mes).
    """
    caplog.set_level("INFO")
    run_family(incident, tmp_path, family, "--mode", "backfill")
    cells = len(incident.months) * len(INCIDENT_THETAS)
    files, footers = (
        (cells, 2 * cells) if family == "summaries" else (2 * cells, 3 * cells)
    )
    assert (
        f"fronteras: {files} archivos verificados ({footers} pies leídos) de 2 θ "
        f"con 64 lectores en " in caplog.text
    )


# El costo de la verificación (ITSC-420): menos viajes y más lectores, el mismo resultado.


class CountingFs(pafs.FileSystemHandler):
    """Un sistema de archivos local que cuenta los `get_file_info` por ruta y las aperturas."""

    def __init__(self) -> None:
        self.inner = pafs.LocalFileSystem()
        self.exists_trips = 0
        self.opened: list[str] = []

    def get_type_name(self):
        return "counting"

    def normalize_path(self, path):
        return path

    def get_file_info(self, paths):
        self.exists_trips += 1
        return self.inner.get_file_info(list(paths))

    def get_file_info_selector(self, selector):
        return self.inner.get_file_info(selector)

    def open_input_file(self, path):
        self.opened.append(path)
        return self.inner.open_input_file(path)

    def open_input_stream(self, path):
        return self.inner.open_input_stream(path)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def _read_only(self, *args, **kwargs):
        raise NotImplementedError

    create_dir = delete_dir = delete_dir_contents = delete_root_dir_contents = (
        _read_only
    )
    delete_file = move = copy_file = open_output_stream = open_append_stream = (
        _read_only
    )


@pytest.fixture
def counting(monkeypatch):
    """El sistema de archivos de la verificación, contado, y las raíces que se resolvieron."""
    handler, resolved = CountingFs(), []
    real = frontier.resolve_fs

    def resolve(root):
        resolved.append(str(root))
        return pafs.PyFileSystem(handler), real(root)[1]

    monkeypatch.setattr(frontier, "resolve_fs", resolve)
    return handler, resolved


def test_the_events_footer_is_opened_once_per_theta_and_month(incident, family):
    """También fuera de S, donde lo piden la familia y el techo de S (cuenta las aperturas)."""
    reader = FooterReader(share_events=family != "summaries")
    handler = CountingFs()
    cells = len(incident.months) * len(INCIDENT_THETAS)
    real = frontier.resolve_fs
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            frontier,
            "resolve_fs",
            lambda root: (pafs.PyFileSystem(handler), real(root)[1]),
        )
        family_scan(incident, family, reader=reader)
    events = [p for p in handler.opened if p.endswith("events.parquet")]
    assert len(events) == cells == len(set(events))
    # Fuera de S: el archivo de la familia, el de summaries y el de events, una vez cada uno.
    assert (
        reader.opened
        == len(handler.opened)
        == cells * (2 if family == "summaries" else 3)
    )


def test_without_sharing_a_family_reads_the_events_footer_twice(incident, family):
    """El caché es lo que evita la segunda lectura: sin él, la familia y el techo la repiten."""
    if family == "summaries":
        pytest.skip("S no tiene techo aparte: un solo recorrido")
    reader = FooterReader(share_events=False)
    family_scan(incident, family, reader=reader)
    cells = len(incident.months) * len(INCIDENT_THETAS)
    assert reader.opened == 4 * cells


def test_the_footers_are_read_without_the_existence_trip_and_with_one_filesystem_per_root(
    incident, family, counting
):
    handler, resolved = counting
    summaries = list_summaries(str(incident.l3), KEY)
    listed = list_summaries(str(incident.l3), KEY, family)
    resolved.clear()
    scan_frontiers(
        INCIDENT_THETAS,
        listed,
        START,
        str(incident.l3),
        str(incident.l2),
        KEY,
        family,
        summaries,
    )
    assert handler.exists_trips == 0
    assert sorted(resolved) == sorted([str(incident.l3), str(incident.l2)])


def test_the_old_open_path_would_pay_the_existence_trip(incident, counting):
    """La razón de `FooterReader`: `open_parquet` hace un `get_file_info` por apertura."""
    handler, _ = counting
    month = incident.months[0]
    path = str(file_of(incident.l3, SMALL, month, "summaries.parquet"))
    # `open_parquet` vive en dc_frames y resuelve con su propio `resolve_fs`.
    with pytest.MonkeyPatch.context() as patch:
        real = lake_module.resolve_fs
        patch.setattr(
            lake_module,
            "resolve_fs",
            lambda root: (pafs.PyFileSystem(handler), real(root)[1]),
        )
        lake_module.open_parquet(path, "la familia").close()
    assert handler.exists_trips == 1


def test_a_file_that_vanishes_between_the_listing_and_the_reading_is_invalid(lake):
    listed = list_summaries(str(lake.l3), KEY)
    file_of(lake.l3, SMALL, M3, "summaries.parquet").unlink()
    found = scan_frontiers([SMALL], listed, START, str(lake.l3), str(lake.l2), KEY)
    assert found[SMALL].reached == ordinal(M2)
    assert "no se pudo leer" in found[SMALL].invalid[ordinal(M3)]
    assert "falta la familia" in found[SMALL].invalid[ordinal(M3)]


def test_the_result_does_not_depend_on_the_number_of_readers(incident, family):
    def outcome(workers: int):
        found = family_scan(incident, family, workers)
        return {
            theta: (f.reached, dict(f.invalid), dict(f.ceiling_invalid), f.checked)
            for theta, f in found.items()
        }

    one = outcome(1)
    assert one[SMALL][0] is None and one[SMALL][1]
    assert outcome(8) == one
    assert outcome(64) == one


def test_a_scan_needs_at_least_one_reader(lake):
    listed = list_summaries(str(lake.l3), KEY)
    with pytest.raises(ValueError, match="workers"):
        scan_frontiers(
            [SMALL], listed, START, str(lake.l3), str(lake.l2), KEY, workers=0
        )


def test_the_log_says_how_many_readers_the_frontier_used(lake, tmp_path, caplog):
    caplog.set_level("INFO")
    catalog = write_catalog(tmp_path / "thetas.yaml", lake.thetas)
    env = env_for(lake, tmp_path, catalog, L3_FRONTIER_WORKERS="3")
    argv = ["--mode", "backfill", "--series-start", "2017-08"]
    assert main(argv, env) == 0
    assert " con 3 lectores en " in caplog.text
