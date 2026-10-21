"""Los errores tipados del núcleo con sus `details` (TRD-L3 §9.3) y la proyección de L2."""

from decimal import Decimal

import pyarrow.parquet as pq
import pytest
from dc_frames import (
    FamilySkeletonMismatch,
    FamilyWriter,
    FrameBoundaryError,
    FramesInputError,
)
from dc_frames.lake import EventFile
from l3_dc_descriptors import SkeletonMismatch, summarize_month
from l3_dc_descriptors.event import EVENT_COLUMNS
from l3_dc_descriptors.summarize import _inconsistent
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts

SMALL, BIG = 100_000, 2_000_000
PARTITION = "provider=binance/market=spot/asset=BTCUSDT"


@pytest.fixture
def lake(tmp_path):
    ticks = fixture_ticks(seed=3, drop=0.1)
    return build_lake(tmp_path, ticks, month_cuts(ticks, 2400))


def theta_text(theta: int) -> str:
    return f"{Decimal(theta).scaleb(-8):.8f}"


def run(lake, thetas=(SMALL,)):
    return summarize_month(
        lake.l1,
        lake.l2,
        lake.l3,
        M2,
        [theta_text(t) for t in thetas],
        series_start=M1,
    )


def test_boundary_error_keeps_a_tiny_theta_as_text():
    # 5E-7 sale como "5.00E-7" en `str(Decimal)`: los details lo traen con 8 decimales.
    error = FrameBoundaryError(
        "mensaje que no se lee",
        theta="0.00000050",
        field="extreme_agg_trade_id",
        got=7,
        wanted="un tick de L1",
    )
    result = _inconsistent(error, M2)
    assert result.details == {
        "theta": "0.00000050",
        "month": "2017-09",
        "field": "extreme_agg_trade_id",
        "got": "7",
        "wanted": "un tick de L1",
        "message": "mensaje que no se lee",
    }


def test_the_summarizer_decodes_only_the_seven_columns_of_l2(lake, monkeypatch):
    seen = []
    original = EventFile.batches

    def spy(self, *args, **kwargs):
        seen.append(set(self.columns))
        return original(self, *args, **kwargs)

    keys = []
    read = pq.ParquetFile.read_row_group

    def spy_keys(self, i, columns=None, **kwargs):
        keys.append(None if columns is None else set(columns))
        return read(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(EventFile, "batches", spy)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_keys)
    run(lake)
    # El motor recibe exactamente las siete columnas de L2.
    assert seen and all(columns == set(EVENT_COLUMNS) for columns in seen)
    # Los row groups de `events.parquet` solo los lee, enteros, el escritor de familias, que
    # compara las claves del esqueleto (`theta` y la confirmación).
    skeleton = {"theta", "confirm_agg_trade_id"}
    assert skeleton in keys and all(columns is not None for columns in keys)


def test_skeleton_mismatch_on_write_names_theta_month_and_family(lake, monkeypatch):
    def refuse(self, batch):
        raise FamilySkeletonMismatch("mensaje del escritor", "fila 3 distinta")

    monkeypatch.setattr(FamilyWriter, "write_row_group", refuse)
    with pytest.raises(SkeletonMismatch) as error:
        run(lake)
    assert error.value.check_type == "family_skeleton_mismatch"
    assert error.value.details == {
        "theta": theta_text(SMALL),
        "month": "2017-09",
        "family": "summaries",
        "detail": "fila 3 distinta",
    }
    assert isinstance(error.value.__cause__, FamilySkeletonMismatch)
    assert [p for p in lake.l3.rglob("*") if p.is_file()] == []


def test_skeleton_mismatch_on_commit_names_theta_month_and_family(lake, monkeypatch):
    def refuse(self):
        raise FamilySkeletonMismatch("mensaje del escritor", "faltan row groups")

    monkeypatch.setattr(FamilyWriter, "commit", refuse)
    with pytest.raises(SkeletonMismatch) as error:
        run(lake, thetas=(BIG,))
    assert error.value.details["theta"] == theta_text(BIG)
    assert error.value.details["detail"] == "faltan row groups"


def test_l1_without_id_statistics_is_an_input_error(lake):
    path = lake.l1 / f"{PARTITION}/year=2017/month=09/consolidated.parquet"
    size = pq.ParquetFile(path).metadata.row_group(0).num_rows
    pq.write_table(
        pq.read_table(path), path, row_group_size=size, write_statistics=False
    )
    with pytest.raises(FramesInputError, match="estadísticas de agg_trade_id"):
        run(lake)


def test_errors_no_evalua_anotaciones_al_importar():
    """En Python 3.12 (Colab) `-> L2StateInconsistent` dentro de su clase falla al importar.

    Con `from __future__ import annotations` la anotación queda como texto en cualquier
    versión; sin él, en 3.14 se evalúa perezosa y devolvería la clase.
    """
    from l3_dc_descriptors import errors

    assert errors.L2StateInconsistent.mismatch.__annotations__["return"] == (
        "L2StateInconsistent"
    )
