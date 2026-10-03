import io
import zipfile

import pyarrow.parquet as pq
import pytest
from dq import Severity, Status
from l1_ingest.marks import ValueRangeError
from l1_ingest.parse import open_zip_batches
from l1_ingest.pipeline import Unit, _materialized
from l1_ingest.seam import check_seam
from l1_ingest.stream import stream_partition

UNIT = Unit(2017, 12, asset="BTCUSDT")
T = 1_512_378_360_000  # 2017-12-04 09:06 UTC, en ms

# Las dos marcas reales de BTCUSDT 2017-12, rodeadas de ticks válidos.
MARKED = [
    f"1195415,11500.00000000,0.10000000,1195415,1195415,{T - 5},True,True",
    f"1195416,0.00000000,0.00000000,-1,-1,{T},True,True",
    f"1195417,0.00000000,0.00000000,-1,-1,{T},True,True",
    f"1195418,11501.50000000,0.20000000,1195418,1195418,{T + 7},False,True",
]


def _zip(rows: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-aggTrades-2017-12.csv", "\n".join(rows) + "\n")
    return buf.getvalue()


def _stream(rows, path):
    with open_zip_batches(_zip(rows), UNIT.asset, UNIT.year, UNIT.month) as (_, b):
        return stream_partition(b, path)


def _by_type(checks):
    return {c.check_type: c for c in checks}


@pytest.mark.parametrize("route", ["stream", "materialized"])
def test_marcas_salen_del_parquet_y_no_son_huecos(tmp_path, route):
    path = str(tmp_path / "c.parquet")
    if route == "stream":
        checks, _ = _stream(MARKED, path)
    else:
        checks, _ = _materialized(_zip(MARKED), path, UNIT)

    assert pq.read_table(path).column("agg_trade_id").to_pylist() == [1195415, 1195418]
    found = _by_type(checks)
    marker = found["provider_invalid_marker"]
    assert marker.metric_value == 2
    assert marker.details == {"ids": [1195416, 1195417]}
    assert (marker.severity, marker.status) == (Severity.WARNING, Status.CORRECTED)
    # Los ids se evalúan crudos: 1195416 y 1195417 existen, no hay hueco.
    assert found["aggid_gap"].status == Status.PASS
    assert found["aggid_duplicate"].status == Status.PASS


def test_ruta_por_lotes_y_materializada_coinciden(tmp_path):
    checks, digest = _stream(MARKED, str(tmp_path / "a.parquet"))
    old_checks, old_digest = _materialized(
        _zip(MARKED), str(tmp_path / "b.parquet"), UNIT
    )
    assert (checks, digest) == (old_checks, old_digest)


def test_sin_marcas_el_hallazgo_pasa(tmp_path):
    checks, _ = _stream([MARKED[0], MARKED[3]], str(tmp_path / "a.parquet"))
    marker = _by_type(checks)["provider_invalid_marker"]
    assert (marker.severity, marker.status, marker.metric_value) == (
        Severity.INFO,
        Status.PASS,
        0.0,
    )


# price = 0 pero con first_trade_id >= 0; quantity = 0 con price válido; y
# price y quantity válidos con first_trade_id = -1 o last_trade_id < first_trade_id.
CORRUPT = [
    [MARKED[0], f"1195416,0.00000000,0.00000000,5,5,{T},True,True", MARKED[3]],
    [MARKED[0], f"1195416,0.00000000,0.00000000,-1,5,{T},True,True", MARKED[3]],
    [MARKED[0], f"1195416,11500.00000000,0.00000000,-1,-1,{T},True,True", MARKED[3]],
    [MARKED[0], f"1195416,11500.00000000,0.10000000,-1,5,{T},True,True", MARKED[3]],
    [MARKED[0], f"1195416,11500.00000000,0.10000000,5,4,{T},True,True", MARKED[3]],
]


@pytest.mark.parametrize("rows", CORRUPT)
@pytest.mark.parametrize("route", ["stream", "materialized"])
def test_fila_corrupta_falla_la_unidad_sin_archivo(tmp_path, rows, route):
    path = str(tmp_path / "c.parquet")
    with pytest.raises(ValueRangeError) as exc:
        if route == "stream":
            _stream(rows, path)
        else:
            _materialized(_zip(rows), path, UNIT)

    check = exc.value.check
    assert (check.check_type, check.severity, check.status) == (
        "price_out_of_range",
        Severity.ERROR,
        Status.FAIL,
    )
    assert check.metric_value == 1
    assert check.details == {"ids": [1195416]}
    assert list(tmp_path.iterdir()) == []


def test_marca_interior_no_mueve_los_bordes_de_la_costura(tmp_path):
    # Una marca nunca es primera ni última fila del mes: los bordes de
    # agg_trade_id del Parquet son los de los trades reales y la costura con
    # el mes vecino sigue cerrando sin hueco.
    prev, nxt = str(tmp_path / "prev.parquet"), str(tmp_path / "next.parquet")
    _stream(MARKED, prev)
    first = f"1195419,11502.00000000,0.30000000,1195419,1195419,{T + 9},True,True"
    _stream([first], nxt)

    check = check_seam(prev, nxt, {"prev": prev, "next": nxt})
    assert (check.status, check.metric_value) == (Status.PASS, 0)
    assert check.details["prev_max"] == 1195418
