import io
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l1_ingest.conform import TimestampUnitError
from l1_ingest.integrity import AggIdConflict
from l1_ingest.parse import open_zip_batches
from l1_ingest.pipeline import Unit, _materialized
from l1_ingest.schema import RAW_SCHEMA
from l1_ingest.stream import NotStreamable, _SequenceCheck, stream_partition

BLOCK = 64 * 1024  # unos mil filas por lote
UNIT = Unit(2024, 3, asset="BTCUSDT")


def _csv(n: int, unit: int = 1000, disorder_at: int | None = None) -> bytes:
    """`n` filas; ids con un hueco cada 997 y, cada 1499 filas, una copia idéntica de la anterior."""
    lines, agg_id = [], 0
    for row in range(n):
        agg_id += 3 if row % 997 == 996 else 1
        time = (1_709_600_000_000 + row) * (1000 // unit)
        if row == disorder_at:
            time -= 5 * (1000 // unit)
        line = (
            f"{agg_id},100.{row % 1000:03d}00000,0.50000000,{row},{row},{time},"
            f"{'True' if row % 2 else 'False'},True"
        )
        lines.append(line)
        if row % 1499 == 1498:
            lines.append(line)
    return ("\n".join(lines) + "\n").encode()


def _zip(csv: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BTCUSDT-aggTrades-2024-03.csv", csv)
    return buf.getvalue()


def _stream(csv: bytes, path: str):
    with open_zip_batches(
        _zip(csv), UNIT.asset, UNIT.year, UNIT.month, block_size=BLOCK
    ) as (_, batches):
        return stream_partition(batches, path)


def test_lotes_dan_mismo_hash_y_hallazgos_que_la_ruta_materializada(tmp_path):
    csv = _csv(25_432)
    checks, digest = _stream(csv, str(tmp_path / "stream.parquet"))
    old_checks, old_digest = _materialized(
        _zip(csv), str(tmp_path / "old.parquet"), UNIT
    )

    assert digest == old_digest
    assert checks == old_checks
    assert [c.check_type for c in checks] == [
        "timestamp_unit_corrected",
        "reorder_applied",
        "aggid_gap",
        "aggid_duplicate",
        "provider_invalid_marker",
    ]
    assert checks[2].metric_value > 0 and checks[3].metric_value > 0


def test_unidad_en_microsegundos_no_se_corrige(tmp_path):
    csv = _csv(3_500, unit=1)
    checks, digest = _stream(csv, str(tmp_path / "a.parquet"))
    old_checks, old_digest = _materialized(_zip(csv), str(tmp_path / "b.parquet"), UNIT)
    assert (checks, digest) == (old_checks, old_digest)
    assert checks[0].status == "pass"


def test_desorden_entre_lotes_cae_a_la_ruta_materializada(tmp_path):
    csv = _csv(5_000, disorder_at=2_500)
    with pytest.raises(NotStreamable):
        _stream(csv, str(tmp_path / "a.parquet"))
    assert list(tmp_path.iterdir()) == []


def test_unidad_invalida_en_un_lote_tardio_reporta_rango_global_sin_archivo(tmp_path):
    csv = _csv(3_000) + b"9999,1.0,1.0,1,1,12345,True,True\n"
    with pytest.raises(TimestampUnitError) as exc:
        _stream(csv, str(tmp_path / "a.parquet"))
    old = pytest.raises(TimestampUnitError)
    with old as old_exc:
        _materialized(_zip(csv), str(tmp_path / "b.parquet"), UNIT)
    assert exc.value.check == old_exc.value.check
    assert not (tmp_path / "a.parquet").exists()
    assert [p.name for p in tmp_path.iterdir()] == []


def _doubled(n: int, divergent_at: int | None = None) -> bytes:
    """`n` filas cuyo id aparece dos veces seguidas; la copia de `divergent_at` cambia el price."""
    lines = []
    for row in range(1, n + 1):
        line = f"{row},100.00000000,0.50000000,{row},{row},{1_709_600_000_000 + row},True,True"
        lines += [line, line.replace("100.0", "100.9") if row == divergent_at else line]
    return ("\n".join(lines) + "\n").encode()


def test_ids_repetidos_entre_lotes_se_eliminan_igual_que_en_la_ruta_materializada(
    tmp_path,
):
    csv = _doubled(4_000)
    checks, digest = _stream(csv, str(tmp_path / "stream.parquet"))
    old_checks, old_digest = _materialized(
        _zip(csv), str(tmp_path / "old.parquet"), UNIT
    )

    assert (checks, digest) == (old_checks, old_digest)
    (dup,) = [c for c in checks if c.check_type == "aggid_duplicate"]
    assert (dup.status, dup.metric_value) == ("corrected", 4_000)
    assert dup.details["ids"] == list(range(1, 11))
    ids = pq.read_table(tmp_path / "stream.parquet")["agg_trade_id"].to_pylist()
    assert ids == list(range(1, 4_001))


def test_id_repetido_con_contenido_distinto_en_un_lote_tardio_aborta_sin_archivo(
    tmp_path,
):
    csv = _doubled(4_000, divergent_at=3_500)
    with pytest.raises(AggIdConflict) as exc:
        _stream(csv, str(tmp_path / "a.parquet"))
    with pytest.raises(AggIdConflict) as old_exc:
        _materialized(_zip(csv), str(tmp_path / "b.parquet"), UNIT)

    assert exc.value.check == old_exc.value.check
    assert exc.value.check.details["agg_trade_id"] == 3_500
    assert [p.name for p in tmp_path.iterdir()] == []


def _batch(ids: list[int], prices: list[str]) -> pa.RecordBatch:
    n = len(ids)
    return pa.record_batch(
        [ids, prices, ["1"] * n, ids, ids, ids, [True] * n, [True] * n],
        schema=RAW_SCHEMA,
    )


def test_repetida_en_la_frontera_entre_lotes():
    seq = _SequenceCheck()
    first = seq.feed(_batch([1, 2], ["1", "1"]))
    second = seq.feed(_batch([2, 3], ["1", "1"]))

    assert first.num_rows == 2
    assert second.column("agg_trade_id").to_pylist() == [3]
    assert seq.n_duplicates == 1 and seq.duplicates == [2]

    seq = _SequenceCheck()
    seq.feed(_batch([1, 2], ["1", "1"]))
    with pytest.raises(AggIdConflict):
        seq.feed(_batch([2, 3], ["9", "1"]))
