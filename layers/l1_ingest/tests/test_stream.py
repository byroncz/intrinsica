import pytest
from l1_ingest.conform import TimestampUnitError
from l1_ingest.parse import detect_header, iter_batches
from l1_ingest.pipeline import _materialized
from l1_ingest.stream import NotStreamable, stream_partition

BATCH = 1_000


def _csv(n: int, unit: int = 1000, disorder_at: int | None = None) -> bytes:
    """`n` filas; ids con un hueco cada 997 y un duplicado cada 1499 filas."""
    lines, agg_id = [], 0
    for row in range(n):
        agg_id += 3 if row % 997 == 996 else 1
        line_id = agg_id - 1 if row % 1499 == 1498 else agg_id
        time = (1_709_600_000_000 + row) * (1000 // unit)
        if row == disorder_at:
            time -= 5 * (1000 // unit)
        lines.append(
            f"{line_id},100.{row % 1000:03d}00000,0.50000000,{row},{row},{time},"
            f"{'True' if row % 2 else 'False'},True"
        )
    return ("\n".join(lines) + "\n").encode()


def _stream(csv: bytes, path: str):
    header, _ = detect_header(csv)
    return stream_partition(iter_batches(csv, header, BATCH), path)


def test_lotes_dan_mismo_hash_y_hallazgos_que_la_ruta_materializada(tmp_path):
    csv = _csv(25_432)
    checks, digest = _stream(csv, str(tmp_path / "stream.parquet"))
    old_checks, old_digest = _materialized(csv, False, str(tmp_path / "old.parquet"))

    assert digest == old_digest
    assert checks == old_checks
    assert [c.check_type for c in checks] == [
        "timestamp_unit_corrected",
        "reorder_applied",
        "aggid_gap",
        "aggid_duplicate",
    ]
    assert checks[2].metric_value > 0 and checks[3].metric_value > 0


def test_unidad_en_microsegundos_no_se_corrige(tmp_path):
    csv = _csv(3_500, unit=1)
    checks, digest = _stream(csv, str(tmp_path / "a.parquet"))
    old_checks, old_digest = _materialized(csv, False, str(tmp_path / "b.parquet"))
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
        _materialized(csv, False, str(tmp_path / "b.parquet"))
    assert exc.value.check == old_exc.value.check
    assert not (tmp_path / "a.parquet").exists()
    assert [p.name for p in tmp_path.iterdir()] == []
