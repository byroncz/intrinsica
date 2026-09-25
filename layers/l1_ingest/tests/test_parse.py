import io
import os
import zipfile

import pyarrow as pa
import pytest
from l1_ingest.parse import read_zip
from l1_ingest.schema import RAW_SCHEMA

HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match"
ROWS = [
    "1,42000.10,0.5,10,10,1700000000000000,True,False",
    "2,42000.20,1.25,11,12,1700000000000001,False,True",
]


def make_zip(*members: tuple[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members:
            zf.writestr(name, content)
    return buf.getvalue()


def csv_text(header: bool) -> str:
    return "\n".join(([HEADER] if header else []) + ROWS) + "\n"


@pytest.mark.parametrize("header", [False, True])
def test_read_zip_con_y_sin_header(header):
    table, check = read_zip(make_zip(("BTCUSDT-aggTrades.csv", csv_text(header))))

    assert table.schema.equals(RAW_SCHEMA)
    assert table.num_rows == 2
    assert table.column("agg_trade_id").to_pylist() == [1, 2]
    assert table.column("price").to_pylist() == ["42000.10", "42000.20"]
    assert table.column("quantity").to_pylist() == ["0.5", "1.25"]
    assert table.column("is_buyer_maker").to_pylist() == [True, False]
    assert table.column("is_best_match").to_pylist() == [False, True]

    assert check.check_type == "header_detected"
    assert check.severity == "info"
    assert check.status == "pass"
    assert check.metric_value == (1.0 if header else 0.0)
    assert check.details["first_line"] == (HEADER if header else ROWS[0])[:80]


def test_first_line_se_recorta_a_80_caracteres():
    _, check = read_zip(make_zip(("a.csv", csv_text(True))))
    assert len(check.details["first_line"]) == 80


def test_first_line_sin_salto_final_no_pierde_el_ultimo_byte():
    _, check = read_zip(make_zip(("a.csv", ROWS[0])))
    assert check.details["first_line"] == ROWS[0]


def test_rechaza_zip_con_dos_miembros():
    data = make_zip(("a.csv", csv_text(False)), ("b.csv", csv_text(False)))
    with pytest.raises(ValueError, match="exactamente un .csv"):
        read_zip(data)


def test_rechaza_zip_con_csv_y_otro_miembro():
    data = make_zip(("a.csv", csv_text(False)), ("LEEME.txt", "x"))
    with pytest.raises(ValueError, match="exactamente un .csv"):
        read_zip(data)


def test_rechaza_zip_sin_csv():
    with pytest.raises(ValueError, match="exactamente un .csv"):
        read_zip(make_zip(("a.txt", "x")))


def test_rechaza_nulos():
    bad = "1,1.0,1.0,1,1,1,True,\n"
    with pytest.raises((ValueError, pa.ArrowInvalid)):
        read_zip(make_zip(("a.csv", bad)))


def test_no_escribe_archivos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = make_zip(("a.csv", csv_text(True)))
    cwd = os.getcwd()
    before = set(os.listdir(tmp_path))
    read_zip(data)
    assert os.getcwd() == cwd
    assert set(os.listdir(tmp_path)) == before


def test_check_result_valida_enumerados():
    from l1_ingest.checks import CheckResult

    with pytest.raises(ValueError):
        CheckResult("x", "critico", "pass")
    ok = CheckResult("x", "info", "pass", 1.0)
    assert ok.details == {}
