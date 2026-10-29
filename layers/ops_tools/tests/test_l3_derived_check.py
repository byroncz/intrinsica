"""`l3_derived_check.py` contra un lago local con las tablas del fixture de `dc_views`.

Las filas A a F y H del fixture (θ = 0,01) van en la partición 2026-09 de un lago en
`tmp_path`, con la disposición hive de L2 y L3; G (θ = 0,02) va en otra. La sonda no importa
`dc_views`; la prueba sí, porque `load_derived` es el cálculo contra el que se compara el CSV.
Los casos que deben fallar alteran el `summaries.parquet` ya escrito.
"""

import csv
import importlib.util
import re
import statistics
import sys
from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_views import DERIVED_VERSION, load_derived

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "shared/dc_views/tests"))

from test_dc_views_derived import (
    COLUMNS_D,
    EVENTS_SCHEMA,
    FLAGS,
    ROWS,
    SUMMARIES_SCHEMA,
    _table,
)

SERIES = "provider=binance/market=spot/asset=BTCUSDT"
THETA = "0.01000000"
IN_MONTH = "ABCDEFH"
DOUBLES = COLUMNS_D[:16]
INVARIANTS = [
    "filas_vista_igual_summaries",
    "union_1a1",
    "c_flow_imbalance_acotado",
    "os_flow_imbalance_acotado",
    "c_range_no_negativo",
    "c_instant_bias_no_negativo",
    "cota_de_confirmacion",
    "atomic_sweep_igual_c_duration_cero",
    "atomic_sweep_igual_c_tick_rate_nulo",
    "empty_overshoot_igual_os_nulos",
    "banderas_sin_nulos",
    "columnas_nunca_nulas",
    "doubles_finitos",
]
LAST_LINE = re.compile(
    r"derived check: mes=(?P<mes>\S+) theta=(?P<theta>\S+) "
    r"derived_version=(?P<version>\S+) filas=(?P<filas>\S+) "
    r"atomic_sweep=(?P<atomic>\S+) empty_overshoot=(?P<empty>\S+) "
    r"provider_gap=(?P<gap>\S+) violaciones=(?P<violaciones>\S+) "
    r"pared_s=(?P<pared>\S+) veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_derived_check", SCRIPTS / "l3_derived_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check():
    return load()


def partition(root: Path, theta: str, month: int = 9) -> Path:
    directory = root / SERIES / f"theta={theta}/year=2026/month={month:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def lake(tmp_path):
    """`(l3, l2)`: la partición del mes con siete eventos y la de otro θ con uno."""
    l3, l2 = tmp_path / "l3", tmp_path / "l2"
    for root, name, schema, side in (
        (l3, "summaries", SUMMARIES_SCHEMA, 0),
        (l2, "events", EVENTS_SCHEMA, 1),
    ):
        pq.write_table(
            _table(schema, [ROWS[r][side] for r in IN_MONTH]),
            partition(root, THETA) / f"{name}.parquet",
        )
        pq.write_table(
            _table(schema, [ROWS["G"][side]]),
            partition(root, "0.02000000") / f"{name}.parquet",
        )
    return l3, l2


def argv(lake, out: Path, *extra: str) -> list[str]:
    l3, l2 = lake
    return [
        "--theta",
        "0.01",
        "--month",
        "2026-09",
        "--summaries-root",
        str(l3),
        "--events-root",
        str(l2),
        "--out",
        str(out),
        *extra,
    ]


def last_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    match = LAST_LINE.match(line)
    assert match, line
    return match.groupdict()


def read_csv(path: Path) -> dict[tuple[str, str], dict]:
    with open(path) as f:
        return {(r["seccion"], r["columna"]): r for r in csv.DictReader(f)}


def expected_rows():
    """Las filas de D que `load_derived` calcula sobre las mismas tablas del mes."""
    con = duckdb.connect()
    load_derived(
        con,
        _table(SUMMARIES_SCHEMA, [ROWS[r][0] for r in IN_MONTH]),
        _table(EVENTS_SCHEMA, [ROWS[r][1] for r in IN_MONTH]),
    )
    names = [c for c in COLUMNS_D]
    rows = con.execute(f"SELECT {', '.join(names)} FROM derived").fetchall()
    con.close()
    return [dict(zip(names, row, strict=True)) for row in rows]


def test_csv_coincide_con_load_derived_cero_violaciones(check, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    line = last_line(capsys)
    rows = read_csv(out)
    derived = expected_rows()

    assert line["mes"] == "2026-09"
    assert line["theta"] == THETA
    assert line["version"] == DERIVED_VERSION
    assert line["filas"] == str(len(IN_MONTH))
    assert line["violaciones"] == "0"
    assert line["veredicto"] == "ok"
    assert float(line["pared"]) >= 0

    for column in DOUBLES:
        values = [r[column] for r in derived if r[column] is not None]
        got = rows["doble", column]
        assert int(got["filas"]) == len(IN_MONTH)
        assert int(got["nulos"]) == len(IN_MONTH) - len(values)
        assert float(got["min"]) == pytest.approx(min(values), rel=1e-12), column
        assert float(got["mediana"]) == pytest.approx(
            statistics.median(values), rel=1e-12
        ), column
        assert float(got["max"]) == pytest.approx(max(values), rel=1e-12), column

    flag_names = {
        "atomic_sweep": "atomic",
        "empty_overshoot": "empty",
        "provider_gap": "gap",
    }
    for flag in FLAGS:
        got = rows["bandera", flag]
        trues = sum(r[flag] for r in derived)
        assert int(got["filas"]) == len(IN_MONTH)
        assert int(got["verdaderas"]) == trues
        assert line[flag_names[flag]] == str(trues)
    # Cruces atómicos: C y F; overshoot vacío: D y F; hueco del proveedor: E.
    assert [line[k] for k in ("atomic", "empty", "gap")] == ["2", "2", "1"]

    assert [c for (s, c) in rows if s == "invariante"] == INVARIANTS
    assert {rows["invariante", n]["violaciones"] for n in INVARIANTS} == {"0"}


def test_no_importa_dc_views_ni_ninguna_capa():
    source = (SCRIPTS / "l3_derived_check.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]


def test_fila_quitada_de_summaries_da_violaciones(check, lake, tmp_path, capsys):
    path = partition(lake[0], THETA) / "summaries.parquet"
    pq.write_table(pq.read_table(path).slice(1), path)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    line = last_line(capsys)
    assert line["veredicto"] == "violaciones"
    assert line["filas"] == str(len(IN_MONTH) - 1)
    rows = read_csv(out)
    # El evento de la fila quitada queda sin su summaries; la vista y summaries coinciden.
    assert rows["invariante", "union_1a1"]["violaciones"] == "1"
    assert rows["invariante", "filas_vista_igual_summaries"]["violaciones"] == "0"
    assert line["violaciones"] == "1"


def test_clave_repetida_en_summaries_falla_union_sin_inflar_filas(
    check, lake, tmp_path, capsys
):
    path = partition(lake[0], THETA) / "summaries.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    line = last_line(capsys)
    rows = read_csv(out)
    # Las cuentas salen de derived sola: la clave repetida suma una fila, no un producto.
    assert line["filas"] == str(len(IN_MONTH) + 1)
    assert rows["invariante", "union_1a1"]["violaciones"] == "1"
    assert rows["invariante", "filas_vista_igual_summaries"]["violaciones"] == "0"
    assert rows["bandera", "atomic_sweep"]["filas"] == str(len(IN_MONTH) + 1)


def test_cantidad_negada_saca_el_flujo_de_cero_a_uno(check, lake, tmp_path, capsys):
    path = partition(lake[0], THETA) / "summaries.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index("c_quantity_buy")
    negated = [
        -v if v is not None else None for v in table["c_quantity_buy"].to_pylist()
    ]
    table = table.set_column(
        index,
        table.schema.field("c_quantity_buy"),
        pa.array(negated, table.schema.field("c_quantity_buy").type),
    )
    pq.write_table(table, path)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    line = last_line(capsys)
    assert line["veredicto"] == "violaciones"
    rows = read_csv(out)
    # Con la compra negada, (−b − s) / (−b + s) sale de [−1, 1] en las filas con b > s.
    assert int(rows["invariante", "c_flow_imbalance_acotado"]["violaciones"]) >= 1
    assert int(line["violaciones"]) >= 1


def test_cota_de_confirmacion_detecta_un_maximo_bajo_el_umbral(
    check, lake, tmp_path, capsys
):
    path = partition(lake[0], THETA) / "summaries.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index("c_price_max")
    prices = table["c_price_max"].to_pylist()
    prices[0] = Decimal("100.5")  # A es un upturn desde 100 con θ = 0,01
    table = table.set_column(
        index,
        table.schema.field("c_price_max"),
        pa.array(prices, table.schema.field("c_price_max").type),
    )
    pq.write_table(table, path)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    rows = read_csv(out)
    assert rows["invariante", "cota_de_confirmacion"]["violaciones"] == "1"


def test_theta_sin_archivos_es_fallo_con_codigo_1(check, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    code = check.main([*argv(lake, out)[:1], "0.05", *argv(lake, out)[2:]])
    text = capsys.readouterr().out
    assert code == 1
    assert "falta el archivo" in text
    assert "summaries.parquet" in text and "events.parquet" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


def test_falta_solo_events_es_fallo(check, lake, tmp_path, capsys):
    (partition(lake[1], THETA) / "events.parquet").unlink()
    assert check.main(argv(lake, tmp_path / "salida.csv")) == 1
    text = capsys.readouterr().out
    assert "events.parquet" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_error_de_e_s_al_abrir_un_archivo_es_fallo_con_codigo_1(
    check, lake, tmp_path, capsys, monkeypatch
):
    def denied(uri):
        raise PermissionError(f"403 {uri}")

    monkeypatch.setattr(check, "open_dataset", denied)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    text = capsys.readouterr().out
    assert "no se pudo abrir" in text and "PermissionError" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


def test_sql_sin_cabecera_es_fallo(check, lake, tmp_path, capsys):
    sql = tmp_path / "derived.sql"
    sql.write_text("SELECT 1;")
    assert check.main(argv(lake, tmp_path / "salida.csv", "--sql", str(sql))) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_sql_explicito_da_su_version(check, lake, tmp_path, capsys):
    source = (ROOT / "shared/dc_views/src/dc_views/derived.sql").read_text()
    sql = tmp_path / "derived.sql"
    sql.write_text(
        source.replace("derived_version: 1.0.0", "derived_version: 9.9.9", 1)
    )
    assert check.main(argv(lake, tmp_path / "salida.csv", "--sql", str(sql))) == 0
    assert last_line(capsys)["version"] == "9.9.9"


def test_out_directorio_usa_el_nombre_por_defecto(check, lake, tmp_path, capsys):
    out = tmp_path / "resultados"
    assert check.main(argv(lake, tmp_path / "x.csv")[:-2] + ["--out", f"{out}/"]) == 0
    capsys.readouterr()
    assert (out / f"l3_derived_check_2026-09_{THETA}.csv").is_file()


@pytest.mark.parametrize(
    "extra",
    [
        ["--theta", "0.01"],  # falta --month
        ["--month", "2026-09"],  # falta --theta
        ["--theta", "0.01", "--month", "2026-13"],
        ["--theta", "0.01", "--month", "2026-9"],
        ["--theta", "0", "--month", "2026-09"],
        ["--theta", "-0.01", "--month", "2026-09"],
        ["--theta", "nan", "--month", "2026-09"],
        ["--theta", "abc", "--month", "2026-09"],
        ["--theta", "0.000000001", "--month", "2026-09"],
    ],
)
def test_args_invalidos_dan_codigo_2(check, extra, tmp_path):
    with pytest.raises(SystemExit) as raised:
        check.main(
            ["--summaries-root", str(tmp_path), "--events-root", str(tmp_path), *extra]
        )
    assert raised.value.code == 2


def test_sin_raices_y_sin_ops_results_uri_es_uso(check, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        check.main(["--theta", "0.01", "--month", "2026-09"])
    assert raised.value.code == 2


def test_out_fuera_de_results_es_uso(check, lake, tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proyecto-ops/results/ejecucion/")
    with pytest.raises(SystemExit) as raised:
        check.main(argv(lake, tmp_path / "x.csv")[:-2] + ["--out", "gs://otro/x.csv"])
    assert raised.value.code == 2
