"""`l3_path_check.py` contra un lago local con los archivos que escribe la CLI de L3.

`summaries.parquet` y `path.parquet` salen de la CLI de `l3_dc_descriptors` (`--family
summaries` y luego `--family path`) sobre el lago sintético de sus pruebas, que se construye una
sola vez; cada prueba copia a `tmp_path` los dos archivos de un θ y un mes (hive de L3) y, los
casos que deben fallar, alteran el `path.parquet` o el `summaries.parquet` ya copiados. La
sonda no importa ninguna capa; la prueba sí, para producir el lago.
"""

import csv
import importlib.util
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "layers/l3_dc_descriptors/tests"))

from cli_support import env_for, write_catalog
from l3_dc_descriptors.cli import main as l3_main
from l3_lake import build_lake, fixture_ticks, month_cuts

SERIES = "provider=binance/market=spot/asset=BTCUSDT"
THETA_RAW = 250_000
THETA = "0.00250000"
MONTH = "2017-08"
PARTITION = f"{SERIES}/theta={THETA}/year=2017/month=08"
COLUMNS = [
    "c_realized_var",
    "os_realized_var",
    "c_tortuosity",
    "os_tortuosity",
    "c_sign_autocorr",
    "os_sign_autocorr",
    "c_max_gap",
    "os_max_gap",
    "c_idle_fraction",
    "os_idle_fraction",
    "c_t25",
    "c_t50",
    "c_t75",
]
INVARIANTS = [
    "filas_path_igual_summaries",
    "union_1a1",
    "row_groups_iguales",
    "c_realized_var_no_negativo",
    "os_realized_var_no_negativo",
    "c_tortuosity_al_menos_1",
    "os_tortuosity_al_menos_1",
    "c_sign_autocorr_acotada",
    "os_sign_autocorr_acotada",
    "c_max_gap_acotado",
    "os_max_gap_acotado",
    "c_idle_fraction_en_0_1",
    "os_idle_fraction_en_0_1",
    "c_idle_fraction_cero_si_silencio_mayor",
    "c_t25_t50_t75_ordenados",
    *(f"nulos_{c}" for c in COLUMNS),
    "doubles_finitos",
]
LAST_LINE = re.compile(
    r"path check: mes=(?P<mes>\S+) theta=(?P<theta>\S+) "
    r"state_version=(?P<version>\S+) silence_us=(?P<silence>\S+) filas=(?P<filas>\S+) "
    r"violaciones=(?P<violaciones>\S+) pared_s=(?P<pared>\S+) veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_path_check", SCRIPTS / "l3_path_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check():
    return load()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """El `l3` de un lago de pruebas con las dos familias escritas por la CLI de L3."""
    base = tmp_path_factory.mktemp("lake")
    ticks = fixture_ticks()
    lake = build_lake(base, ticks, month_cuts(ticks, 1500, 3000))
    catalog = write_catalog(base / "thetas.yaml", lake.thetas)
    env = env_for(lake, base, catalog)
    backfill = ["--mode", "backfill", "--series-start", "2017-08"]
    for family in ("summaries", "path"):
        assert l3_main([*backfill, "--family", family], env) == 0
    return lake.l3


@pytest.fixture
def l3(built, tmp_path):
    """Una copia de la partición del θ y el mes en `tmp_path/l3`, que cada prueba altera."""
    root = tmp_path / "l3"
    (root / PARTITION).mkdir(parents=True)
    for name in ("summaries.parquet", "path.parquet"):
        shutil.copy(built / PARTITION / name, root / PARTITION / name)
    return root


def file(l3: Path, name: str) -> Path:
    return l3 / PARTITION / name


def argv(l3: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--theta",
        "0.0025",
        "--month",
        MONTH,
        "--summaries-root",
        str(l3),
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


def alter(path: Path, column: str, change) -> None:
    """Reescribe el archivo con `change(valores)` en `column`; conserva esquema y metadatos."""
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    field = table.schema.field(column).with_nullable(True)
    values = change(table[column].to_pylist())
    table = table.set_column(index, field, pa.array(values, field.type))
    pq.write_table(table, path, row_group_size=5)


def violations(rows) -> dict[str, int]:
    return {c: int(r["violaciones"]) for (s, c), r in rows.items() if s == "invariante"}


def first(l3: Path, condition) -> int:
    """La posición de la primera fila de `summaries` que cumple `condition`."""
    rows = pq.read_table(file(l3, "summaries.parquet")).to_pylist()
    return next(i for i, row in enumerate(rows) if condition(row))


def set_at(index: int, value):
    def change(values):
        values = list(values)
        values[index] = value
        return values

    return change


def test_csv_coincide_con_el_archivo_y_cero_violaciones(check, l3, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 0
    line = last_line(capsys)
    rows = read_csv(out)
    table = pq.read_table(file(l3, "path.parquet"))

    assert line["mes"] == MONTH
    assert line["theta"] == THETA
    assert line["version"] == "1.0.0"
    assert line["silence"] == "10000000"
    assert line["filas"] == str(table.num_rows) and table.num_rows > 5
    assert line["violaciones"] == "0"
    assert line["veredicto"] == "ok"
    assert float(line["pared"]) >= 0

    for column in COLUMNS:
        values = [v for v in table[column].to_pylist() if v is not None]
        got = rows["columna", column]
        assert int(got["filas"]) == table.num_rows
        assert int(got["nulos"]) == table.num_rows - len(values), column
        assert float(got["min"]) == pytest.approx(min(values), rel=1e-12), column
        assert float(got["mediana"]) == pytest.approx(
            statistics.median(values), rel=1e-12
        ), column
        assert float(got["max"]) == pytest.approx(max(values), rel=1e-12), column

    assert [c for (s, c) in rows if s == "invariante"] == INVARIANTS
    assert set(violations(rows).values()) == {0}


def test_el_lago_ejerce_los_casos_de_la_regla_de_nulos(l3):
    """Sin estos casos en el lago, «cero violaciones» no probaría la regla de nulos."""
    summaries = pq.read_table(file(l3, "summaries.parquet")).to_pylist()
    path = pq.read_table(file(l3, "path.parquet")).to_pylist()
    assert any(s["os_ticks"] == 0 for s in summaries)
    assert any(s["os_ticks"] > 0 for s in summaries)
    assert any(s["c_duration"] == 0 for s in summaries)
    assert any(p["c_sign_autocorr"] is None for p in path)
    assert any(p["c_sign_autocorr"] is not None for p in path)
    # Hay filas con ticks de un solo lado y más de uno: la nulidad no es solo `ticks < 2`.
    assert any(
        s["c_ticks"] >= 2 and 0 in (s["c_quantity_buy"], s["c_quantity_sell"])
        for s in summaries
    )
    assert any(s["c_duration"] > 0 for s in summaries)
    assert any(0 < s["c_duration"] < 10_000_000 for s in summaries)


def test_tortuosidad_de_overshoot_nula_solo_si_el_extremo_es_la_confirmacion(
    check, l3, tmp_path, capsys
):
    """El lago no trae `p_E = p_C`: se fabrica igualando el extremo del overshoot al de la
    confirmación en `summaries` (la condición necesaria que la sonda ve) y anulando T-04."""
    index = first(
        l3,
        lambda s: (
            s["os_ticks"] > 0
            and s["os_price_max"] != s["c_price_max"]
            and s["os_price_min"] != s["c_price_min"]
        ),
    )
    alter(file(l3, "path.parquet"), "os_tortuosity", set_at(index, None))
    out = tmp_path / "salida.csv"
    # Con el overshoot aún distinto de la confirmación, la nula es una violación.
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["nulos_os_tortuosity"] == 1

    summaries = file(l3, "summaries.parquet")
    c_max = pq.read_table(summaries)["c_price_max"].to_pylist()
    alter(
        summaries,
        "os_price_max",
        lambda values: [
            c_max[index] if i == index else v for i, v in enumerate(values)
        ],
    )
    assert check.main(argv(l3, out)) == 0
    capsys.readouterr()


def test_no_importa_ninguna_capa():
    source = (SCRIPTS / "l3_path_check.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]


def test_fila_quitada_de_path_da_violaciones(check, l3, tmp_path, capsys):
    path = file(l3, "path.parquet")
    pq.write_table(pq.read_table(path).slice(1), path, row_group_size=5)
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    line = last_line(capsys)
    got = violations(read_csv(out))
    assert line["veredicto"] == "violaciones"
    assert got["filas_path_igual_summaries"] == 1
    assert got["union_1a1"] == 1
    # Cada row group de path sale una fila corrido respecto del de summaries.
    assert got["row_groups_iguales"] >= 1
    assert int(line["violaciones"]) == sum(got.values())


def test_row_groups_distintos_con_las_mismas_filas_dan_violacion(
    check, l3, tmp_path, capsys
):
    path = file(l3, "path.parquet")
    pq.write_table(pq.read_table(path), path, row_group_size=7)
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["row_groups_iguales"] >= 1
    assert sum(got.values()) == got["row_groups_iguales"]


def test_tortuosidad_bajo_uno_da_violacion(check, l3, tmp_path, capsys):
    alter(file(l3, "path.parquet"), "c_tortuosity", set_at(0, 0.5))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["c_tortuosity_al_menos_1"] == 1
    assert sum(got.values()) == 1


def test_nulo_donde_la_regla_dice_no_nulo_da_violacion(check, l3, tmp_path, capsys):
    index = first(l3, lambda s: s["c_duration"] > 0)
    alter(file(l3, "path.parquet"), "c_t50", set_at(index, None))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["nulos_c_t50"] == 1
    assert sum(got.values()) == 1


def test_valor_donde_la_regla_dice_nulo_da_violacion(check, l3, tmp_path, capsys):
    index = first(l3, lambda s: s["os_ticks"] == 0)
    alter(file(l3, "path.parquet"), "os_tortuosity", set_at(index, 2.0))
    alter(file(l3, "path.parquet"), "os_max_gap", set_at(index, 0))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["nulos_os_tortuosity"] == 1
    assert got["nulos_os_max_gap"] == 1


def test_autocorrelacion_nula_con_ticks_de_ambos_lados_da_violacion(
    check, l3, tmp_path, capsys
):
    s = pq.read_table(file(l3, "summaries.parquet")).to_pylist()
    p = pq.read_table(file(l3, "path.parquet")).to_pylist()
    index = next(
        i
        for i, (a, b) in enumerate(zip(s, p, strict=True))
        if b["c_sign_autocorr"] is not None
    )
    alter(file(l3, "path.parquet"), "c_sign_autocorr", set_at(index, None))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["nulos_c_sign_autocorr"] == 1


def test_autocorrelacion_con_todos_los_ticks_del_mismo_lado_puede_ser_nula(
    check, l3, tmp_path, capsys
):
    """La regla del TRD: nula también con todos los ticks del mismo lado, no solo con < 2."""
    s = pq.read_table(file(l3, "summaries.parquet")).to_pylist()
    p = pq.read_table(file(l3, "path.parquet")).to_pylist()
    one_sided = [
        i
        for i, (a, b) in enumerate(zip(s, p, strict=True))
        if a["c_ticks"] >= 2 and 0 in (a["c_quantity_buy"], a["c_quantity_sell"])
    ]
    assert one_sided
    for i in one_sided:
        assert p[i]["c_sign_autocorr"] is None
    assert check.main(argv(l3, tmp_path / "salida.csv")) == 0
    capsys.readouterr()


def test_inf_y_nan_dan_violaciones(check, l3, tmp_path, capsys):
    alter(file(l3, "path.parquet"), "c_realized_var", set_at(0, math.inf))
    alter(file(l3, "path.parquet"), "c_tortuosity", set_at(1, math.nan))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["doubles_finitos"] == 2


def test_hueco_mayor_que_la_duracion_da_violacion(check, l3, tmp_path, capsys):
    index = first(l3, lambda s: s["c_duration"] > 0)
    duration = pq.read_table(file(l3, "summaries.parquet"))["c_duration"][index].as_py()
    alter(file(l3, "path.parquet"), "c_max_gap", set_at(index, duration + 1))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["c_max_gap_acotado"] == 1


def test_fraccion_ociosa_con_silencio_mayor_que_la_duracion_da_violacion(
    check, l3, tmp_path, capsys
):
    index = first(l3, lambda s: 0 < s["c_duration"] < 10_000_000)
    alter(file(l3, "path.parquet"), "c_idle_fraction", set_at(index, 0.5))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["c_idle_fraction_cero_si_silencio_mayor"] == 1


def test_niveles_desordenados_dan_violacion(check, l3, tmp_path, capsys):
    index = first(l3, lambda s: s["c_duration"] > 0)
    alter(file(l3, "path.parquet"), "c_t25", set_at(index, 1.0))
    alter(file(l3, "path.parquet"), "c_t75", set_at(index, 0.0))
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["c_t25_t50_t75_ordenados"] == 1


def test_clave_repetida_en_path_falla_la_union(check, l3, tmp_path, capsys):
    path = file(l3, "path.parquet")
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path, row_group_size=5)
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["union_1a1"] >= 1
    assert got["filas_path_igual_summaries"] == 1


def test_theta_sin_archivos_es_fallo_con_codigo_1(check, l3, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    code = check.main([*argv(l3, out)[:1], "0.05", *argv(l3, out)[2:]])
    text = capsys.readouterr().out
    assert code == 1
    assert "falta el archivo" in text
    assert "path.parquet" in text and "summaries.parquet" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


def test_falta_solo_summaries_es_fallo(check, l3, tmp_path, capsys):
    file(l3, "summaries.parquet").unlink()
    assert check.main(argv(l3, tmp_path / "salida.csv")) == 1
    text = capsys.readouterr().out
    assert "summaries.parquet" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_error_de_e_s_al_abrir_un_archivo_es_fallo_con_codigo_1(
    check, l3, tmp_path, capsys, monkeypatch
):
    def denied(uri):
        raise PermissionError(f"403 {uri}")

    monkeypatch.setattr(check, "open_dataset", denied)
    out = tmp_path / "salida.csv"
    assert check.main(argv(l3, out)) == 1
    text = capsys.readouterr().out
    assert "no se pudo abrir" in text and "PermissionError" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


def test_path_sin_silence_us_en_los_metadatos_es_fallo(check, l3, tmp_path, capsys):
    path = file(l3, "path.parquet")
    table = pq.read_table(path)
    pq.write_table(
        table.replace_schema_metadata({"state_version": "1.0.0"}),
        path,
        row_group_size=5,
    )
    assert check.main(argv(l3, tmp_path / "salida.csv")) == 1
    text = capsys.readouterr().out
    assert "silence_us" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_out_directorio_usa_el_nombre_por_defecto(check, l3, tmp_path, capsys):
    out = tmp_path / "resultados"
    assert check.main(argv(l3, tmp_path / "x.csv")[:-2] + ["--out", f"{out}/"]) == 0
    capsys.readouterr()
    assert (out / f"l3_path_check_{MONTH}_{THETA}.csv").is_file()


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
        check.main(["--summaries-root", str(tmp_path), *extra])
    assert raised.value.code == 2


def test_sin_raiz_y_sin_ops_results_uri_es_uso(check, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        check.main(["--theta", "0.01", "--month", "2026-09"])
    assert raised.value.code == 2


def test_out_fuera_de_results_es_uso(check, l3, tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proyecto-ops/results/ejecucion/")
    with pytest.raises(SystemExit) as raised:
        check.main(argv(l3, tmp_path / "x.csv")[:-2] + ["--out", "gs://otro/x.csv"])
    assert raised.value.code == 2
