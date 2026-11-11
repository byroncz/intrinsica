"""`l3_spectrum_check.py` contra un lago local con los archivos que escribe la CLI de L3.

`summaries.parquet` y `spectrum.parquet` salen de la CLI de `l3_dc_descriptors` (`--family
summaries` y luego `--family spectrum`) sobre el lago sintético de sus pruebas, que se construye una
sola vez; el `events.parquet` es el de L2 de ese mismo lago. Cada prueba copia a `tmp_path` los
tres archivos de un θ y un mes (hive de L3 y de L2) y los casos que deben fallar alteran el
`spectrum.parquet`, el `summaries.parquet` o el `events.parquet` ya copiados. La sonda no importa
ninguna capa; la prueba sí, para producir el lago.
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
THETA = "0.00250000"
MONTH = "2017-08"
PARTITION = f"{SERIES}/theta={THETA}/year=2017/month=08"
N_MIN = 16
BANDS = 7
REGISTERS = ("fourier", "haar")
PHASES = ("c", "os")
UNIT = {"fourier": "band", "haar": "level"}
# Las 28 columnas de datos de F en el orden del archivo (TRD-L3 §7.2.3).
COLUMNS = [
    f"{phase}_{register}_{UNIT[register]}_{b}"
    for register in REGISTERS
    for phase in PHASES
    for b in range(1, BANDS + 1)
]
INVARIANTS = [
    "filas_spectrum_igual_summaries",
    "union_1a1",
    "union_events_1a1",
    "row_groups_iguales",
    "fracciones_en_0_1",
    *(f"suma_bandas_{p}_{r}" for r in REGISTERS for p in PHASES),
    "nulos_presupuesto",
    *(
        f"nulos_{p}_{r}_{kind}"
        for r in REGISTERS
        for p in PHASES
        for kind in ("valor_donde_nulo", "nulo_donde_valor")
    ),
    "doubles_finitos",
]
INFORMATIVE = [
    "sobre_presupuesto",
    *(f"energia_cero_{p}_{r}" for r in REGISTERS for p in PHASES),
    *(f"fase_corta_{p}" for p in PHASES),
]
LAST_LINE = re.compile(
    r"spectrum check: mes=(?P<mes>\S+) theta=(?P<theta>\S+) "
    r"state_version=(?P<version>\S+) filas=(?P<filas>\S+) "
    r"violaciones=(?P<violaciones>\S+) over_budget=(?P<over>\S+) "
    r"pared_s=(?P<pared>\S+) veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_spectrum_check", SCRIPTS / "l3_spectrum_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check():
    return load()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """El `l2` y el `l3` de un lago de pruebas con S y F escritas por la CLI."""
    base = tmp_path_factory.mktemp("lake")
    ticks = fixture_ticks()
    lake = build_lake(base, ticks, month_cuts(ticks, 1500, 3000))
    catalog = write_catalog(base / "thetas.yaml", lake.thetas)
    env = env_for(lake, base, catalog)
    backfill = ["--mode", "backfill", "--series-start", "2017-08"]
    for family in ("summaries", "spectrum"):
        assert l3_main([*backfill, "--family", family], env) == 0
    return lake.l2, lake.l3


@pytest.fixture
def lake(built, tmp_path):
    """Una copia de la partición del θ y el mes en `tmp_path`, que cada prueba altera."""
    l2, l3 = built
    for root, source, names in (
        (tmp_path / "l2", l2, ("events.parquet",)),
        (tmp_path / "l3", l3, ("summaries.parquet", "spectrum.parquet")),
    ):
        (root / PARTITION).mkdir(parents=True)
        for name in names:
            shutil.copy(source / PARTITION / name, root / PARTITION / name)
    return tmp_path / "l2", tmp_path / "l3"


def file(lake, name: str) -> Path:
    l2, l3 = lake
    return (l2 if name == "events.parquet" else l3) / PARTITION / name


def argv(lake, out: Path, *extra: str) -> list[str]:
    l2, l3 = lake
    return [
        "--theta",
        "0.0025",
        "--month",
        MONTH,
        "--summaries-root",
        str(l3),
        "--l2-root",
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


def section(rows, name: str, column: str) -> dict[str, int]:
    return {c: int(r[column]) for (s, c), r in rows.items() if s == name}


def violations(rows) -> dict[str, int]:
    return section(rows, "invariante", "violaciones")


def informative(rows) -> dict[str, int]:
    return section(rows, "informativo", "conteo")


def rewrite(path: Path, table: pa.Table, **meta: str) -> None:
    """Escribe `table` con las filas por row group del lago (5) y con `meta` en el pie."""
    if meta:
        table = table.replace_schema_metadata(
            {
                **{k.decode(): v.decode() for k, v in table.schema.metadata.items()},
                **meta,
            }
        )
    pq.write_table(table, path, row_group_size=5)


def alter(path: Path, column: str, change) -> None:
    """Reescribe el archivo con `change(valores)` en `column`; conserva esquema y metadatos."""
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    field = table.schema.field(column).with_nullable(True)
    values = change(table[column].to_pylist())
    rewrite(path, table.set_column(index, field, pa.array(values, field.type)))


def set_at(index: int, value):
    def change(values):
        values = list(values)
        values[index] = value
        return values

    return change


def put(lake, index: int, **cells) -> None:
    """Pone `cells` (columna=valor) en la fila `index` de `spectrum.parquet`."""
    for column, value in cells.items():
        alter(file(lake, "spectrum.parquet"), column, set_at(index, value))


def joined(lake) -> list[tuple[dict, dict, dict]]:
    """(spectrum, summaries, events) de cada fila; los tres archivos tienen el mismo orden."""
    spectrum = pq.read_table(file(lake, "spectrum.parquet")).to_pylist()
    summaries = pq.read_table(file(lake, "summaries.parquet")).to_pylist()
    events = pq.read_table(file(lake, "events.parquet")).to_pylist()
    return list(zip(spectrum, summaries, events, strict=True))


def first(lake, condition) -> int:
    """La posición de la primera fila que cumple `condition(spectrum, summaries, events)`."""
    return next(i for i, row in enumerate(joined(lake)) if condition(*row))


def ticks(s: dict, phase: str) -> int:
    return s[f"{phase}_ticks"]


def span(e: dict) -> int:
    return e["extreme_agg_trade_id"] - e["reference_agg_trade_id"]


def group(sp: dict, phase: str, register: str) -> list:
    return [sp[f"{phase}_{register}_{UNIT[register]}_{b}"] for b in range(1, BANDS + 1)]


def column(phase: str, register: str, b: int) -> str:
    return f"{phase}_{register}_{UNIT[register]}_{b}"


def run_ok(check, lake, tmp_path, capsys, expected_code: int) -> dict:
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == expected_code
    last_line(capsys)
    return read_csv(out)


def test_csv_coincide_con_el_archivo_y_cero_violaciones(check, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    line = last_line(capsys)
    rows = read_csv(out)
    table = pq.read_table(file(lake, "spectrum.parquet"))

    assert line["mes"] == MONTH
    assert line["theta"] == THETA
    assert line["version"] == "1.0.0"
    assert line["filas"] == str(table.num_rows) and table.num_rows > 5
    assert line["violaciones"] == "0"
    assert line["over"] == "0"
    assert line["veredicto"] == "ok"
    assert float(line["pared"]) >= 0

    assert [c for (s, c) in rows if s == "columna"] == COLUMNS
    for name in COLUMNS:
        values = [v for v in table[name].to_pylist() if v is not None]
        got = rows["columna", name]
        assert int(got["filas"]) == table.num_rows
        assert int(got["nulos"]) == table.num_rows - len(values), name
        if values:
            assert float(got["min"]) == pytest.approx(min(values), rel=1e-12), name
            assert float(got["mediana"]) == pytest.approx(
                statistics.median(values), rel=1e-12
            ), name
            assert float(got["max"]) == pytest.approx(max(values), rel=1e-12), name
        else:
            assert got["min"] == got["mediana"] == got["max"] == ""
    assert [c for (s, c) in rows if s == "invariante"] == INVARIANTS
    assert [c for (s, c) in rows if s == "informativo"] == INFORMATIVE
    assert set(violations(rows).values()) == {0}


def test_el_lago_ejerce_los_casos_de_la_regla_de_nulos(lake):
    """Sin estos casos en el lago, «cero violaciones» no probaría la regla de nulos."""
    rows = joined(lake)
    for phase in PHASES:
        # Una fase corta, con todo nulo; una de 16 a 31 retornos (sin las bandas 5 a 7) y una de
        # 32 a 63 (con la 5 y sin la 6 y la 7).
        assert any(
            0 < ticks(s, phase) < N_MIN
            and all(v is None for v in group(sp, phase, "fourier"))
            for sp, s, _ in rows
        )
        assert any(16 <= ticks(s, phase) < 32 for _, s, _ in rows)
        assert any(32 <= ticks(s, phase) < 64 for _, s, _ in rows)
    assert any(ticks(s, "os") == 0 for _, s, _ in rows)
    for sp, s, _ in rows:
        for phase in PHASES:
            for register in REGISTERS:
                values = group(sp, phase, register)
                assert [v is not None for v in values[4:]] == [
                    ticks(s, phase) >= 2**b and ticks(s, phase) >= N_MIN
                    for b in (5, 6, 7)
                ] or all(v is None for v in values)
    assert any(sp["c_fourier_band_5"] is not None for sp, _, _ in rows)
    assert any(sp["os_haar_level_5"] is not None for sp, _, _ in rows)
    assert any(sp["c_fourier_band_4"] is not None for sp, _, _ in rows)
    assert any(
        sp["c_fourier_band_5"] is None for sp, s, _ in rows if ticks(s, "c") >= 16
    )


def test_las_cifras_informativas_coinciden_con_un_oraculo_en_python(
    check, lake, tmp_path, capsys
):
    got = informative(run_ok(check, lake, tmp_path, capsys, 0))
    expected = dict.fromkeys(INFORMATIVE, 0)
    for sp, s, e in joined(lake):
        expected["sobre_presupuesto"] += span(e) > 20_000_000
        for phase in PHASES:
            expected[f"fase_corta_{phase}"] += ticks(s, phase) < N_MIN
            for register in REGISTERS:
                expected[f"energia_cero_{phase}_{register}"] += ticks(
                    s, phase
                ) >= N_MIN and all(v is None for v in group(sp, phase, register))
    assert got == expected
    assert expected["fase_corta_c"] > 0 and expected["fase_corta_os"] > 0


def test_no_importa_ninguna_capa():
    source = (SCRIPTS / "l3_spectrum_check.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]


def test_fila_quitada_de_spectrum_da_violaciones(check, lake, tmp_path, capsys):
    spectrum = file(lake, "spectrum.parquet")
    rewrite(spectrum, pq.read_table(spectrum).slice(1))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    line = last_line(capsys)
    got = violations(read_csv(out))
    assert line["veredicto"] == "violaciones"
    assert got["filas_spectrum_igual_summaries"] == 1
    assert got["union_1a1"] == 1
    assert got["union_events_1a1"] == 1
    # Cada row group de spectrum sale una fila corrido respecto del de summaries.
    assert got["row_groups_iguales"] >= 1
    assert int(line["violaciones"]) == sum(got.values())


def test_row_groups_distintos_con_las_mismas_filas_dan_violacion(
    check, lake, tmp_path, capsys
):
    spectrum = file(lake, "spectrum.parquet")
    pq.write_table(pq.read_table(spectrum), spectrum, row_group_size=7)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["row_groups_iguales"] >= 1
    assert sum(got.values()) == got["row_groups_iguales"]


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_una_fraccion_fuera_de_0_1_da_violacion(check, lake, tmp_path, capsys, value):
    index = first(lake, lambda sp, s, e: sp["c_fourier_band_1"] is not None)
    put(lake, index, c_fourier_band_1=value)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["fracciones_en_0_1"] == 1
    # La suma de la fase deja de ser 1 por el mismo valor.
    assert got["suma_bandas_c_fourier"] == 1
    assert sum(got.values()) == 2


def test_una_fase_cuyas_bandas_no_suman_1_da_violacion(check, lake, tmp_path, capsys):
    index = first(lake, lambda sp, s, e: (sp["c_haar_level_1"] or 0) > 0.1)
    value = joined(lake)[index][0]["c_haar_level_1"] / 2
    put(lake, index, c_haar_level_1=value)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["suma_bandas_c_haar"] == 1
    assert sum(got.values()) == 1


def test_la_suma_tolera_1e_9_y_no_mas(check, lake, tmp_path, capsys):
    index = first(lake, lambda sp, s, e: (sp["os_fourier_band_1"] or 0) > 0.1)
    base = joined(lake)[index][0]["os_fourier_band_1"]
    put(lake, index, os_fourier_band_1=base + 5e-10)
    assert set(violations(run_ok(check, lake, tmp_path, capsys, 0)).values()) == {0}
    put(lake, index, os_fourier_band_1=base + 5e-9)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["suma_bandas_os_fourier"] == 1
    assert sum(got.values()) == 1


def test_un_nulo_donde_la_regla_dice_no_nulo_da_violacion(
    check, lake, tmp_path, capsys
):
    index = first(lake, lambda sp, s, e: (sp["c_fourier_band_2"] or 0) > 0.01)
    put(lake, index, c_fourier_band_2=None)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["nulos_c_fourier_nulo_donde_valor"] == 1
    # Falta una de las bandas, así que la fase tampoco suma 1.
    assert got["suma_bandas_c_fourier"] == 1
    assert sum(got.values()) == 2


def test_un_valor_en_un_overshoot_vacio_da_violacion(check, lake, tmp_path, capsys):
    index = first(lake, lambda sp, s, e: ticks(s, "os") == 0)
    put(lake, index, os_fourier_band_1=1.0)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["nulos_os_fourier_valor_donde_nulo"] == 1
    assert sum(got.values()) == 1


def test_un_valor_en_una_fase_corta_da_violacion(check, lake, tmp_path, capsys):
    index = first(lake, lambda sp, s, e: 0 < ticks(s, "c") < N_MIN)
    put(lake, index, c_haar_level_1=1.0)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["nulos_c_haar_valor_donde_nulo"] == 1
    assert sum(got.values()) == 1


@pytest.mark.parametrize(
    ("low", "high", "register", "band"),
    [(16, 32, "fourier", 5), (16, 32, "haar", 6), (32, 64, "fourier", 6)],
)
def test_una_banda_sin_bins_con_valor_da_violacion(
    check, lake, tmp_path, capsys, low, high, register, band
):
    """Con 16 a 31 retornos no hay bins en la banda 5 ni en las siguientes; con 32 a 63, en la 6
    ni en la 7. Un `0.0` no cambia la suma de la fase: solo viola la regla de nulos."""
    index = first(
        lake, lambda sp, s, e: low <= ticks(s, "c") < high and sp["c_fourier_band_1"]
    )
    put(lake, index, **{column("c", register, band): 0.0})
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got[f"nulos_c_{register}_valor_donde_nulo"] == 1
    assert sum(got.values()) == 1


def test_una_banda_con_bins_en_la_frontera_no_es_violacion(
    check, lake, tmp_path, capsys
):
    """Con 32 retornos o más la banda 5 tiene bins: es no nula."""
    index = first(lake, lambda sp, s, e: 32 <= ticks(s, "os") < 64)
    row = joined(lake)[index][0]
    assert row["os_fourier_band_5"] is not None and row["os_fourier_band_6"] is None
    assert set(violations(run_ok(check, lake, tmp_path, capsys, 0)).values()) == {0}


def test_un_evento_con_toda_la_fase_plana_es_informativo_y_no_violacion(
    check, lake, tmp_path, capsys
):
    index = first(lake, lambda sp, s, e: sp["c_haar_level_1"] is not None)
    base = informative(run_ok(check, lake, tmp_path, capsys, 0))
    put(lake, index, **{column("c", "haar", b): None for b in range(1, BANDS + 1)})
    rows = run_ok(check, lake, tmp_path, capsys, 0)
    assert set(violations(rows).values()) == {0}
    got = informative(rows)
    assert got["energia_cero_c_haar"] == base["energia_cero_c_haar"] + 1
    assert got["energia_cero_c_fourier"] == base["energia_cero_c_fourier"]


def test_inf_y_nan_dan_violaciones(check, lake, tmp_path, capsys):
    indices = [i for i, (sp, _, _) in enumerate(joined(lake)) if sp["c_fourier_band_1"]]
    put(lake, indices[0], c_fourier_band_1=math.inf)
    put(lake, indices[1], c_fourier_band_1=math.nan)
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["doubles_finitos"] == 2


def low_budget(lake) -> tuple[int, list[int]]:
    """Un presupuesto que deja sobre él a varias filas con datos, y sus posiciones."""
    rows = joined(lake)
    withdata = sorted(
        span(e) for sp, _, e in rows if sp["c_fourier_band_1"] is not None
    )
    budget = withdata[len(withdata) // 2] - 1
    return budget, [i for i, (_, _, e) in enumerate(rows) if span(e) > budget]


def test_un_evento_sobre_el_presupuesto_con_valores_da_violacion(
    check, lake, tmp_path, capsys
):
    """El presupuesto sale del pie de `spectrum.parquet`, no de una constante de la sonda."""
    budget, over = low_budget(lake)
    spectrum = file(lake, "spectrum.parquet")
    rewrite(spectrum, pq.read_table(spectrum), budget_ticks=str(budget))
    rows = joined(lake)
    expected = sum(
        any(sp[c] is not None for c in COLUMNS)
        for i, (sp, _, _) in enumerate(rows)
        if i in over
    )
    assert expected > 0
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    line = last_line(capsys)
    got = violations(read_csv(out))
    assert got["nulos_presupuesto"] == expected
    assert line["over"] == str(len(over))
    assert informative(read_csv(out))["sobre_presupuesto"] == len(over)


def test_un_evento_sobre_el_presupuesto_con_todo_nulo_cumple(
    check, lake, tmp_path, capsys
):
    budget, over = low_budget(lake)
    spectrum = file(lake, "spectrum.parquet")
    table = pq.read_table(spectrum)
    for name in COLUMNS:
        values = table[name].to_pylist()
        for i in over:
            values[i] = None
        field = table.schema.field(name)
        table = table.set_column(
            table.schema.get_field_index(name), field, pa.array(values, field.type)
        )
    rewrite(spectrum, table, budget_ticks=str(budget))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    line = last_line(capsys)
    assert line["violaciones"] == "0"
    assert line["over"] == str(len(over)) and int(line["over"]) > 0
    assert set(violations(read_csv(out)).values()) == {0}


def test_n_min_sale_del_pie_del_archivo(check, lake, tmp_path, capsys):
    """Con `n_min = 32` en el pie, las fases de 16 a 31 retornos deben ser nulas."""
    spectrum = file(lake, "spectrum.parquet")
    rewrite(spectrum, pq.read_table(spectrum), n_min="32")
    rows = joined(lake)
    expected = {
        phase: sum(
            ticks(s, phase) < 32
            and any(v is not None for v in group(sp, phase, "fourier"))
            for sp, s, _ in rows
        )
        for phase in PHASES
    }
    assert all(expected.values())
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    for phase in PHASES:
        assert got[f"nulos_{phase}_fourier_valor_donde_nulo"] == expected[phase]


def test_bands_sale_del_pie_del_archivo(check, lake, tmp_path, capsys):
    spectrum = file(lake, "spectrum.parquet")
    rewrite(spectrum, pq.read_table(spectrum), bands="6")
    rows = run_ok(check, lake, tmp_path, capsys, 0)
    assert len([c for (s, c) in rows if s == "columna"]) == 4 * 6
    assert ("columna", "c_fourier_band_7") not in rows


@pytest.mark.parametrize("key", ["state_version", "n_min", "bands", "budget_ticks"])
def test_un_metadato_que_falta_es_fallo(check, lake, tmp_path, capsys, key):
    spectrum = file(lake, "spectrum.parquet")
    table = pq.read_table(spectrum)
    meta = {k: v for k, v in table.schema.metadata.items() if k.decode() != key}
    pq.write_table(table.replace_schema_metadata(meta), spectrum, row_group_size=5)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    text = capsys.readouterr().out
    assert "faltan metadatos" in text and key in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


def test_un_metadato_no_entero_es_fallo(check, lake, tmp_path, capsys):
    spectrum = file(lake, "spectrum.parquet")
    rewrite(spectrum, pq.read_table(spectrum), n_min="dieciseis")
    assert check.main(argv(lake, tmp_path / "salida.csv")) == 1
    assert "no enteros" in capsys.readouterr().out


def test_una_columna_que_falta_es_fallo(check, lake, tmp_path, capsys):
    spectrum = file(lake, "spectrum.parquet")
    rewrite(spectrum, pq.read_table(spectrum).drop_columns(["os_haar_level_7"]))
    assert check.main(argv(lake, tmp_path / "salida.csv")) == 1
    text = capsys.readouterr().out
    assert "os_haar_level_7" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_clave_repetida_en_spectrum_falla_la_union(check, lake, tmp_path, capsys):
    spectrum = file(lake, "spectrum.parquet")
    table = pq.read_table(spectrum)
    rewrite(spectrum, pa.concat_tables([table, table.slice(0, 1)]))
    got = violations(run_ok(check, lake, tmp_path, capsys, 1))
    assert got["union_1a1"] >= 1
    assert got["filas_spectrum_igual_summaries"] == 1


def test_theta_sin_archivos_es_fallo_con_codigo_1(check, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    code = check.main([*argv(lake, out)[:1], "0.05", *argv(lake, out)[2:]])
    text = capsys.readouterr().out
    assert code == 1
    assert "falta el archivo" in text
    for name in ("spectrum.parquet", "summaries.parquet", "events.parquet"):
        assert name in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


@pytest.mark.parametrize(
    "name", ["spectrum.parquet", "summaries.parquet", "events.parquet"]
)
def test_falta_uno_de_los_tres_archivos_es_fallo(check, lake, tmp_path, capsys, name):
    file(lake, name).unlink()
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    text = capsys.readouterr().out
    assert name in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


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


def test_out_directorio_usa_el_nombre_por_defecto(check, lake, tmp_path, capsys):
    out = tmp_path / "resultados"
    assert check.main(argv(lake, tmp_path / "x.csv")[:-2] + ["--out", f"{out}/"]) == 0
    capsys.readouterr()
    assert (out / f"l3_spectrum_check_{MONTH}_{THETA}.csv").is_file()


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
            ["--summaries-root", str(tmp_path), "--l2-root", str(tmp_path), *extra]
        )
    assert raised.value.code == 2


def test_sin_raices_y_sin_ops_results_uri_es_uso(check, tmp_path, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    for roots in (
        [],
        ["--summaries-root", str(tmp_path)],
        ["--l2-root", str(tmp_path)],
    ):
        with pytest.raises(SystemExit) as raised:
            check.main(["--theta", "0.01", "--month", "2026-09", *roots])
        assert raised.value.code == 2


def test_out_fuera_de_results_es_uso(check, lake, tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proyecto-ops/results/ejecucion/")
    with pytest.raises(SystemExit) as raised:
        check.main(argv(lake, tmp_path / "x.csv")[:-2] + ["--out", "gs://otro/x.csv"])
    assert raised.value.code == 2
