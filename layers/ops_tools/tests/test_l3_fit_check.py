"""`l3_fit_check.py` contra un lago local con los archivos que escribe la CLI de L3.

`summaries.parquet` y `fit.parquet` salen de la CLI de `l3_dc_descriptors` (`--family
summaries` y luego `--family fit`) sobre el lago sintético de sus pruebas, que se construye una
sola vez; el `events.parquet` es el de L2 de ese mismo lago. Cada prueba copia a `tmp_path` los
tres archivos de un θ y un mes (hive de L3 y de L2) y los casos que deben fallar alteran el
`fit.parquet` o el `summaries.parquet` ya copiados. La sonda no importa ninguna capa; la prueba
sí, para producir el lago.
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
COLUMNS = [
    "c_slope",
    "os_slope",
    "ev_slope",
    "c_accel",
    "os_accel",
    "ev_accel",
    "c_jump_share",
    "os_jump_share",
    "ev_jump_share",
]
INVARIANTS = [
    "filas_fit_igual_summaries",
    "union_1a1",
    "union_events_1a1",
    "row_groups_iguales",
    "c_jump_share_en_0_1",
    "os_jump_share_no_negativo",
    "ev_jump_share_no_negativo",
    *(f"nulos_{p}_slope" for p in ("c", "os", "ev")),
    *(f"nulos_{p}_accel" for p in ("c", "os", "ev")),
    *(f"nulos_{p}_jump_share" for p in ("c", "os", "ev")),
    "doubles_finitos",
]
INFORMATIVE = [
    *(f"mismo_t_{p}_slope" for p in ("c", "os", "ev")),
    *(f"accel_sin_explicar_{p}_accel" for p in ("c", "os", "ev")),
    "ev_jump_share_fuera_de_cota",
    *(f"abs_{c}_mayor_2" for c in ("c_accel", "os_accel", "ev_accel")),
]
LAST_LINE = re.compile(
    r"fit check: mes=(?P<mes>\S+) theta=(?P<theta>\S+) "
    r"state_version=(?P<version>\S+) filas=(?P<filas>\S+) "
    r"violaciones=(?P<violaciones>\S+) mismo_t=(?P<mismo_t>\S+) "
    r"pared_s=(?P<pared>\S+) veredicto=(?P<veredicto>\S+)$"
)
# Ancla y cierre de cada fase, como en TRD-L3 §7.2.2.
PHASES = {
    "c": ("reference", "confirm", lambda s: s["c_ticks"]),
    "os": ("confirm", "extreme", lambda s: s["os_ticks"]),
    "ev": ("reference", "extreme", lambda s: s["c_ticks"] + s["os_ticks"]),
}


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_fit_check", SCRIPTS / "l3_fit_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check():
    return load()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """El `l2` y el `l3` de un lago de pruebas con las dos familias escritas por la CLI."""
    base = tmp_path_factory.mktemp("lake")
    ticks = fixture_ticks()
    lake = build_lake(base, ticks, month_cuts(ticks, 1500, 3000))
    catalog = write_catalog(base / "thetas.yaml", lake.thetas)
    env = env_for(lake, base, catalog)
    backfill = ["--mode", "backfill", "--series-start", "2017-08"]
    for family in ("summaries", "fit"):
        assert l3_main([*backfill, "--family", family], env) == 0
    return lake.l2, lake.l3


@pytest.fixture
def lake(built, tmp_path):
    """Una copia de la partición del θ y el mes en `tmp_path`, que cada prueba altera."""
    l2, l3 = built
    for root, source, names in (
        (tmp_path / "l2", l2, ("events.parquet",)),
        (tmp_path / "l3", l3, ("summaries.parquet", "fit.parquet")),
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


def alter(path: Path, column: str, change) -> None:
    """Reescribe el archivo con `change(valores)` en `column`; conserva esquema y metadatos."""
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    field = table.schema.field(column).with_nullable(True)
    values = change(table[column].to_pylist())
    table = table.set_column(index, field, pa.array(values, field.type))
    pq.write_table(table, path, row_group_size=5)


def section(rows, name: str, column: str) -> dict[str, int]:
    return {c: int(r[column]) for (s, c), r in rows.items() if s == name}


def violations(rows) -> dict[str, int]:
    return section(rows, "invariante", "violaciones")


def informative(rows) -> dict[str, int]:
    return section(rows, "informativo", "conteo")


def joined(lake) -> list[tuple[dict, dict, dict]]:
    """(fit, summaries, events) de cada fila; los tres archivos tienen el mismo orden."""
    fit = pq.read_table(file(lake, "fit.parquet")).to_pylist()
    summaries = pq.read_table(file(lake, "summaries.parquet")).to_pylist()
    events = pq.read_table(file(lake, "events.parquet")).to_pylist()
    return list(zip(fit, summaries, events, strict=True))


def first(lake, condition) -> int:
    """La posición de la primera fila que cumple `condition(fit, summaries, events)`."""
    return next(i for i, row in enumerate(joined(lake)) if condition(*row))


def distinct_times(phase: str, s: dict, e: dict) -> bool:
    start, end, ticks = PHASES[phase]
    return ticks(s) >= 2 and e[f"{end}_time"] != e[f"{start}_time"]


def set_at(index: int, value):
    def change(values):
        values = list(values)
        values[index] = value
        return values

    return change


def test_csv_coincide_con_el_archivo_y_cero_violaciones(check, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    line = last_line(capsys)
    rows = read_csv(out)
    table = pq.read_table(file(lake, "fit.parquet"))

    assert line["mes"] == MONTH
    assert line["theta"] == THETA
    assert line["version"] == "1.0.0"
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
    for column in ("c_accel", "os_accel", "ev_accel"):
        values = sorted(v for v in table[column].to_pylist() if v is not None)
        got = rows["columna", column]
        assert float(got["q01"]) <= float(got["q05"]) <= float(got["mediana"])
        assert float(got["mediana"]) <= float(got["q95"]) <= float(got["q99"])
        assert float(got["q99"]) <= values[-1] and float(got["q01"]) >= values[0]

    assert [c for (s, c) in rows if s == "invariante"] == INVARIANTS
    assert [c for (s, c) in rows if s == "informativo"] == INFORMATIVE
    assert set(violations(rows).values()) == {0}


def test_mismo_t_y_las_cifras_informativas_coinciden_con_un_oraculo_en_python(
    check, lake, tmp_path, capsys
):
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    line = last_line(capsys)
    got = informative(read_csv(out))

    same_t = dict.fromkeys(PHASES, 0)
    singular = dict.fromkeys(PHASES, 0)
    above_two = dict.fromkeys(PHASES, 0)
    for f, s, e in joined(lake):
        for phase, (start, end, _) in PHASES.items():
            if distinct_times(phase, s, e):
                same_t[phase] += f[f"{phase}_slope"] is None
                priced = e[f"{end}_price"] != e[f"{start}_price"]
                singular[phase] += (
                    f[f"{phase}_accel"] is None
                    and f[f"{phase}_slope"] is not None
                    and priced
                )
            above_two[phase] += abs(f[f"{phase}_accel"] or 0) > 2
    assert {p: got[f"mismo_t_{p}_slope"] for p in PHASES} == same_t
    assert {p: got[f"accel_sin_explicar_{p}_accel"] for p in PHASES} == singular
    assert {p: got[f"abs_{p}_accel_mayor_2"] for p in PHASES} == above_two
    assert line["mismo_t"] == str(sum(same_t.values()))


def test_el_lago_ejerce_los_casos_de_la_regla_de_nulos(lake):
    """Sin estos casos en el lago, «cero violaciones» no probaría la regla de nulos."""
    rows = joined(lake)
    assert any(s["os_ticks"] == 0 for _, s, _ in rows)
    assert any(s["os_ticks"] > 0 for _, s, _ in rows)
    assert any(f["os_slope"] is None for f, s, _ in rows if s["os_ticks"] == 0)
    assert any(f["os_slope"] is not None for f, s, _ in rows if s["os_ticks"] >= 2)
    assert any(f["os_jump_share"] is None for f, _, _ in rows)
    assert any(f["os_jump_share"] is not None for f, _, _ in rows)
    assert any(f["c_slope"] is None for f, _, _ in rows)
    assert any(f["c_slope"] is not None for f, _, _ in rows)
    assert any(f["c_accel"] is not None for f, _, _ in rows)
    assert any(f["ev_accel"] is not None for f, _, _ in rows)
    assert all(f["c_jump_share"] is not None for f, _, _ in rows)


def test_no_importa_ninguna_capa():
    source = (SCRIPTS / "l3_fit_check.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]


def test_fila_quitada_de_fit_da_violaciones(check, lake, tmp_path, capsys):
    fit = file(lake, "fit.parquet")
    pq.write_table(pq.read_table(fit).slice(1), fit, row_group_size=5)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    line = last_line(capsys)
    got = violations(read_csv(out))
    assert line["veredicto"] == "violaciones"
    assert got["filas_fit_igual_summaries"] == 1
    assert got["union_1a1"] == 1
    assert got["union_events_1a1"] == 1
    # Cada row group de fit sale una fila corrido respecto del de summaries.
    assert got["row_groups_iguales"] >= 1
    assert int(line["violaciones"]) == sum(got.values())


def test_row_groups_distintos_con_las_mismas_filas_dan_violacion(
    check, lake, tmp_path, capsys
):
    fit = file(lake, "fit.parquet")
    pq.write_table(pq.read_table(fit), fit, row_group_size=7)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["row_groups_iguales"] >= 1
    assert sum(got.values()) == got["row_groups_iguales"]


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
def test_c_jump_share_fuera_de_0_1_da_violacion(check, lake, tmp_path, capsys, value):
    alter(file(lake, "fit.parquet"), "c_jump_share", set_at(0, value))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["c_jump_share_en_0_1"] == 1
    assert sum(got.values()) == 1


def test_jump_share_negativo_de_overshoot_y_de_evento_da_violacion(
    check, lake, tmp_path, capsys
):
    index = first(lake, lambda f, s, e: f["os_jump_share"] is not None)
    alter(file(lake, "fit.parquet"), "os_jump_share", set_at(index, -0.1))
    alter(file(lake, "fit.parquet"), "ev_jump_share", set_at(index, -0.1))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["os_jump_share_no_negativo"] == 1
    assert got["ev_jump_share_no_negativo"] == 1
    assert sum(got.values()) == 2


def test_ev_jump_share_sobre_1_mas_theta_es_informativo_y_no_violacion(
    check, lake, tmp_path, capsys
):
    alter(file(lake, "fit.parquet"), "ev_jump_share", set_at(0, 1.5))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    assert last_line(capsys)["violaciones"] == "0"
    rows = read_csv(out)
    assert informative(rows)["ev_jump_share_fuera_de_cota"] == 1
    assert set(violations(rows).values()) == {0}


def test_ev_jump_share_dentro_de_1_mas_theta_no_cuenta_como_fuera_de_cota(
    check, lake, tmp_path, capsys
):
    # 1 + θ = 1,0025: un valor entre 1 y 1 + θ está dentro de la cota.
    alter(file(lake, "fit.parquet"), "ev_jump_share", set_at(0, 1.002))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    capsys.readouterr()
    assert informative(read_csv(out))["ev_jump_share_fuera_de_cota"] == 0


def test_accel_sobre_2_es_informativo_y_no_violacion(check, lake, tmp_path, capsys):
    index = first(lake, lambda f, s, e: f["c_accel"] is not None)
    before = pq.read_table(file(lake, "fit.parquet"))["c_accel"].to_pylist()
    alter(file(lake, "fit.parquet"), "c_accel", set_at(index, 1e6))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    capsys.readouterr()
    rows = read_csv(out)
    assert set(violations(rows).values()) == {0}
    expected = sum(
        abs(v) > 2 for i, v in enumerate(before) if v is not None and i != index
    )
    assert informative(rows)["abs_c_accel_mayor_2"] == expected + 1
    valued = sum(v is not None for v in before)
    assert float(
        rows["informativo", "abs_c_accel_mayor_2"]["fraccion"]
    ) == pytest.approx((expected + 1) / valued)
    assert float(rows["columna", "c_accel"]["max"]) == 1e6


def test_nulo_donde_la_regla_dice_no_nulo_en_jump_share_da_violacion(
    check, lake, tmp_path, capsys
):
    alter(file(lake, "fit.parquet"), "c_jump_share", set_at(0, None))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["nulos_c_jump_share"] == 1
    assert sum(got.values()) == 1


def test_valor_donde_la_regla_dice_nulo_da_violacion(check, lake, tmp_path, capsys):
    index = first(lake, lambda f, s, e: s["os_ticks"] == 0)
    for column in ("os_slope", "os_accel", "os_jump_share"):
        alter(file(lake, "fit.parquet"), column, set_at(index, 0.5))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["nulos_os_slope"] == 1
    assert got["nulos_os_accel"] == 1
    assert got["nulos_os_jump_share"] == 1
    assert sum(got.values()) == 3


def test_accel_con_precio_de_cierre_igual_al_ancla_debe_ser_nula(
    check, lake, tmp_path, capsys
):
    """El lago no trae `p_fin = p_0` en la confirmación: se fabrica en `events`. Con él,
    `c_accel` y `c_jump_share` no nulos violan la regla, y nulos la cumplen."""
    index = first(lake, lambda f, s, e: f["c_accel"] is not None)
    events = file(lake, "events.parquet")
    reference = pq.read_table(events)["reference_price"].to_pylist()
    alter(
        events,
        "confirm_price",
        lambda values: [
            reference[index] if i == index else v for i, v in enumerate(values)
        ],
    )
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["nulos_c_accel"] == 1
    assert got["nulos_c_jump_share"] == 1

    alter(file(lake, "fit.parquet"), "c_accel", set_at(index, None))
    alter(file(lake, "fit.parquet"), "c_jump_share", set_at(index, None))
    assert check.main(argv(lake, out)) == 0
    capsys.readouterr()
    assert violations(read_csv(out))["nulos_c_accel"] == 0


def test_slope_no_nula_con_menos_de_dos_ticks_da_violacion(
    check, lake, tmp_path, capsys
):
    index = first(lake, lambda f, s, e: s["c_ticks"] < 2)
    alter(file(lake, "fit.parquet"), "c_slope", set_at(index, 1.0))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["nulos_c_slope"] == 1


def test_slope_nula_con_ticks_y_tiempos_distintos_es_mismo_t_y_no_violacion(
    check, lake, tmp_path, capsys
):
    index = first(
        lake,
        lambda f, s, e: distinct_times("c", s, e) and f["c_slope"] is not None,
    )
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    base = last_line(capsys)["mismo_t"]
    alter(file(lake, "fit.parquet"), "c_slope", set_at(index, None))
    assert check.main(argv(lake, out)) == 0
    line = last_line(capsys)
    assert line["violaciones"] == "0"
    assert int(line["mismo_t"]) == int(base) + 1
    assert informative(read_csv(out))["mismo_t_c_slope"] >= 1


def test_accel_nula_con_slope_y_precios_distintos_es_informativa(
    check, lake, tmp_path, capsys
):
    index = first(
        lake,
        lambda f, s, e: (
            distinct_times("c", s, e)
            and f["c_accel"] is not None
            and f["c_slope"] is not None
        ),
    )
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 0
    base = informative(read_csv(out))["accel_sin_explicar_c_accel"]
    capsys.readouterr()
    alter(file(lake, "fit.parquet"), "c_accel", set_at(index, None))
    assert check.main(argv(lake, out)) == 0
    capsys.readouterr()
    rows = read_csv(out)
    assert informative(rows)["accel_sin_explicar_c_accel"] == base + 1
    assert set(violations(rows).values()) == {0}


def test_inf_y_nan_dan_violaciones(check, lake, tmp_path, capsys):
    index = first(
        lake,
        lambda f, s, e: distinct_times("c", s, e) and distinct_times("ev", s, e),
    )
    alter(file(lake, "fit.parquet"), "c_slope", set_at(index, math.inf))
    alter(file(lake, "fit.parquet"), "ev_slope", set_at(index, math.nan))
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    assert violations(read_csv(out))["doubles_finitos"] == 2


def test_clave_repetida_en_fit_falla_la_union(check, lake, tmp_path, capsys):
    fit = file(lake, "fit.parquet")
    table = pq.read_table(fit)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), fit, row_group_size=5)
    out = tmp_path / "salida.csv"
    assert check.main(argv(lake, out)) == 1
    capsys.readouterr()
    got = violations(read_csv(out))
    assert got["union_1a1"] >= 1
    assert got["filas_fit_igual_summaries"] == 1


def test_theta_sin_archivos_es_fallo_con_codigo_1(check, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    code = check.main([*argv(lake, out)[:1], "0.05", *argv(lake, out)[2:]])
    text = capsys.readouterr().out
    assert code == 1
    assert "falta el archivo" in text
    for name in ("fit.parquet", "summaries.parquet", "events.parquet"):
        assert name in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


@pytest.mark.parametrize("name", ["fit.parquet", "summaries.parquet", "events.parquet"])
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
    assert (out / f"l3_fit_check_{MONTH}_{THETA}.csv").is_file()


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
