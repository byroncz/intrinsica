"""`l3_spectrum_xcheck.py` contra un lago local con el `spectrum.parquet` que escribe el motor real.

`summaries.parquet` y `spectrum.parquet` salen de la CLI de `l3_dc_descriptors` (`--family
summaries` y luego `--family spectrum`) sobre el lago sintético de sus pruebas, que se construye una
sola vez; `events.parquet` es el de L2 y `consolidated.parquet` el de L1 de ese mismo lago. Cada
prueba copia a `tmp_path` los tres archivos de un θ y un mes y los casos que deben dar `revisar`
alteran el `spectrum.parquet` ya copiado. La sonda no importa ninguna capa; la prueba sí, para
producir el lago.
"""

import csv
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import numpy as np
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
L1_PARTITION = f"{SERIES}/year=2017/month=08"
N_MIN = 16
BANDS = 7
UNIT = {"fourier": "band", "haar": "level"}
COLUMNS = [
    *(
        f"{phase}_{register}_{UNIT[register]}_{b}"
        for register in UNIT
        for phase in ("c", "os")
        for b in range(1, BANDS + 1)
    ),
    "c_energy_loc",
    "os_energy_loc",
]
LAST_LINE = re.compile(
    r"spectrum xcheck: mes=(?P<mes>\S+) theta=(?P<theta>\S+) filas=(?P<filas>\S+) "
    r"comparadas=(?P<comparadas>\S+) omitidas=(?P<omitidas>\S+) "
    r"fuera_tolerancia=(?P<fuera>\S+) nulos_distintos=(?P<nulos>\S+) "
    r"max_diff=(?P<max_diff>\S+) haar_cero_f64=(?P<haar64>\S+) "
    r"haar_cero_f32=(?P<haar32>\S+) veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_spectrum_xcheck", SCRIPTS / "l3_spectrum_xcheck.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def xcheck():
    return load()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """El `l1`, el `l2` y el `l3` de un lago de pruebas con S y F escritas por la CLI."""
    base = tmp_path_factory.mktemp("lake")
    ticks = fixture_ticks()
    lake = build_lake(base, ticks, month_cuts(ticks, 1500, 3000))
    catalog = write_catalog(base / "thetas.yaml", lake.thetas)
    env = env_for(lake, base, catalog)
    backfill = ["--mode", "backfill", "--series-start", "2017-08"]
    for family in ("summaries", "spectrum"):
        assert l3_main([*backfill, "--family", family], env) == 0
    return lake


@pytest.fixture
def lake(built, tmp_path):
    """Una copia de la partición del θ y el mes en `tmp_path`, que cada prueba altera."""
    for root, source, name, part in (
        (tmp_path / "l1", built.l1, "consolidated.parquet", L1_PARTITION),
        (tmp_path / "l2", built.l2, "events.parquet", PARTITION),
        (tmp_path / "l3", built.l3, "spectrum.parquet", PARTITION),
    ):
        (root / part).mkdir(parents=True)
        shutil.copy(source / part / name, root / part / name)
    return tmp_path


def argv(lake: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--theta",
        "0.0025",
        "--month",
        MONTH,
        "--l1-root",
        str(lake / "l1"),
        "--l2-root",
        str(lake / "l2"),
        "--l3-root",
        str(lake / "l3"),
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


def spectrum_file(lake: Path) -> Path:
    return lake / "l3" / PARTITION / "spectrum.parquet"


def alter(lake: Path, column: str, index: int, value) -> None:
    """Reescribe `spectrum.parquet` con `value` en la fila `index` de `column`."""
    path = spectrum_file(lake)
    table = pq.read_table(path)
    position = table.schema.get_field_index(column)
    field = table.schema.field(column).with_nullable(True)
    values = table[column].to_pylist()
    values[index] = value
    table = table.set_column(position, field, pa.array(values, field.type))
    pq.write_table(table, path, row_group_size=5)


def first_row(lake: Path, column: str, *, null: bool) -> int:
    values = pq.read_table(spectrum_file(lake), columns=[column])[column].to_pylist()
    return next(i for i, v in enumerate(values) if (v is None) == null)


def test_the_real_engine_matches_the_oracle(xcheck, lake, tmp_path, capsys):
    out = tmp_path / "x.csv"
    assert xcheck.main(argv(lake, out)) == 0
    line = last_line(capsys)
    assert line["veredicto"] == "ok"
    assert line["mes"] == MONTH and line["theta"] == THETA
    assert line["fuera"] == "0" and line["nulos"] == "0"
    # Las dos fases de cada evento entran; ninguna se omite en el lago de pruebas.
    assert int(line["filas"]) > 0
    assert int(line["comparadas"]) + int(line["omitidas"]) <= 2 * int(line["filas"])
    assert int(line["comparadas"]) > int(line["filas"])
    assert float(line["max_diff"]) <= xcheck.TOL

    rows = read_csv(out)
    columns = [c for (s, c) in rows if s == "columna"]
    assert columns == xcheck.data_columns() == COLUMNS
    assert len(columns) == 30
    # Cada fase comparada aporta sus 15 columnas, y solo ellas.
    per_column = [int(rows["columna", c]["comparadas"]) for c in columns]
    assert sum(per_column) == 15 * int(line["comparadas"])
    assert all(
        int(rows["columna", c]["comparadas"]) == per_column[0]
        for c in columns
        if c.startswith("c_")
    )
    # Las colas planas de Haar y las fases cortas se cuentan aparte.
    info = {c: int(r["conteo"]) for (s, c), r in rows.items() if s == "informativo"}
    assert info["comparadas"] == int(line["comparadas"])
    assert info["haar_cero_f64"] == int(line["haar64"])
    assert info["haar_cero_f32"] == int(line["haar32"])
    assert info["fases_cortas"] > 0
    assert info["antes_del_mes"] == 0 and info["despues_del_mes"] == 0


def test_a_value_changed_by_hand_gives_revisar(xcheck, lake, tmp_path, capsys):
    column = "c_fourier_band_1"
    index = first_row(lake, column, null=False)
    value = pq.read_table(spectrum_file(lake), columns=[column])[column][index].as_py()
    alter(lake, column, index, value + 0.01)
    out = tmp_path / "x.csv"
    assert xcheck.main(argv(lake, out)) == 1
    line = last_line(capsys)
    assert line["veredicto"] == "revisar"
    assert line["fuera"] == "1" and line["nulos"] == "0"
    assert float(line["max_diff"]) == pytest.approx(0.01, rel=1e-3)
    row = read_csv(out)["columna", column]
    assert row["fuera_tolerancia"] == "1" and row["nulos_distintos"] == "0"


def test_a_difference_below_the_tolerance_is_not_a_violation(
    xcheck, lake, tmp_path, capsys
):
    column = "c_haar_level_1"
    index = first_row(lake, column, null=False)
    value = pq.read_table(spectrum_file(lake), columns=[column])[column][index].as_py()
    alter(lake, column, index, value + 1e-11)
    assert xcheck.main(argv(lake, tmp_path / "x.csv")) == 0
    assert last_line(capsys)["veredicto"] == "ok"


def test_a_null_where_the_oracle_has_a_value_gives_revisar(
    xcheck, lake, tmp_path, capsys
):
    column = "c_energy_loc"
    alter(lake, column, first_row(lake, column, null=False), None)
    assert xcheck.main(argv(lake, tmp_path / "a.csv")) == 1
    line = last_line(capsys)
    assert line["veredicto"] == "revisar" and line["nulos"] == "1"
    assert line["fuera"] == "0"


def test_a_value_where_the_oracle_has_a_null_gives_revisar(
    xcheck, lake, tmp_path, capsys
):
    # Una fase `os` con menos de 16 retornos (o vacía) debe traer la fase entera nula.
    column = "os_energy_loc"
    alter(lake, column, first_row(lake, column, null=True), 0.5)
    assert xcheck.main(argv(lake, tmp_path / "a.csv")) == 1
    line = last_line(capsys)
    assert line["veredicto"] == "revisar" and line["nulos"] == "1"


def test_max_ticks_skips_and_counts_the_long_phases(xcheck, lake, tmp_path, capsys):
    out = tmp_path / "x.csv"
    assert xcheck.main(argv(lake, out, "--max-ticks", "20")) == 0
    line = last_line(capsys)
    assert line["veredicto"] == "ok"
    assert int(line["omitidas"]) > 0
    full = tmp_path / "full.csv"
    assert xcheck.main(argv(lake, full)) == 0
    complete = last_line(capsys)
    assert int(line["comparadas"]) + int(line["omitidas"]) == int(
        complete["comparadas"]
    )


def test_a_flat_haar_tail_is_counted_with_both_arithmetics(xcheck):
    n = 64
    prices = np.full(n, 100_000_000_000, dtype=np.int64)
    returns, flat_f64 = xcheck.engine_returns(prices, 100_000_000_000)
    assert returns.dtype == np.float32 and not returns.any() and flat_f64
    values, flat_f32 = xcheck.expected_columns("c", returns, N_MIN)
    assert flat_f32
    # Todos los retornos son cero: sin Fourier, sin Haar y sin ubicación.
    assert values == [None] * 15

    # Un paso al inicio: la cola de Haar (los últimos 64 de 65) es plana, el registro entero no.
    prices = np.full(n + 1, 100_000_000_001, dtype=np.int64)
    returns, flat_f64 = xcheck.engine_returns(prices, 100_000_000_000)
    values, flat_f32 = xcheck.expected_columns("c", returns, N_MIN)
    assert flat_f64 and flat_f32
    assert values[7:14] == [None] * 7
    assert (
        values[14] is not None
    )  # la ubicación sí: la energía está toda en el primer retorno


def test_the_returns_are_the_ones_of_the_engine(xcheck):
    prices = np.array(
        [100_000_000_000, 100_000_000_003, 99_999_999_000], dtype=np.int64
    )
    returns, _ = xcheck.engine_returns(prices, 100_000_000_000)
    # El primero sale del ancla (sin paso: 0), el segundo del paso de +3 y el tercero del de −1003,
    # cada uno como `ln_1p(Δp / p_previo)` en f64 guardado en f32 (`log_return` de spectrum.rs).
    assert returns.dtype == np.float32 and len(returns) == 3
    assert returns[0] == 0.0
    assert returns[1] == np.float32(np.log1p(3 / 100_000_000_000))
    assert returns[2] == np.float32(np.log1p(-1003 / 100_000_000_003))


def test_a_missing_file_is_fallo(xcheck, lake, tmp_path, capsys):
    spectrum_file(lake).unlink()
    assert xcheck.main(argv(lake, tmp_path / "x.csv")) == 1
    out = capsys.readouterr().out
    assert "FALLO" in out and out.strip().splitlines()[-1].endswith("veredicto=fallo")


def test_usage_errors_exit_2(xcheck, lake, tmp_path):
    for extra in (
        ["--month", "2017-13"],
        ["--theta", "0.123456789"],
        ["--max-ticks", "0"],
    ):
        with pytest.raises(SystemExit) as raised:
            xcheck.main([*argv(lake, tmp_path / "x.csv"), *extra])
        assert raised.value.code == 2


def phase_of(xcheck, n: int):
    return xcheck.Phase("c", 0, n, 100, 7, {}, False, 1_000)


def test_haar_zero_cases_are_counted_and_listed(xcheck):
    cols = xcheck.group("c", "haar")
    nulls = dict.fromkeys(cols)
    valued = {**nulls, cols[0]: 0.5}
    wanted = [None] * 15
    tally = xcheck.Tally()
    # Plana en las dos aritméticas y nula en el motor: se cuenta y no es discrepancia.
    tally.haar_zero(phase_of(xcheck, 64), nulls, wanted, (True, True))
    assert tally.counts == {
        "haar_cero_f64": 1,
        "haar_cero_f64_motor_nulo": 1,
        "haar_cero_f32": 1,
        "haar_cero_f32_motor_nulo": 1,
    }
    assert tally.discrepancies == 0
    # Plana solo en f64 y con valor en el motor: discrepancia, con los dos valores.
    tally.haar_zero(phase_of(xcheck, 80), valued, wanted, (True, False))
    assert tally.counts["haar_cero_f64_motor_valor"] == 1
    assert "haar_cero_f32_motor_valor" not in tally.counts
    assert tally.discrepancies == 1
    assert tally.listed == [
        {
            "event_id": 7,
            "fase": "c",
            "n": 0,
            "plano_f64": True,
            "plano_f32": False,
            "motor": 0.5,
            "oraculo": None,
        }
    ]
    # No plana en ninguna y con valor: ni se cuenta ni es discrepancia. La lista corta en 20.
    tally.haar_zero(phase_of(xcheck, 64), valued, wanted, (False, False))
    assert tally.discrepancies == 1
    for _ in range(30):
        tally.haar_zero(phase_of(xcheck, 64), valued, wanted, (True, True))
    assert tally.discrepancies == 31 and len(tally.listed) == xcheck.MAX_LISTED
