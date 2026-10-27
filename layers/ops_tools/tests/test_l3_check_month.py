"""`l3_check_month.py` contra un `summaries.parquet` que escribe el resumidor de verdad.

El lago es el de los fixtures de viz (L1 y L2 reales de 2017-08, 5 θ) y el archivo lo genera
`summarize_month` de L3; el script no importa esa capa, la prueba sí, porque aquí ella es el
sistema bajo prueba. Los casos que deben fallar alteran el archivo ya escrito.
"""

import importlib.util
import re
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l3_dc_descriptors.summarize import summarize_month

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "layers/viz_tiles/tests"))
sys.path.insert(0, str(ROOT / "layers/l3_dc_descriptors/tests"))

import l3_lake
from dc_frames import Frame
from lake_fixture import MONTH, build_lake, read_events, theta_text

MONTH_TEXT = f"{MONTH[0]:04d}-{MONTH[1]:02d}"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
DEC = pa.decimal128(18, 8)
THETA = 2_000_000
SEED = ROOT / "layers/l2_dc_events/src/l2_dc_events/config/thetas.yaml"


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_check_month", SCRIPTS / "l3_check_month.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check():
    return load()


@pytest.fixture
def lake(tmp_path):
    """L1, L2 y L3 de 2017-08 del fixture de viz; devuelve `(l1, l2, l3, thetas)`.

    Los eventos de ese fixture son `events_v0`, anteriores al instante de confirmación
    atómico (ADR-L2-03): el resumidor rechaza cuatro de los cinco θ por el invariante 4 de
    §7.2 (`l2_state_inconsistent`). Solo θ = 0,02 pasa su compuerta, con 11 eventos en 4 row
    groups, y es el que la prueba usa; el resto de la cobertura está en el lago de L3.
    """
    roots, _ticks, _pending = build_lake(tmp_path)
    l3 = tmp_path / "l3"
    assert max(read_events()) == THETA
    thetas = [theta_text(THETA)]
    summarize_month(roots.landing, roots.events, l3, MONTH, thetas, series_start=MONTH)
    return roots.landing, roots.events, l3, thetas


def argv(l1, l2, l3, *extra: str) -> list[str]:
    return [
        "--month",
        MONTH_TEXT,
        "--l1-root",
        str(l1),
        "--l2-root",
        str(l2),
        "--l3-root",
        str(l3),
        *extra,
    ]


def summaries(l3: Path, theta: str) -> Path:
    return (
        l3
        / SERIES
        / f"theta={theta}/year={MONTH[0]:04d}/month={MONTH[1]:02d}/summaries.parquet"
    )


def alter(path: Path, column: str, row: int, value) -> int:
    """Pisa una celda de `summaries.parquet` y conserva sus row groups; devuelve el id de esa fila."""
    parquet = pq.ParquetFile(path)
    groups = [parquet.read_row_group(i) for i in range(parquet.num_row_groups)]
    schema, meta = parquet.schema_arrow, parquet.schema_arrow.metadata
    parquet.close()
    start = 0
    for i, group in enumerate(groups):
        if row < start + group.num_rows:
            index = group.schema.get_field_index(column)
            values = group.column(column).to_pylist()
            values[row - start] = value
            kind = group.schema.field(column).type
            groups[i] = group.set_column(
                index, group.schema.field(column), pa.array(values, kind)
            )
            ident = group.column("confirm_agg_trade_id")[row - start].as_py()
            break
        start += group.num_rows
    with pq.ParquetWriter(path, schema.with_metadata(meta)) as writer:
        for group in groups:
            writer.write_table(group)
    return ident


def last_line(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_cero_diferencias_contra_lo_que_escribe_el_resumidor(check, lake, capsys):
    l1, l2, l3, thetas = lake
    code = check.main(argv(l1, l2, l3, "--thetas", ",".join(thetas)))
    out = capsys.readouterr().out
    assert code == 0, out
    rows = sum(pq.ParquetFile(summaries(l3, t)).metadata.num_rows for t in thetas)
    assert rows > 0
    assert (
        out.strip()
        .splitlines()[-1]
        .startswith(
            f"check mes: mes={MONTH_TEXT} thetas={len(thetas)} filas={rows} diferencias=0 "
        )
    )
    assert out.strip().endswith("veredicto=ok")
    assert "DIFERENCIA" not in out


def test_all_toma_los_theta_que_tiene_l3_y_el_catalogo_por_defecto(
    check, lake, tmp_path, capsys
):
    l1, l2, l3, thetas = lake
    assert check.main(argv(l1, l2, l3, "--thetas", "all")) == 0
    assert f"thetas={len(thetas)} " in last_line(capsys)
    # La semilla real de L2 con la lista de θ cambiada por los del lago: la línea `scale`
    # (con su comentario) y el formato de cada θ son los de producción.
    catalog = tmp_path / "thetas.yaml"
    head = [x for x in SEED.read_text().splitlines() if not re.match(r"\s*-\s*\d+", x)]
    integers = [int(Decimal(t).scaleb(8)) for t in thetas]
    catalog.write_text(
        "\n".join(head + [f"  - {i}  # {t}" for i, t in zip(integers, thetas)]) + "\n"
    )
    assert check.main(argv(l1, l2, l3, "--catalog", str(catalog))) == 0
    assert f"thetas={len(thetas)} " in last_line(capsys)


def test_la_semilla_real_del_catalogo_se_lee_con_su_comentario_de_escala(check):
    assert re.search(r"^scale:\s*100000000\s*#", SEED.read_text(), re.MULTILINE)
    thetas = check.read_catalog(str(SEED))
    assert len(thetas) == 50
    assert thetas == sorted(thetas)
    assert thetas[0] == "0.00010000"
    assert thetas[-1] == "0.05000000"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("c_vwap", Decimal("1.00000001")),
        ("c_ticks", 99999),
        ("confirm_instant_price_best", Decimal("2.00000000")),
        ("os_ticks", 7),
    ],
)
def test_una_celda_alterada_se_detecta_con_su_fila_y_columna(
    check, lake, capsys, column, value
):
    l1, l2, l3, thetas = lake
    theta = thetas[0]
    row = 2
    original = pq.ParquetFile(summaries(l3, theta)).read().column(column)[row].as_py()
    ident = alter(summaries(l3, theta), column, row, value)
    code = check.main(argv(l1, l2, l3, "--thetas", ",".join(thetas)))
    out = capsys.readouterr().out
    assert code == 1, out
    line = next(x for x in out.splitlines() if x.startswith("DIFERENCIA"))
    assert f"θ={theta} confirm_agg_trade_id={ident} columna={column} " in line
    shown = f"{original:.8f}" if isinstance(original, Decimal) else str(original)
    assert f"oráculo={shown} L3=" in line
    assert out.strip().endswith("veredicto=diferencias")
    # El resto de las columnas no se contagia: la única con diferencias es la alterada.
    columns = {
        re.search(r"columna=(\w+)", x)[1] for x in out.splitlines() if "columna=" in x
    }
    assert columns == {column}


def test_un_nulo_donde_el_overshoot_no_es_vacio_se_detecta(check, lake, capsys):
    l1, l2, l3, thetas = lake
    theta = thetas[-1]
    table = pq.ParquetFile(summaries(l3, theta)).read()
    row = next(i for i, n in enumerate(table.column("os_ticks").to_pylist()) if n > 0)
    ident = alter(summaries(l3, theta), "os_vwap", row, None)
    assert check.main(argv(l1, l2, l3, "--thetas", theta)) == 1
    out = capsys.readouterr().out
    assert f"confirm_agg_trade_id={ident} columna=os_vwap " in out
    assert "L3=None" in out


def test_un_theta_sin_archivo_en_l3_es_una_diferencia(check, lake, capsys):
    l1, l2, l3, thetas = lake
    summaries(l3, thetas[0]).unlink()
    assert check.main(argv(l1, l2, l3, "--thetas", thetas[0])) == 1
    out = capsys.readouterr().out
    assert f"DIFERENCIA θ={thetas[0]} falta el archivo" in out
    assert "veredicto=diferencias" in last_line_of(out)


def last_line_of(out: str) -> str:
    return out.strip().splitlines()[-1]


def rewrite(path: Path, pick) -> None:
    """Reescribe `summaries.parquet` con los row groups que dice `pick(groups)`."""
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    groups = [parquet.read_row_group(i) for i in range(parquet.num_row_groups)]
    parquet.close()
    with pq.ParquetWriter(path, schema) as writer:
        for group in pick(groups):
            writer.write_table(group)


def test_un_l3_con_menos_filas_que_el_oraculo_es_una_diferencia(check, lake, capsys):
    l1, l2, l3, thetas = lake
    rewrite(summaries(l3, thetas[0]), lambda groups: groups[:2])
    assert check.main(argv(l1, l2, l3, "--thetas", thetas[0])) == 1
    out = capsys.readouterr().out
    assert (
        f"DIFERENCIA θ={thetas[0]} filas: el oráculo tiene 5 evento(s) más que L3"
        in out
    )
    assert "(11 contra 6)" in out


def test_un_l3_con_mas_filas_que_el_oraculo_es_una_diferencia(check, lake, capsys):
    l1, l2, l3, thetas = lake
    rewrite(summaries(l3, thetas[0]), lambda groups: [*groups, groups[0]])
    assert check.main(argv(l1, l2, l3, "--thetas", thetas[0])) == 1
    out = capsys.readouterr().out
    assert (
        f"DIFERENCIA θ={thetas[0]} filas: L3 tiene 3 fila(s) más que el oráculo" in out
    )


def test_el_lector_que_falla_es_un_fallo_con_codigo_1(check, lake, capsys):
    l1, l2, l3, thetas = lake
    next(Path(l1).rglob("consolidated.parquet")).unlink()
    assert check.main(argv(l1, l2, l3, "--thetas", thetas[0])) == 1
    out = capsys.readouterr().out
    assert "FALLO lector: FramesInputError" in out
    assert last_line_of(out).endswith("veredicto=fallo")


def test_el_uso_incorrecto_sale_con_2(check, lake, monkeypatch):
    l1, l2, l3, thetas = lake
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    theta = thetas[0]
    for month, others in (
        ("2017-13", ["--thetas", theta]),
        ("2017-08", ["--thetas", "0.123456789"]),
        ("2017-08", ["--thetas", f"{theta},{theta}"]),
        ("2017-08", ["--thetas", "all", "--l3-root", str(l3 / "vacio")]),
        ("2017-08", ["--thetas", "abc"]),
        ("2017-08", ["--thetas", f"{theta},"]),
        ("2017-08", ["--thetas", "Infinity"]),
        ("2017-08", ["--thetas", "1e30"]),
        ("2017-08", ["--thetas", "0"]),
        ("2017-08", ["--thetas", "-0.001"]),
        ("2017-08", ["--catalog", str(l3 / "no-existe.yaml")]),
    ):
        roots = ["--l1-root", str(l1), "--l2-root", str(l2), "--l3-root", str(l3)]
        with pytest.raises(SystemExit) as exit_:
            check.main(["--month", month, *roots, *others])
        assert exit_.value.code == 2
    # Sin raíces ni OPS_RESULTS_URI no sabe de qué proyecto es.
    with pytest.raises(SystemExit) as exit_:
        check.main(["--month", "2017-08", "--thetas", theta])
    assert exit_.value.code == 2


def test_el_script_no_importa_l3_ni_pyarrow_dataset():
    text = (SCRIPTS / "l3_check_month.py").read_text()
    assert not re.search(r"^\s*(import|from)\s+l3_dc_descriptors", text, re.MULTILINE)
    assert not re.search(r"^\s*(import|from)\s+pyarrow\.dataset", text, re.MULTILINE)
    assert "ds.dataset(" not in text and "pq.read_table(" not in text


def test_la_tabla_por_theta_se_guarda_en_ops_results(check, lake, monkeypatch, capsys):
    l1, l2, l3, thetas = lake
    saved = {}

    class Blob:
        def __init__(self, name):
            self.name = name

        def upload_from_string(self, body):
            saved[self.name] = body

    class Bucket:
        def blob(self, name):
            return Blob(name)

    class Client:
        def bucket(self, name):
            saved["bucket"] = name
            return Bucket()

    from google.cloud import storage

    monkeypatch.setattr(storage, "Client", Client)
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://p-ops/results/ejec/")
    assert check.main(argv(l1, l2, l3, "--thetas", ",".join(thetas))) == 0
    (name,) = [k for k in saved if k.startswith("results/ejec/l3_check_month_2017-08_")]
    lines = saved[name].splitlines()
    assert lines[0] == "theta,filas,diferencias,segundos"
    assert [x.split(",")[0] for x in lines[1:]] == thetas
    assert all(x.split(",")[2] == "0" for x in lines[1:])
    assert saved["bucket"] == "p-ops"


# --- la agregación con números que desbordan int64 si se hace de frente -----------------


def frame(times, prices, quantities, sold) -> Frame:
    return Frame(
        pa.array(times, pa.int64()),
        pa.array([Decimal(p).scaleb(-8) for p in prices], DEC),
        pa.array([Decimal(q).scaleb(-8) for q in quantities], DEC),
        pa.array(sold, pa.bool_()),
    )


def naive(prices, quantities, sold):
    buy = sum(q for q, s in zip(quantities, sold) if not s)
    sell = sum(q for q, s in zip(quantities, sold) if s)
    notional = sum(p * q for p, q in zip(prices, quantities))
    return buy, sell, notional


@pytest.mark.parametrize("n", [10, 200, 5000])
@pytest.mark.parametrize("scale", [10**6, 10**12, 10**17])
def test_la_fase_suma_exacto_con_cifras_grandes(check, n, scale):
    rng = np.random.default_rng(n)
    prices = [int(x) + 1 for x in rng.integers(1, scale, n)]
    quantities = [int(x) + 1 for x in rng.integers(1, scale, n)]
    sold = [bool(x) for x in rng.integers(0, 2, n)]
    times = sorted(int(x) for x in rng.integers(0, 50, n))
    phase = check.Phase(times[-1])
    half = n // 3
    phase.consume(
        [
            frame(times[:half], prices[:half], quantities[:half], sold[:half]),
            frame(times[half:], prices[half:], quantities[half:], sold[half:]),
        ]
    )
    buy, sell, notional = naive(prices, quantities, sold)
    assert (phase.ticks, phase.buy, phase.sell, phase.notional) == (
        n,
        buy,
        sell,
        notional,
    )
    assert (phase.high, phase.low) == (max(prices), min(prices))
    assert phase.duration() == times[-1] - times[0]
    at = [p for p, t in zip(prices, times) if t == times[-1]]
    assert (phase.instant_ticks, phase.instant_high, phase.instant_low) == (
        len(at),
        max(at),
        min(at),
    )
    assert phase.vwap() == check.half_even(notional, buy + sell)


def test_la_division_redondea_half_even(check):
    assert check.half_even(5, 2) == 2
    assert check.half_even(7, 2) == 4
    assert check.half_even(-5, 2) == -2
    assert check.half_even(10, 4) == 2 and check.half_even(11, 4) == 3


# --- un lago con cantidades y lados al azar, de varios meses --------------------------------


def test_cero_diferencias_con_lados_y_cantidades_al_azar_en_dos_meses(
    check, tmp_path, capsys
):
    ticks = l3_lake.fixture_ticks(seed=11)
    cuts = l3_lake.month_cuts(ticks, len(ticks) // 2)
    lake = l3_lake.build_lake(tmp_path, ticks, cuts, l1_row_group=150)
    first, second = lake.months
    for month in (first, second):
        summarize_month(
            lake.l1,
            lake.l2,
            lake.l3,
            month,
            lake.theta_texts,
            series_start=first,
        )
        code = check.main(
            [
                "--month",
                f"{month[0]:04d}-{month[1]:02d}",
                "--l1-root",
                str(lake.l1),
                "--l2-root",
                str(lake.l2),
                "--l3-root",
                str(lake.l3),
                "--thetas",
                "all",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0, out
        assert re.search(r"filas=[1-9]\d* diferencias=0 ", out)
