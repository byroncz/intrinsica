"""`l3_tick_pairs.py` contra un lago local de L1 con `consolidated.parquet` sintéticos.

Precios en centavos (en el archivo van como `decimal128(18, 8)`); tiempos en µs sobre BASE, que es
múltiplo de 1 000. Lado: C = compra (`is_buyer_maker` falso), V = venta (verdadero).

Mes 2026-08, dos row groups (5 y 4 ticks), 8 pares:

    tick  grupo  tiempo  precio  lado   par que lo cierra
     0     0     +0      10000   C
     1     0     +500    10001   C      0: espera 500 µs, mismo ms; C→C sube
     2     0     +500    10002   C      1: espera 0;                C→C sube   } escalera de
     3     0     +500    10003   C      2: espera 0;                C→C sube   } tres ticks
     4     0     +2000   10003   V      3: espera 1 500 µs, otro ms; C→V igual
     5     1     +2000   10002   V      4: cruza grupos; espera 0;  V→V baja
     6     1     +2400   10002   C      5: espera 400 µs, mismo ms; V→C igual
     7     1     +3000   10004   V      6: espera 600 µs, otro ms;  C→V sube
     8     1     +3000   10003   C      7: espera 0;                V→C baja

Tabla `todos` (filas: transición; columnas: baja, igual, sube):
    C→C 0 0 3 | C→V 0 1 1 | V→C 1 1 0 | V→V 1 0 0
`cero` (pares 1, 2, 4, 7): C→C 0 0 2 | V→C 1 0 0 | V→V 1 0 0, y las demás en 0.
`ms` (los de `cero` más 0 y 5): C→C 0 0 3 | V→C 1 1 0 | V→V 1 0 0.
espera_cero = 4, mismo_ms = 6, pares = 8: frac_cero = 0,5 y frac_mismo_ms = 0,75.
Escalera (C→C sube + V→V baja): en `cero` 3 de 4 = 0,75; en `ms` 4 de 6.

Cambios de precio (centavos): +1 +1 +1 0 −1 0 +2 −1. Los distintos de cero: + + + − + −, cinco
pasos entre uno y el siguiente y tres cambian de signo (+→−, −→+, +→−). El signo + del par 2 (grupo
0) se arrastra hasta el par 4 (grupo 1), que baja: ese cambio solo sale si se conserva el signo
entre row groups. Reversión 3/5 = 0,6.
Autocorrelación: N = 8, Σd = 3, m = 3/8, Σd² = 9, Σdᵢdᵢ₊₁ = 0, d₁ = 1 y d₈ = −1:
numerador = 0 − m·((3 + 1) + (3 − 1)) + 7·m² = −81/64; denominador = 9 − 8·m² = 63/8; r = −9/56.

Mes 2026-09: precios 100 101 100 101 100 (todos compra, 1 ms entre uno y otro, dos row groups de
3 y 2): cambios + − + −, reversión 1; Σdᵢdᵢ₊₁ = −3, Σd² = 4, m = 0: autocorrelación −3/4. Sin
pares de espera cero ni de mismo ms.

Mes 2026-10: precios 100 101 103 106 110 en row groups de 2 y 3: cambios 1 2 3 4, reversión 0;
Σd = 10, Σd² = 30, Σdᵢdᵢ₊₁ = 20, m = 5/2: numerador = 20 − m·(6 + 9) + 3·m² = 5/4; denominador =
30 − 4·m² = 5; autocorrelación 1/4.
"""

import csv
import importlib.util
import re
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
BASE = 1_700_000_000_000_000
DEC = pa.decimal128(18, 8)
SCHEMA = pa.schema(
    [
        ("agg_trade_id", pa.int64()),
        ("price", DEC),
        ("is_buyer_maker", pa.bool_()),
        ("transact_time", pa.int64()),
    ]
)
COLUMNS = ("agg_trade_id", "price", "is_buyer_maker", "transact_time")
LAST_LINE = re.compile(
    r"tick pairs: meses=(?P<meses>\d+) ticks=(?P<ticks>\d+) pared_s=(?P<pared>\S+)$"
)

C, V = False, True
MIXED = [
    [  # row group 0
        (BASE + 0, 10000, C),
        (BASE + 500, 10001, C),
        (BASE + 500, 10002, C),
        (BASE + 500, 10003, C),
        (BASE + 2000, 10003, V),
    ],
    [  # row group 1
        (BASE + 2000, 10002, V),
        (BASE + 2400, 10002, C),
        (BASE + 3000, 10004, V),
        (BASE + 3000, 10003, C),
    ],
]
ALTERNATING = [
    [(BASE + 1000 * i, p, C) for i, p in enumerate((100, 101, 100))],
    [(BASE + 1000 * i, p, C) for i, p in enumerate((101, 100), start=3)],
]
RISING = [
    [(BASE + 1000 * i, p, C) for i, p in enumerate((100, 101))],
    [(BASE + 1000 * i, p, C) for i, p in enumerate((103, 106, 110), start=2)],
]


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_tick_pairs", SCRIPTS / "l3_tick_pairs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load()


def write_month(root: Path, month: str, groups: list[list[tuple]], ids=None):
    """Un `consolidated.parquet` con un row group por lista de `(tiempo, centavos, vendió)`.

    `agg_trade_id` = 1000 + posición global del tick, salvo que `ids` lo fije.
    """
    year, number = month.split("-")
    directory = root / SERIES / f"year={year}/month={number}"
    directory.mkdir(parents=True, exist_ok=True)
    position = 0
    with pq.ParquetWriter(directory / "consolidated.parquet", SCHEMA) as writer:
        for group in groups:
            size = len(group)
            writer.write_table(
                pa.table(
                    {
                        "agg_trade_id": (
                            ids[position : position + size]
                            if ids
                            else [1000 + position + i for i in range(size)]
                        ),
                        "price": pa.array(
                            [Decimal(cents).scaleb(-2) for _, cents, _ in group],
                            type=DEC,
                        ),
                        "is_buyer_maker": [sold for _, _, sold in group],
                        "transact_time": [time for time, _, _ in group],
                    },
                    schema=SCHEMA,
                )
            )
            position += size


@pytest.fixture
def lake(tmp_path):
    root = tmp_path / "l1"
    write_month(root, "2026-08", MIXED)
    write_month(root, "2026-09", ALTERNATING)
    write_month(root, "2026-10", RISING)
    return root


def argv(root: Path, out: Path, months="2026-08,2026-09,2026-10", *extra) -> list:
    return ["--months", months, "--landing-root", str(root), "--out", str(out), *extra]


def last_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    match = LAST_LINE.match(line)
    assert match, line
    return match.groupdict()


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    """`{mes: {medida: valor}}`."""
    result: dict[str, dict[str, str]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            result.setdefault(row["mes"], {})[row["medida"]] = row["valor"]
    return result


def table(rows: dict[str, str], name: str) -> list[list[int]]:
    return [
        [int(rows[f"{name}.{t}.{c}"]) for c in ("baja", "igual", "sube")]
        for t in ("compra_compra", "compra_venta", "venta_compra", "venta_venta")
    ]


def test_conteos_tablas_y_derivados_a_mano(probe, lake, tmp_path, capsys):
    out = tmp_path / "salida.csv"
    assert probe.main(argv(lake, out)) == 0
    stdout = capsys.readouterr().out
    rows = read_csv(out)["2026-08"]

    assert {k: rows[k] for k in ("ticks", "pares", "espera_cero", "mismo_ms")} == {
        "ticks": "9",
        "pares": "8",
        "espera_cero": "4",
        "mismo_ms": "6",
    }
    assert table(rows, "todos") == [[0, 0, 3], [0, 1, 1], [1, 1, 0], [1, 0, 0]]
    assert table(rows, "cero") == [[0, 0, 2], [0, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert table(rows, "ms") == [[0, 0, 3], [0, 0, 0], [1, 1, 0], [1, 0, 0]]
    assert float(rows["frac_cero"]) == 0.5
    assert float(rows["frac_mismo_ms"]) == 0.75
    assert float(rows["frac_escalera_cero"]) == 0.75
    assert float(rows["frac_escalera_ms"]) == pytest.approx(4 / 6)
    assert float(rows["reversion"]) == pytest.approx(0.6)
    assert float(rows["autocorr_1"]) == pytest.approx(-9 / 56)

    # Todos los derivados van en la línea del mes, no solo en el CSV.
    line = next(row for row in stdout.splitlines() if row.startswith("mes 2026-08:"))
    for expected in (
        "ticks=9",
        "pares=8",
        "espera_cero=4",
        "mismo_ms=6",
        "frac_cero=0.500000",
        "frac_mismo_ms=0.750000",
        "frac_escalera_cero=0.750000",
        "frac_escalera_ms=0.666667",
        "reversion=0.600000",
        "autocorr_1=-0.160714",
    ):
        assert expected in line
    assert re.search(r"pared_s=\d", line)
    assert "  todos compra→compra  baja=0 igual=0 sube=3" in stdout


def test_el_par_que_cruza_row_groups_cuenta(probe, tmp_path):
    """Los ticks 4 y 5 son de grupos distintos: su par (V→V, baja, espera 0) está en las tablas."""
    root = tmp_path / "l1"
    write_month(root, "2026-08", MIXED)
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out, "2026-08")) == 0
    rows = read_csv(out)["2026-08"]
    # Sin ese par, `venta→venta` no tendría ninguna baja, y `espera_cero` sería 3.
    assert rows["todos.venta_venta.baja"] == "1"
    assert rows["cero.venta_venta.baja"] == "1"
    assert rows["espera_cero"] == "4"
    assert rows["pares"] == "8"


def test_mismo_milisegundo_con_distinto_microsegundo(probe, tmp_path):
    """Dos ticks a 500 µs: mismo ms sin ser espera cero."""
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[(BASE + 0, 100, C), (BASE + 500, 101, C)]])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out, "2026-08")) == 0
    rows = read_csv(out)["2026-08"]
    assert (rows["pares"], rows["espera_cero"], rows["mismo_ms"]) == ("1", "0", "1")
    assert rows["frac_cero"] == "0"
    assert float(rows["frac_mismo_ms"]) == 1
    assert rows["frac_escalera_cero"] == ""  # el conjunto `cero` está vacío
    assert float(rows["frac_escalera_ms"]) == 1  # C→C y sube


def test_escalera_de_compras_y_de_ventas(probe, tmp_path):
    """Compra→compra sube y venta→venta baja son escalera; el mismo lado en sentido contrario, no."""
    root = tmp_path / "l1"
    ticks = [
        (BASE, 100, C),
        (BASE, 101, C),  # C→C sube: escalera
        (BASE, 100, C),  # C→C baja: no
        (BASE, 100, V),  # C→V igual: no
        (BASE, 99, V),  # V→V baja: escalera
        (BASE, 100, V),  # V→V sube: no
    ]
    write_month(root, "2026-08", [ticks])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out, "2026-08")) == 0
    rows = read_csv(out)["2026-08"]
    assert rows["espera_cero"] == "5"
    assert float(rows["frac_escalera_cero"]) == pytest.approx(2 / 5)
    assert float(rows["frac_cero"]) == 1


def test_serie_que_alterna_tiene_reversion_1(probe, lake, tmp_path):
    out = tmp_path / "salida.csv"
    assert probe.main(argv(lake, out)) == 0
    rows = read_csv(out)["2026-09"]
    assert float(rows["reversion"]) == 1
    assert float(rows["autocorr_1"]) == pytest.approx(-0.75)
    assert (rows["espera_cero"], rows["mismo_ms"]) == ("0", "0")
    assert rows["frac_escalera_cero"] == ""


def test_serie_que_solo_sube_tiene_reversion_0(probe, lake, tmp_path):
    out = tmp_path / "salida.csv"
    assert probe.main(argv(lake, out)) == 0
    rows = read_csv(out)["2026-10"]
    assert float(rows["reversion"]) == 0
    assert float(rows["autocorr_1"]) == pytest.approx(0.25)
    assert table(rows, "todos")[0] == [0, 0, 4]  # compra→compra: cuatro subidas


def test_cambios_constantes_o_sin_cambios_no_tienen_autocorrelacion_ni_reversion(
    probe, tmp_path
):
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[(BASE + 1000 * i, 100, C) for i in range(4)]])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out, "2026-08")) == 0
    rows = read_csv(out)["2026-08"]
    assert (rows["reversion"], rows["autocorr_1"]) == ("", "")
    assert rows["pares"] == "3"


def test_un_solo_tick_no_tiene_pares(probe, tmp_path, capsys):
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[(BASE, 100, C)]])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out, "2026-08")) == 0
    rows = read_csv(out)["2026-08"]
    assert (rows["ticks"], rows["pares"], rows["frac_cero"]) == ("1", "0", "")
    assert last_line(capsys)["ticks"] == "1"


def test_la_ultima_linea_y_una_linea_por_mes(probe, lake, tmp_path, capsys):
    assert probe.main(argv(lake, tmp_path / "salida.csv")) == 0
    stdout = capsys.readouterr().out
    final = LAST_LINE.match(stdout.strip().splitlines()[-1])
    assert final and (final["meses"], final["ticks"]) == ("3", "19")  # 9 + 5 + 5
    assert float(final["pared"]) >= 0
    months = [row for row in stdout.splitlines() if row.startswith("mes ")]
    assert [row.split(":")[0] for row in months] == [
        "mes 2026-08",
        "mes 2026-09",
        "mes 2026-10",
    ]


def test_lee_un_row_group_a_la_vez_y_las_cuatro_columnas(
    probe, lake, tmp_path, monkeypatch
):
    calls = []
    original = pq.ParquetFile.read_row_group

    def spy(self, i, columns=None, **kwargs):
        calls.append((i, tuple(columns)))
        return original(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy)
    assert probe.main(argv(lake, tmp_path / "salida.csv", "2026-08")) == 0
    assert calls == [(0, COLUMNS), (1, COLUMNS)]


def test_out_directorio_usa_el_nombre_por_defecto(probe, lake, tmp_path):
    out = tmp_path / "resultados"
    assert probe.main(argv(lake, out, "2026-08")[:-2] + ["--out", f"{out}/"]) == 0
    assert (out / "l3_tick_pairs.csv").is_file()


def test_un_mes_sin_archivo_da_codigo_1_y_sigue_con_los_demas(
    probe, lake, tmp_path, capsys
):
    out = tmp_path / "salida.csv"
    assert probe.main(argv(lake, out, "2026-07,2026-08")) == 1
    stdout = capsys.readouterr().out
    assert "FALLO mes 2026-07: falta el archivo" in stdout
    assert last_line_text(stdout).startswith("tick pairs: meses=1 ticks=9 ")
    assert list(read_csv(out)) == ["2026-08"]


def last_line_text(stdout: str) -> str:
    return stdout.strip().splitlines()[-1]


def test_archivo_ilegible_da_codigo_1(probe, lake, tmp_path, capsys):
    path = lake / SERIES / "year=2026/month=08/consolidated.parquet"
    path.write_bytes(b"no es parquet")
    assert probe.main(argv(lake, tmp_path / "salida.csv", "2026-08")) == 1
    stdout = capsys.readouterr().out
    assert "FALLO mes 2026-08: no se pudo leer" in stdout
    assert last_line_text(stdout).startswith("tick pairs: meses=0 ticks=0 ")
    assert not (tmp_path / "salida.csv").exists()


def test_tiempos_que_decrecen_dan_codigo_1(probe, tmp_path, capsys):
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[(BASE + 5, 100, C), (BASE + 1, 101, C)]])
    assert probe.main(argv(root, tmp_path / "salida.csv", "2026-08")) == 1
    assert "transact_time decrece" in capsys.readouterr().out


def test_agg_trade_id_que_no_crece_da_codigo_1(probe, tmp_path, capsys):
    root = tmp_path / "l1"
    write_month(root, "2026-08", MIXED, ids=[1, 2, 3, 4, 5, 5, 6, 7, 8])  # repite el 5
    assert probe.main(argv(root, tmp_path / "salida.csv", "2026-08")) == 1
    assert "agg_trade_id no crece" in capsys.readouterr().out


def test_no_importa_ninguna_capa():
    source = (SCRIPTS / "l3_tick_pairs.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]


@pytest.mark.parametrize(
    "extra",
    [
        [],  # falta --months
        ["--months", ""],
        ["--months", "2026-13"],
        ["--months", "2026-8"],
        ["--months", "agosto"],
        ["--months", "2026-08,2026-08"],
    ],
)
def test_args_invalidos_dan_codigo_2(probe, extra, tmp_path):
    with pytest.raises(SystemExit) as raised:
        probe.main(["--landing-root", str(tmp_path), *extra])
    assert raised.value.code == 2


def test_sin_raiz_y_sin_ops_results_uri_es_uso(probe, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        probe.main(["--months", "2026-08"])
    assert raised.value.code == 2


def test_out_fuera_de_results_es_uso(probe, lake, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proyecto-ops/results/ejecucion/")
    with pytest.raises(SystemExit) as raised:
        probe.main(
            [
                "--months",
                "2026-08",
                "--landing-root",
                str(lake),
                "--out",
                "gs://otro/x.csv",
            ]
        )
    assert raised.value.code == 2
