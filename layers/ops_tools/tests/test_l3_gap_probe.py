"""`l3_gap_probe.py` contra un lago local de L1 con un `consolidated.parquet` sintético.

El mes tiene 21 ticks en dos row groups (11 y 10) y 20 huecos elegidos a mano:

    row group 0 (10 huecos): 0 0 5 5 5 10 10 20 40 1 000 000
    entre grupos (1 hueco):  12 000 000
    row group 1 (9 huecos):  5 10 20 90 000 000 100 100 100 7 0

Ordenados, con la cuenta acumulada:

    0 ×3 (3)  5 ×4 (7)  7 (8)  10 ×3 (11)  20 ×2 (13)  40 (14)  100 ×3 (17)
    1 000 000 (18)  12 000 000 (19)  90 000 000 (20)

Cuantil por rango más cercano, lugar ⌈20·q⌉: p50 → 10.º = 10; p90 → 18.º = 1 000 000 (justo
en el borde del histograma); p99, p99,9 y p99,99 → 20.º = 90 000 000. El máximo es el hueco
de 90 s, que cierra el tick 15 (el quinto del row group 1). Pasan de 1 s: 3 (1 s cuenta);
de 10 s: 2; de 60 s: 1; de 600 s: 0.
"""

import csv
import importlib.util
import itertools
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
BASE = 1_700_000_000_000_000
GAPS_0 = [0, 0, 5, 5, 5, 10, 10, 20, 40, 1_000_000]
GAP_BETWEEN = 12_000_000
GAPS_1 = [5, 10, 20, 90_000_000, 100, 100, 100, 7, 0]
ALL_GAPS = [*GAPS_0, GAP_BETWEEN, *GAPS_1]
LAST_LINE = re.compile(
    r"gap probe: mes=(?P<mes>\S+) ticks=(?P<ticks>\S+) p50_us=(?P<p50>\S+) "
    r"p90_us=(?P<p90>\S+) p99_us=(?P<p99>\S+) p999_us=(?P<p999>\S+) "
    r"p9999_us=(?P<p9999>\S+) max_us=(?P<max>\S+) ge_1s=(?P<ge1>\S+) "
    r"ge_10s=(?P<ge10>\S+) ge_60s=(?P<ge60>\S+) ge_600s=(?P<ge600>\S+) "
    r"pared_s=(?P<pared>\S+) veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_gap_probe", SCRIPTS / "l3_gap_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load()


def write_month(root: Path, month: str, gaps_by_group: list[list[int]], first=BASE):
    """Un `consolidated.parquet` con un row group por lista de huecos.

    Cada grupo empieza con el hueco que lo une al anterior, salvo el primero, que parte de
    `first`. `agg_trade_id` = 1000 + posición global del tick.
    """
    year, number = month.split("-")
    directory = root / SERIES / f"year={year}/month={number}"
    directory.mkdir(parents=True, exist_ok=True)
    times = [first]
    sizes = [len(gaps_by_group[0]) + 1]
    times += list(itertools.accumulate(gaps_by_group[0], initial=first))[1:]
    for gaps in gaps_by_group[1:]:
        times += list(itertools.accumulate(gaps, initial=times[-1]))[1:]
        sizes.append(len(gaps))
    ids = [1000 + i for i in range(len(times))]
    schema = pa.schema([("agg_trade_id", pa.int64()), ("transact_time", pa.int64())])
    with pq.ParquetWriter(directory / "consolidated.parquet", schema) as writer:
        start = 0
        for size in sizes:
            writer.write_table(
                pa.table(
                    {
                        "agg_trade_id": ids[start : start + size],
                        "transact_time": times[start : start + size],
                    },
                    schema=schema,
                )
            )
            start += size
    return times


@pytest.fixture
def lake(tmp_path):
    root = tmp_path / "l1"
    times = write_month(root, "2026-08", [GAPS_0, [GAP_BETWEEN, *GAPS_1]])
    return root, times


def argv(root: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--month",
        "2026-08",
        "--landing-root",
        str(root),
        "--out",
        str(out),
        *extra,
    ]


def last_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    match = LAST_LINE.match(line)
    assert match, line
    return match.groupdict()


def read_csv(path: Path) -> dict[str, dict]:
    with open(path) as f:
        return {r["medida"]: r for r in csv.DictReader(f)}


def test_cuantiles_maximo_y_conteos_a_mano(probe, lake, tmp_path, capsys):
    root, times = lake
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out)) == 0
    line = last_line(capsys)
    rows = read_csv(out)

    expected = {
        "ticks": 21,
        "huecos": 20,
        "p50_us": 10,
        "p90_us": 1_000_000,
        "p99_us": 90_000_000,
        "p999_us": 90_000_000,
        "p9999_us": 90_000_000,
        "max_us": 90_000_000,
        "ge_1s": 3,
        "ge_10s": 2,
        "ge_60s": 1,
        "ge_600s": 0,
    }
    assert {m: int(r["valor"]) for m, r in rows.items()} == expected
    assert list(rows) == list(expected)
    assert line["mes"] == "2026-08"
    assert line["veredicto"] == "ok"
    assert float(line["pared"]) >= 0
    assert {k: int(line[k]) for k in ("ticks", "p50", "p90", "p99", "max")} == {
        "ticks": 21,
        "p50": 10,
        "p90": 1_000_000,
        "p99": 90_000_000,
        "max": 90_000_000,
    }
    assert (line["p999"], line["p9999"]) == ("90000000", "90000000")
    assert [line[k] for k in ("ge1", "ge10", "ge60", "ge600")] == ["3", "2", "1", "0"]

    # El hueco máximo cierra el tick 15 (índice global), el quinto del segundo row group.
    assert rows["max_us"]["agg_trade_id"] == "1015"
    assert int(rows["max_us"]["transact_time"]) == times[15]
    assert times[15] - times[14] == 90_000_000
    assert rows["max_us"]["legible"] == "1.50 min"
    assert rows["p50_us"]["legible"] == "0.00001 s"
    assert rows["p90_us"]["legible"] == "1 s"


def test_hueco_entre_row_groups_cuenta(probe, tmp_path, capsys):
    """El único hueco grande está entre grupos: el máximo cierra el primer tick del segundo."""
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[1, 2], [700_000_000, 3]])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out)) == 0
    rows = read_csv(out)
    assert rows["max_us"]["valor"] == "700000000"
    assert rows["max_us"]["agg_trade_id"] == "1003"
    assert rows["ge_600s"]["valor"] == "1"
    assert rows["huecos"]["valor"] == "4"
    assert last_line(capsys)["max"] == "700000000"


def test_maximo_repetido_toma_el_primero(probe, tmp_path):
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[5, 9], [3, 9]])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out)) == 0
    assert read_csv(out)["max_us"]["agg_trade_id"] == "1002"


def test_lee_un_row_group_a_la_vez_y_solo_transact_time(
    probe, lake, tmp_path, monkeypatch
):
    calls = []
    original = pq.ParquetFile.read_row_group

    def spy(self, i, columns=None, **kwargs):
        calls.append((i, tuple(columns)))
        return original(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy)
    assert probe.main(argv(lake[0], tmp_path / "salida.csv")) == 0
    assert calls == [
        (0, ("transact_time",)),
        (1, ("transact_time",)),
        (1, ("agg_trade_id",)),  # solo el row group del máximo
    ]


def test_out_directorio_usa_el_nombre_por_defecto(probe, lake, tmp_path):
    out = tmp_path / "resultados"
    assert probe.main(argv(lake[0], out)[:-2] + ["--out", f"{out}/"]) == 0
    assert (out / "l3_gap_probe_2026-08.csv").is_file()


def test_un_solo_tick_no_tiene_huecos(probe, tmp_path, capsys):
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[]])
    out = tmp_path / "salida.csv"
    assert probe.main(argv(root, out)) == 0
    line = last_line(capsys)
    assert (line["ticks"], line["p50"], line["max"], line["ge1"]) == (
        "1",
        "-",
        "-",
        "0",
    )
    assert read_csv(out)["huecos"]["valor"] == "0"


def test_mes_sin_archivo_da_codigo_1(probe, lake, tmp_path, capsys):
    argv_ = ["--month", "2026-07", "--landing-root", str(lake[0])]
    assert probe.main([*argv_, "--out", str(tmp_path / "salida.csv")]) == 1
    assert not (tmp_path / "salida.csv").exists()
    out = capsys.readouterr().out
    assert "FALLO falta el archivo" in out
    line = out.strip().splitlines()[-1]
    assert line.startswith("gap probe: mes=2026-07 ticks=- ")
    assert line.endswith("veredicto=fallo")


def test_archivo_ilegible_da_codigo_1(probe, lake, tmp_path, capsys):
    path = lake[0] / SERIES / "year=2026/month=08/consolidated.parquet"
    path.write_bytes(b"no es parquet")
    assert probe.main(argv(lake[0], tmp_path / "salida.csv")) == 1
    assert "FALLO no se pudo leer" in capsys.readouterr().out


def test_tiempos_que_decrecen_dan_codigo_1(probe, tmp_path, capsys):
    root = tmp_path / "l1"
    write_month(root, "2026-08", [[5, -3, 4]])
    assert probe.main(argv(root, tmp_path / "salida.csv")) == 1
    assert "decrece" in capsys.readouterr().out


def test_no_importa_ninguna_capa():
    source = (SCRIPTS / "l3_gap_probe.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]


@pytest.mark.parametrize(
    "extra",
    [
        [],  # falta --month
        ["--month", "2026-13"],
        ["--month", "2026-8"],
        ["--month", "agosto"],
    ],
)
def test_args_invalidos_dan_codigo_2(probe, extra, tmp_path):
    with pytest.raises(SystemExit) as raised:
        probe.main(["--landing-root", str(tmp_path), *extra])
    assert raised.value.code == 2


def test_sin_raiz_y_sin_ops_results_uri_es_uso(probe, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        probe.main(["--month", "2026-08"])
    assert raised.value.code == 2


def test_out_fuera_de_results_es_uso(probe, lake, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proyecto-ops/results/ejecucion/")
    with pytest.raises(SystemExit) as raised:
        probe.main(
            [
                "--month",
                "2026-08",
                "--landing-root",
                str(lake[0]),
                "--out",
                "gs://otro/x.csv",
            ]
        )
    assert raised.value.code == 2
