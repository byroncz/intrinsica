"""`l3_phase_sizes.py` contra un lago local con `summaries.parquet` sintéticos (ITSC-431).

El lago de cada prueba se escribe en `tmp_path` con la misma disposición hive que el real. Cada
`summaries.parquet` solo trae `c_ticks` y `os_ticks`, que son las dos columnas que lee la sonda, y
se escribe con row groups de 2 filas para que cada archivo tenga varios.
"""

import csv
import importlib.util
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
THETAS = ("0.00010000", "0.00031313")
THRESHOLDS = (16, 64, 256, 1024, 4096, 10_000, 16_384, 65_536)
LAST_LINE = re.compile(
    r"phase sizes: thetas=(?P<thetas>\d+) archivos=(?P<archivos>\d+) "
    r"faltantes=(?P<faltantes>\d+) fases_c=(?P<fases_c>\d+) fases_os=(?P<fases_os>\d+)$"
)
# Los bordes: 15 y 16 (el mínimo de la familia F), 9 999, 10 000 y 10 001 (el `--sample`).
EDGE_C = [15, 16, 9_999, 10_000, 10_001]
EDGE_OS = [0, 15, 16, 10_000, 0]  # dos overshoots vacíos


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_phase_sizes", SCRIPTS / "l3_phase_sizes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summaries(root: Path, theta: str, month: str) -> Path:
    year, number = month.split("-")
    base = root / SERIES / f"theta={theta}/year={year}/month={number}"
    return base / "summaries.parquet"


def write_summaries(root: Path, theta: str, month: str, c: list[int], os: list[int]):
    path = summaries(root, theta, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"c_ticks": pa.array(c, pa.int64()), "os_ticks": pa.array(os, pa.int64())}
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def write_catalog(root: Path, thetas=THETAS) -> None:
    numbers = [int(float(t) * 10**8 + 0.5) for t in thetas]
    (root / "thetas.yaml").write_text(
        "scale: 100000000\nthetas:\n" + "".join(f"  - {n}\n" for n in numbers)
    )


@pytest.fixture
def lake(tmp_path):
    """θ0 con los bordes en enero y datos menores en febrero y marzo; θ1 sin febrero."""
    l3 = tmp_path / "l3"
    write_summaries(l3, THETAS[0], "2023-01", EDGE_C, EDGE_OS)
    write_summaries(l3, THETAS[0], "2023-02", [20, 300, 5_000], [0, 0, 70])
    write_summaries(l3, THETAS[0], "2023-03", [1, 70_000], [40_000, 0])
    write_summaries(l3, THETAS[1], "2023-01", [1_024, 4_096, 16_384], [16, 64, 256])
    write_summaries(l3, THETAS[1], "2023-03", [65_536], [0])
    write_catalog(tmp_path)
    return tmp_path


def sizes(lake: Path, capsys, *extra: str) -> tuple[int, str, list[dict]]:
    out = lake / "salida" / "l3_phase_sizes.csv"
    code = load().main(
        [
            "--catalog",
            str(lake / "thetas.yaml"),
            "--l3-root",
            str(lake / "l3"),
            "--out",
            str(out),
            *extra,
        ]
    )
    text = capsys.readouterr().out
    rows = list(csv.DictReader(out.open())) if out.exists() else []
    return code, text, rows


def last_line(text: str) -> dict[str, str]:
    match = LAST_LINE.fullmatch(text.strip().splitlines()[-1])
    assert match, text
    return match.groupdict()


def by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["theta"], r["fase"]): r for r in rows}


def expected(ticks: list[int], sample: int = 10_000) -> dict[str, int]:
    """El oráculo: los conteos de una lista de ticks, sin pyarrow."""
    capped = [min(t, sample) for t in ticks]
    return {
        "fases": len(ticks),
        **{f"ge_{n}": sum(t >= n for t in ticks) for n in THRESHOLDS},
        "puntos": sum(capped),
        "pares": sum(t * t for t in capped),
    }


def check(row: dict, ticks: list[int], sample: int = 10_000) -> None:
    for name, value in expected(ticks, sample).items():
        assert row[name] == str(value), name


def test_the_edges_are_counted_exactly(lake, capsys):
    code, _, rows = sizes(lake, capsys, "--to", "2023-01")
    assert code == 0
    c, os = by_key(rows)[THETAS[0], "c"], by_key(rows)[THETAS[0], "os"]
    # Valores a mano: 15 no llega a 16; 9 999 y 10 000 pasan 4 096; 10 000 y 10 001, 10 000.
    assert c["fases"] == "5"
    assert [c[f"ge_{n}"] for n in THRESHOLDS] == [
        "4",
        "3",
        "3",
        "3",
        "3",
        "2",
        "0",
        "0",
    ]
    assert c["puntos"] == str(15 + 16 + 9_999 + 10_000 + 10_000)
    assert c["pares"] == str(15**2 + 16**2 + 9_999**2 + 2 * 10_000**2)
    # `os` cuenta solo las de `os_ticks > 0`; las otras dos son overshoots vacíos.
    assert os["fases"] == "3" and os["os_vacias"] == "2"
    assert [os[f"ge_{n}"] for n in THRESHOLDS] == [
        "2",
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
        "0",
    ]
    assert os["puntos"] == str(15 + 16 + 10_000)
    assert os["pares"] == str(15**2 + 16**2 + 10_000**2)
    assert c["os_vacias"] == ""  # `c` no tiene overshoot vacío


def test_every_file_matches_the_oracle_and_the_totals_add_up(lake, capsys):
    code, text, rows = sizes(lake, capsys)
    assert code == 0
    table = by_key(rows)
    c0 = EDGE_C + [20, 300, 5_000] + [1, 70_000]
    os0 = [t for t in EDGE_OS + [0, 0, 70] + [40_000, 0] if t > 0]
    c1 = [1_024, 4_096, 16_384] + [65_536]
    os1 = [16, 64, 256]
    check(table[THETAS[0], "c"], c0)
    check(table[THETAS[0], "os"], os0)
    check(table["total", "c"], c0 + c1)
    check(table["total", "os"], os0 + os1)
    assert table["total", "os"]["os_vacias"] == "6"
    assert last_line(text) == {
        "thetas": "2",
        "archivos": "5",
        "faltantes": "1",
        "fases_c": str(len(c0 + c1)),
        "fases_os": str(len(os0 + os1)),
    }


def test_a_theta_line_is_printed_as_each_one_closes_and_the_totals_follow(lake, capsys):
    _, text, _ = sizes(lake, capsys)
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith(f"θ={THETAS[0]}"))
    assert lines[first + 1].startswith(f"θ={THETAS[1]}")
    assert lines[first + 2].startswith("total ")


def test_an_overshoot_that_is_empty_everywhere_leaves_zero_os_phases(lake, capsys):
    write_summaries(lake / "l3", THETAS[0], "2023-01", [10, 20, 30], [0, 0, 0])
    code, _, rows = sizes(lake, capsys, "--from", "2023-01", "--to", "2023-01")
    assert code == 0
    os = by_key(rows)[THETAS[0], "os"]
    assert (os["fases"], os["os_vacias"], os["puntos"], os["pares"]) == (
        "0",
        "3",
        "0",
        "0",
    )
    assert by_key(rows)[THETAS[0], "c"]["fases"] == "3"


def test_a_missing_month_counts_as_missing_and_does_not_stop_the_rest(lake, capsys):
    code, text, rows = sizes(lake, capsys)
    assert code == 0
    table = by_key(rows)
    assert (table[THETAS[0], "c"]["archivos"], table[THETAS[0], "c"]["faltantes"]) == (
        "3",
        "0",
    )
    assert (table[THETAS[1], "c"]["archivos"], table[THETAS[1], "c"]["faltantes"]) == (
        "2",
        "1",
    )
    assert last_line(text)["faltantes"] == "1"


def test_a_theta_without_any_file_is_all_missing_and_a_theta_outside_the_catalog_is_ignored(
    lake, capsys
):
    write_summaries(
        lake / "l3", "0.00500000", "2023-02", [999], [999]
    )  # fuera del catálogo
    write_catalog(lake, (*THETAS, "0.00098052"))  # en el catálogo, sin archivos
    code, text, rows = sizes(lake, capsys)
    assert code == 0
    table = by_key(rows)
    assert (
        table["0.00098052", "c"]["archivos"],
        table["0.00098052", "c"]["faltantes"],
    ) == (
        "0",
        "3",
    )
    assert table["0.00098052", "c"]["fases"] == "0"
    assert last_line(text)["thetas"] == "2"
    assert ("0.00500000", "c") not in table


def test_from_and_to_filter_the_months(lake, capsys):
    code, text, rows = sizes(lake, capsys, "--from", "2023-02", "--to", "2023-03")
    assert code == 0
    table = by_key(rows)
    check(table[THETAS[0], "c"], [20, 300, 5_000, 1, 70_000])
    check(table[THETAS[1], "c"], [65_536])
    assert last_line(text)["archivos"] == "3"
    assert last_line(text)["faltantes"] == "1"  # θ1 sin febrero


def test_months_without_files_inside_an_explicit_range_are_missing(lake, capsys):
    code, text, _ = sizes(lake, capsys, "--from", "2023-03", "--to", "2023-05")
    assert code == 0
    assert last_line(text)["archivos"] == "2"
    assert last_line(text)["faltantes"] == "4"  # abril y mayo, para los dos θ


def test_the_same_result_with_1_reader_and_with_8(lake, capsys):
    results = {}
    for readers in ("1", "8"):
        code, text, rows = sizes(lake, capsys, "--readers", readers)
        assert code == 0
        deterministic = [line for line in text.splitlines() if "pared_s" not in line]
        deterministic = [line for line in deterministic if "readers=" not in line]
        results[readers] = (deterministic, rows)
    assert results["1"] == results["8"]


def test_sample_caps_puntos_and_pares(lake, capsys):
    code, _, rows = sizes(lake, capsys, "--to", "2023-01", "--sample", "100")
    assert code == 0
    check(by_key(rows)[THETAS[0], "c"], EDGE_C, sample=100)
    check(by_key(rows)[THETAS[0], "os"], [t for t in EDGE_OS if t > 0], sample=100)


def test_a_file_that_cannot_be_read_is_missing_and_the_others_are_counted(lake, capsys):
    summaries(lake / "l3", THETAS[0], "2023-02").write_bytes(b"no es parquet")
    code, text, rows = sizes(lake, capsys)
    assert code == 0
    assert "FALLO" in text and "2023-02" in text
    assert last_line(text)["archivos"] == "4"
    assert last_line(text)["faltantes"] == "2"
    check(by_key(rows)[THETAS[0], "c"], EDGE_C + [1, 70_000])


def test_without_files_it_exits_with_1_and_still_ends_with_the_last_line(
    tmp_path, capsys
):
    (tmp_path / "l3").mkdir()
    write_catalog(tmp_path)
    code, text, rows = sizes(tmp_path, capsys)
    assert code == 1
    assert last_line(text) == {
        "thetas": "0",
        "archivos": "0",
        "faltantes": "0",
        "fases_c": "0",
        "fases_os": "0",
    }
    assert rows


def test_a_missing_catalog_fails_with_1(lake, capsys):
    (lake / "thetas.yaml").unlink()
    code, text, rows = sizes(lake, capsys)
    assert code == 1
    assert "FALLO" in text and not rows


@pytest.mark.parametrize(
    "extra",
    [
        ("--from", "2023-1"),
        ("--to", "2023-13"),
        ("--from", "2023-03", "--to", "2023-01"),
        ("--readers", "0"),
        ("--sample", "0"),
        ("--sample", "1000001"),
    ],
)
def test_bad_arguments_are_usage_errors(lake, capsys, extra):
    with pytest.raises(SystemExit) as exc:
        sizes(lake, capsys, *extra)
    assert exc.value.code == 2


def test_without_arguments_and_without_project_it_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as exc:
        load().main([])
    assert exc.value.code == 2
