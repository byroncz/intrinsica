"""`l3_version_audit.py` contra un lago local con archivos de familia sintéticos (ITSC-419).

El lago de cada prueba se escribe en `tmp_path` con la misma disposición hive que el real: un
`events.parquet` de L2 por (θ, mes) y, por cada familia, un `<familia>.parquet` de L3 cuyo pie lleva
`state_version` (y `budget_ticks` en `spectrum`). La sonda solo lee el pie: los archivos no tienen
datos. Dos θ y tres meses por familia dan 24 archivos.
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
MONTHS = ("2023-01", "2023-02", "2023-03")
FAMILIES = ("summaries", "path", "fit", "spectrum")
VERSIONS = {"summaries": "1.0.0", "path": "1.0.0", "fit": "1.0.0", "spectrum": "1.2.0"}
BUDGET = "130000000"
LAST_LINE = re.compile(
    r"version audit: familias=(?P<familias>\d+) archivos=(?P<archivos>\d+) "
    r"faltantes=(?P<faltantes>\d+) versiones_mezcladas=(?P<mezcladas>\d+) "
    r"veredicto=(?P<veredicto>ok|mezcla|faltantes)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_version_audit", SCRIPTS / "l3_version_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partition(root: Path, theta: str, month: str, name: str) -> Path:
    year, number = month.split("-")
    return root / SERIES / f"theta={theta}/year={year}/month={number}" / name


def write_family(root: Path, theta: str, month: str, family: str, version: str, **meta):
    path = partition(root, theta, month, f"{family}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"theta": pa.array([], pa.int64())})
    metadata = {"state_version": version, **meta}
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


@pytest.fixture
def lake(tmp_path):
    l2, l3 = tmp_path / "l2", tmp_path / "l3"
    for theta in THETAS:
        for month in MONTHS:
            events = partition(l2, theta, month, "events.parquet")
            events.parent.mkdir(parents=True, exist_ok=True)
            events.write_bytes(b"")  # la sonda solo lista L2
            for family in FAMILIES:
                meta = {"budget_ticks": BUDGET} if family == "spectrum" else {}
                write_family(l3, theta, month, family, VERSIONS[family], **meta)
    catalog = tmp_path / "thetas.yaml"
    numbers = [int(float(t) * 10**8 + 0.5) for t in THETAS]
    catalog.write_text(
        "scale: 100000000\nthetas:\n" + "".join(f"  - {n}\n" for n in numbers)
    )
    return tmp_path


def audit(lake: Path, capsys, *extra: str) -> tuple[int, str, list[dict]]:
    out = lake / "salida" / "l3_version_audit.csv"
    code = load().main(
        [
            "--catalog",
            str(lake / "thetas.yaml"),
            "--l2-root",
            str(lake / "l2"),
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


def test_a_uniform_lake_is_ok_and_leaves_one_row_per_family_and_version(lake, capsys):
    code, text, rows = audit(lake, capsys)
    assert code == 0
    assert last_line(text) == {
        "familias": "4",
        "archivos": "24",
        "faltantes": "0",
        "mezcladas": "0",
        "veredicto": "ok",
    }
    assert [r["fila"] for r in rows] == ["version"] * 4
    assert {r["familia"]: (r["state_version"], r["archivos"]) for r in rows} == {
        "summaries": ("1.0.0", "6"),
        "path": ("1.0.0", "6"),
        "fit": ("1.0.0", "6"),
        "spectrum": ("1.2.0", "6"),
    }
    (spectrum,) = (r for r in rows if r["familia"] == "spectrum")
    assert spectrum["budget_ticks"] == BUDGET


def test_a_mixed_version_leaves_the_exception_and_exits_with_1(lake, capsys):
    write_family(lake / "l3", THETAS[0], "2023-02", "path", "0.9.0")
    code, text, rows = audit(lake, capsys)
    assert code == 1
    line = last_line(text)
    assert (line["faltantes"], line["mezcladas"], line["veredicto"]) == (
        "0",
        "1",
        "mezcla",
    )
    assert line["archivos"] == "24"
    versions = {
        (r["familia"], r["state_version"]): r["archivos"]
        for r in rows
        if r["fila"] == "version"
    }
    assert versions[("path", "1.0.0")] == "5"
    assert versions[("path", "0.9.0")] == "1"
    (exception,) = (r for r in rows if r["fila"] == "excepcion")
    assert (exception["familia"], exception["theta"], exception["mes"]) == (
        "path",
        THETAS[0],
        "2023-02",
    )
    assert exception["state_version"] == "0.9.0"
    assert "1.0.0" in exception["motivo"]


def test_a_different_budget_in_spectrum_is_a_mix_even_with_the_same_version(
    lake, capsys
):
    write_family(
        lake / "l3", THETAS[1], "2023-03", "spectrum", "1.2.0", budget_ticks="20000000"
    )
    code, text, rows = audit(lake, capsys)
    assert code == 1
    assert last_line(text)["veredicto"] == "mezcla"
    (exception,) = (r for r in rows if r["fila"] == "excepcion")
    assert exception["budget_ticks"] == "20000000"


def test_a_missing_file_is_reported_and_wins_over_the_mix(lake, capsys):
    partition(lake / "l3", THETAS[1], "2023-03", "fit.parquet").unlink()
    write_family(lake / "l3", THETAS[0], "2023-01", "summaries", "0.9.0")
    code, text, rows = audit(lake, capsys)
    assert code == 1
    line = last_line(text)
    assert (line["archivos"], line["faltantes"], line["mezcladas"]) == ("23", "1", "1")
    assert line["veredicto"] == "faltantes"
    missing = [r for r in rows if r["motivo"] == "falta el archivo"]
    assert [(r["familia"], r["theta"], r["mes"]) for r in missing] == [
        ("fit", THETAS[1], "2023-03")
    ]


def test_a_file_that_cannot_be_read_counts_as_another_version(lake, capsys):
    partition(lake / "l3", THETAS[0], "2023-01", "path.parquet").write_bytes(b"x")
    code, text, rows = audit(lake, capsys)
    assert code == 1
    assert last_line(text)["veredicto"] == "mezcla"
    (exception,) = (r for r in rows if r["fila"] == "excepcion")
    assert exception["state_version"] == "ilegible"


def test_months_without_events_in_l2_and_thetas_out_of_the_catalog_do_not_count(
    lake, capsys
):
    # Un mes de L3 sin `events.parquet` en L2 y un θ fuera del catálogo con la versión vieja.
    write_family(lake / "l3", THETAS[0], "2023-04", "path", "0.9.0")
    for month in MONTHS:
        events = partition(lake / "l2", "0.00500000", month, "events.parquet")
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_bytes(b"")
        write_family(lake / "l3", "0.00500000", month, "path", "0.9.0")
    code, text, _ = audit(lake, capsys)
    assert code == 0
    assert last_line(text)["archivos"] == "24"


def test_a_missing_catalog_fails_with_1(lake, capsys):
    (lake / "thetas.yaml").unlink()
    code, text, rows = audit(lake, capsys)
    assert code == 1
    assert "FALLO" in text and not rows


def test_without_arguments_and_without_project_it_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as exc:
        load().main([])
    assert exc.value.code == 2
