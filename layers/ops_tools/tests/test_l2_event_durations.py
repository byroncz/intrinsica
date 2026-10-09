"""`l2_event_durations.py` contra el lago de fixtures y contra eventos fabricados.

El lago de fixtures (L2 real de 2017-08, 5 θ) prueba que los números cuadran con lo que
dicen los `events.parquet`; los eventos fabricados prueban lo que el fixture no tiene:
eventos de más de 90 días, varios meses por θ y los fallos.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "layers/viz_tiles/tests"))

from lake_fixture import MONTH, build_lake, events_path
from viz_tiles.lake import EVENTS

SERIES = "provider=binance/market=spot/asset=BTCUSDT"
DAY_US = 86_400_000_000


def load():
    spec = importlib.util.spec_from_file_location(
        "l2_event_durations", SCRIPTS / "l2_event_durations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(root: Path, theta: str, month: str, events: list[tuple[int, int, int, int]]):
    """Un `events.parquet` con `(ref_time, ext_time, ref_id, ext_id)` por evento."""
    year, number = month.split("-")
    path = root / SERIES / f"theta={theta}/year={year}/month={number}/events.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = zip(*events, strict=True) if events else [[], [], [], []]
    names = [
        "reference_time",
        "extreme_time",
        "reference_agg_trade_id",
        "extreme_agg_trade_id",
    ]
    pq.write_table(
        pa.table({n: pa.array(list(c), pa.int64()) for n, c in zip(names, cols)}), path
    )


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_lago_de_fixtures_cuadra_con_los_events_parquet(tmp_path, capsys):
    roots, _ticks, _pending = build_lake(tmp_path)
    out = tmp_path / "salida.csv"
    mod = load()
    assert mod.main(["--events-root", str(roots.events), "--out", str(out)]) == 0
    table = read_csv(out)
    assert len(table) == 5
    for row in table:
        theta = int(row["theta"].split(".")[1])
        events = pq.read_table(events_path(roots, theta, MONTH, EVENTS)).to_pylist()
        dur = sorted(e["extreme_time"] - e["reference_time"] for e in events)
        size = sorted(
            e["extreme_agg_trade_id"] - e["reference_agg_trade_id"] for e in events
        )
        assert int(row["eventos"]) == len(events)
        assert float(row["dias_max"]) == pytest.approx(dur[-1] / DAY_US)
        assert int(row["ticks_max"]) == size[-1]
        for d in (1, 7, 30, 60, 90):
            assert int(row[f"ge_{d}d"]) == sum(x >= d * DAY_US for x in dur)
        # La mediana es aproximada: el error relativo del histograma es <= 0,25 %.
        exact = size[(len(size) + 1) // 2 - 1]
        assert float(row["ticks_mediana"]) == pytest.approx(exact, rel=0.01)
        assert row["mes_ticks_max"] == "2017-08"
    out_text = capsys.readouterr().out
    assert "θ mínimo con eventos >= 90 días: ninguno" in out_text
    assert "evento más grande en ticks: θ=" in out_text
    assert "tabla: " in out_text


def test_theta_minimo_y_evento_mas_grande(tmp_path, capsys):
    # θ chico: eventos cortos. θ mediano: uno de 45 días. θ grande: uno de 100 días en
    # el segundo mes y otro de 10 días en el primero.
    t0 = 1_500_000_000_000_000
    put(tmp_path, "0.00010000", "2020-01", [(t0, t0 + 3_600_000_000, 10, 50)])
    put(tmp_path, "0.00500000", "2020-01", [(t0, t0 + 45 * DAY_US, 10, 5_000)])
    put(tmp_path, "0.02000000", "2020-01", [(t0, t0 + 10 * DAY_US, 10, 900)])
    put(tmp_path, "0.02000000", "2020-02", [(t0, t0 + 100 * DAY_US, 100, 7_000_100)])
    out = tmp_path / "dir" / ""
    mod = load()
    assert mod.main(["--events-root", str(tmp_path), "--out", f"{out}/"]) == 0
    text = capsys.readouterr().out
    assert "θ mínimo con eventos >= 90 días: 0.02000000 (1 eventos)" in text
    assert "θ mínimo con eventos >= 60 días: 0.02000000 (1 eventos)" in text
    assert "θ mínimo con eventos >= 30 días: 0.00500000 (1 eventos)" in text
    mib = 7_000_000 * 48 / 2**20
    assert (
        f"θ=0.02000000 mes 2020-02 (inicia 2017-07-14): 7,000,000 ticks = {mib:,.1f} MiB"
        in text
    )
    big = {r["theta"]: r for r in read_csv(tmp_path / "dir" / "l2_event_durations.csv")}
    assert int(big["0.02000000"]["eventos"]) == 2
    assert float(big["0.02000000"]["dias_max"]) == pytest.approx(100)
    assert big["0.02000000"]["mes_dias_max"] == "2020-02"
    assert [int(big["0.02000000"][f"ge_{d}d"]) for d in (1, 7, 30, 60, 90)] == [
        2,
        2,
        1,
        1,
        1,
    ]


def test_eventos_por_umbral_de_ticks(tmp_path, capsys):
    # Tamaños de θ chico: 999_999 (bajo 1 M), 1_000_000 (justo en el umbral), 6 M y, en
    # otro mes, 12 M. θ grande: 60 M y uno de 2 M. Los umbrales cuentan con `>=`.
    t0 = 1_500_000_000_000_000
    one = (t0, t0 + DAY_US)
    put(
        tmp_path,
        "0.00100000",
        "2020-01",
        [(*one, 0, 999_999), (*one, 0, 1_000_000), (*one, 0, 6_000_000)],
    )
    put(tmp_path, "0.00100000", "2020-02", [(*one, 0, 12_000_000)])
    put(
        tmp_path, "0.02000000", "2020-01", [(*one, 0, 60_000_000), (*one, 0, 2_000_000)]
    )
    out = tmp_path / "salida.csv"
    assert load().main(["--events-root", str(tmp_path), "--out", str(out)]) == 0
    by_theta = {r["theta"]: r for r in read_csv(out)}
    names = [f"ticks_ge_{m}M" for m in (1, 5, 10, 20, 50)]
    assert [int(by_theta["0.00100000"][n]) for n in names] == [3, 2, 1, 0, 0]
    assert [int(by_theta["0.02000000"][n]) for n in names] == [2, 1, 1, 1, 1]
    text = capsys.readouterr().out
    header = next(x for x in text.splitlines() if x.lstrip().startswith("theta"))
    assert all(n in header.split() for n in names)
    assert (
        "eventos con >= 1 M ticks (0.04 GiB a 48 B por tick), suma de los θ: 5" in text
    )
    assert (
        "eventos con >= 5 M ticks (0.22 GiB a 48 B por tick), suma de los θ: 3" in text
    )
    assert (
        "eventos con >= 10 M ticks (0.45 GiB a 48 B por tick), suma de los θ: 2" in text
    )
    assert (
        "eventos con >= 20 M ticks (0.89 GiB a 48 B por tick), suma de los θ: 1" in text
    )
    assert (
        "eventos con >= 50 M ticks (2.24 GiB a 48 B por tick), suma de los θ: 1" in text
    )


def test_mediana_y_p99_aproximadas(tmp_path):
    t0 = 1_500_000_000_000_000
    events = [(t0, t0 + i * DAY_US, 0, 1000 * i) for i in range(1, 201)]
    put(tmp_path, "0.00100000", "2020-01", events[:100])
    put(tmp_path, "0.00100000", "2020-02", events[100:])
    out = tmp_path / "salida.csv"
    assert load().main(["--events-root", str(tmp_path), "--out", str(out)]) == 0
    (row,) = read_csv(out)
    assert float(row["dias_mediana"]) == pytest.approx(100, rel=0.01)
    assert float(row["dias_p99"]) == pytest.approx(198, rel=0.01)
    assert float(row["dias_max"]) == 200
    assert int(row["ticks_max"]) == 200_000


def test_sin_eventos_falla(tmp_path, capsys):
    put(tmp_path, "0.00100000", "2020-01", [])
    assert (
        load().main(["--events-root", str(tmp_path), "--out", str(tmp_path / "x.csv")])
        == 1
    )
    assert "FALLO sin_eventos" in capsys.readouterr().out
    assert not (tmp_path / "x.csv").exists()


def test_extremo_antes_de_la_referencia(tmp_path):
    put(tmp_path, "0.00100000", "2020-01", [(2_000, 1_000, 5, 9)])
    with pytest.raises(SystemExit, match="extremo_antes_de_la_referencia"):
        load().main(["--events-root", str(tmp_path), "--out", str(tmp_path / "x.csv")])


def test_sin_events_root_ni_proyecto(monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit, match="OPS_RESULTS_URI"):
        load().main([])


def test_raiz_por_defecto_sale_del_proyecto(monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://intrinsica-dc-ops/results/abc/")
    mod = load()
    assert mod.default_events_root() == "gs://intrinsica-dc-dc-events/l2"
    assert (
        mod.default_out() == "gs://intrinsica-dc-ops/results/abc/l2_event_durations.csv"
    )


def test_nulos_fallan(tmp_path):
    put(tmp_path, "0.00100000", "2020-01", [(1_000, 2_000, None, 9)])
    with pytest.raises(SystemExit, match="FALLO nulos: θ=0.00100000 2020-01"):
        load().main(["--events-root", str(tmp_path), "--out", str(tmp_path / "x.csv")])


def test_out_gs_fuera_de_results_falla_al_arrancar(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://intrinsica-dc-ops/results/abc/")
    # El lago no existe: si leyera antes de validar, fallaría por otra cosa.
    with pytest.raises(SystemExit, match="out_fuera_de_results"):
        load().main(
            [
                "--events-root",
                str(tmp_path),
                "--out",
                "gs://intrinsica-dc-ops/scripts/l2_event_durations.csv",
            ]
        )


def test_imprime_cada_theta_al_cerrarlo(tmp_path, capsys):
    t0 = 1_500_000_000_000_000
    put(tmp_path, "0.00100000", "2020-01", [(t0, t0 + DAY_US, 10, 50)])
    put(tmp_path, "0.00100000", "2020-02", [(t0, t0 + 2 * DAY_US, 10, 60)])
    put(tmp_path, "0.02000000", "2020-01", [(t0, t0 + 100 * DAY_US, 10, 900)])
    assert load().main(["--events-root", str(tmp_path), "--out", f"{tmp_path}/"]) == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if "cerrado:" in x]
    assert len(lines) == 2
    assert lines[0].startswith("θ=0.00100000 cerrado: eventos=2 dias_max=2.000")
    assert lines[1].endswith("ge_90d=1")
