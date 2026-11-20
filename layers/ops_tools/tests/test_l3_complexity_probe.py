"""`l3_complexity_probe.py` contra un lago local con los cinco archivos sintéticos (ITSC-432).

El lago de cada prueba se escribe a mano en `tmp_path` con la misma disposición hive que el real:
`consolidated.parquet` de L1, `events.parquet` de L2 y `summaries.parquet`, `path.parquet` y
`spectrum.parquet` de L3, con solo las columnas que la sonda lee. Los ticks de L1 van en row groups
de pocos ticks, así que cada fase cruza varios. Un evento se escribe como su tick de referencia (el
extremo del anterior), los ticks de su confirmación `c` y los de su overshoot `os`; los precios son
enteros sin escalar (`DECIMAL(18,8)`) y las medidas no dependen de la escala.
"""

import csv
import importlib.util
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
THETA = "0.05"
THETA_FILE = "0.05000000"
MONTH = "2023-03"
DECIMAL = pa.decimal128(18, 8)
BASE = 10**15
LAST_LINE = re.compile(
    r"complexity probe: theta=(?P<theta>\S+) meses=(?P<meses>\S+) "
    r"fases_c=(?P<fases_c>\S+) fases_os=(?P<fases_os>\S+) "
    r"D_hurst_sub=(?P<D_hurst_sub>\S+) D_pe_sub=(?P<D_pe_sub>\S+) "
    r"D_hurst_bar=(?P<D_hurst_bar>\S+) D_pe_bar=(?P<D_pe_bar>\S+) "
    r"piso_c=(?P<piso_c>\S+) rho_hurst=(?P<rho_hurst>\S+) rho_pe=(?P<rho_pe>\S+) "
    r"us_hurst=(?P<us_hurst>\S+) us_pe=(?P<us_pe>\S+) pared_s=(?P<pared>\S+) "
    r"veredicto_hurst=(?P<veredicto_hurst>\S+) veredicto_pe=(?P<veredicto_pe>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_complexity_probe", SCRIPTS / "l3_complexity_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load()


@dataclass
class Ev:
    """Un evento: los precios de los ticks de `c` y de `os`, y los registros de las otras familias.

    El tick de referencia es el último del evento anterior (o `first` en el primero).
    """

    c: list[int]
    os: list[int] = field(default_factory=list)
    sign: float | None = 0.2
    bands: list[float | None] = field(default_factory=lambda: [0.1] * 7)


def walk(returns, start: int = BASE) -> list[int]:
    """Precios enteros que parten de `start` y siguen los retornos logarítmicos dados."""
    return [round(start * math.exp(x)) for x in np.cumsum(returns)]


def noise_events(count: int, n_c: int, n_os: int, seed: int = 1) -> list[Ev]:
    """`count` eventos de ruido blanco con `n_c` y `n_os` ticks; encadenados en precio."""
    rng = np.random.default_rng(seed)
    events, start = [], BASE
    for _ in range(count):
        c = walk(rng.standard_normal(n_c) * 1e-4, start)
        os = walk(rng.standard_normal(n_os) * 1e-4, c[-1]) if n_os else []
        events.append(
            Ev(c, os, sign=float(rng.uniform(-1, 1)), bands=list(rng.uniform(0, 1, 7)))
        )
        start = (os or c)[-1]
    return events


def write_lake(
    root: Path,
    events: list[Ev],
    *,
    month: str = MONTH,
    first: int = BASE,
    rows_per_group: int = 61,
    drop_first: int = 0,
) -> dict[str, Path]:
    """Escribe los cinco archivos de `(THETA, month)`; devuelve sus rutas.

    `drop_first` quita esos primeros ticks del L1 (el primer evento arranca en el mes anterior).
    """
    ticks = [(1_000, first)]
    ref_id, ref_price = 1_000, first
    rows = []
    for event in events:
        c_ids = list(range(ref_id + 1, ref_id + 1 + len(event.c)))
        conf_id = c_ids[-1]
        os_ids = list(range(conf_id + 1, conf_id + 1 + len(event.os)))
        ext_id = os_ids[-1] if os_ids else conf_id
        ticks += list(zip(c_ids, event.c, strict=True))
        ticks += list(zip(os_ids, event.os, strict=True))
        rows.append(
            {
                "ref_price": ref_price,
                "ref_id": ref_id,
                "conf_price": event.c[-1],
                "conf_id": conf_id,
                "ext_id": ext_id,
                "c_ticks": len(event.c),
                "os_ticks": len(event.os),
                "sign": event.sign,
                "bands": event.bands,
            }
        )
        ref_id, ref_price = ext_id, (event.os or event.c)[-1]
    in_month = ticks[drop_first:]

    def exact(values):
        return pa.array([Decimal(v).scaleb(-8) for v in values], DECIMAL)

    year, number = month.split("-")
    partition = f"{SERIES}/theta={THETA_FILE}/year={year}/month={number}"
    paths = {
        "l1": root / "l1" / SERIES / f"year={year}/month={number}/consolidated.parquet",
        "events": root / "l2" / partition / "events.parquet",
        "summaries": root / "l3" / partition / "summaries.parquet",
        "path": root / "l3" / partition / "path.parquet",
        "spectrum": root / "l3" / partition / "spectrum.parquet",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "agg_trade_id": pa.array([t[0] for t in in_month], pa.int64()),
                "price": exact([t[1] for t in in_month]),
            }
        ),
        paths["l1"],
        row_group_size=rows_per_group,
    )
    ids = pa.array([r["conf_id"] for r in rows], pa.int64())

    def column(name, kind=None):
        return pa.array([r[name] for r in rows], kind or pa.int64())

    pq.write_table(
        pa.table(
            {
                "reference_price": exact([r["ref_price"] for r in rows]),
                "reference_agg_trade_id": column("ref_id"),
                "confirm_price": exact([r["conf_price"] for r in rows]),
                "confirm_agg_trade_id": ids,
                "extreme_agg_trade_id": column("ext_id"),
            }
        ),
        paths["events"],
        row_group_size=3,
    )
    pq.write_table(
        pa.table(
            {
                "confirm_agg_trade_id": ids,
                "c_ticks": column("c_ticks"),
                "os_ticks": column("os_ticks"),
            }
        ),
        paths["summaries"],
        row_group_size=2,
    )
    pq.write_table(
        pa.table(
            {
                "confirm_agg_trade_id": ids,
                "c_sign_autocorr": column("sign", pa.float64()),
            }
        ),
        paths["path"],
        row_group_size=5,
    )
    bands = {
        f"c_fourier_band_{b}": pa.array([r["bands"][b - 1] for r in rows], pa.float32())
        for b in range(1, 8)
    }
    pq.write_table(
        pa.table({"confirm_agg_trade_id": ids, **bands}),
        paths["spectrum"],
        row_group_size=4,
    )
    return paths


def argv(root: Path, out: Path, *extra: str, months: str = MONTH) -> list[str]:
    return [
        "--theta",
        THETA,
        "--months",
        months,
        "--l1-root",
        str(root / "l1"),
        "--l2-root",
        str(root / "l2"),
        "--l3-root",
        str(root / "l3"),
        "--out",
        str(out),
        "--min-ticks",
        "128",
        *extra,
    ]


def last_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    match = LAST_LINE.match(line)
    assert match, line
    return match.groupdict()


def table(path: Path) -> dict:
    """El CSV como `{(seccion, medida, nulo, fase, estadistico): valor}`."""
    with path.open() as handle:
        return {
            (r["seccion"], r["medida"], r["nulo"], r["fase"], r["estadistico"]): r[
                "valor"
            ]
            for r in csv.DictReader(handle)
        }


def spy_windows(monkeypatch, probe) -> list[np.ndarray]:
    """Registra el buffer `[p_0, p_1 … p_w]` de cada ventana que mide la sonda."""
    seen = []
    original = probe.measure_window

    def spy(buffer, rng, nulls):
        seen.append(buffer.copy())
        return original(buffer, rng, nulls)

    monkeypatch.setattr(probe, "measure_window", spy)
    return seen


# ----------------------------------------------------------------------------------------------
# Las medidas
# ----------------------------------------------------------------------------------------------


def test_hurst_of_white_noise_is_near_half(probe):
    returns = np.random.default_rng(7).standard_normal((20, 4096))
    assert abs(probe.hurst(returns).mean() - 0.5) < 0.05


def test_hurst_of_an_alternating_series_is_low(probe):
    returns = np.array([1e-3, -1e-3] * 1024)
    assert probe.hurst(returns)[0] < 0.1


def test_hurst_of_a_trending_series_is_high(probe):
    returns = np.cumsum(np.random.default_rng(3).standard_normal(4096))
    assert probe.hurst(returns)[0] > 1.0


def test_hurst_matches_a_plain_dfa(probe):
    """Una implementación ingenua (`polyfit` por ventana) da la misma H."""
    returns = np.random.default_rng(5).standard_normal(1000)
    profile = np.cumsum(returns - returns.mean())
    scales = [16, 32, 64, 128]  # n / 4 = 250
    assert probe.dfa_scales(1000) == scales
    log_f = []
    for s in scales:
        squares = []
        for k in range(1000 // s):
            segment = profile[k * s : (k + 1) * s]
            x = np.arange(s)
            fit = np.polyval(np.polyfit(x, segment, 1), x)
            squares.append(np.mean((segment - fit) ** 2))
        log_f.append(0.5 * math.log(np.mean(squares)))
    expected = np.polyfit(np.log(scales), log_f, 1)[0]
    assert probe.hurst(returns)[0] == pytest.approx(expected, rel=1e-9)


def test_hurst_is_null_when_a_scale_has_no_fluctuation(probe):
    assert math.isnan(probe.hurst(np.zeros(512))[0])


def test_entropy_of_monotone_prices_is_zero(probe):
    assert probe.permutation_entropy(np.arange(1.0, 200.0))[0] == 0.0
    assert probe.permutation_entropy(np.arange(200.0, 1.0, -1.0))[0] == 0.0


def test_entropy_of_constant_prices_is_zero(probe):
    """Los empates se ordenan por aparición: una serie plana es un solo orden."""
    assert probe.permutation_entropy(np.full(100, 5.0))[0] == 0.0


def test_entropy_matches_a_plain_count_and_ties_keep_appearance_order(probe):
    from collections import Counter

    prices = (
        np.random.default_rng(2).integers(0, 4, 500).astype(float)
    )  # muchos empates
    counts = Counter(
        tuple(np.argsort(prices[i : i + 4], kind="stable"))
        for i in range(len(prices) - 3)
    )
    p = np.array(list(counts.values())) / (len(prices) - 3)
    expected = -(p * np.log(p)).sum() / math.log(24)
    assert probe.permutation_entropy(prices)[0] == pytest.approx(expected, rel=1e-12)


def test_entropy_is_normalized(probe):
    values = probe.permutation_entropy(
        np.random.default_rng(4).standard_normal((5, 5000))
    )
    assert ((values > 0.99) & (values <= 1.0)).all()


# ----------------------------------------------------------------------------------------------
# El sustituto
# ----------------------------------------------------------------------------------------------


def test_surrogate_keeps_the_sorted_values_and_the_fourier_amplitudes(probe):
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(4096)
    series = np.empty_like(noise)  # AR(1): un espectro muy lejos de ser plano
    series[0] = noise[0]
    for i in range(1, len(noise)):
        series[i] = 0.8 * series[i - 1] + noise[i]
    series = np.sign(series) * np.abs(series) ** 1.5  # no gaussiana
    shuffled = np.tile(series, (5, 1))
    rng.permuted(shuffled, axis=1, out=shuffled)
    surrogate = probe.iaaft(shuffled.copy(), series)

    # Los valores ordenados son exactos: el sustituto termina en la reasignación por rango.
    assert np.array_equal(np.sort(surrogate, axis=1), np.tile(np.sort(series), (5, 1)))
    target = np.abs(np.fft.rfft(series))
    for row, mixed in zip(surrogate, shuffled, strict=True):
        error = np.linalg.norm(np.abs(np.fft.rfft(row)) - target) / np.linalg.norm(
            target
        )
        barajado = np.linalg.norm(np.abs(np.fft.rfft(mixed)) - target) / np.linalg.norm(
            target
        )
        assert error < 0.1
        assert error < barajado / 3  # y mucho más cerca del espectro que el barajado


def test_surrogate_stops_at_once_when_the_ranks_do_not_change(probe):
    """Una serie ya ordenada es un punto fijo: se queda igual."""
    series = np.sort(np.random.default_rng(1).standard_normal(256))
    assert np.array_equal(probe.iaaft(series[None, :].copy(), series)[0], series)


# ----------------------------------------------------------------------------------------------
# La regla del veredicto
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "floor", "rho", "expected"),
    [
        (0.30, 0.15, 0.5, "sigue"),  # pasa las tres condiciones
        (0.08, 0.05, 0.5, "retirada"),  # D bajo 0,10
        (0.12, 0.20, 0.5, "retirada"),  # D ≥ 0,10 pero bajo el piso
        (0.30, 0.15, 0.95, "retirada"),  # repite una columna que ya existe
        (0.10, 0.10, 0.9, "sigue"),  # los bordes pasan
        (math.nan, 0.15, 0.5, "retirada"),
        (0.30, 0.15, math.nan, "sigue"),  # una ρ indefinida no cuenta como redundancia
    ],
)
def test_verdict_rule(probe, d, floor, rho, expected):
    assert probe.verdict_of(d, floor, rho) == expected


def test_floor_and_ks(probe):
    assert probe.floor_of(200, 2000) == pytest.approx(1.36 * math.sqrt(2200 / 400000))
    assert probe.ks_distance(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert probe.ks_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_spearman(probe):
    x = np.arange(10.0)
    assert probe.spearman(x, x**3) == pytest.approx(1.0)
    assert probe.spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(probe.spearman(x, np.ones(10)))
    assert math.isnan(probe.max_abs([math.nan, math.nan]))
    assert probe.max_abs([0.2, -0.7, math.nan]) == pytest.approx(0.7)


# ----------------------------------------------------------------------------------------------
# La sonda entera contra el lago
# ----------------------------------------------------------------------------------------------


def test_window_takes_the_last_of_c_and_the_first_of_os(
    probe, tmp_path, monkeypatch, capsys
):
    events = noise_events(2, 300, 200)
    # Un evento más corto que la ventana: se usa entero, con el ancla como precio previo.
    short = noise_events(1, 140, 130, seed=9)[0]
    short.c = walk(
        np.random.default_rng(9).standard_normal(140) * 1e-4, events[-1].os[-1]
    )
    short.os = walk(np.random.default_rng(10).standard_normal(130) * 1e-4, short.c[-1])
    events.append(short)
    write_lake(tmp_path, events)
    seen = spy_windows(monkeypatch, probe)

    assert probe.main(argv(tmp_path, tmp_path / "o.csv", "--window", "150")) == 0

    first, second, third = events
    expected = [
        # c: p_{n−w} … p_n, el precio previo a los últimos w retornos y esos w precios
        [*first.c[300 - 151 : 300]],
        # os: el ancla (el precio de confirmación) y los primeros w precios
        [first.c[-1], *first.os[:150]],
        [*second.c[300 - 151 : 300]],
        [second.c[-1], *second.os[:150]],
        # c con n = 140 ≤ w: el ancla (el último tick del evento anterior) y los 140 precios
        [second.os[-1], *third.c],
        [third.c[-1], *third.os],
    ]
    assert len(seen) == len(expected)
    for buffer, want in zip(seen, expected, strict=True):
        assert buffer.tolist() == [float(v) for v in want]
    assert last_line(capsys)["fases_c"] == "3"


def test_max_phases_takes_evenly_spaced_phases(probe, tmp_path, monkeypatch, capsys):
    events = noise_events(10, 200, 0)
    write_lake(tmp_path, events)
    seen = spy_windows(monkeypatch, probe)

    assert probe.main(argv(tmp_path, tmp_path / "o.csv", "--max-phases", "4")) == 0

    # ⌊j · 10 / 4⌋ = 0, 2, 5, 7
    taken = [events[i].c[-1] for i in (0, 2, 5, 7)]
    assert [b[-1] for b in seen] == [float(v) for v in taken]
    assert last_line(capsys)["fases_c"] == "4"
    assert table(tmp_path / "o.csv")[("excluidos", "", "", "c", "tope")] == "6"


def test_max_phases_counts_across_months(probe, tmp_path, monkeypatch, capsys):
    write_lake(tmp_path, noise_events(3, 200, 0, seed=1), month="2023-03")
    write_lake(tmp_path, noise_events(3, 200, 0, seed=2), month="2023-04")
    seen = spy_windows(monkeypatch, probe)
    out = tmp_path / "o.csv"

    code = probe.main(
        argv(tmp_path, out, "--max-phases", "3", months="2023-04,2023-03")
    )

    assert code == 0
    assert len(seen) == 3  # 6 en total: ⌊j · 6 / 3⌋ = 0, 2, 4
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines if line.startswith("mes ")] == [
        "mes 2023-03",
        "mes 2023-04",
    ]
    assert table(out)[("config", "", "", "", "meses")] == "2023-03,2023-04"
    assert table(out)[("muestras", "", "", "c", "N")] == "3"


def test_same_seed_same_csv_and_another_seed_another(probe, tmp_path):
    write_lake(tmp_path, noise_events(8, 300, 200))
    outs = []
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):
        out = tmp_path / f"{name}.csv"
        assert probe.main(argv(tmp_path, out, "--seed", str(seed))) == 0
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert outs[0] != outs[2]


def test_last_line_csv_and_os_line(probe, tmp_path, capsys):
    write_lake(tmp_path, noise_events(8, 300, 200))
    out = tmp_path / "o.csv"
    assert probe.main(argv(tmp_path, out)) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2].startswith("complexity probe os: fases_os=8 ")
    assert "D_hurst_sub=" in lines[-2] and "piso_os=" in lines[-2]
    figures = LAST_LINE.match(lines[-1]).groupdict()
    assert figures["theta"] == "0.05000000"
    assert figures["fases_c"] == figures["fases_os"] == "8"
    assert figures["veredicto_hurst"] in ("sigue", "retirada")
    assert float(figures["us_hurst"]) > 0

    values = table(out)
    n, m = 8, 80
    assert values[("D", "hurst", "sustituto", "c", "N")] == str(n)
    assert values[("D", "hurst", "sustituto", "c", "M")] == str(m)
    floor = float(values[("D", "pe", "barajado", "os", "piso")])
    assert floor == pytest.approx(1.36 * math.sqrt((n + m) / (n * m)))
    assert float(figures["piso_c"]) == pytest.approx(floor, abs=1e-6)
    assert float(figures["D_hurst_sub"]) == pytest.approx(
        float(values[("D", "hurst", "sustituto", "c", "D")]), abs=1e-6
    )
    for stat in ("p5", "p50", "p95", "N"):
        for source in ("observado", "barajado", "sustituto"):
            assert ("cuantiles", "pe", source, "os", stat) in values
    for column in (
        *(f"c_fourier_band_{b}" for b in range(1, 8)),
        "c_sign_autocorr",
        "c_ticks",
    ):
        assert ("rho", "hurst", "", "c", column) in values
    # Ninguna cifra que dependa del reloj va al CSV.
    assert not any("us" in k[4] or "pared" in k[4] for k in values)
    assert values[("veredicto", "pe", "", "c", "veredicto")] == figures["veredicto_pe"]


def test_skips_are_counted_by_reason(probe, tmp_path, capsys):
    events = noise_events(5, 300, 200)
    events[1].c = events[1].c[:100]  # c bajo el mínimo (128)
    events[1].os = []  # overshoot vacío
    # Sin movimiento desde el ancla (el último tick del evento anterior): F(s) = 0, fase nula.
    events[2].c = [events[1].c[-1]] * 300
    write_lake(tmp_path, events)
    out = tmp_path / "o.csv"

    assert probe.main(argv(tmp_path, out)) == 0

    values = table(out)
    assert values[("excluidos", "", "", "c", "bajo_minimo")] == "1"
    assert values[("excluidos", "", "", "os", "overshoot_vacio")] == "1"
    assert values[("excluidos", "", "", "c", "nulas")] == "1"
    assert values[("muestras", "", "", "c", "N")] == "3"
    assert values[("muestras", "", "", "os", "N")] == "4"
    assert "mes 2023-03: fases_c=3 fases_os=4 saltadas c[overshoot_vacio=0 " in (
        capsys.readouterr().out
    )


def test_event_that_starts_in_the_previous_month_is_skipped(probe, tmp_path):
    write_lake(tmp_path, noise_events(4, 300, 200), drop_first=50)
    out = tmp_path / "o.csv"
    assert probe.main(argv(tmp_path, out)) == 0
    values = table(out)
    assert values[("excluidos", "", "", "", "evento_mes_anterior")] == "1"
    assert values[("muestras", "", "", "c", "N")] == "3"
    assert values[("muestras", "", "", "os", "N")] == "3"


def test_failures_exit_1_with_a_failure_line(probe, tmp_path, capsys):
    paths = write_lake(tmp_path, noise_events(3, 300, 200))
    paths["spectrum"].unlink()
    assert probe.main(argv(tmp_path, tmp_path / "o.csv")) == 1
    out = capsys.readouterr().out
    assert "FALLO InputError: falta el archivo" in out
    assert (
        out.strip()
        .splitlines()[-1]
        .endswith("veredicto_hurst=fallo veredicto_pe=fallo")
    )


def test_no_measurable_c_phase_exits_1(probe, tmp_path, capsys):
    write_lake(tmp_path, noise_events(3, 100, 100))
    assert probe.main(argv(tmp_path, tmp_path / "o.csv")) == 1
    assert "ninguna fase c medible" in capsys.readouterr().out


def test_misaligned_files_exit_1(probe, tmp_path, capsys):
    paths = write_lake(tmp_path, noise_events(3, 300, 200))
    table_ = pq.read_table(paths["path"])
    bad = table_.set_column(0, "confirm_agg_trade_id", pa.array([1, 2, 3], pa.int64()))
    pq.write_table(bad, paths["path"])
    assert probe.main(argv(tmp_path, tmp_path / "o.csv")) == 1
    assert "no están alineados" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--theta", "0.123456789"),
        ("--months", "2023-13"),
        ("--months", "2023-03,2023-03"),
        ("--window", "64"),
        ("--min-ticks", "100"),
        ("--nulls", "0"),
        ("--max-phases", "1"),
    ],
)
def test_usage_errors_exit_2(probe, tmp_path, flag, value):
    args = argv(tmp_path, tmp_path / "o.csv")
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    with pytest.raises(SystemExit) as exit_info:
        probe.main(args)
    assert exit_info.value.code == 2
