"""`l3_spectrum_probe.py` contra un lago local con los cinco archivos sintéticos (ITSC-411).

El lago de cada prueba se escribe a mano en `tmp_path` con la misma disposición hive que el real:
`consolidated.parquet` de L1, `events.parquet` de L2 y `summaries.parquet`, `path.parquet` y
`fit.parquet` de L3, con solo las columnas que la sonda lee. Los ticks de L1 van en row groups de
pocos ticks, así que cada fase cruza varios. Un evento se escribe como su tick de referencia (el
extremo del anterior), los ticks de su confirmación `c` y los de su overshoot `os`; los precios son
enteros sin escalar (`DECIMAL(18,8)`) y las fracciones no dependen de la escala.

El oráculo (`ORACLE`) son fases cuyas fracciones por banda de Fourier y por nivel de Haar están
calculadas a mano abajo con tolerancia 1e-9; la hija 2 de la Épica ITSC-410 (el motor nativo) lo
reutiliza.
"""

import csv
import importlib.util
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SERIES = "provider=binance/market=spot/asset=BTCUSDT"
THETA = "0.00098052"
MONTH = "2023-03"
DECIMAL = pa.decimal128(18, 8)
BASE = 10**15
LAST_LINE = re.compile(
    r"spectrum probe: mes=(?P<mes>\S+) theta=(?P<theta>\S+) eventos=(?P<eventos>\S+) "
    r"D_ret_fast=(?P<D_ret_fast>\S+) D_ret_slow=(?P<D_ret_slow>\S+) "
    r"D_price_fast=(?P<D_price_fast>\S+) D_price_slow=(?P<D_price_slow>\S+) "
    r"rho_price_ret=(?P<rho_price_ret>\S+) rho_haar_fourier=(?P<rho_haar_fourier>\S+) "
    r"rho_ret_accel=(?P<rho_ret_accel>\S+) rho_ret_t50=(?P<rho_ret_t50>\S+) "
    r"rho_price_accel=(?P<rho_price_accel>\S+) rho_price_t50=(?P<rho_price_t50>\S+) "
    r"f32_err=(?P<f32_err>\S+) n_lt16=(?P<n_lt16>\S+) "
    r"inflight_peak_ticks=(?P<inflight_peak_ticks>\S+) pared_s=(?P<pared>\S+) "
    r"veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_spectrum_probe", SCRIPTS / "l3_spectrum_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load()


@dataclass
class Ev:
    """Un evento: los precios de los ticks de `c` y de `os`, y los registros de A y T.

    `gap` salta ids antes del último tick del evento (lo agranda sin agregar ticks). El tick de
    referencia es el último del evento anterior (o `first` en el primero).
    """

    c: list[int]
    os: list[int] = field(default_factory=list)
    gap: int = 0
    accel: float | None = 0.1
    t50: float | None = 0.5


def walk(returns, start: int = BASE) -> list[int]:
    """Precios enteros que parten de `start` y siguen los retornos logarítmicos dados."""
    return [round(start * math.exp(x)) for x in np.cumsum(returns)]


R = 1e-3
ALTERNATING = [R, -R] * 8  # n = 16: toda la energía en el período 2 a 4
SQUARE = [R, R, -R, -R] * 4  # n = 16: período 4
SINE = [R * math.sin(2 * math.pi * k / 8) for k in range(1, 65)]  # n = 64: período 8
# Oráculo a mano (ventana de Hann periódica; el centro del bin se lleva 2/3 y cada vecino 1/6):
#  - alternados (k = 8 de 16, período 2) y vecino k = 7 (período 2,29): todo en la banda 1.
#  - cuadrada de período 4, n = 16: bin 4 (período 4, banda 2) y vecinos 3 (período 5,33, banda 2)
#    y 5 (período 3,2, banda 1): banda 1 = 1/6, banda 2 = 5/6.
#  - senoide de período 8, n = 64: bin 8 (período 8) y vecino 7 (9,14) en la banda 3, vecino 9
#    (7,11) en la banda 2: banda 2 = 1/6, banda 3 = 5/6.
# Haar (energía por nivel de la serie centrada, que suma la energía total):
#  - alternados: pares (r, −r), d = √2·r por par, todo en el nivel 1.
#  - cuadrada: los pares (r, r) y (−r, −r) no tienen detalle; a1 = ±√2·r; el nivel 2 lo tiene todo.
#  - senoide, con a = √2/2: nivel 1 = (2 − √2)/4, nivel 2 = 1/8, nivel 3 = (3 + 2√2)/8.
ORACLE_FOURIER = {
    "alternados": [1, 0, 0, 0, 0, 0, 0],
    "cuadrada": [1 / 6, 5 / 6, 0, 0, 0, 0, 0],
    "senoide": [0, 1 / 6, 5 / 6, 0, 0, 0, 0],
}
ORACLE_HAAR = {
    "alternados": [1, 0, 0, 0, 0, 0, 0],
    "cuadrada": [0, 1, 0, 0, 0, 0, 0],
    "senoide": [(2 - math.sqrt(2)) / 4, 1 / 8, (3 + 2 * math.sqrt(2)) / 8, 0, 0, 0, 0],
}
ORACLE_SERIES = {"alternados": ALTERNATING, "cuadrada": SQUARE, "senoide": SINE}


def oracle_events() -> list[Ev]:
    """Evento 1: c alternados y os senoide; evento 2: c cuadrada, sin overshoot."""
    first = walk(ALTERNATING)
    sine = walk(SINE, start=first[-1])
    second = walk(SQUARE, start=sine[-1])
    return [Ev(first, sine), Ev(second)]


def realized(anchor: int, prices: list[int]) -> float:
    steps = [anchor, *prices]
    return sum(math.log(b / a) ** 2 for a, b in pairwise(steps))


def write_lake(
    root: Path,
    events: list[Ev],
    *,
    first: int = BASE,
    rows_per_group: int = 7,
    drop_first: int = 0,
    ticks_override: dict[int, int] | None = None,
) -> dict[str, Path]:
    """Escribe los cinco archivos de `(THETA, MONTH)`; devuelve sus rutas.

    `drop_first` quita esos primeros ticks del L1 (el evento arranca en el mes anterior);
    `ticks_override[i]` cambia `c_ticks` del evento `i`.
    """
    ticks = [(1_000, first)]
    ref_id, ref_price = 1_000, first
    rows = []
    for index, event in enumerate(events):
        c_ids = list(range(ref_id + 1, ref_id + 1 + len(event.c)))
        conf_id = c_ids[-1]
        os_ids = list(range(conf_id + 1, conf_id + 1 + len(event.os)))
        if event.gap:
            (os_ids if os_ids else c_ids)[-1] += event.gap
        conf_id = c_ids[-1]
        ext_id = os_ids[-1] if os_ids else conf_id
        ticks += list(zip(c_ids, event.c, strict=True))
        ticks += list(zip(os_ids, event.os, strict=True))
        conf_price = event.c[-1]
        rows.append(
            {
                "ref_price": ref_price,
                "ref_id": ref_id,
                "conf_price": conf_price,
                "conf_id": conf_id,
                "ext_id": ext_id,
                "c_ticks": (ticks_override or {}).get(index, len(event.c)),
                "os_ticks": len(event.os),
                "c_var": realized(ref_price, event.c),
                "os_var": realized(conf_price, event.os) if event.os else None,
                "t50": event.t50,
                "accel": event.accel,
            }
        )
        ref_id, ref_price = ext_id, (event.os or event.c)[-1]
    in_month = ticks[drop_first:]

    def exact(values):
        return pa.array([Decimal(v).scaleb(-8) for v in values], DECIMAL)

    partition = f"{SERIES}/theta={THETA}/year=2023/month=03"
    paths = {
        "l1": root / "l1" / SERIES / "year=2023/month=03" / "consolidated.parquet",
        "events": root / "l2" / partition / "events.parquet",
        "summaries": root / "l3" / partition / "summaries.parquet",
        "path": root / "l3" / partition / "path.parquet",
        "fit": root / "l3" / partition / "fit.parquet",
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
                "c_realized_var": column("c_var", pa.float64()),
                "os_realized_var": column("os_var", pa.float64()),
                "c_t50": column("t50", pa.float64()),
            }
        ),
        paths["path"],
        row_group_size=5,
    )
    pq.write_table(
        pa.table(
            {"confirm_agg_trade_id": ids, "c_accel": column("accel", pa.float64())}
        ),
        paths["fit"],
        row_group_size=4,
    )
    return paths


def write_l2_events(root: Path, theta: str, month: str, triples: list[tuple]) -> Path:
    """Un `events.parquet` de L2 con solo `(R, C, E)`, en row groups de 4 eventos."""
    year, number = month.split("-")
    path = (
        root
        / "l2"
        / SERIES
        / f"theta={theta}/year={year}/month={number}"
        / "events.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "reference_agg_trade_id": pa.array([t[0] for t in triples], pa.int64()),
                "confirm_agg_trade_id": pa.array([t[1] for t in triples], pa.int64()),
                "extreme_agg_trade_id": pa.array([t[2] for t in triples], pa.int64()),
            }
        ),
        path,
        row_group_size=4,
    )
    return path


def write_summaries(
    root: Path, theta: str, c_ticks: list[int], os_ticks: list[int]
) -> Path:
    path = (
        root / "l3" / SERIES / f"theta={theta}/year=2023/month=03" / "summaries.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "confirm_agg_trade_id": pa.array(range(len(c_ticks)), pa.int64()),
                "c_ticks": pa.array(c_ticks, pa.int64()),
                "os_ticks": pa.array(os_ticks, pa.int64()),
            }
        ),
        path,
        row_group_size=3,
    )
    return path


def argv(root: Path, out: Path, *extra: str, theta: str = THETA) -> list[str]:
    return [
        "--theta",
        theta,
        "--month",
        MONTH,
        "--l1-root",
        str(root / "l1"),
        "--l2-root",
        str(root / "l2"),
        "--l3-root",
        str(root / "l3"),
        "--out",
        str(out),
        *extra,
    ]


def uris(paths: dict[str, Path]) -> dict[str, str]:
    return {label: str(path) for label, path in paths.items()}


def last_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    match = LAST_LINE.match(line)
    assert match, line
    return match.groupdict()


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def value(rows, **where):
    found = [r for r in rows if all(r[k] == str(v) for k, v in where.items())]
    assert len(found) == 1, (where, len(found))
    return found[0]["valor"]


# --- (a) el oráculo a mano -----------------------------------------------------------------


@pytest.mark.parametrize("name", ORACLE_SERIES)
def test_fracciones_de_fourier_y_de_haar_a_mano(probe, name):
    series = np.array(ORACLE_SERIES[name])
    window = probe.hann(len(series))
    fourier = probe.fractions(series, window)
    assert fourier.tolist() == pytest.approx(ORACLE_FOURIER[name], abs=1e-9)
    assert fourier.sum() == pytest.approx(1.0, abs=1e-9)
    levels = probe.haar(series)
    assert levels.tolist() == pytest.approx(ORACLE_HAAR[name], abs=1e-9)
    assert levels.sum() == pytest.approx(1.0, abs=1e-9)


def test_las_bandas_son_octavas_de_periodo(probe):
    """Un coseno de un solo bin, sin ventana, cae en la banda de su período `n/k`."""
    n = 1024
    # (k, período n/k, índice de banda): 2 → 0, 3,41 → 0, 3,98 → 0, 5,12 → 1, 8 → 2, 16 → 3,
    # 51,2 → 4, 102,4 → 5, 128 → 6 (la banda de 128 o más incluye el borde), 256 → 6.
    cases = (
        (512, 0), (300, 0), (257, 0), (200, 1), (128, 2), (64, 3),
        (20, 4), (10, 5), (8, 6), (4, 6),
    )  # fmt: skip
    for k, band in cases:
        series = np.cos(2 * np.pi * k * np.arange(n) / n)
        fractions = probe.fractions(series, np.ones(n))
        assert fractions.argmax() == band, (k, band)
        assert fractions[band] == pytest.approx(1.0, abs=1e-9)


def test_energia_cero_da_nan(probe):
    series = np.full(20, 0.5)
    assert np.isnan(probe.fractions(series, probe.hann(20))).all()
    assert np.isnan(probe.haar(series)).all()


def test_haar_usa_los_ultimos_2_a_la_parte_entera_de_log2_n(probe):
    """Con n = 20 se leen los últimos 16 valores: los primeros 4 no cuentan."""
    tail = np.array(SQUARE)
    series = np.concatenate(([9.0, -9.0, 4.0, 1.0], tail))
    assert probe.haar(series).tolist() == pytest.approx(
        ORACLE_HAAR["cuadrada"], abs=1e-9
    )


def test_spearman(probe):
    assert probe.ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    x = np.arange(10.0)
    assert probe.spearman(x, x**3) == pytest.approx(1.0)
    assert probe.spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(probe.spearman(x, np.ones(10)))
    assert probe.spearman(
        np.array([1.0, 2.0, np.nan, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])
    ) == (pytest.approx(1.0))


def test_oraculo_de_punta_a_punta_sobre_el_lago(probe, tmp_path):
    paths = write_lake(tmp_path, oracle_events())
    assert (
        pq.ParquetFile(paths["l1"]).num_row_groups >= 3
    )  # las fases cruzan row groups
    samples, counters = probe.measure(uris(paths), seed=7, nulls=5)
    assert samples.n == {"c": 2, "os": 1}
    assert counters == {("os", "vacias"): 1}  # el evento 2 no tiene overshoot
    for candidate, table in (("retornos", ORACLE_FOURIER), ("haar", ORACLE_HAAR)):
        c = samples.observed(candidate, "c")
        assert c[0].tolist() == pytest.approx(table["alternados"], abs=1e-9)
        assert c[1].tolist() == pytest.approx(table["cuadrada"], abs=1e-9)
        os = samples.observed(candidate, "os")
        assert os[0].tolist() == pytest.approx(table["senoide"], abs=1e-9)
    # Los registros de A y T siguen a las fases de c, y el precio de c es alternado como los retornos.
    assert samples.accel[:2].tolist() == [0.1, 0.1]
    assert samples.observed("precios", "c")[0].tolist() == pytest.approx(
        ORACLE_FOURIER["alternados"], abs=1e-9
    )
    for candidate in probe.CANDIDATES:
        for phase in probe.PHASES:
            observed, simulated = (
                samples.observed(candidate, phase),
                samples.simulated(candidate, phase),
            )
            assert observed.sum(axis=1) == pytest.approx(1.0, abs=1e-9)
            assert simulated.sum(axis=1) == pytest.approx(1.0, abs=1e-5)
            assert len(simulated) == 5 * len(observed)
    assert samples.f32_err < 1e-4


@pytest.mark.parametrize("rows_per_group", [1, 3, 50, 1000])
def test_el_resultado_no_depende_de_los_row_groups(probe, tmp_path, rows_per_group):
    reference = write_lake(tmp_path / "ref", oracle_events(), rows_per_group=1000)
    expected, _ = probe.measure(uris(reference), seed=7, nulls=4)
    paths = write_lake(tmp_path / "x", oracle_events(), rows_per_group=rows_per_group)
    samples, _ = probe.measure(uris(paths), seed=7, nulls=4)
    for key, values in expected.obs.items():
        n = expected.n[key[1]]
        assert samples.obs[key][:n].tolist() == values[:n].tolist()
        assert samples.sim[key][: 4 * n].tolist() == expected.sim[key][: 4 * n].tolist()


def test_csv_de_un_evento_trae_las_cifras_a_mano(probe, tmp_path, capsys):
    paths = write_lake(tmp_path, [oracle_events()[0]])
    out = tmp_path / "salida" / "tabla.csv"
    assert probe.main(argv(tmp_path, out, "--nulls", "3", "--inflight-months", "")) == 0
    rows = read_csv(out)
    assert list(rows[0]) == [
        "seccion",
        "candidato",
        "fase",
        "banda",
        "fuente",
        "medida",
        "valor",
    ]
    cell = {
        "seccion": "cuantiles",
        "candidato": "retornos",
        "fase": "c",
        "fuente": "observado",
    }
    for band, expected in enumerate(ORACLE_FOURIER["alternados"], start=1):
        # Con N = 1 todos los cuantiles son el único valor.
        for percent in (1, 50, 99):
            got = float(value(rows, **cell, banda=band, medida=f"p{percent}"))
            assert got == pytest.approx(expected, abs=1e-9)
        assert value(rows, **cell, banda=band, medida="N") == "1"
    sim = {**cell, "fuente": "simulado"}
    assert value(rows, **sim, banda=1, medida="N") == "3"
    haar_os = {
        "seccion": "cuantiles",
        "candidato": "haar",
        "fase": "os",
        "fuente": "observado",
    }
    for band, expected in enumerate(ORACLE_HAAR["senoide"], start=1):
        got = float(value(rows, **haar_os, banda=band, medida="p50"))
        assert got == pytest.approx(expected, abs=1e-9)
    assert value(rows, seccion="muestras", fase="c", medida="N") == "1"
    assert value(rows, seccion="config", medida="nulls") == "3"
    assert paths["l1"].exists()
    line = last_line(capsys)
    assert line["eventos"] == "1"
    assert line["veredicto"] in ("sigue", "retirada")


# --- (b) D alto con un ritmo, bajo con ruido -------------------------------------------------


def rhythmic_events(count: int, seed: int = 5) -> list[Ev]:
    """c de 32 ticks con un ritmo de período 8 (fase y amplitud al azar) y un poco de ruido."""
    rng = np.random.default_rng(seed)
    events, start = [], BASE
    for _ in range(count):
        phase = rng.uniform(0, 2 * math.pi)
        amplitude = R * rng.uniform(0.5, 2.0)
        k = np.arange(1, 33)
        returns = amplitude * np.sin(
            2 * np.pi * k / 8 + phase
        ) + 0.05 * R * rng.standard_normal(32)
        prices = walk(returns, start)
        events.append(Ev(prices, accel=float(rng.normal()), t50=float(rng.uniform())))
        start = prices[-1]
    return events


def iid_events(count: int, seed: int = 6) -> list[Ev]:
    rng = np.random.default_rng(seed)
    events, start = [], BASE
    for _ in range(count):
        prices = walk(R * rng.standard_normal(32), start)
        events.append(Ev(prices, accel=float(rng.normal()), t50=float(rng.uniform())))
        start = prices[-1]
    return events


def test_un_ritmo_de_periodo_8_da_D_alto_y_el_ruido_iid_da_D_bajo(
    probe, tmp_path, capsys
):
    write_lake(tmp_path / "ritmo", rhythmic_events(300), rows_per_group=500)
    out = tmp_path / "ritmo.csv"
    assert probe.main(argv(tmp_path / "ritmo", out, "--inflight-months", "")) == 0
    rhythm = last_line(capsys)
    assert (
        float(rhythm["D_ret_fast"]) > 0.9
    )  # los retornos barajados sí tienen energía rápida
    assert float(rhythm["D_ret_slow"]) < 1.0
    assert float(rhythm["D_price_fast"]) > 0.5
    assert rhythm["veredicto"] == "sigue"  # el ritmo no repite a c_accel ni a c_t50

    write_lake(tmp_path / "ruido", iid_events(500), rows_per_group=500)
    assert (
        probe.main(
            argv(tmp_path / "ruido", tmp_path / "ruido.csv", "--inflight-months", "")
        )
        == 0
    )
    noise = last_line(capsys)
    for name in ("D_ret_fast", "D_ret_slow"):
        assert float(noise[name]) < 0.10, noise
    assert noise["veredicto"] in ("retirada", "sigue")
    assert float(noise["f32_err"]) < 1e-4


def test_veredicto_exige_que_no_repita_a_A_ni_a_T(probe):
    """D alto con |ρ| ≤ 0,9 sigue; con |ρ| > 0,9 contra c_accel o c_t50 se retira."""
    distances = {("retornos", "c"): (0.5, 0.0), ("precios", "c"): (0.0, 0.0)}
    low = np.full(7, 0.3)
    high = np.array([0.1, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0])
    ok = {("retornos_accel", "c"): low, ("retornos_t50", "c"): low}
    assert probe.verdict_of(distances, ok) == "sigue"
    for pair in ("retornos_accel", "retornos_t50"):
        assert probe.verdict_of(distances, {**ok, (pair, "c"): high}) == "retirada"
    quiet = {("retornos", "c"): (0.099, 0.099), ("precios", "c"): (0.0, 0.0)}
    assert probe.verdict_of(quiet, ok) == "retirada"
    assert (
        probe.verdict_of(
            {**quiet, ("precios", "c"): (0.0, 0.2)}, {**ok, ("precios_accel", "c"): low}
        )
        == "sigue"
    )


# --- exclusiones ----------------------------------------------------------------------------


def test_cuenta_las_fases_cortas_vacias_de_energia_cero_y_el_presupuesto(
    probe, tmp_path
):
    short = walk(SQUARE[:15], BASE)
    flat = [short[-1]] * 16  # el ancla es el último tick del evento anterior
    events = [
        Ev(walk(ALTERNATING), walk(SINE[:3], BASE)),  # os de 3 ticks: corta
        Ev(short),  # c de 15 ticks: corta; sin os: vacía
        Ev(flat),  # c de energía cero (todos los retornos iguales)
        Ev(
            walk(SQUARE), gap=probe.BUDGET
        ),  # extreme − reference > 20 M: sobre el presupuesto
        Ev(walk(SQUARE)),  # el último, medible
    ]
    paths = write_lake(tmp_path, events)
    samples, counters = probe.measure(uris(paths), seed=3, nulls=2)
    assert samples.n == {"c": 2, "os": 0}
    assert counters["os", "cortas"] == 1
    assert counters["c", "cortas"] == 1
    # Eventos 2, 3 y 5; el 4, sobre el presupuesto, no cuenta sus fases.
    assert counters["os", "vacias"] == 3
    assert counters["c", "energia_cero"] == 1
    assert counters["c", "energia_cero", "retornos"] == 1
    assert counters["c", "energia_cero", "precios"] == 1
    assert counters["eventos_sobre_presupuesto"] == 1


def test_el_evento_que_arranca_en_el_mes_anterior_se_cuenta_y_no_falla(probe, tmp_path):
    events = [Ev(walk(SQUARE), walk(SINE)), Ev(walk(ALTERNATING))]
    paths = write_lake(
        tmp_path, events, drop_first=4
    )  # al primero le faltan 3 ticks de c
    samples, counters = probe.measure(uris(paths), seed=3, nulls=2)
    assert counters["evento_mes_anterior"] == 1
    assert samples.n == {"c": 1, "os": 0}


def test_summaries_que_no_cuadra_con_L1_es_un_error(probe, tmp_path):
    events = [Ev(walk(SQUARE)), Ev(walk(ALTERNATING))]
    paths = write_lake(tmp_path, events, ticks_override={1: 17})
    with pytest.raises(probe.InputError, match="summaries dice 17"):
        probe.measure(uris(paths), seed=3, nulls=2)


def test_misma_semilla_mismo_csv_y_otra_semilla_otro_csv(probe, tmp_path):
    write_lake(tmp_path, oracle_events())
    outputs = {}
    for label, seed in (("a", 11), ("b", 11), ("c", 12)):
        out = tmp_path / f"{label}.csv"
        args = argv(
            tmp_path, out, "--nulls", "4", "--seed", str(seed), "--inflight-months", ""
        )
        assert probe.main(args) == 0
        outputs[label] = out.read_bytes()
    assert outputs["a"] == outputs["b"]
    assert outputs["a"] != outputs["c"]


def test_el_grupo_de_nulos_no_cambia_las_cifras_observadas(
    probe, tmp_path, monkeypatch
):
    paths = write_lake(tmp_path, oracle_events())
    base, _ = probe.measure(uris(paths), seed=1, nulls=6)
    monkeypatch.setattr(probe, "GROUP", 32)  # un nulo por grupo: n = 16 y 64
    small, _ = probe.measure(uris(paths), seed=1, nulls=6)
    for key, values in base.obs.items():
        n = base.n[key[1]]
        assert small.obs[key][:n].tolist() == values[:n].tolist()


# --- (c) el pico de ticks en vuelo -----------------------------------------------------------

# Tres θ, a mano. Fases (a, b]: cuentan x − a en el tick x.
#   θ1: (0,10] (10,30] (30,50] (50,60]
#   θ2: (0,20] (20,25] (25,45] (45,100]
#   θ3: (5,15] y un evento sobre el presupuesto (15, 40, 20 000 016]: cuenta 0.
# Sumas en los fines de fase: x=10: 10+10+5 = 25; x=15: 5+15+10 = 30; x=20: 10+20 = 30; x=25: 15+5;
# x=30: 20+5; x=45: 15+20 = 35; x=50: 20+5; x=60: 10+15; x=100: 55 (solo θ2) ← el pico.
# Si el evento sobre el presupuesto contara, el pico sería de unos 20 M.
THREE_THETAS = {
    "0.00098052": [(0, 10, 30), (30, 50, 60)],
    "0.00238246": [(0, 20, 25), (25, 45, 100)],
    "0.00500000": [(5, 15, 15), (15, 40, 20_000_016)],
}
THREE_PEAK = 55


def brute_peak(thetas: dict[str, list[tuple]], budget: int) -> int:
    """La definición, tick a tick."""
    top = max(e for events in thetas.values() for _, _, e in events)
    peak = 0
    for x in range(top + 2):
        total = 0
        for events in thetas.values():
            for ref, conf, ext in events:
                if ext - ref > budget:
                    continue
                if ref < x <= conf:
                    total += x - ref
                elif conf < x <= ext:
                    total += x - conf
        peak = max(peak, total)
    return peak


def test_pico_de_ticks_en_vuelo_a_mano(probe, tmp_path, monkeypatch):
    paths = {
        theta: str(write_l2_events(tmp_path, theta, MONTH, events))
        for theta, events in THREE_THETAS.items()
    }
    assert probe.month_inflight(paths) == THREE_PEAK
    # Con lotes de un solo evento el resultado es el mismo.
    monkeypatch.setattr(probe, "INFLIGHT_BATCH", 1)
    assert probe.month_inflight(paths) == THREE_PEAK


def test_pico_de_ticks_en_vuelo_contra_la_definicion(probe, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "BUDGET", 40)
    monkeypatch.setattr(probe, "INFLIGHT_BATCH", 3)
    rng = np.random.default_rng(9)
    for case in range(6):
        thetas = {}
        for index in range(5):
            events, ref = [], int(rng.integers(0, 20))
            for _ in range(int(rng.integers(1, 14))):
                conf = ref + int(rng.integers(1, 15))
                ext = conf + int(rng.integers(0, 45))
                events.append((ref, conf, ext))
                ref = ext
            thetas[f"0.0{case}00000{index}"] = events
        paths = {
            theta: str(write_l2_events(tmp_path / str(case), theta, MONTH, events))
            for theta, events in thetas.items()
        }
        assert probe.month_inflight(paths) == brute_peak(thetas, 40), case


def test_pico_en_el_csv_y_en_la_ultima_linea(probe, tmp_path, capsys):
    """El lago principal es el θ1 de verdad; los otros dos θ solo traen su L2 del mes."""
    events = [Ev(walk(SQUARE), walk(SINE)), Ev(walk(ALTERNATING))]
    write_lake(tmp_path, events)
    main_triples = [
        tuple(int(v) for v in row)
        for row in zip(
            *(
                pq.read_table(
                    tmp_path
                    / "l2"
                    / SERIES
                    / f"theta={THETA}/year=2023/month=03/events.parquet"
                )
                .column(name)
                .to_pylist()
                for name in (
                    "reference_agg_trade_id",
                    "confirm_agg_trade_id",
                    "extreme_agg_trade_id",
                )
            ),
            strict=True,
        )
    ]
    others = {t: e for t, e in THREE_THETAS.items() if t != THETA}
    for theta, triples in others.items():
        write_l2_events(tmp_path, theta, MONTH, triples)
    # Otro mes con un solo θ: el pico es el de sus ticks.
    write_l2_events(tmp_path, THETA, "2021-01", [(0, 7, 7), (7, 9, 9)])
    out = tmp_path / "tabla.csv"
    args = argv(tmp_path, out, "--nulls", "2", "--inflight-months", f"2021-01,{MONTH}")
    assert probe.main(args) == 0
    rows = read_csv(out)
    expected = brute_peak({THETA: main_triples, **others}, probe.BUDGET)
    assert (
        value(
            rows,
            seccion="ticks_en_vuelo",
            fuente="mes=2021-01",
            medida="inflight_peak_ticks",
        )
        == "7"
    )
    assert value(
        rows,
        seccion="ticks_en_vuelo",
        fuente=f"mes={MONTH}",
        medida="inflight_peak_ticks",
    ) == str(expected)
    assert value(
        rows, seccion="ticks_en_vuelo", fuente=f"mes={MONTH}", medida="bytes_8B"
    ) == str(expected * 8)
    assert value(
        rows, seccion="ticks_en_vuelo", fuente=f"mes={MONTH}", medida="bytes_4B"
    ) == str(expected * 4)
    assert (
        expected >= THREE_PEAK - 0 and expected < 1_000_000
    )  # el evento sobre presupuesto no cuenta
    assert last_line(capsys)["inflight_peak_ticks"] == str(expected)


# --- (d) la fracción bajo 16 ticks por θ -------------------------------------------------------


def test_fraccion_bajo_16_por_theta_a_mano(probe, tmp_path, capsys):
    write_lake(tmp_path, [Ev(walk(SQUARE), walk(SINE)), Ev(walk(ALTERNATING))])
    # θ principal: c = [16, 16], os = [64, 0] (de las fases escritas arriba).
    write_summaries(tmp_path, "0.00238246", [5, 16, 20, 15], [0, 3, 16, 100])
    write_summaries(tmp_path, "0.00500000", [1, 2, 3, 4, 5, 6, 7, 8], [16] * 8)
    out = tmp_path / "tabla.csv"
    assert probe.main(argv(tmp_path, out, "--nulls", "2", "--inflight-months", "")) == 0
    rows = read_csv(out)

    def fraction(theta, phase, measure):
        return value(
            rows,
            seccion="fases_lt16",
            fuente=f"theta={theta}",
            fase=phase,
            medida=measure,
        )

    assert fraction(THETA, "c", "lt16") == "0"
    assert fraction(THETA, "os", "lt16") == "1"  # la fase os vacía cuenta como bajo 16
    assert fraction(THETA, "os", "vacias") == "1"
    assert float(fraction(THETA, "os", "fraccion_lt16")) == 0.5
    assert fraction("0.00238246", "c", "lt16") == "2"
    assert float(fraction("0.00238246", "c", "fraccion_lt16")) == 0.5
    assert fraction("0.00238246", "os", "lt16") == "2"
    assert fraction("0.00238246", "os", "vacias") == "1"
    assert float(fraction("0.00500000", "c", "fraccion_lt16")) == 1.0
    assert float(fraction("0.00500000", "os", "fraccion_lt16")) == 0.0
    line = last_line(capsys)
    assert line["n_lt16"] == "0"
    assert line["inflight_peak_ticks"] == "-"


def test_n_lt16_de_la_ultima_linea_es_el_del_theta_medido(probe, tmp_path, capsys):
    events = [Ev(walk(SQUARE)), Ev(walk(SQUARE[:15], BASE)), Ev(walk(SQUARE[:9], BASE))]
    write_lake(tmp_path, events)
    out = tmp_path / "tabla.csv"
    assert probe.main(argv(tmp_path, out, "--nulls", "2", "--inflight-months", "")) == 0
    assert last_line(capsys)["n_lt16"] == "2"


# --- (e) códigos de salida ---------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["l1", "events", "summaries", "path", "fit"])
def test_un_archivo_faltante_da_codigo_1(probe, tmp_path, capsys, missing):
    paths = write_lake(tmp_path, oracle_events())
    paths[missing].unlink()
    out = tmp_path / "tabla.csv"
    assert probe.main(argv(tmp_path, out, "--inflight-months", "")) == 1
    output = capsys.readouterr().out
    assert "falta el archivo" in output
    assert output.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not out.exists()


def test_un_mes_de_ticks_en_vuelo_sin_L2_da_codigo_1(probe, tmp_path, capsys):
    write_lake(tmp_path, oracle_events())
    out = tmp_path / "tabla.csv"
    assert probe.main(argv(tmp_path, out, "--inflight-months", "2017-12")) == 1
    assert "L2 no tiene eventos de 2017-12" in capsys.readouterr().out


def test_los_archivos_desalineados_dan_codigo_1(probe, tmp_path, capsys):
    paths = write_lake(tmp_path, oracle_events())
    table = pq.read_table(paths["fit"])
    pq.write_table(table.slice(0, 1), paths["fit"])
    assert probe.main(argv(tmp_path, tmp_path / "t.csv", "--inflight-months", "")) == 1
    assert "distinto número de filas" in capsys.readouterr().out


def test_sin_eventos_medibles_da_codigo_1(probe, tmp_path, capsys):
    write_lake(tmp_path, [Ev(walk(SQUARE[:15], BASE))])
    assert probe.main(argv(tmp_path, tmp_path / "t.csv", "--inflight-months", "")) == 1
    assert "ningún evento medible" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [
        ["--theta", "0.123456789"],  # más de 8 decimales
        ["--theta", "-1"],
        ["--theta", "abc"],
        ["--month", "2023-13"],
        ["--month", "marzo"],
        ["--nulls", "0"],
        ["--nulls", "veinte"],
        ["--seed", "1.5"],
        ["--inflight-months", "2023-03,xx"],
    ],
)
def test_args_invalidos_dan_codigo_2(probe, tmp_path, extra):
    args = argv(tmp_path, tmp_path / "t.csv")
    for flag, new in zip(extra[::2], extra[1::2], strict=True):
        if flag in args:
            args[args.index(flag) + 1] = new
        else:
            args += [flag, new]
    with pytest.raises(SystemExit) as raised:
        probe.main(args)
    assert raised.value.code == 2


@pytest.mark.parametrize("missing", ["--theta", "--month"])
def test_args_obligatorios_dan_codigo_2(probe, tmp_path, missing):
    args = argv(tmp_path, tmp_path / "t.csv")
    del args[args.index(missing) : args.index(missing) + 2]
    with pytest.raises(SystemExit) as raised:
        probe.main(args)
    assert raised.value.code == 2


def test_sin_raices_ni_OPS_RESULTS_URI_da_codigo_2(probe, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        probe.main(["--theta", THETA, "--month", MONTH])
    assert raised.value.code == 2


def test_out_gs_fuera_de_results_da_codigo_2(probe, monkeypatch, tmp_path):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proy-ops/results/run1/")
    args = argv(tmp_path, tmp_path / "t.csv")
    args[args.index("--out") + 1] = "gs://otro-bucket/x.csv"
    with pytest.raises(SystemExit) as raised:
        probe.main(args)
    assert raised.value.code == 2


def test_out_que_termina_en_barra_es_un_directorio(probe, tmp_path):
    write_lake(tmp_path, oracle_events())
    out = tmp_path / "carpeta" / ""
    args = argv(tmp_path, out, "--nulls", "2", "--inflight-months", "")
    args[args.index("--out") + 1] = f"{out}/"
    assert probe.main(args) == 0
    assert (tmp_path / "carpeta" / f"l3_spectrum_probe_{MONTH}_{THETA}.csv").exists()
