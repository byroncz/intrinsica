"""`l3_fit_bridge_probe.py` contra un lago local con los cuatro archivos sintéticos.

El lago de cada prueba se escribe a mano en `tmp_path` con la misma disposición hive que el
real: `consolidated.parquet` de L1, `events.parquet` de L2 y `summaries.parquet` y
`path.parquet` de L3, con solo las columnas que la sonda lee. Los ticks de L1 van en row groups
de pocos ticks, así que cada fase de confirmación cruza varios. Un evento se escribe como su tick
de referencia, los ticks de su confirmación y un tick de relleno (el overshoot, que la sonda no
lee); los precios son enteros sin escalar (`DECIMAL(18,8)`), y los registros no dependen de la
escala.

El oráculo (`ORACLE`) son tres eventos de pocos ticks cuyos `c_accel` y `c_jump_share` están
calculados a mano abajo; la hija 2 de la Épica ITSC-397 (el motor nativo) lo reutiliza.
"""

import csv
import importlib.util
import math
import re
from dataclasses import dataclass
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
LAST_LINE = re.compile(
    r"fit probe: mes=(?P<mes>\S+) theta=(?P<theta>\S+) eventos=(?P<eventos>\S+) "
    r"excluidos=(?P<excluidos>\S+) puentes=(?P<puentes>\S+) semilla=(?P<semilla>\S+) "
    r"D_accel=(?P<D_accel>\S+) D_jump=(?P<D_jump>\S+) pared_s=(?P<pared>\S+) "
    r"veredicto=(?P<veredicto>\S+)$"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_fit_bridge_probe", SCRIPTS / "l3_fit_bridge_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load()


@dataclass
class Event:
    """Un evento: el tick de referencia, los de la confirmación y el precio de L2 del cierre.

    Los tiempos de `phase` son los del tick relativos a `base`; el último tick es el de
    `confirm_agg_trade_id`. `confirm_price` es el de L2 y puede diferir del último tick (grupo
    de empate). Con `before_month` ticks, el tick de referencia y esos primeros ticks de la fase
    quedan fuera del L1 del mes (el evento arranca en el mes anterior).
    """

    ref: tuple[int, int]
    phase: list[tuple[int, int]]
    confirm_price: int | None = None
    before_month: int = -1


# Oráculo a mano. τ = (t − t_R)/(t_C − t_R), y = (p − p_R)/(p_C − p_R); con los momentos
# T2 = Στ², T3 = Στ³, T4 = Στ⁴, S1 = Στy, S2 = Στ²y:
#   a = (S2·T2 − S1·T3) / (T4·T2 − T3²)
# Evento 1 (alza): R (0; 100), C (10; 110), ticks (2; 102) (5; 104) (8; 109) (10; 110).
#   τ = .2 .5 .8 1, y = .2 .4 .9 1; T2 = 1,93, T3 = 1,645, T4 = 1,4737, S1 = 1,96, S2 = 1,684.
#   det = 1,4737·1,93 − 1,645² = 0,138216; a = (1,684·1,93 − 1,96·1,645)/0,138216 = 1080/5759.
#   El último tick con t < 10 es (8; 109): salto = (110 − 109)/(110 − 100) = 1/10.
# Evento 2 (baja, con empate en C; L2 cierra en 190 aunque el último tick vale 191): R (100; 200),
#   C (140; 190), ticks (110; 198) (110; 195) (130; 192) (140; 190) (140; 191).
#   τ = .25 .25 .75 1 1, y = .2 .5 .8 1 .9; T2 = 43/16, T3 = 157/64, T4 = 595/256, S1 = 107/40,
#   S2 = 383/160; a = −22/39. Último tick con t < 140: (130; 192): salto = (190 − 192)/(190 − 200)
#   = 1/5.
# Evento 3 (alza con exceso): R (0; 1000), C (20; 1010), ticks (5; 1012) (10; 1004) (20; 1010).
#   τ = .25 .5 1, y = 1,2 .4 1; T2 = 21/16, T3 = 73/64, T4 = 273/256, S1 = 3/2, S2 = 47/40;
#   a = −864/505. Último tick con t < 20: (10; 1004): salto = (1010 − 1004)/10 = 3/5.
ORACLE = [
    Event((0, 100), [(2, 102), (5, 104), (8, 109), (10, 110)]),
    Event(
        (100, 200), [(110, 198), (110, 195), (130, 192), (140, 190), (140, 191)], 190
    ),
    Event((0, 1000), [(5, 1012), (10, 1004), (20, 1010)]),
]
ORACLE_ACCEL = [1080 / 5759, -22 / 39, -864 / 505]
ORACLE_JUMP = [1 / 10, 1 / 5, 3 / 5]


def realized_var(ref_price: int, phase: list[tuple[int, int]]) -> float:
    """`Σ (ln(p_i / p_{i−1}))²` de la confirmación, con el paso del ancla al primer tick."""
    prices = [ref_price, *(p for _, p in phase)]
    return sum(math.log(b / a) ** 2 for a, b in pairwise(prices))


def write_lake(
    root: Path,
    events: list[Event],
    *,
    rows_per_group: int = 4,
    ticks_override: dict[int, int] | None = None,
) -> dict[str, Path]:
    """Escribe los cuatro archivos; devuelve sus rutas. `ticks_override[i]` cambia `c_ticks`."""
    ticks, rows = [], []  # ticks: (id, t, price, en_el_mes)
    next_id = 5_000
    for index, event in enumerate(events):
        base = 1_000_000 * (index + 1)
        ref_id = next_id
        outside = event.before_month
        ticks.append((ref_id, base + event.ref[0], event.ref[1], outside < 0))
        for position, (dt, price) in enumerate(event.phase, start=1):
            ticks.append((ref_id + position, base + dt, price, position > outside))
        last_id = ref_id + len(event.phase)
        last_time = base + event.phase[-1][0]
        ticks.append((last_id + 1, last_time + 1, event.phase[-1][1], True))  # relleno
        next_id = last_id + 2
        rows.append(
            {
                "ref_price": event.ref[1],
                "ref_time": base + event.ref[0],
                "ref_id": ref_id,
                "conf_price": event.confirm_price or event.phase[-1][1],
                "conf_time": last_time,
                "conf_id": last_id,
                "ticks": (ticks_override or {}).get(index, len(event.phase)),
                "var": realized_var(event.ref[1], event.phase),
            }
        )
    in_month = [t for t in ticks if t[3]]

    def decimals(values):
        return pa.array([Decimal(v).scaleb(-8) for v in values], DECIMAL)

    l1 = root / "l1" / SERIES / "year=2023/month=03"
    partition = f"{SERIES}/theta={THETA}/year=2023/month=03"
    paths = {
        "l1": l1 / "consolidated.parquet",
        "events": root / "l2" / partition / "events.parquet",
        "summaries": root / "l3" / partition / "summaries.parquet",
        "path": root / "l3" / partition / "path.parquet",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "agg_trade_id": [t[0] for t in in_month],
                "transact_time": [t[1] for t in in_month],
                "price": decimals([t[2] for t in in_month]),
            }
        ),
        paths["l1"],
        row_group_size=rows_per_group,
    )
    ids = pa.array([r["conf_id"] for r in rows], pa.int64())
    pq.write_table(
        pa.table(
            {
                "reference_price": decimals([r["ref_price"] for r in rows]),
                "reference_time": pa.array([r["ref_time"] for r in rows], pa.int64()),
                "reference_agg_trade_id": pa.array(
                    [r["ref_id"] for r in rows], pa.int64()
                ),
                "confirm_price": decimals([r["conf_price"] for r in rows]),
                "confirm_time": pa.array([r["conf_time"] for r in rows], pa.int64()),
                "confirm_agg_trade_id": ids,
            }
        ),
        paths["events"],
        row_group_size=3,
    )
    pq.write_table(
        pa.table(
            {
                "confirm_agg_trade_id": ids,
                "c_ticks": pa.array([r["ticks"] for r in rows], pa.int64()),
            }
        ),
        paths["summaries"],
        row_group_size=2,
    )
    pq.write_table(
        pa.table(
            {
                "confirm_agg_trade_id": ids,
                "c_realized_var": pa.array([r["var"] for r in rows], pa.float64()),
            }
        ),
        paths["path"],
        row_group_size=5,
    )
    return paths


def argv(root: Path, out: Path, *extra: str, theta: str = "0.00098052") -> list[str]:
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


def read_csv(path: Path) -> dict[tuple[str, str, str], str]:
    with path.open() as handle:
        return {
            (r["registro"], r["fuente"], r["medida"]): r["valor"]
            for r in csv.DictReader(handle)
        }


def shaped_events(count: int = 150, seed: int = 11) -> list[Event]:
    """Trayectorias con forma: casi todo el recorrido en el último 10 % del tiempo."""
    rng = np.random.default_rng(seed)
    events = []
    for _ in range(count):
        ref, span = 2_000_000_000_000, 20_000
        move = int(ref * 0.001 * rng.uniform(1.0, 1.5))
        phase = []
        for k in range(1, 16):  # 15 ticks en τ ∈ (0, 0,9], casi sin moverse
            phase.append(
                (round(0.9 * span * k / 15), ref + round(0.02 * move * k / 15))
            )
        for j in range(1, 6):  # 5 ticks en τ ∈ (0,9, 1] que hacen todo el recorrido
            phase.append(
                (
                    round(span * (0.9 + 0.1 * j / 5)),
                    ref + round(move * (0.02 + 0.98 * j / 5)),
                )
            )
        events.append(Event((0, ref), phase))
    return events


def bridge_events(count: int = 400, seed: int = 4242) -> list[Event]:
    """Puentes brownianos puros: otra semilla que la de la sonda y σ√n = |ℓ_C − ℓ_R|."""
    rng = np.random.default_rng(seed)
    events = []
    for _ in range(count):
        n, ref, span = 15, 2_000_000_000_000, 20_000
        d = math.log(1.001 if rng.random() < 0.5 else 1 / 1.001)
        times = np.sort(rng.choice(np.arange(1, span), n - 1, replace=False))
        times = [*times.tolist(), span]
        walk = np.cumsum(rng.standard_normal(n))
        k = np.arange(1, n + 1)
        sigma = abs(d) / math.sqrt(n)
        log_price = math.log(ref) + d * k / n + sigma * (walk - k / n * walk[-1])
        prices = np.round(np.exp(log_price)).astype(np.int64).tolist()
        prices[-1] = round(ref * math.exp(d))
        events.append(Event((0, ref), list(zip(times, prices, strict=True))))
    return events


def test_oraculo_a_mano_coincide_con_la_sonda(probe, tmp_path):
    paths = write_lake(tmp_path, ORACLE)
    # El lago cruza row groups: L1 de 4 ticks por grupo y fases de 3 a 5 ticks.
    assert pq.ParquetFile(paths["l1"]).num_row_groups >= 3
    samples, pairs = probe.measure(uris(paths), seed=7, bridges=5)
    assert samples.n == 3
    assert samples.excluded == 0
    accel, jump = pairs["c_accel"][0], pairs["c_jump_share"][0]
    assert accel.tolist() == pytest.approx(sorted(ORACLE_ACCEL), rel=1e-12)
    assert jump.tolist() == pytest.approx(sorted(ORACLE_JUMP), rel=1e-12)
    assert len(pairs["c_accel"][1]) == 15


@pytest.mark.parametrize("rows_per_group", [1, 2, 3, 100])
def test_el_resultado_no_depende_de_los_row_groups(probe, tmp_path, rows_per_group):
    paths = write_lake(tmp_path, ORACLE, rows_per_group=rows_per_group)
    _, pairs = probe.measure(uris(paths), seed=7, bridges=5)
    assert pairs["c_accel"][0].tolist() == pytest.approx(
        sorted(ORACLE_ACCEL), rel=1e-12
    )
    assert pairs["c_jump_share"][0].tolist() == pytest.approx(
        sorted(ORACLE_JUMP), rel=1e-12
    )
    reference = write_lake(tmp_path / "ref", ORACLE, rows_per_group=100)
    _, expected = probe.measure(uris(reference), seed=7, bridges=5)
    for register in probe.REGISTERS:
        assert pairs[register][1].tolist() == pytest.approx(
            expected[register][1].tolist(), rel=1e-9, abs=1e-12
        )


def test_los_puentes_coinciden_con_una_simulacion_explicita(probe, tmp_path):
    """Reconstruye cada puente tick a tick y ajusta por mínimos cuadrados con NumPy."""
    paths = write_lake(tmp_path, ORACLE)
    seed, count = 7, 6
    _, pairs = probe.measure(uris(paths), seed=seed, bridges=count)
    rng = np.random.default_rng(seed)
    accel, jump = [], []
    for index, event in enumerate(ORACLE):
        n = len(event.phase)
        walk = np.cumsum(rng.standard_normal((n, count)), axis=0)
        sigma = math.sqrt(realized_var(event.ref[1], event.phase) / n)
        conf_price = event.confirm_price or event.phase[-1][1]
        d = math.log(conf_price / event.ref[1])
        k = np.arange(1, n + 1)[:, None]
        y = k / n + (sigma / d) * (walk - k / n * walk[-1])
        t_ref = 0 if index != 1 else 100
        span = event.phase[-1][0] - t_ref
        tau = np.array([(t - t_ref) / span for t, _ in event.phase])
        before = [i for i, (t, _) in enumerate(event.phase) if t < event.phase[-1][0]]
        for b in range(count):
            design = np.column_stack([tau**2, tau])
            accel.append(np.linalg.lstsq(design, y[:, b], rcond=None)[0][0])
            jump.append(1 - y[before[-1], b])
    assert pairs["c_accel"][1].tolist() == pytest.approx(sorted(accel), rel=1e-9)
    assert pairs["c_jump_share"][1].tolist() == pytest.approx(sorted(jump), rel=1e-9)


def test_trayectorias_con_forma_dan_d_alto_y_sigue(probe, tmp_path, capsys):
    write_lake(tmp_path, shaped_events())
    out = tmp_path / "out.csv"
    assert probe.main(argv(tmp_path, out)) == 0
    line = last_line(capsys)
    assert float(line["D_accel"]) >= 0.10
    assert line["veredicto"] == "sigue"
    assert line["eventos"] == "150"
    assert line["excluidos"] == "0"
    assert line["puentes"] == "20"
    assert line["semilla"] == str(probe.DEFAULT_SEED)


def test_puentes_puros_con_otra_semilla_dan_d_bajo_y_retirada(probe, tmp_path, capsys):
    write_lake(tmp_path, bridge_events())
    out = tmp_path / "out.csv"
    assert probe.main(argv(tmp_path, out)) == 0
    line = last_line(capsys)
    assert float(line["D_accel"]) < 0.10
    assert line["veredicto"] == "retirada"
    assert line["eventos"] == "400"
    table = read_csv(out)
    assert float(table[("c_accel", "ambas", "D")]) == pytest.approx(
        float(line["D_accel"]), abs=1e-6
    )


def test_csv_tiene_las_medidas_y_no_depende_del_reloj(probe, tmp_path, capsys):
    write_lake(tmp_path, ORACLE)
    first, second, other = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    assert probe.main(argv(tmp_path, first, "--bridges", "4")) == 0
    assert probe.main(argv(tmp_path, second, "--bridges", "4")) == 0
    assert probe.main(argv(tmp_path, other, "--bridges", "4", "--seed", "99")) == 0
    capsys.readouterr()
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != other.read_bytes()
    table = read_csv(first)
    for register in ("c_accel", "c_jump_share"):
        for percent in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            for source in ("observado", "simulado"):
                assert (register, source, f"p{percent}") in table
        assert table[(register, "observado", "N")] == "3"
        assert table[(register, "simulado", "N")] == "12"
        assert (register, "ambas", "D") in table
    assert table[("", "", "excluidos")] == "0"
    assert table[("", "", "puentes")] == "4"
    # Con tres valores, el rango más cercano da el mínimo (p1), la mediana (p50) y el máximo (p99).
    assert float(table[("c_accel", "observado", "p1")]) == pytest.approx(
        min(ORACLE_ACCEL)
    )
    assert float(table[("c_accel", "observado", "p50")]) == pytest.approx(
        sorted(ORACLE_ACCEL)[1]
    )
    assert float(table[("c_accel", "observado", "p99")]) == pytest.approx(
        max(ORACLE_ACCEL)
    )


def test_distancia_de_ks_exacta_con_empates(probe):
    a = np.array([1.0, 2.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 4.0, 5.0, 5.0, 6.0])
    # F_a: 1→.25, 2→.75, 3→1; F_b: 2→1/3, 4→.5, 5→5/6, 6→1. El supremo está en 3: 1 − 1/3.
    assert probe.ks_distance(a, b) == pytest.approx(2 / 3, rel=1e-12)
    assert probe.ks_distance(a, a) == 0.0
    assert probe.ks_distance(np.array([0.0, 1.0]), np.array([5.0, 6.0])) == 1.0
    assert probe.quantile(np.array([1.0, 2.0, 3.0, 4.0]), 50) == 2.0
    assert probe.quantile(np.array([1.0, 2.0, 3.0, 4.0]), 99) == 4.0


def test_eventos_sin_registro_se_excluyen_y_se_cuentan(probe, tmp_path):
    degenerate = [
        Event((0, 100), [(10, 110)]),  # un solo tick
        Event((0, 100), [(10, 105), (10, 110)]),  # todos los ticks en el mismo t
        Event(
            (0, 100), [(0, 105), (10, 110), (10, 111)], confirm_price=110
        ),  # τ = 0 y un τ
    ]
    paths = write_lake(tmp_path, [*ORACLE, *degenerate])
    samples, pairs = probe.measure(uris(paths), seed=7, bridges=5)
    assert (samples.n, samples.excluded) == (3, 3)
    assert pairs["c_accel"][0].tolist() == pytest.approx(
        sorted(ORACLE_ACCEL), rel=1e-12
    )


def test_evento_que_arranca_en_el_mes_anterior_se_excluye(probe, tmp_path):
    early = Event((0, 100), [(2, 102), (5, 104), (8, 109), (10, 110)], before_month=2)
    paths = write_lake(tmp_path, [early, *ORACLE])
    samples, pairs = probe.measure(uris(paths), seed=7, bridges=5)
    assert (samples.n, samples.excluded) == (3, 1)
    assert pairs["c_accel"][0].tolist() == pytest.approx(
        sorted(ORACLE_ACCEL), rel=1e-12
    )


def test_ticks_de_l1_que_no_cuadran_con_summaries_son_fallo(probe, tmp_path, capsys):
    write_lake(tmp_path, ORACLE, ticks_override={1: 4})
    assert probe.main(argv(tmp_path, tmp_path / "x.csv")) == 1
    text = capsys.readouterr().out
    assert "summaries dice 4" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not (tmp_path / "x.csv").exists()


def test_archivos_desalineados_son_fallo(probe, tmp_path, capsys):
    paths = write_lake(tmp_path, ORACLE)
    table = pq.read_table(paths["path"])
    pq.write_table(table.slice(0, 2), paths["path"])
    assert probe.main(argv(tmp_path, tmp_path / "x.csv")) == 1
    assert "distinto número de filas" in capsys.readouterr().out
    pq.write_table(
        pa.table(
            {
                "confirm_agg_trade_id": table.column("confirm_agg_trade_id")[::-1],
                "c_realized_var": table.column("c_realized_var"),
            }
        ),
        paths["path"],
    )
    assert probe.main(argv(tmp_path, tmp_path / "x.csv")) == 1
    assert "no están alineados" in capsys.readouterr().out


def test_eventos_mas_alla_de_los_ticks_son_fallo(probe, tmp_path, capsys):
    paths = write_lake(tmp_path, ORACLE)
    table = pq.read_table(paths["l1"])
    pq.write_table(table.slice(0, 8), paths["l1"], row_group_size=3)
    assert probe.main(argv(tmp_path, tmp_path / "x.csv")) == 1
    assert "más allá de los ticks" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["l1", "events", "summaries", "path"])
def test_archivo_faltante_da_codigo_1(probe, tmp_path, capsys, missing):
    paths = write_lake(tmp_path, ORACLE)
    paths[missing].unlink()
    assert probe.main(argv(tmp_path, tmp_path / "x.csv")) == 1
    text = capsys.readouterr().out
    assert f"falta el archivo {paths[missing]}" in text
    assert text.strip().splitlines()[-1].endswith("veredicto=fallo")
    assert not (tmp_path / "x.csv").exists()


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
        ["--theta", "0.01", "--month", "2026-09", "--bridges", "0"],
        ["--theta", "0.01", "--month", "2026-09", "--bridges", "x"],
        ["--theta", "0.01", "--month", "2026-09", "--seed", "x"],
    ],
)
def test_args_invalidos_dan_codigo_2(probe, extra, tmp_path):
    roots = ["--l1-root", str(tmp_path), "--l2-root", str(tmp_path)]
    with pytest.raises(SystemExit) as raised:
        probe.main([*roots, "--l3-root", str(tmp_path), *extra])
    assert raised.value.code == 2


def test_sin_raices_y_sin_ops_results_uri_es_uso(probe, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as raised:
        probe.main(["--theta", "0.01", "--month", "2026-09"])
    assert raised.value.code == 2


def test_out_fuera_de_results_es_uso(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proyecto-ops/results/ejecucion/")
    with pytest.raises(SystemExit) as raised:
        probe.main(
            argv(tmp_path, tmp_path / "x.csv")[:-2] + ["--out", "gs://otro/x.csv"]
        )
    assert raised.value.code == 2


def test_out_directorio_usa_el_nombre_por_defecto(probe, tmp_path, capsys):
    write_lake(tmp_path, ORACLE)
    out = tmp_path / "resultados"
    assert (
        probe.main(argv(tmp_path, tmp_path / "x.csv")[:-2] + ["--out", f"{out}/"]) == 0
    )
    capsys.readouterr()
    assert (out / f"l3_fit_bridge_probe_{MONTH}_{THETA}.csv").is_file()


def test_no_importa_ninguna_capa():
    source = (SCRIPTS / "l3_fit_bridge_probe.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", source, re.MULTILINE)
    assert not [m for m in imports if re.match(r"(dc_|l[123]_|viz_|ops_|pyutils)", m)]
