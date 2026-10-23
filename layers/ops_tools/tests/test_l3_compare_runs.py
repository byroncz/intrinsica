"""`l3_compare_runs.py` contra hallazgos reales de `shared/dq` escritos en un lago local.

Cada corrida se emite con `emit_findings` y un `detected_at` propio, así que cae en su carpeta
`detected_date=` como en `dq-findings/l3/`. El script no importa nada de L3.
"""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dq import Finding, Severity, Stage, Status, emit_findings

SCRIPTS = Path(__file__).parents[3] / "layers/ops_tools/scripts"
THETAS = ("0.00100000", "0.00250000")
MONTHS = ((2021, 1), (2021, 2))


def load():
    spec = importlib.util.spec_from_file_location(
        "l3_compare_runs", SCRIPTS / "l3_compare_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def micros(day: str, hour: int = 12) -> int:
    moment = datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000)


def finding(run, day, check, year, month, details, asset="BTCUSDT", hour=12) -> Finding:
    return Finding(
        detected_at=micros(day, hour),
        layer="l3",
        mode="backfill",
        check_type=check,
        severity=Severity.INFO,
        stage=Stage.CANONICAL,
        status=Status.PASS,
        provider="binance",
        market="spot",
        asset=asset,
        year=year,
        month=month,
        details=details,
        run_id=run,
        image_version="test",
    )


def summary(
    run, day, theta, year, month, content_hash=None, rows=10, family=None, **extra
):
    details = {
        "theta": theta,
        "rows": rows,
        "content_hash": content_hash or f"h-{theta}-{year}-{month}",
    }
    if family is not None:
        details["family"] = family
    return finding(run, day, "summaries_summary", year, month, details, **extra)


def run_of(run, day, change=None):
    """Un `summaries_summary` por (θ, mes), con `change[(θ, año, mes)] = {campo: valor}`."""
    out = []
    for theta in THETAS:
        for year, month in MONTHS:
            override = (change or {}).get((theta, year, month), {})
            out.append(summary(run, day, theta, year, month, **override))
    return out


@pytest.fixture
def lake(tmp_path):
    root = tmp_path / "dq/l3"

    def add(*findings: Finding) -> None:
        emit_findings(list(findings), root)

    add.root = str(root)
    return add


@pytest.fixture
def compare():
    return load()


def run(compare, lake, capsys, *argv: str) -> tuple[int, str]:
    code = compare.main(["--dq-root", lake.root, *argv])
    return code, capsys.readouterr().out


BOTH = ("--base-dates", "2026-10-10", "--new-dates", "2026-10-22")


def test_corridas_iguales_dan_cero(compare, lake, capsys):
    lake(*run_of("base", "2026-10-10"), *run_of("new", "2026-10-22"))
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 0
    assert out.strip().splitlines()[-1] == (
        "compare runs: pares=4 coinciden=4 difieren=0 solo_base=0 solo_nueva=0 "
        "inconsistentes=0 veredicto=ok"
    )
    assert "DIFERENCIA" not in out


def test_un_hash_distinto_es_una_diferencia(compare, lake, capsys):
    change = {("0.00250000", 2021, 2): {"content_hash": "otro"}}
    lake(*run_of("base", "2026-10-10"), *run_of("new", "2026-10-22", change))
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 1
    assert (
        "DIFERENCIA θ=0.00250000 2021-02: content_hash base=h-0.00250000-2021-2 nueva=otro"
        in out
    )
    assert "coinciden=3 difieren=1" in out
    assert "veredicto=diferencias" in out


def test_las_filas_distintas_son_una_diferencia_aunque_el_hash_coincida(
    compare, lake, capsys
):
    change = {("0.00100000", 2021, 1): {"rows": 11}}
    lake(*run_of("base", "2026-10-10"), *run_of("new", "2026-10-22", change))
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 1
    assert "DIFERENCIA θ=0.00100000 2021-01: rows base=10 nueva=11" in out
    assert "content_hash base" not in out


def test_un_par_que_falta_en_un_lado_no_coincide(compare, lake, capsys):
    new = [f for f in run_of("new", "2026-10-22") if f.details["theta"] != THETAS[1]]
    new.append(summary("new", "2026-10-22", "0.00500000", 2021, 1))
    lake(*run_of("base", "2026-10-10"), *new)
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 1
    assert "SOLO EN BASE θ=0.00250000 2021-01" in out
    assert "SOLO EN BASE θ=0.00250000 2021-02" in out
    assert "SOLO EN NUEVA θ=0.00500000 2021-01" in out
    assert "solo_base=2 solo_nueva=1" in out


def test_corridas_de_un_lado_que_discrepan_son_inconsistentes(compare, lake, capsys):
    bad = {("0.00100000", 2021, 1): {"content_hash": "viejo"}}
    lake(
        *run_of("a", "2026-10-10", bad),
        *run_of("b", "2026-10-10"),
        *run_of("new", "2026-10-22"),
    )
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 1
    assert (
        "INCONSISTENTE base θ=0.00100000 2021-01: sus corridas dejaron 2 pares" in out
    )
    assert "inconsistentes=1" in out
    assert "coinciden=3" in out


def test_run_id_acota_el_lado_dentro_de_un_mismo_dia(compare, lake, capsys):
    bad = {("0.00100000", 2021, 1): {"content_hash": "viejo"}}
    lake(
        *run_of("a", "2026-10-10", bad),
        *run_of("b", "2026-10-10"),
        *run_of("new", "2026-10-22"),
    )
    code, out = run(
        compare,
        lake,
        capsys,
        "--base-dates",
        "2026-10-10",
        "--base-run",
        "b",
        "--new-run",
        "new",
    )
    assert code == 0, out
    assert "pares=4 coinciden=4" in out


def test_una_corrida_repetida_gana_la_ultima_por_detected_at(compare, lake, capsys):
    """Un mes que se reconstruye con otro contenido cuenta con lo último que quedó."""
    old = run_of("old", "2026-10-09", {("0.00100000", 2021, 1): {"content_hash": "x"}})
    lake(*old, *run_of("base", "2026-10-10"), *run_of("new", "2026-10-22"))
    code, out = run(
        compare,
        lake,
        capsys,
        "--base-dates",
        "2026-10-10:",
        "--new-dates",
        "2026-10-22",
    )
    assert code == 0, out


def test_solo_se_compara_la_familia_pedida(compare, lake, capsys):
    """Los `summaries_summary` de otra familia no se mezclan con los de `summaries`."""
    other = [
        summary("base", "2026-10-10", t, y, m, "otra", family="otra")
        for t in THETAS
        for y, m in MONTHS
    ]
    lake(*run_of("base", "2026-10-10"), *other, *run_of("new", "2026-10-22"))
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 0, out
    assert "INCONSISTENTE" not in out
    # Una corrida cuyos `summaries_summary` traen `family` es de esa familia.
    tagged = [
        summary("new2", "2026-10-23", t, y, m, family="summaries")
        for t in THETAS
        for y, m in MONTHS
    ]
    lake(*tagged)
    code, out = run(
        compare, lake, capsys, "--base-dates", "2026-10-10", "--new-dates", "2026-10-23"
    )
    assert code == 0, out
    code, out = run(compare, lake, capsys, *BOTH, "--family", "otra")
    assert code == 1
    assert "FALLO sin_hallazgos" in out


def test_el_rango_de_fechas_es_cerrado_y_puede_quedar_abierto(compare, lake, capsys):
    bad = {("0.00100000", 2021, 1): {"content_hash": "fuera"}}
    lake(
        *run_of("a", "2026-10-09"),
        *run_of("b", "2026-10-10"),
        *run_of("c", "2026-10-11", bad),
        *run_of("new", "2026-10-22"),
    )
    closed = ("--base-dates", "2026-10-09:2026-10-10", "--new-dates", "2026-10-22:")
    code, out = run(compare, lake, capsys, *closed)
    assert code == 0, out
    assert "base: 4 pares; nueva: 4 pares" in out
    # Abierto por el final, el rango alcanza el 11 y sus corridas discrepan.
    code, out = run(
        compare,
        lake,
        capsys,
        "--base-dates",
        "2026-10-09:2026-10-11",
        "--new-dates",
        "2026-10-22",
    )
    assert code == 1
    assert "INCONSISTENTE base θ=0.00100000 2021-01" in out


def test_meses_y_serie_acotan_la_comparacion(compare, lake, capsys):
    change = {("0.00100000", 2021, 2): {"content_hash": "otro"}}
    other = summary("base", "2026-10-10", "0.00100000", 2021, 1, "z", asset="ETHUSDT")
    lake(*run_of("base", "2026-10-10"), other, *run_of("new", "2026-10-22", change))
    code, out = run(
        compare, lake, capsys, *BOTH, "--from", "2021-01", "--to", "2021-01"
    )
    assert code == 0, out
    assert "pares=2 coinciden=2" in out


def test_un_lado_sin_hallazgos_falla(compare, lake, capsys):
    lake(*run_of("base", "2026-10-10"))
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 1
    assert "FALLO sin_hallazgos: el lado nueva" in out
    assert "veredicto=diferencias" in out


def test_un_lago_vacio_falla_sin_romper(compare, lake, capsys):
    code, out = run(compare, lake, capsys, *BOTH)
    assert code == 1
    assert "FALLO sin_hallazgos" in out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--base-dates", "2026-10-10"],
        ["--base-dates", "2026-10-1x", "--new-dates", "2026-10-22"],
        [*BOTH, "--from", "2021-13"],
    ],
)
def test_el_uso_incorrecto_sale_con_2(compare, lake, argv):
    with pytest.raises(SystemExit) as exit_:
        compare.main(["--dq-root", lake.root, *argv])
    assert exit_.value.code == 2


def test_sin_dq_root_ni_proyecto_es_un_error_de_uso(compare, monkeypatch):
    monkeypatch.delenv("OPS_RESULTS_URI", raising=False)
    with pytest.raises(SystemExit) as exit_:
        compare.main([*BOTH])
    assert exit_.value.code == 2


def timing(run, day, year, month, **details):
    base = {
        "wall_s": 100.0,
        "cores": 4,
        "cpu_utilization": 0.95,
        "rss_peak_mib": 2000.0,
        "engine_busy": 0.9,
    }
    return finding(run, day, "unit_timing", year, month, {**base, **details})


def test_timing_resume_percentiles_y_metas(compare, lake, capsys):
    months = [(2021, m) for m in range(1, 6)]
    utilization = [0.97, 0.95, 0.93, 0.91, 0.70]
    new = [
        timing("new", "2026-10-22", y, m, cpu_utilization=u, wall_s=100.0 * (i + 1))
        for i, ((y, m), u) in enumerate(zip(months, utilization, strict=True))
    ]
    new[2].details["rss_peak_mib"] = 3100.5
    lake(
        *run_of("base", "2026-10-10"),
        *run_of("new", "2026-10-22"),
        *new,
        # un unit_timing anterior del mismo mes no cuenta: gana el último
        timing("old", "2026-10-21", 2021, 5, cpu_utilization=0.99),
    )
    code, out = run(
        compare, lake, capsys, *BOTH, "--timing", "--baseline-vcpu-s", "10000"
    )
    assert code == 0
    assert "medición: 5 meses con unit_timing" in out
    assert "cpu_utilization: p10=0.784 p50=0.930 mínimo=0.700 (2021-05)" in out
    assert "meses bajo 0.80: 1: 2021-05 (0.700)" in out
    assert "metas: p50 ≥ 0.90: sí; ningún mes bajo 0.80: no" in out
    # Σ wall_s = 1500 s; × 4 vCPU = 6000
    assert "wall_s: suma=1500 p50=300.0 máximo=500.0" in out
    assert "vCPU-s (Σ wall_s × cores): 6000 contra 10000 (60.0 %)" in out
    assert "rss_peak_mib: máximo=3100.5 (2021-03)" in out
    assert "engine_busy: p50=0.900" in out


def test_timing_no_cambia_el_codigo_de_salida(compare, lake, capsys):
    lake(
        *run_of("base", "2026-10-10"),
        *run_of("new", "2026-10-22"),
        timing("new", "2026-10-22", 2021, 1, cpu_utilization=0.1),
    )
    code, out = run(compare, lake, capsys, *BOTH, "--timing")
    assert code == 0
    assert "metas: p50 ≥ 0.90: no; ningún mes bajo 0.80: no" in out


def test_timing_sin_hallazgos_lo_dice(compare, lake, capsys):
    lake(*run_of("base", "2026-10-10"), *run_of("new", "2026-10-22"))
    code, out = run(compare, lake, capsys, *BOTH, "--timing")
    assert code == 0
    assert "medición: 0 meses con unit_timing" in out


def test_los_unit_timing_no_se_cuentan_como_pares(compare, lake, capsys):
    """Si `details` de `unit_timing` trajera `theta`, no debe mezclarse con los resúmenes."""
    stray = timing("new", "2026-10-22", 2021, 1, theta="0.00100000", content_hash="zz")
    lake(*run_of("base", "2026-10-10"), *run_of("new", "2026-10-22"), stray)
    code, out = run(compare, lake, capsys, *BOTH, "--timing")
    assert code == 0
    assert "pares=4 coinciden=4" in out
