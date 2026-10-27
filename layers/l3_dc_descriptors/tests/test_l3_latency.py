"""El costo fijo de una unidad no es espera en serie (ITSC-369, TRD-L3 §8.1 y §9.3).

En GCS cada archivo que la unidad abre antes del primer row group (`events.parquet`, el
carry-over del mes y del anterior, la huella) y cada archivo que publica al final cuesta un
viaje. En el lago de pruebas no hay red: aquí el sistema de archivos duerme `DELAY_S` por
apertura (`latency.py`) y la unidad de 50 θ debe tardar lo que el ancho del pool de metadatos
permite, no la suma de los viajes. Si la espera vuelve a ser en serie estas pruebas fallan.
"""

import json
from dataclasses import replace

import latency
import pytest
from l3_dc_descriptors import summarize
from l3_dc_descriptors.pipeline import process_unit
from l3_lake import M1, M2, build_lake, fixture_ticks, month_cuts
from test_l3_timing import context, unit_of

# Retardo por apertura (el de GCS ronda 50 a 200 ms). Se puede subir sin tocar las cotas: ellas
# están en viajes, no en segundos.
DELAY_S = 0.05
THETAS = 50
TICKS = fixture_ticks()
# Los viajes de la compuerta y la apertura por θ: `events.parquet`, carry-over del mes, carry-over
# del mes anterior y huella.
TRIPS_PER_THETA = 4
# Cota de setup_s + gate_s con el pool por defecto: los 200 viajes en 16 hilos son 0,63 s y el
# resto es lo que cuesta arrancar. En serie son 10 s y con el pool de E/S de 3 hilos (1 vCPU),
# 3,7 s.
FIXED_COST_S = 1.5


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    thetas = tuple(range(100_000, 100_000 + THETAS * 40_000, 40_000))
    return build_lake(
        tmp_path_factory.mktemp("latency"),
        TICKS,
        month_cuts(TICKS, 1500, 3000),
        thetas=thetas,
    )


def run(lake, tmp_path, monkeypatch, thetas=None, **knobs):
    """Una unidad de 2017-09 con `DELAY_S` por apertura y por publicación; devuelve sus
    tiempos y lo que contaron las dos esperas.
    """
    opened = latency.install(monkeypatch, DELAY_S)
    published = latency.slow_commit(monkeypatch, DELAY_S)
    ctx = replace(context(lake, tmp_path), **knobs)
    result = process_unit(unit_of(M2), ctx, thetas or lake.thetas)
    assert result.ok, result.error
    return result.timing.details(), opened, published


def test_the_fixed_cost_overlaps_with_the_default_metadata_pool(
    lake, tmp_path, monkeypatch
):
    # `threads=3` es el pool de E/S de 1 vCPU: el costo fijo no depende de él.
    details, opened, published = run(lake, tmp_path, monkeypatch, threads=3)
    trips = DELAY_S * len(lake.thetas)
    # La espera existió: la suma de los viajes de cada paso es la de hacerlos en serie.
    assert details["gate_open_work_s"] >= trips
    assert details["setup_carry_work_s"] >= 3 * trips
    assert details["publish_work_s"] >= trips
    # Y se solapó: la pared del costo fijo cabe en la cota (en serie son ≥ 5 s con los viajes
    # de `events.parquet` y carry-over que cuenta la card, 10 s con la huella y el mes anterior).
    assert details["gate_s"] + details["setup_s"] <= FIXED_COST_S
    assert details["gate_s"] + details["setup_s"] < trips * TRIPS_PER_THETA / 4
    # Publicar: a lo más un tercio de lo que costaría en serie, que es la suma de sus tareas.
    assert details["publish_s"] <= details["publish_work_s"] / 3
    assert details["publish_serial"] <= len(lake.thetas) // 5
    assert opened.peak > 1 and published.peak > 1
    assert details["meta_workers"] == summarize.DEFAULT_META_WORKERS


def test_the_metadata_pool_bounds_the_waits_in_flight(lake, tmp_path, monkeypatch):
    details, opened, published = run(
        lake, tmp_path, monkeypatch, thetas=lake.thetas[:12], threads=4, meta_workers=3
    )
    assert details["meta_workers"] == 3
    assert 2 <= opened.peak <= 3
    assert 2 <= published.peak <= 3


def test_without_the_pool_every_wait_is_in_series(lake, tmp_path, monkeypatch):
    # Con `threads=1` nada se solapa: la pared de cada paso es su suma y las 12 publicaciones
    # corrieron solas. Es la contraparte de las pruebas de arriba.
    thetas = lake.thetas[:12]
    details, opened, published = run(
        lake, tmp_path, monkeypatch, thetas=thetas, threads=1
    )
    assert details["meta_workers"] == 0
    assert opened.peak == published.peak == 1
    assert details["publish_serial"] == len(thetas)
    assert details["publish_s"] >= DELAY_S * len(thetas)
    assert details["gate_s"] + details["setup_s"] >= (
        DELAY_S * len(thetas) * TRIPS_PER_THETA
    )


def test_the_hashes_do_not_depend_on_the_latency_or_the_pool(
    lake, tmp_path, monkeypatch
):
    thetas = lake.thetas[:6]

    def hashes(**knobs) -> list[str]:
        ctx = replace(context(lake, tmp_path), **knobs)
        result = process_unit(unit_of(M2), ctx, thetas)
        assert result.ok, result.error
        return [
            json.dumps(f.details["content_hash"])
            for f in result.findings
            if f.check_type == "summaries_summary"
        ]

    quiet = hashes(threads=1)
    assert hashes(threads=4, meta_workers=3) == quiet
    latency.install(monkeypatch, 0.005)
    assert hashes(threads=4) == quiet


def test_the_first_month_of_the_series_opens_one_carry_over_less(
    lake, tmp_path, monkeypatch
):
    # El primer mes de la serie abre un carry-over menos por θ.
    opened = latency.install(monkeypatch, DELAY_S / 5)
    ctx = replace(context(lake, tmp_path), threads=4)
    result = process_unit(unit_of(M1), ctx, lake.thetas[:5])
    assert result.ok, result.error
    assert result.timing.details()["setup_carry_work_s"] >= 2 * 5 * DELAY_S / 5
    assert opened.opens >= 5 * 3
