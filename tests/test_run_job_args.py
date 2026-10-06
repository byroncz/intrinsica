import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / ".github/scripts/run-job-args.sh"


def args(*entrada: str) -> str:
    out = subprocess.run([SCRIPT, *entrada], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_l2_backfill_un_mes_conserva_to():
    # ITSC-292: sin --to, l2-backfill recorre hasta el último mes cerrado de L1.
    assert (
        args("l2-backfill", "2023-03", "2023-03", "true", "2023-03")
        == "--mode,backfill,--from=2023-03,--to=2023-03,--force,--series-start=2023-03"
    )


def test_l2_backfill_sin_to_queda_abierto():
    assert args("l2-backfill", "2023-03") == "--mode,backfill,--from=2023-03"


def test_l2_backfill_rango_pasa_to():
    assert (
        args("l2-backfill", "2023-03", "2023-05")
        == "--mode,backfill,--from=2023-03,--to=2023-05"
    )


@pytest.mark.parametrize(
    ("job", "unidad"),
    [
        ("l1-backfill", "2023-03"),
        ("l1-monthly-close", "2023-03"),
        ("l1-daily", "2023-03-01"),
        ("l1-seam-check", "2023-03"),
        ("l2-monthly", "2023-03"),
    ],
)
def test_otros_jobs_omiten_to_igual_a_from(job, unidad):
    modo = job.split("-", 1)[1]
    assert args(job, unidad, unidad) == f"--mode,{modo},--from={unidad}"


def test_l1_rango_pasa_to():
    assert (
        args("l1-backfill", "2023-03", "2023-05")
        == "--mode,backfill,--from=2023-03,--to=2023-05"
    )


def test_sin_from_ni_to():
    assert args("l2-backfill") == "--mode,backfill"


SCRIPT_URI = "gs://proj-ops/scripts/seam.py"


def ops(script: str = SCRIPT_URI, texto: str = "") -> str:
    return args("ops-script", "", "", "", "", script, texto)


def test_ops_script_sin_args_solo_pasa_el_script():
    assert ops() == f"^;;^--script={SCRIPT_URI}"


def test_ops_script_args_van_como_una_sola_pieza():
    # Comas y valores repetidos rompen la lista de gcloud: por eso no se parten.
    assert ops(texto="--a 5 --b 5, 'dos palabras'") == (
        f"^;;^--script={SCRIPT_URI};;--script-args=--a 5 --b 5, 'dos palabras'"
    )


def test_ops_script_sin_mode():
    assert "--mode" not in ops()


def test_ops_script_rechaza_el_separador_en_el_texto():
    out = subprocess.run(
        [SCRIPT, "ops-script", "", "", "", "", SCRIPT_URI, "a;;b"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert ";;" in out.stderr


def test_los_demas_jobs_ignoran_script_y_args():
    assert (
        args("l2-monthly", "2023-03", "2023-03", "", "", SCRIPT_URI, "x")
        == "--mode,monthly,--from=2023-03"
    )


def test_viz_tiles_sin_rango_solo_pasa_el_modo():
    # Vacío, la CLI toma el mes anterior.
    assert args("viz-tiles") == "--mode,tiles"


def test_viz_tiles_un_mes_omite_to_igual_a_from():
    assert args("viz-tiles", "2023-03", "2023-03") == "--mode,tiles,--from=2023-03"


def test_viz_tiles_rango_pasa_to():
    assert (
        args("viz-tiles", "2023-03", "2023-05")
        == "--mode,tiles,--from=2023-03,--to=2023-05"
    )


def test_viz_tiles_force_va_al_final():
    assert (
        args("viz-tiles", "2023-03", "2023-05", "true")
        == "--mode,tiles,--from=2023-03,--to=2023-05,--force"
    )


@pytest.mark.parametrize(
    "extra",
    [
        ("2023-03",),  # series_start
        ("", SCRIPT_URI),  # script
        ("", "", "--a 1"),  # args
    ],
)
def test_viz_tiles_rechaza_series_start_script_y_args(extra):
    out = subprocess.run(
        [SCRIPT, "viz-tiles", "2023-03", "", "", *extra],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert "viz-tiles no usa" in out.stderr


def test_viz_render_sin_rango_solo_pasa_el_modo():
    # Vacío, la CLI toma el mes anterior.
    assert args("viz-render") == "--mode,render"


def test_viz_render_un_mes_omite_to_igual_a_from():
    assert args("viz-render", "2023-03", "2023-03") == "--mode,render,--from=2023-03"


def test_viz_render_rango_pasa_to():
    assert (
        args("viz-render", "2023-03", "2023-05")
        == "--mode,render,--from=2023-03,--to=2023-05"
    )


def test_viz_render_force_va_al_final():
    assert (
        args("viz-render", "2023-03", "2023-05", "true")
        == "--mode,render,--from=2023-03,--to=2023-05,--force"
    )


@pytest.mark.parametrize(
    "extra",
    [
        ("2023-03",),  # series_start
        ("", SCRIPT_URI),  # script
        ("", "", "--a 1"),  # args
    ],
)
def test_viz_render_rechaza_series_start_script_y_args(extra):
    out = subprocess.run(
        [SCRIPT, "viz-render", "2023-03", "", "", *extra],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert "viz-render no usa" in out.stderr
