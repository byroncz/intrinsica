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


def test_l3_backfill_un_mes_conserva_to():
    # Como l2-backfill: sin --to, la CLI de L3 recorre hasta el último mes de L1.
    assert (
        args("l3-backfill", "2023-03", "2023-03", "true", "2017-08")
        == "--mode,backfill,--from=2023-03,--to=2023-03,--force,--series-start=2017-08"
    )


def test_l3_backfill_sin_to_queda_abierto():
    assert args("l3-backfill", "2023-03") == "--mode,backfill,--from=2023-03"


def test_l3_backfill_sin_from_deja_decidir_a_la_frontera():
    assert args("l3-backfill") == "--mode,backfill"
    assert args("l3-backfill", "", "2024-01") == "--mode,backfill,--to=2024-01"


def test_l3_monthly_omite_to_igual_a_from_y_admite_series_start():
    assert (
        args("l3-monthly", "2023-03", "2023-03", "", "2017-08")
        == "--mode,monthly,--from=2023-03,--series-start=2017-08"
    )


def test_l3_monthly_sin_from_solo_pasa_el_modo():
    # Vacío, la CLI toma el mes anterior.
    assert args("l3-monthly") == "--mode,monthly"


def test_l3_monthly_rechaza_force():
    out = subprocess.run(
        [SCRIPT, "l3-monthly", "2023-03", "", "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert "l3-monthly no admite force" in out.stderr


# ITSC-394: la entrada family (summaries o path) es de l3-backfill y l3-monthly.
def test_l3_backfill_family_path_va_al_final():
    assert (
        args("l3-backfill", "", "", "", "", "", "", "path")
        == "--mode,backfill,--family=path"
    )


def test_l3_backfill_family_con_rango_y_series_start():
    assert (
        args("l3-backfill", "2023-03", "2023-05", "true", "2023-03", "", "", "path")
        == "--mode,backfill,--from=2023-03,--to=2023-05,--force,"
        "--series-start=2023-03,--family=path"
    )


def test_l3_monthly_family_path():
    assert (
        args("l3-monthly", "2023-03", "2023-03", "", "", "", "", "path")
        == "--mode,monthly,--from=2023-03,--family=path"
    )


@pytest.mark.parametrize("job", ["l3-backfill", "l3-monthly"])
def test_l3_family_summaries_se_pasa_explicita(job):
    modo = job.split("-", 1)[1]
    assert args(job, "", "", "", "", "", "", "summaries") == (
        f"--mode,{modo},--family=summaries"
    )


@pytest.mark.parametrize("job", ["l3-backfill", "l3-monthly"])
def test_l3_sin_family_no_agrega_la_opcion(job):
    modo = job.split("-", 1)[1]
    assert args(job) == f"--mode,{modo}"


@pytest.mark.parametrize("job", ["l3-backfill", "l3-monthly"])
@pytest.mark.parametrize("family", ["fit", "Path", "path,fit"])
def test_l3_rechaza_una_familia_que_no_existe(job, family):
    with pytest.raises(subprocess.CalledProcessError) as error:
        args(job, "", "", "", "", "", "", family)
    assert error.value.returncode == 2


@pytest.mark.parametrize(
    "job",
    [
        "l1-backfill",
        "l1-daily",
        "l1-monthly-close",
        "l1-seam-check",
        "l2-backfill",
        "l2-monthly",
        "viz-render",
        "viz-tiles",
        "ops-script",
    ],
)
def test_los_demas_jobs_rechazan_family_path(job):
    with pytest.raises(subprocess.CalledProcessError) as error:
        args(job, "", "", "", "", "", "", "path")
    assert error.value.returncode == 2


@pytest.mark.parametrize("job", ["l1-backfill", "l2-backfill", "viz-tiles"])
def test_los_demas_jobs_ignoran_el_valor_por_defecto_summaries(job):
    # El input del workflow trae summaries siempre: no es pedir una familia.
    assert args(job, "2023-03", "", "", "", "", "", "summaries") == args(job, "2023-03")
