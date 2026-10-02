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
