"""Punto de entrada de la imagen: `python -m l2_dc_events --mode ... --from ...`."""

import argparse
import logging
import math
import os
import re
import resource
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from l2_dc_events.carry import CarryOverError
from l2_dc_events.landing import LandingError
from l2_dc_events.pipeline import Result, RunContext, Unit, process_unit

MODES = ("backfill", "monthly")
EXIT_USAGE = 2
ROOT_VARS = ("L2_LANDING_ROOT", "L2_EVENTS_ROOT", "L2_DQ_ROOT")

logger = logging.getLogger(__name__)

_MONTH = re.compile(r"(\d{4})-(\d{2})")


class UsageError(ValueError):
    """Argumento o entorno inválido: el proceso termina con código 2."""


def _parse(text: str) -> tuple[int, int]:
    match = _MONTH.fullmatch(text)
    if match is None:
        raise UsageError(f"{text!r} no cumple YYYY-MM")
    try:
        day = date(*map(int, match.groups()), 1)
    except ValueError as exc:
        raise UsageError(f"{text!r} no es una fecha válida: {exc}") from exc
    return day.year, day.month


def resolve_unit(from_: str, to: str | None, index: int) -> tuple[int, int]:
    """Devuelve (año, mes) de la unidad `from_ + index` dentro de [from_, to].

    Pura: no lee entorno ni reloj.
    """
    (y0, m0), (y1, m1) = _parse(from_), _parse(to or from_)
    first, last = y0 * 12 + m0 - 1, y1 * 12 + m1 - 1
    if last < first:
        raise UsageError(f"--to {to} es anterior a --from {from_}")
    if not 0 <= index <= last - first:
        raise UsageError(
            f"CLOUD_RUN_TASK_INDEX={index} fuera del rango de {last - first + 1} "
            "unidades"
        )
    year, month = divmod(first + index, 12)
    return year, month + 1


def _image_version(env: Mapping[str, str]) -> str:
    if env.get("IMAGE_VERSION"):
        return env["IMAGE_VERSION"]
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    version = version_file.read_text().strip() if version_file.exists() else "unknown"
    return f"{version}+local"


def _context(
    mode: str, series_start: tuple[int, int], env: Mapping[str, str]
) -> RunContext:
    missing = [name for name in ROOT_VARS if not env.get(name)]
    if missing:
        raise UsageError(f"falta la variable de entorno {', '.join(missing)}")
    return RunContext(
        mode=mode,
        run_id=str(uuid.uuid4()),
        image_version=_image_version(env),
        series_start=series_start,
        landing_root=env["L2_LANDING_ROOT"],
        events_root=env["L2_EVENTS_ROOT"],
        dq_root=env["L2_DQ_ROOT"],
    )


def _index(env: Mapping[str, str]) -> int:
    raw = env.get("CLOUD_RUN_TASK_INDEX", "0")
    try:
        return int(raw)
    except ValueError:
        raise UsageError(f"CLOUD_RUN_TASK_INDEX={raw!r} no es un entero") from None


def _log_probe(
    unit: Unit, mode: str, started: float, result: Result | None = None
) -> None:
    """Línea de cierre de la sonda §14.1: RSS pico (KiB en Linux → MiB) y pared.

    La pared se redondea hacia arriba a 0.1 s para que nunca salga 0.0. Con
    `result`, agrega los ticks y los ticks/s por core (los de la unidad entera:
    lectura, 50 θ y escritura, sobre los núcleos disponibles), y θ·ticks/s por
    core, la unidad con que `dc_core` reporta su benchmark.
    """
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    wall_s = math.ceil((time.monotonic() - started) * 10) / 10
    line = f"sonda: unit={unit} mode={mode} rss_peak_mib={rss_mib} wall_s={wall_s:.1f}"
    if result is not None:
        cores = len(os.sched_getaffinity(0))
        per_core = result.n_ticks / wall_s / cores
        line += (
            f" ticks={result.n_ticks} cores={cores} ticks_s_core={per_core:.0f}"
            f" theta_ticks_s_core={per_core * len(result.events_per_theta):.0f}"
        )
    logger.info(line)


def main(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> int:
    started = time.monotonic()
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(prog="l2_dc_events")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--from", dest="from_", required=True, metavar="DESDE")
    parser.add_argument("--to", metavar="HASTA")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--series-start",
        metavar="YYYY-MM",
        help="primer mes de la serie, el único que arranca sin carry-over "
        "(ADR-L2-08); por defecto --from en backfill, obligatorio en monthly",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse ya imprimió el motivo
        return int(exc.code or 0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        year, month = resolve_unit(args.from_, args.to, _index(env))
        if args.series_start is None and args.mode == "monthly":
            raise UsageError("--mode monthly exige --series-start")
        series_start = _parse(args.series_start or args.from_)
        ctx = _context(args.mode, series_start, env)
    except UsageError as exc:
        print(f"l2_dc_events: {exc}", file=sys.stderr)
        return EXIT_USAGE

    unit = Unit(year, month, asset=args.asset)
    result = None
    try:
        result = process_unit(unit, ctx)
    except (LandingError, CarryOverError) as exc:
        print(f"l2_dc_events: {exc}", file=sys.stderr)
        return 1
    finally:
        _log_probe(unit, args.mode, started, result)
    logger.info(
        "eventos cerrados por θ: min=%d max=%d",
        min(result.events_per_theta),
        max(result.events_per_theta),
    )
    return 0
