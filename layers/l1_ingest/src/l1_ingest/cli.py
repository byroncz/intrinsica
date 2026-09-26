"""Punto de entrada de la imagen: `python -m l1_ingest --mode ... --from ...`."""

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
from datetime import date, timedelta
from pathlib import Path

from l1_ingest.pipeline import RunContext, Unit, process_unit
from l1_ingest.seam import run_seam_check

MODES = ("backfill", "daily", "monthly-close", "seam-check")
NOT_IMPLEMENTED = ("monthly-close",)
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3
ROOT_VARS = ("L1_LANDING_ROOT", "L1_DQ_ROOT", "L1_MANIFEST_ROOT")

logger = logging.getLogger(__name__)

_MONTH = re.compile(r"(\d{4})-(\d{2})")
_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


class UsageError(ValueError):
    """Argumento o entorno inválido: el proceso termina con código 2."""


def _parse(mode: str, text: str) -> tuple[int, int, int | None]:
    daily = mode == "daily"
    match = (_DAY if daily else _MONTH).fullmatch(text)
    if match is None:
        raise UsageError(
            f"{text!r} no cumple {'YYYY-MM-DD' if daily else 'YYYY-MM'} "
            f"para --mode {mode}"
        )
    try:
        day = date(*map(int, match.groups()), *([1] if not daily else []))
    except ValueError as exc:
        raise UsageError(f"{text!r} no es una fecha válida: {exc}") from exc
    return day.year, day.month, day.day if daily else None


def _ordinal(year: int, month: int, day: int | None) -> int:
    """Posición de la unidad: días desde 0001-01-01 o meses desde el año 0."""
    if day is None:
        return year * 12 + month - 1
    return date(year, month, day).toordinal()


def resolve_unit(
    mode: str, from_: str, to: str | None, index: int
) -> tuple[int, int, int | None]:
    """Devuelve (año, mes, día) de la unidad `from_ + index` dentro de [from_, to].

    `day` es None en los modos mensuales. Pura: no lee entorno ni reloj.
    """
    start = _parse(mode, from_)
    end = _parse(mode, to or from_)
    first, last = _ordinal(*start), _ordinal(*end)
    if last < first:
        raise UsageError(f"--to {to} es anterior a --from {from_}")
    if not 0 <= index <= last - first:
        raise UsageError(
            f"CLOUD_RUN_TASK_INDEX={index} fuera del rango de {last - first + 1} "
            "unidades"
        )
    if mode == "daily":
        day = date.fromordinal(first) + timedelta(days=index)
        return day.year, day.month, day.day
    year, month = divmod(first + index, 12)
    return year, month + 1, None


def _image_version(env: Mapping[str, str]) -> str:
    if env.get("IMAGE_VERSION"):
        return env["IMAGE_VERSION"]
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    version = version_file.read_text().strip() if version_file.exists() else "unknown"
    return f"{version}+local"


def _context(mode: str, env: Mapping[str, str], force: bool) -> RunContext:
    missing = [name for name in ROOT_VARS if not env.get(name)]
    if missing:
        raise UsageError(f"falta la variable de entorno {', '.join(missing)}")
    extra = {}
    if env.get("L1_SOURCE_BASE_URL"):
        extra["source_base_url"] = env["L1_SOURCE_BASE_URL"]
    return RunContext(
        mode=mode,
        run_id=str(uuid.uuid4()),
        image_version=_image_version(env),
        landing_root=env["L1_LANDING_ROOT"],
        dq_root=env["L1_DQ_ROOT"],
        manifest_root=env["L1_MANIFEST_ROOT"],
        force=force,
        **extra,
    )


def _index(env: Mapping[str, str]) -> int:
    raw = env.get("CLOUD_RUN_TASK_INDEX", "0")
    try:
        return int(raw)
    except ValueError:
        raise UsageError(f"CLOUD_RUN_TASK_INDEX={raw!r} no es un entero") from None


def _log_probe(unit: Unit, mode: str, started: float) -> None:
    """Línea de cierre de la sonda §14.1: RSS pico (KiB en Linux → MiB) y pared.

    La pared se redondea hacia arriba a 0.1 s para que nunca salga 0.0.
    """
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    wall_s = math.ceil((time.monotonic() - started) * 10) / 10
    logger.info(
        "sonda: unit=%s mode=%s rss_peak_mib=%d wall_s=%.1f",
        unit,
        mode,
        rss_mib,
        wall_s,
    )


def _seam_check(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    """Una sola tarea recorre todos los bordes del rango; ignora el índice."""
    try:
        start = _parse(args.mode, args.from_)[:2]
        end = _parse(args.mode, args.to or args.from_)[:2]
        if end < start:
            raise UsageError(f"--to {args.to} es anterior a --from {args.from_}")
        ctx = _context(args.mode, env, args.force)
    except UsageError as exc:
        print(f"l1_ingest: {exc}", file=sys.stderr)
        return EXIT_USAGE
    run_seam_check(start, end, args.asset, ctx)
    return 0


def main(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> int:
    started = time.monotonic()
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(prog="l1_ingest")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--from", dest="from_", required=True, metavar="DESDE")
    parser.add_argument("--to", metavar="HASTA")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--force",
        action="store_true",
        help="reprocesa aunque el .CHECKSUM coincida con el manifiesto",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse ya imprimió el motivo
        return int(exc.code or 0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.mode == "seam-check":
        return _seam_check(args, env)
    try:
        year, month, day = resolve_unit(args.mode, args.from_, args.to, _index(env))
        if args.mode in NOT_IMPLEMENTED:
            print(f"--mode {args.mode}: no implementado hasta E3", file=sys.stderr)
            return EXIT_NOT_IMPLEMENTED
        ctx = _context(args.mode, env, args.force)
    except UsageError as exc:
        print(f"l1_ingest: {exc}", file=sys.stderr)
        return EXIT_USAGE

    unit = Unit(year, month, day, asset=args.asset)
    try:
        process_unit(unit, ctx)
    finally:
        _log_probe(unit, args.mode, started)
    return 0
