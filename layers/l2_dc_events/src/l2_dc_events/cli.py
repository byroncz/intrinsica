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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from l2_dc_events.carry import CarryOverError
from l2_dc_events.landing import LandingError
from l2_dc_events.memory import tune_allocators
from l2_dc_events.pipeline import (
    Result,
    RunContext,
    Unit,
    carry_over_complete,
    process_unit,
)

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


def resolve_range(from_: str, to: str | None) -> list[tuple[int, int]]:
    """Los meses (año, mes) de [from_, to] en orden; solo `from_` si falta `to`.

    Pura: no lee entorno ni reloj.
    """
    (y0, m0), (y1, m1) = _parse(from_), _parse(to or from_)
    first, last = y0 * 12 + m0 - 1, y1 * 12 + m1 - 1
    if last < first:
        raise UsageError(f"--to {to} es anterior a --from {from_}")
    return [(n // 12, n % 12 + 1) for n in range(first, last + 1)]


def _today() -> date:
    return datetime.now(UTC).date()


def previous_month(today: date) -> str:
    """El mes anterior a `today` como YYYY-MM (el que cierra `monthly`)."""
    last_month = today.replace(day=1) - timedelta(days=1)
    return f"{last_month.year:04d}-{last_month.month:02d}"


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


def _require_single_task(env: Mapping[str, str]) -> None:
    """L2 corre en una sola tarea: la cadena de carry-over no admite task array."""
    raw = env.get("CLOUD_RUN_TASK_INDEX", "0")
    try:
        index = int(raw)
    except ValueError:
        raise UsageError(f"CLOUD_RUN_TASK_INDEX={raw!r} no es un entero") from None
    if index != 0:
        raise UsageError(
            f"CLOUD_RUN_TASK_INDEX={index}: L2 corre en una sola tarea, nunca en "
            "un task array (los meses se encadenan por carry-over)"
        )


def _pending_range(
    months: list[tuple[int, int]], ctx: RunContext, asset: str, force: bool
) -> list[tuple[int, int]]:
    """Desde dónde procesa el backfill (RF-L2-09).

    Arranca en el primer mes con algún θ sin `carry_over.parquet` válido y desde
    ahí sigue con todos, tengan salida o no: un mes escrito en frío por la sonda
    se reescribe encadenado. Con `force`, arranca en el primero.
    """
    if force:
        return months
    for i, (year, month) in enumerate(months):
        unit = Unit(year, month, asset=asset)
        if not carry_over_complete(unit, ctx):
            return months[i:]
        logger.info("unidad %s: carry-over completo, se salta", unit)
    return []


def _log_probe(
    unit: Unit, mode: str, started: float, result: Result | None = None
) -> None:
    """Línea de cierre de la sonda §14.1: RSS pico (KiB en Linux → MiB) y pared.

    La pared se redondea hacia arriba a 0.1 s para que nunca salga 0.0. Con
    `result`, agrega los ticks y los ticks/s por core (los de la unidad entera:
    lectura, 50 θ y escritura, sobre el límite efectivo de CPU), y θ·ticks/s por
    core, la unidad con que `dc_core` reporta su benchmark.

    `cores` es el límite efectivo (cuota del cgroup o, sin ella, los cores
    visibles) y `cores_visible` lo que ve la máquina. Las fases son las de
    `timing.py`: `read_s`, `decode_s`, `detect_s`, `carry_s` y `wait_s` son
    pared del hilo principal; `write_s` es la suma de los hilos de escritura
    y `other_s` lo que queda de `wall_s` sin explicar (incluye emitir los
    hallazgos, que el hallazgo `unit_timing` no cuenta).
    """
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    wall_s = math.ceil((time.monotonic() - started) * 10) / 10
    line = f"sonda: unit={unit} mode={mode} rss_peak_mib={rss_mib} wall_s={wall_s:.1f}"
    if result is not None:
        t = result.timing
        per_core = result.n_ticks / wall_s / t.cores
        other_s = max(
            0.0, wall_s - (t.carry_s + t.read_s + t.decode_s + t.detect_s + t.wait_s)
        )
        line += (
            f" ticks={result.n_ticks} cores={t.cores:g} cores_visible={t.cores_visible}"
            f" cores_source={t.cores_source} ticks_s_core={per_core:.0f}"
            f" theta_ticks_s_core={per_core * len(result.events_per_theta):.0f}"
            f" read_s={t.read_s:.1f} decode_s={t.decode_s:.1f}"
            f" detect_s={t.detect_s:.1f} write_s={t.write_s:.1f}"
            f" carry_s={t.carry_s:.1f} wait_s={t.wait_s:.1f} other_s={other_s:.1f}"
            f" row_groups={t.row_groups} bytes_in={t.bytes_in}"
        )
        if t.cpu_throttled_s is not None:
            line += f" cpu_throttled_s={t.cpu_throttled_s:.1f}"
    logger.info(line)


def _run_unit(unit: Unit, ctx: RunContext) -> bool:
    """Procesa un mes y deja su sonda. False si la entrada o el carry-over faltan."""
    started = time.monotonic()
    result = None
    try:
        result = process_unit(unit, ctx)
    except (LandingError, CarryOverError) as exc:
        print(f"l2_dc_events: {exc}", file=sys.stderr)
        return False
    finally:
        _log_probe(unit, ctx.mode, started, result)
    logger.info(
        "eventos cerrados por θ: min=%d max=%d",
        min(result.events_per_theta),
        max(result.events_per_theta),
    )
    return True


def main(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> int:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(prog="l2_dc_events")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--from",
        dest="from_",
        metavar="DESDE",
        help="primer mes (YYYY-MM); obligatorio en backfill. En monthly, por "
        "defecto el mes anterior al actual (UTC)",
    )
    parser.add_argument("--to", metavar="HASTA", help="último mes; solo backfill")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--series-start",
        metavar="YYYY-MM",
        help="primer mes de la serie, el único que arranca sin carry-over "
        "(ADR-L2-08); si falta, se toma de L2_SERIES_START, y sin ninguno de los "
        "dos es un error: se declara, no se infiere",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="backfill: arranca en --from aunque los meses ya tengan carry-over",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse ya imprimió el motivo
        return int(exc.code or 0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        _require_single_task(env)
        if args.from_ is None:
            if args.mode != "monthly":
                raise UsageError("--from es obligatorio (solo monthly lo omite)")
            # Se resuelve aquí para que resolve_range siga pura.
            args.from_ = previous_month(_today())
        if args.mode == "monthly" and args.to is not None:
            raise UsageError("monthly procesa un solo mes: no admite --to")
        months = resolve_range(args.from_, args.to)
        declared = args.series_start or env.get("L2_SERIES_START")
        if not declared:
            raise UsageError(
                "falta --series-start (o L2_SERIES_START): el primer mes se declara"
            )
        series_start = _parse(declared)
        if months[0] < series_start:
            raise UsageError(
                f"la unidad {months[0][0]:04d}-{months[0][1]:02d} es anterior a "
                f"la serie ({declared})"
            )
        ctx = _context(args.mode, series_start, env)
    except UsageError as exc:
        print(f"l2_dc_events: {exc}", file=sys.stderr)
        return EXIT_USAGE

    tune_allocators()
    if args.mode == "backfill":
        months = _pending_range(months, ctx, args.asset, args.force)
        if not months:
            logger.info("backfill: todos los meses del rango tienen carry-over")
    # En orden y en este proceso: cada mes lee el carry-over que el anterior
    # acaba de escribir. Un fallo detiene el rango; los siguientes no se tocan.
    for year, month in months:
        if not _run_unit(Unit(year, month, asset=args.asset), ctx):
            return 1
    return 0
