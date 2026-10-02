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
from decimal import Decimal
from pathlib import Path

from l2_dc_events import findings
from l2_dc_events.carry import CarryOverError
from l2_dc_events.frontier import frontiers, label, list_carry_overs, ordinal
from l2_dc_events.landing import READ_AHEAD, LandingError, last_consolidated
from l2_dc_events.memory import tune_allocators
from l2_dc_events.pipeline import Result, RunContext, Unit, process_unit
from l2_dc_events.thetas import THETAS_CONFIG, ThetasError, load_thetas

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
        read_ahead=_read_ahead(env),
    )


def _read_ahead(env: Mapping[str, str]) -> int:
    """`L2_READ_AHEAD`: row groups que el lector pide por delante (0 = en serie)."""
    raw = env.get("L2_READ_AHEAD")
    if raw is None:
        return READ_AHEAD
    try:
        value = int(raw)
    except ValueError:
        raise UsageError(f"L2_READ_AHEAD={raw!r} no es un entero") from None
    if value < 0:
        raise UsageError(f"L2_READ_AHEAD={value} debe ser 0 o mayor")
    return value


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


def _catalog(
    args: argparse.Namespace, env: Mapping[str, str], ctx: RunContext, reference: Unit
) -> list[int]:
    """El catálogo de θ, validado antes de tocar nada (TRD-L2 §7.3).

    Si no se puede leer o no cumple el contrato, deja el hallazgo
    `theta_catalog_invalid` y termina con código 2 sin escribir salida.
    """
    source = args.thetas_uri or env.get("L2_THETAS_URI") or THETAS_CONFIG
    try:
        return load_thetas(source)
    except ThetasError as exc:
        findings.emit(
            ctx,
            [
                findings.theta_catalog_invalid(
                    ctx, reference, source=exc.source, problems=exc.problems
                )
            ],
        )
        raise UsageError(f"catálogo de θ inválido: {exc}") from exc


def _select(catalog: list[int], raw: str | None) -> list[int]:
    """`--thetas` (decimales como en la ruta, p. ej. `0.00010000`) acota el catálogo."""
    if raw is None:
        return catalog
    chosen = set()
    for text in raw.split(","):
        try:
            scaled = Decimal(text.strip()).scaleb(8)
            valid = scaled.is_finite() and scaled == scaled.to_integral_value()
        except ArithmeticError:
            valid = False
        if not valid:
            raise UsageError(f"--thetas: {text!r} no es un θ con a lo más 8 decimales")
        if int(scaled) not in catalog:
            raise UsageError(f"--thetas: {text.strip()} no está en el catálogo")
        chosen.add(int(scaled))
    return sorted(chosen)


def _report_drift(
    ctx: RunContext,
    reference: Unit,
    catalog: list[int],
    listed: dict[int, set[int]],
) -> None:
    """Informa los θ con particiones en el lago que el catálogo ya no incluye.

    Quitar un θ del catálogo no borra nada: es un hecho a la vista, no un error.
    """
    removed = sorted(set(listed) - set(catalog))
    for theta in removed:
        logger.info(
            "θ=%d tiene %d meses en el lago y ya no está en el catálogo",
            theta,
            len(listed[theta]),
        )
    findings.emit(
        ctx,
        [
            findings.theta_config_drift(
                ctx,
                reference,
                theta=theta,
                months=len(listed[theta]),
                last_month=label(max(listed[theta])),
            )
            for theta in removed
        ],
    )


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
    `timing.py`: `read_s`, `detect_s`, `carry_s` y `wait_s` son pared del hilo
    principal; `decode_s` (CPU del lector), `detect_cpu_s` (CPU del principal
    durante el fan-out) y `write_s` (suma de los hilos de escritura) no entran
    en esa suma, y `other_s` es lo que queda de `wall_s` sin explicar (incluye
    emitir los hallazgos, que el hallazgo `unit_timing` no cuenta).
    """
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    wall_s = math.ceil((time.monotonic() - started) * 10) / 10
    line = f"sonda: unit={unit} mode={mode} rss_peak_mib={rss_mib} wall_s={wall_s:.1f}"
    if result is not None:
        t = result.timing
        per_core = result.n_ticks / wall_s / t.cores
        other_s = max(0.0, wall_s - (t.carry_s + t.read_s + t.detect_s + t.wait_s))
        line += (
            f" ticks={result.n_ticks} cores={t.cores:g} cores_visible={t.cores_visible}"
            f" cores_source={t.cores_source} ticks_s_core={per_core:.0f}"
            f" theta_ticks_s_core={per_core * len(result.events_per_theta):.0f}"
            f" read_s={t.read_s:.1f} decode_s={t.decode_s:.1f}"
            f" detect_s={t.detect_s:.1f} detect_cpu_s={t.detect_cpu_s:.1f}"
            f" write_s={t.write_s:.1f}"
            f" carry_s={t.carry_s:.1f} wait_s={t.wait_s:.1f} other_s={other_s:.1f}"
            f" row_groups={t.row_groups} bytes_in={t.bytes_in}"
            f" fanout_threads={t.fanout_threads} write_workers={t.write_workers}"
        )
        if t.cpu_throttled_s is not None:
            line += f" cpu_throttled_s={t.cpu_throttled_s:.1f}"
    logger.info(line)


def _run_unit(unit: Unit, ctx: RunContext, thetas: list[int]) -> bool:
    """Procesa un mes para `thetas` y deja su sonda. False si la entrada o el
    carry-over faltan."""
    started = time.monotonic()
    result = None
    try:
        result = process_unit(unit, ctx, thetas=thetas)
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


def _before(reached: int | None, n: int) -> bool:
    """¿El θ, que llegó hasta `reached`, necesita el mes `n`?"""
    return reached is None or reached < n


def _backfill(
    months: list[tuple[int, int]],
    ctx: RunContext,
    asset: str,
    thetas: list[int],
    frontier: dict[int, int | None],
    force: bool,
) -> int:
    """Recorre `months` en orden; cada mes lleva solo los θ cuya frontera es anterior.

    Un mes se procesa una sola vez para todos los θ que lo necesitan, y uno que
    ningún θ necesita se salta sin leer L1. Con `force`, todos los θ de `thetas`
    reprocesan todos los meses. Al terminar un mes, la frontera de sus θ es ese
    mes. Un fallo detiene el rango; los siguientes no se tocan.
    """
    processed = 0
    for year, month in months:
        n = ordinal((year, month))
        unit = Unit(year, month, asset=asset)
        pending = [t for t in thetas if force or _before(frontier.get(t), n)]
        if not pending:
            logger.info("unidad %s: ningún θ la necesita, se salta", unit)
            continue
        # En orden y en este proceso: cada mes lee el carry-over que el anterior
        # acaba de escribir.
        if not _run_unit(unit, ctx, pending):
            return 1
        frontier.update(dict.fromkeys(pending, n))
        processed += 1
    if not processed:
        logger.info("backfill: los %d θ están al día en el rango", len(thetas))
    return 0


def _monthly(
    month: tuple[int, int],
    ctx: RunContext,
    asset: str,
    thetas: list[int],
    frontier: dict[int, int | None],
) -> int:
    """Procesa el mes solo para los θ cuya frontera es el mes previo.

    Un θ que ya llegó al mes se salta (re-ejecutarlo daría los mismos archivos).
    Uno rezagado (agregado sin backfill, o con un hueco) no se procesa: deja el
    hallazgo `theta_behind_frontier` con su frontera y la unidad no falla, salvo
    que ninguno esté listo: con rezagados y sin nada que avanzar termina con
    código 1 (fail-closed), para que un mes perdido no detenga a L2 en silencio.
    """
    n = ordinal(month)
    unit = Unit(*month, asset=asset)
    ready, behind = [], []
    for theta in thetas:
        reached = frontier[theta]
        if reached is not None and reached >= n:
            continue
        # Sin frontera solo puede arrancar el primer mes de la serie.
        if reached == n - 1 or (reached is None and ctx.series_start == month):
            ready.append(theta)
        else:
            behind.append(theta)
    for theta in behind:
        logger.warning(
            "unidad %s: θ=%d rezagado (frontera %s), no se procesa; lanzar backfill",
            unit,
            theta,
            label(frontier[theta]),
        )
    findings.emit(
        ctx,
        [
            findings.theta_behind_frontier(
                ctx, unit, theta=theta, frontier=label(frontier[theta])
            )
            for theta in behind
        ],
    )
    if not ready:
        if behind:
            logger.error(
                "unidad %s: ningún θ está listo y %d rezagados; lanzar backfill",
                unit,
                len(behind),
            )
            return 1
        logger.info("unidad %s: ningún θ la necesita, se salta", unit)
        return 0
    return 0 if _run_unit(unit, ctx, ready) else 1


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
        help="primer mes (YYYY-MM). En backfill es opcional: sin él, la frontera "
        "de cada θ decide desde dónde avanza (el piso es --series-start). En "
        "monthly, por defecto el mes anterior al actual (UTC)",
    )
    parser.add_argument(
        "--to",
        metavar="HASTA",
        help="último mes; solo backfill. Por defecto, el último mes con "
        "consolidated.parquet en L1 (nunca provisionales)",
    )
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--series-start",
        metavar="YYYY-MM",
        help="primer mes de la serie, el único que arranca sin carry-over "
        "(ADR-L2-08); si falta, se toma de L2_SERIES_START, y sin ninguno de los "
        "dos es un error: se declara, no se infiere",
    )
    parser.add_argument(
        "--thetas-uri",
        metavar="URI",
        help="catálogo de θ (ruta local o gs://); si falta, L2_THETAS_URI, y sin "
        "ninguno de los dos, la semilla del paquete (config/thetas.yaml)",
    )
    parser.add_argument(
        "--thetas",
        metavar="θ,θ,...",
        help="acota la corrida a estos θ del catálogo, en decimal como en la ruta "
        "de la partición (p. ej. 0.00010000,0.00031313)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="backfill: reprocesa el rango completo aunque los θ ya tengan "
        "carry-over (ignora la frontera); exige --from",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse ya imprimió el motivo
        return int(exc.code or 0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        _require_single_task(env)
        declared = args.series_start or env.get("L2_SERIES_START")
        if not declared:
            raise UsageError(
                "falta --series-start (o L2_SERIES_START): el primer mes se declara"
            )
        series_start = _parse(declared)
        if args.mode == "monthly":
            if args.to is not None:
                raise UsageError("monthly procesa un solo mes: no admite --to")
            # Se resuelve aquí para que resolve_range siga pura.
            first = args.from_ or previous_month(_today())
        else:
            if args.force and args.from_ is None:
                raise UsageError(
                    "--force exige --from: sin él reprocesaría toda la serie"
                )
            first = args.from_ or declared
        start = resolve_range(first, args.to)[0]
        if start < series_start:
            raise UsageError(
                f"la unidad {start[0]:04d}-{start[1]:02d} es anterior a "
                f"la serie ({declared})"
            )
        ctx = _context(args.mode, series_start, env)
        reference = Unit(*start, asset=args.asset)
        catalog = _catalog(args, env, ctx, reference)
        thetas = _select(catalog, args.thetas)
    except UsageError as exc:
        print(f"l2_dc_events: {exc}", file=sys.stderr)
        return EXIT_USAGE

    tune_allocators()
    listed = list_carry_overs(str(ctx.events_root), reference)
    _report_drift(ctx, reference, catalog, listed)
    forced = args.force and args.mode == "backfill"
    frontier = (
        {}
        if forced
        else frontiers(thetas, listed, series_start, str(ctx.events_root), reference)
    )
    for theta in thetas:
        logger.info("θ=%d frontera=%s", theta, label(frontier.get(theta)))

    if args.mode == "monthly":
        return _monthly(start, ctx, args.asset, thetas, frontier)

    if args.to is None:
        last = last_consolidated(
            str(ctx.landing_root), reference.provider, reference.market, args.asset
        )
        if last is None:
            print(
                f"l2_dc_events: L1 no ha publicado ningún mes de {args.asset} "
                f"en {ctx.landing_root}",
                file=sys.stderr,
            )
            return 1
        args.to = f"{last[0]:04d}-{last[1]:02d}"
        if last < start:
            logger.info(
                "backfill: el último mes cerrado de L1 es %s, anterior a %s; "
                "no hay nada que procesar",
                args.to,
                first,
            )
            return 0
    months = resolve_range(first, args.to)
    return _backfill(months, ctx, args.asset, thetas, frontier, forced)
