"""Punto de entrada de la imagen: `python -m l3_dc_descriptors --mode ... --from ...`.

Un mes a la vez, y en cada mes el fan-out de los θ que lo necesitan (TRD-L3 §8.2 a §8.4).
En backfill cada mes corre en un proceso hijo (`L3_UNIT_PROCESS`, ITSC-372): el padre solo
recorre el rango y emite los hallazgos. Esta CLI solo decide **qué** meses y θ procesa; cada mes lo hace
`pipeline.process_unit`, una función sin estado entre meses. Nada aquí guarda estado fuera
de las variables de `main` y de las funciones que reciben lo que necesitan: ni siquiera los
hallazgos, que viajan en el `Result` de la unidad hasta que se emiten.
"""

import argparse
import logging
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from dc_frames.lake import Month, month_ordinal
from dq import configure_logging

from l3_dc_descriptors import findings
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.frontier import (
    Listing,
    frontiers,
    l2_missing,
    label,
    list_l2,
    list_summaries,
    theta_text,
)
from l3_dc_descriptors.landing import last_consolidated
from l3_dc_descriptors.memory import tune_allocators
from l3_dc_descriptors.pipeline import process_unit
from l3_dc_descriptors.schema import FAMILY
from l3_dc_descriptors.thetas import ThetasError, load_thetas
from l3_dc_descriptors.unit_process import run_unit

MODES = ("backfill", "monthly")
FAMILIES = (FAMILY,)
EXIT_USAGE = 2
ROOT_VARS = (
    "L3_LANDING_ROOT",
    "L3_EVENTS_ROOT",
    "L3_SUMMARIES_ROOT",
    "L3_DQ_ROOT",
)

logger = logging.getLogger(__name__)

_MONTH = re.compile(r"(\d{4})-(\d{2})")


class UsageError(ValueError):
    """Argumento o entorno inválido: el proceso termina con código 2."""


def _parse(text: str) -> Month:
    match = _MONTH.fullmatch(text)
    if match is None:
        raise UsageError(f"{text!r} no cumple YYYY-MM")
    try:
        day = date(*map(int, match.groups()), 1)
    except ValueError as exc:
        raise UsageError(f"{text!r} no es una fecha válida: {exc}") from exc
    return day.year, day.month


def resolve_range(from_: str, to: str | None) -> list[Month]:
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


def _count(env: Mapping[str, str], name: str, *, minimum: int) -> int | None:
    """Un entero de entorno (`L3_THREADS`, `L3_READERS`, ...), o `None` si no está o está vacío."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise UsageError(f"{name}={raw!r} no es un entero") from None
    if value < minimum:
        raise UsageError(f"{name}={value} debe ser al menos {minimum}")
    return value


def _flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    """Un interruptor de entorno (`L3_UNIT_PROCESS`): `0` o `1`, o `default` si falta o está vacío."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    if raw.strip() not in ("0", "1"):
        raise UsageError(f"{name}={raw!r} debe ser 0 o 1")
    return raw.strip() == "1"


def _context(mode: str, series_start: Month, env: Mapping[str, str]) -> RunContext:
    missing = [name for name in ROOT_VARS if not env.get(name)]
    if missing:
        raise UsageError(f"falta la variable de entorno {', '.join(missing)}")
    return RunContext(
        mode=mode,
        run_id=str(uuid.uuid4()),
        image_version=_image_version(env),
        series_start=series_start,
        landing_root=env["L3_LANDING_ROOT"],
        events_root=env["L3_EVENTS_ROOT"],
        summaries_root=env["L3_SUMMARIES_ROOT"],
        dq_root=env["L3_DQ_ROOT"],
        profile=env.get("L3_PROFILE") == "1",
        threads=_count(env, "L3_THREADS", minimum=1),
        readers=_count(env, "L3_READERS", minimum=1),
        inflight=_count(env, "L3_INFLIGHT", minimum=1),
        writers=_count(env, "L3_WRITERS", minimum=1),
        engine_threads=_count(env, "L3_ENGINE_THREADS", minimum=1),
        inflight_bytes=_count(env, "L3_INFLIGHT_BYTES", minimum=1),
        lot_events=_count(env, "L3_LOT_EVENTS", minimum=0),
        unit_process=_flag(env, "L3_UNIT_PROCESS", default=True),
    )


def _require_single_task(env: Mapping[str, str]) -> None:
    """L3 corre en una sola tarea por ahora: la paralelización por mes la decide ITSC-335."""
    raw = env.get("CLOUD_RUN_TASK_INDEX", "0")
    try:
        index = int(raw)
    except ValueError:
        raise UsageError(f"CLOUD_RUN_TASK_INDEX={raw!r} no es un entero") from None
    if index != 0:
        raise UsageError(
            f"CLOUD_RUN_TASK_INDEX={index}: L3 corre en una sola tarea, nunca en un "
            "task array (la partición por mes no está decidida, ITSC-335)"
        )


def _catalog(
    args: argparse.Namespace, env: Mapping[str, str], ctx: RunContext, reference: Unit
) -> list[int]:
    """El catálogo de θ de L2, validado antes de tocar nada (TRD-L2 §7.3).

    No hay copia local: el URI es obligatorio. Si falta, no se puede leer o no cumple el
    contrato, deja el hallazgo `theta_catalog_invalid` y termina con código 2 sin escribir
    salida.
    """
    source = args.thetas_uri or env.get("L3_THETAS_URI")
    if not source:
        findings.emit(
            ctx,
            [
                findings.theta_catalog_invalid(
                    ctx,
                    reference,
                    source=None,
                    problems=["falta --thetas-uri (o L3_THETAS_URI)"],
                )
            ],
        )
        raise UsageError(
            "falta el catálogo de θ: --thetas-uri o L3_THETAS_URI (el de L2, l2/thetas.yaml)"
        )
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
    ctx: RunContext, reference: Unit, catalog: list[int], listed: Listing
) -> None:
    """Informa los θ con particiones en el lago de L3 que el catálogo ya no incluye.

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
                theta=theta_text(theta),
                months=len(listed[theta]),
                last_month=label(max(listed[theta])),
            )
            for theta in removed
        ],
    )


def _run_unit(
    unit: Unit, ctx: RunContext, thetas: list[int], *, isolated: bool = False
) -> bool:
    """Procesa un mes para `thetas`, emite sus hallazgos y dice si se escribió.

    Con `isolated` la unidad corre en un proceso hijo (`unit_process`) y su memoria muere con
    ella. Los hallazgos se emiten al terminar la unidad, aunque haya fallado: es la única
    evidencia de por qué.
    """
    result = (run_unit if isolated else process_unit)(unit, ctx, thetas)
    findings.emit(ctx, list(result.findings))
    if not result.ok:
        print(f"l3_dc_descriptors: {result.error}", file=sys.stderr)
        return False
    # Las medidas de la unidad salen del hallazgo `unit_timing`, que viaja igual desde el
    # proceso hijo (el `Timing` no se serializa).
    timing = next(f for f in result.findings if f.check_type == "unit_timing").details
    logger.info(
        "unidad %s: θ=%d eventos=%d ticks=%d wall_s=%.1f rss_peak_mib=%.0f "
        "row_groups_month=%d row_groups_previous=%d spawn_s=%s",
        unit,
        len(thetas),
        timing["events"],
        timing["ticks"],
        timing["wall_s"],
        timing["rss_peak_mib"],
        timing["row_groups_month"],
        timing["row_groups_previous"],
        timing["spawn_s"],
    )
    return True


def _before(reached: int | None, n: int) -> bool:
    """¿El θ, que llegó hasta `reached`, necesita el mes `n`?"""
    return reached is None or reached < n


def _backfill(
    months: list[Month],
    ctx: RunContext,
    asset: Unit,
    thetas: list[int],
    frontier: dict[int, int | None],
    l2: Listing,
    force: bool,
) -> int:
    """Recorre `months` en orden; cada mes lleva solo los θ cuya frontera es anterior.

    Un mes se procesa una sola vez para todos los θ que lo necesitan, y uno que ningún θ
    necesita se salta sin leer L1. Con `force`, todos los θ de `thetas` reprocesan todos
    los meses. Un θ al que L2 no le ha llegado al mes deja `l2_input_missing`, sale del
    rango y la corrida termina con código 1 (la frontera de L3 nunca supera la de L2); los
    demás siguen. Un fallo de la unidad detiene el rango; los meses siguientes no se tocan.
    """
    stopped: set[int] = set()
    processed = 0
    for month in months:
        n = month_ordinal(month)
        unit = asset.with_month(month)
        pending = [
            t
            for t in thetas
            if t not in stopped and (force or _before(frontier.get(t), n))
        ]
        if not pending:
            logger.info("unidad %s: ningún θ la necesita, se salta", unit)
            continue
        ready = []
        for theta in pending:
            absent = l2_missing(l2, theta, n)
            if not absent:
                ready.append(theta)
                continue
            stopped.add(theta)
            logger.error(
                "unidad %s: θ=%d sin %s de L2, su rango se detiene",
                unit,
                theta,
                " ni ".join(absent),
            )
            findings.emit(
                ctx,
                [
                    findings.l2_input_missing(
                        ctx, unit, theta=theta_text(theta), file=name
                    )
                    for name in absent
                ],
            )
        if not ready:
            continue
        if not _run_unit(unit, ctx, ready, isolated=ctx.unit_process):
            return 1
        frontier.update(dict.fromkeys(ready, n))
        processed += 1
    if stopped:
        return 1
    if not processed:
        logger.info("backfill: los %d θ están al día en el rango", len(thetas))
    return 0


def _monthly(
    ctx: RunContext,
    unit: Unit,
    thetas: list[int],
    frontier: dict[int, int | None],
    l2: Listing,
) -> int:
    """Procesa el mes solo para los θ cuya frontera es el mes previo y cuyo L2 ya lo tiene.

    Un θ que ya llegó al mes se salta (re-ejecutarlo daría los mismos archivos). Uno
    rezagado no se procesa y deja `theta_behind_frontier` con su frontera y el `reason`:
    `l2_missing` si L2 no tiene el mes de ese θ (es lo que bloquea, también si además hay
    un hueco en L3) o `l3_frontier` si su serie de L3 no llega al mes previo. La unidad no
    falla por ellos, salvo que ninguno esté listo ni tenga ya el mes: con rezagados y sin
    nada que avanzar termina con código 1 sin leer L1 (fail-closed), para que un mes perdido
    no pase como éxito.
    """
    n = month_ordinal(unit.at)
    ready, behind, done = [], {}, []
    for theta in thetas:
        reached = frontier[theta]
        if reached is not None and reached >= n:
            done.append(theta)
        elif l2_missing(l2, theta, n):
            behind[theta] = "l2_missing"
        # Sin frontera solo puede arrancar el primer mes de la serie.
        elif reached == n - 1 or (reached is None and ctx.series_start == unit.at):
            ready.append(theta)
        else:
            behind[theta] = "l3_frontier"
    for theta, reason in behind.items():
        logger.warning(
            "unidad %s: θ=%d rezagado (%s, frontera %s), no se procesa",
            unit,
            theta,
            reason,
            label(frontier[theta]),
        )
    findings.emit(
        ctx,
        [
            findings.theta_behind_frontier(
                ctx,
                unit,
                theta=theta_text(theta),
                frontier=label(frontier[theta]),
                reason=reason,
            )
            for theta, reason in behind.items()
        ],
    )
    if not ready:
        # Con algún θ ya al día el mes se procesó: los rezagados no son un mes perdido.
        if behind and not done:
            logger.error(
                "unidad %s: ningún θ está listo y %d rezagados; lanzar backfill",
                unit,
                len(behind),
            )
            return 1
        logger.info("unidad %s: ningún θ la necesita, se salta", unit)
        return 0
    return 0 if _run_unit(unit, ctx, ready) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l3_dc_descriptors")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--from",
        dest="from_",
        metavar="DESDE",
        help="primer mes (YYYY-MM). En backfill es opcional: sin él, la frontera de "
        "cada θ decide desde dónde avanza (el piso es --series-start). En monthly, por "
        "defecto el mes anterior al actual (UTC)",
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
        help="primer mes de la serie de L2, el único sin mes anterior; si falta, se "
        "toma de L3_SERIES_START, y sin ninguno de los dos es un error: se declara, "
        "no se infiere",
    )
    parser.add_argument(
        "--thetas-uri",
        metavar="URI",
        help="catálogo de θ (ruta local o gs://), el mismo objeto l2/thetas.yaml que "
        "usa L2; si falta, L3_THETAS_URI, y sin ninguno de los dos es un error: L3 no "
        "tiene catálogo propio",
    )
    parser.add_argument(
        "--thetas",
        metavar="θ,θ,...",
        help="acota la corrida a estos θ del catálogo, en decimal como en la ruta de la "
        "partición (p. ej. 0.00010000,0.00031313)",
    )
    parser.add_argument(
        "--family",
        default=FAMILY,
        help=f"familia de descriptores; por ahora solo {FAMILY!r} (la predeterminada)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="backfill: reprocesa el rango completo aunque los θ ya tengan la familia "
        "(ignora la frontera); exige --from",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> int:
    env = os.environ if env is None else env
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse ya imprimió el motivo
        return int(exc.code or 0)

    configure_logging(env)
    try:
        _require_single_task(env)
        if args.family not in FAMILIES:
            raise UsageError(
                f"--family {args.family!r}: por ahora solo existe {FAMILY!r}"
            )
        declared = args.series_start or env.get("L3_SERIES_START")
        if not declared:
            raise UsageError(
                "falta --series-start (o L3_SERIES_START): el primer mes se declara"
            )
        series_start = _parse(declared)
        if args.mode == "monthly":
            if args.to is not None:
                raise UsageError("monthly procesa un solo mes: no admite --to")
            if args.force:
                raise UsageError("--force es de backfill: monthly no reprocesa")
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
        asset = Unit(*start, asset=args.asset)
        catalog = _catalog(args, env, ctx, asset)
        thetas = _select(catalog, args.thetas)
    except UsageError as exc:
        print(f"l3_dc_descriptors: {exc}", file=sys.stderr)
        return EXIT_USAGE

    tune_allocators()
    summaries_root, events_root = str(ctx.summaries_root), str(ctx.events_root)
    listed = list_summaries(summaries_root, asset.key)
    _report_drift(ctx, asset, catalog, listed)
    l2 = list_l2(events_root, asset.key)
    forced = args.force
    frontier = (
        {}
        if forced
        else frontiers(
            thetas, listed, series_start, summaries_root, events_root, asset.key
        )
    )
    for theta in thetas:
        logger.info("θ=%d frontera=%s", theta, label(frontier.get(theta)))

    if args.mode == "monthly":
        return _monthly(ctx, asset, thetas, frontier, l2)

    if args.to is None:
        last = last_consolidated(str(ctx.landing_root), asset.key)
        if last is None:
            print(
                f"l3_dc_descriptors: L1 no ha publicado ningún mes de {args.asset} "
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
    return _backfill(months, ctx, asset, thetas, frontier, l2, forced)
