"""Punto de entrada de la imagen: `python -m viz_tiles --mode tiles ...` (TRD-viz §8.2)."""

import argparse
import logging
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from dq import configure_logging

from viz_tiles.context import RunContext
from viz_tiles.contract import price_scale
from viz_tiles.lake import EventsIndex, Month, month_of, ordinal
from viz_tiles.memory import tune_allocators
from viz_tiles.pipeline import days_of_month, process_month
from viz_tiles.write import SeriesMismatch, find_index

MODES = ("tiles",)
EXIT_USAGE = 2
ROOT_VARS = ("VIZ_LANDING_ROOT", "VIZ_EVENTS_ROOT", "VIZ_TILES_ROOT", "VIZ_DQ_ROOT")

logger = logging.getLogger(__name__)

_MONTH = re.compile(r"(\d{4})-(\d{2})")
_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


class UsageError(ValueError):
    """Argumento o entorno inválido: el proceso termina con código 2."""


def parse_month(text: str) -> Month:
    match = _MONTH.fullmatch(text)
    if match is None:
        raise UsageError(f"{text!r} no cumple YYYY-MM")
    try:
        day = date(*map(int, match.groups()), 1)
    except ValueError as exc:
        raise UsageError(f"{text!r} no es una fecha válida: {exc}") from exc
    return day.year, day.month


def parse_day(text: str) -> date:
    match = _DAY.fullmatch(text)
    if match is None:
        raise UsageError(f"{text!r} no cumple YYYY-MM-DD")
    try:
        return date(*map(int, match.groups()))
    except ValueError as exc:
        raise UsageError(f"{text!r} no es una fecha válida: {exc}") from exc


def resolve_range(from_: str, to: str | None) -> list[Month]:
    """Los meses de [from_, to] en orden; solo `from_` si falta `to`. Pura."""
    first, last = ordinal(parse_month(from_)), ordinal(parse_month(to or from_))
    if last < first:
        raise UsageError(f"--to {to} es anterior a --from {from_}")
    return [month_of(n) for n in range(first, last + 1)]


def _today() -> date:
    return datetime.now(UTC).date()


def previous_month(today: date) -> Month:
    """El mes anterior a `today`: el que cierra `l2-monthly`."""
    last_month = today.replace(day=1) - timedelta(days=1)
    return last_month.year, last_month.month


def _image_version(env: Mapping[str, str]) -> str:
    if env.get("IMAGE_VERSION"):
        return env["IMAGE_VERSION"]
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    version = version_file.read_text().strip() if version_file.exists() else "unknown"
    return f"{version}+local"


def _context(args: argparse.Namespace, env: Mapping[str, str]) -> RunContext:
    missing = [name for name in ROOT_VARS if not env.get(name)]
    if missing:
        raise UsageError(f"falta la variable de entorno {', '.join(missing)}")
    try:
        price_scale(args.asset)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return RunContext(
        run_id=str(uuid.uuid4()),
        image_version=_image_version(env),
        landing_root=env["VIZ_LANDING_ROOT"],
        events_root=env["VIZ_EVENTS_ROOT"],
        tiles_root=env["VIZ_TILES_ROOT"],
        dq_root=env["VIZ_DQ_ROOT"],
        asset=args.asset,
        force=args.force,
    )


def _has_provisional(index: dict) -> bool:
    return any(t["provisional_from_s"] is not None for t in index["thetas"])


def review_months(ctx: RunContext, month: Month) -> list[tuple[Month, list[date]]]:
    """Los meses anteriores a `month` con cola provisional y sus días (TRD-viz §7.6).

    Desde `month − 1` mira el `index.json` del último día del mes: si algún θ
    trae `provisional_from_s`, retrocede por los días del mes mientras los
    encuentre provisionales y pasa al mes anterior. Se detiene en el primer mes
    cuyo último día es definitivo para todos los θ (o no tiene índice). Sale del
    más antiguo al más reciente. Los días que se rehacen pasan por la idempotencia:
    solo cambia el `input_hash` de los que la cadena de carry-overs alcanzó.
    """
    found: list[tuple[Month, list[date]]] = []
    n = ordinal(month) - 1
    while n >= 0:
        candidate = month_of(n)
        days = []
        for day in reversed(days_of_month(candidate)):
            index = find_index(ctx.tiles_root, ctx.provider, ctx.market, ctx.asset, day)
            if index is None or not _has_provisional(index):
                break
            days.append(day)
        if not days:
            break
        found.append((candidate, sorted(days)))
        n -= 1
    return found[::-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="viz_tiles")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--day",
        metavar="YYYY-MM-DD",
        help="un solo día (no se combina con --from ni --to)",
    )
    parser.add_argument(
        "--from",
        dest="from_",
        metavar="DESDE",
        help="primer mes (YYYY-MM): todos los días de cada mes del rango. Sin "
        "--day ni --from, el mes anterior al actual (UTC) más la revisión de los "
        "meses con cola provisional",
    )
    parser.add_argument(
        "--to", metavar="HASTA", help="último mes del rango; por defecto, --from"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignora el input_hash y regenera lo seleccionado",
    )
    parser.add_argument("--asset", default="BTCUSDT")
    return parser


def _plan(
    args: argparse.Namespace, ctx: RunContext
) -> list[tuple[Month, list[date] | None]]:
    """Las unidades a procesar: `(mes, días)`, con `None` para todos los días de L1."""
    if args.day is not None:
        if args.from_ is not None or args.to is not None:
            raise UsageError("--day no se combina con --from ni --to")
        day = parse_day(args.day)
        return [((day.year, day.month), [day])]
    if args.to is not None and args.from_ is None:
        raise UsageError("--to exige --from")
    if args.from_ is not None:
        return [(m, None) for m in resolve_range(args.from_, args.to)]
    month = previous_month(_today())
    return [*review_months(ctx, month), (month, None)]


def main(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> int:
    env = os.environ if env is None else env
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse ya imprimió el motivo
        return int(exc.code or 0)

    configure_logging(env)
    try:
        ctx = _context(args, env)
        plan = _plan(args, ctx)
    except UsageError as exc:
        print(f"viz_tiles: {exc}", file=sys.stderr)
        return EXIT_USAGE

    tune_allocators()
    lake = EventsIndex.list(ctx.events_root, ctx.provider, ctx.market, ctx.asset)
    ok = True
    for month, days in plan:
        try:
            ok &= process_month(ctx, month, days, lake)
        except SeriesMismatch as exc:
            print(f"viz_tiles: {exc}", file=sys.stderr)
            return 1
    return 0 if ok else 1
