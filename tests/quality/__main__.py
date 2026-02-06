"""CLI entry point para el framework de calidad.

Uso:
    python -m tests.quality --year 2025 --month 11
    python -m tests.quality --year 2025 --month 11 --level strict
    python -m tests.quality --year 2025 --month 11 --output report.json
"""

import argparse
import sys
from pathlib import Path

from .config import ValidationLevel
from .runner import run_quality_check


def main():
    parser = argparse.ArgumentParser(
        description="Silver Data Quality Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tests.quality --year 2025 --month 11
    python -m tests.quality --year 2025 --month 11 --level strict
    python -m tests.quality --year 2025 --month 11 --ticker BTCUSDT --theta 0.005
    python -m tests.quality --year 2025 --month 11 --output quality_report.json
        """,
    )

    parser.add_argument(
        "--year",
        "-y",
        type=int,
        required=True,
        help="Año a validar",
    )
    parser.add_argument(
        "--month",
        "-m",
        type=int,
        required=True,
        help="Mes a validar (1-12)",
    )
    parser.add_argument(
        "--ticker",
        "-t",
        type=str,
        default="BTCUSDT",
        help="Símbolo del instrumento (default: BTCUSDT)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.005,
        help="Umbral DC (default: 0.005)",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=str,
        choices=["strict", "normal", "permissive"],
        default="normal",
        help="Nivel de exigencia (default: normal)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path para guardar reporte JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Modo detallado para estadísticas (muestra cada día)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Modo silencioso (sin output a consola)",
    )

    args = parser.parse_args()

    # Validar mes
    if not 1 <= args.month <= 12:
        print(f"Error: month must be 1-12, got {args.month}", file=sys.stderr)
        sys.exit(1)

    # Mapear nivel
    level_map = {
        "strict": ValidationLevel.STRICT,
        "normal": ValidationLevel.NORMAL,
        "permissive": ValidationLevel.PERMISSIVE,
    }
    level = level_map[args.level]

    # Output path
    output_path = Path(args.output) if args.output else None

    try:
        report = run_quality_check(
            year=args.year,
            month=args.month,
            ticker=args.ticker,
            theta=args.theta,
            level=level,
            output_path=output_path,
            verbose=not args.quiet,
            verbose_stats=args.verbose,
        )

        # Exit code basado en resultado
        if report.has_critical_failures:
            sys.exit(1)
        sys.exit(0)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except AssertionError as e:
        print(f"Quality check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
