"""Orquestador del framework de calidad.

Ejecuta todos los validadores y genera el reporte.
"""

from datetime import datetime
from pathlib import Path

from .config import CONFIG, ValidationLevel
from .conftest import load_silver_month
from .report import QualityReport
from .validators.base import BaseValidator
from .validators.collision import CollisionValidator
from .validators.cross_day import CrossDayValidator
from .validators.determinism import DeterminismValidator
from .validators.intra_event import IntraEventValidator
from .validators.price import PriceValidator
from .validators.scale_laws import ScaleLawsValidator
from .validators.structural import StructuralValidator
from .validators.temporal import TemporalValidator
from .validators.threshold import ThresholdValidator

# Registro de validadores en orden de ejecución
VALIDATORS: list[type[BaseValidator]] = [
    StructuralValidator,  # Nivel 1
    IntraEventValidator,  # Nivel 2
    TemporalValidator,  # Nivel 3
    PriceValidator,  # Nivel 4
    CollisionValidator,  # Nivel 6 (crítico, antes de threshold)
    ThresholdValidator,  # Nivel 5
    CrossDayValidator,  # Nivel 7
    ScaleLawsValidator,  # Nivel 8
    DeterminismValidator,  # Nivel 9
]


def run_quality_check(
    year: int,
    month: int,
    ticker: str = "BTCUSDT",
    theta: float = 0.005,
    level: ValidationLevel = ValidationLevel.NORMAL,
    output_path: Path | None = None,
    verbose: bool = True,
    verbose_stats: bool = False,
) -> QualityReport:
    """Ejecuta el framework completo de calidad.

    Args:
        year: Año a validar
        month: Mes a validar
        ticker: Símbolo del instrumento
        theta: Umbral DC
        level: Nivel de exigencia
        output_path: Path para guardar reporte JSON (opcional)
        verbose: Si True, imprime en consola
        verbose_stats: Si True, muestra estadísticas detalladas por día

    Returns:
        QualityReport con resultados

    Raises:
        ValueError: Si no hay datos para el mes
        AssertionError: Si hay fallos críticos y level != PERMISSIVE
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console() if verbose else None

    # Iniciar reporte
    report = QualityReport(
        ticker=ticker,
        theta=theta,
        year=year,
        month=month,
        execution_start=datetime.now(),
        silver_path=str(CONFIG.silver_base_path / ticker / f"theta={theta}"),
    )

    # Cargar datos
    if verbose:
        console.print(f"[bold]Loading Silver data for {ticker} {year}-{month:02d}...[/bold]")

    data = load_silver_month(ticker, theta, year, month)

    if not data:
        raise ValueError(f"No Silver data found for {ticker} {year}-{month:02d} theta={theta}")

    if verbose:
        console.print(f"  Found [cyan]{len(data)}[/cyan] days with data")

    # Ejecutar validadores
    if verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            for validator_class in VALIDATORS:
                validator = validator_class()
                task = progress.add_task(f"Running {validator.name}...", total=None)
                validator_report = validator.validate_month(data, theta)
                report.add_validator_report(validator_report)
                progress.remove_task(task)
    else:
        for validator_class in VALIDATORS:
            validator = validator_class()
            validator_report = validator.validate_month(data, theta)
            report.add_validator_report(validator_report)

    # Finalizar
    report.finalize()

    # Guardar JSON si se especificó
    if output_path:
        report.save_json(output_path)
        if verbose:
            console.print(f"Report saved to [cyan]{output_path}[/cyan]")

    # Imprimir en consola
    if verbose:
        report.print_console(console, verbose_stats=verbose_stats)

    # Verificar si debe fallar
    if report.should_fail(level):
        raise AssertionError(
            f"Quality check failed with {report.total_failed} failures (level={level.value})"
        )

    return report
