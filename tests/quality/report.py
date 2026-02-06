"""Generador de reportes de calidad.

Genera reportes en formato JSON y consola.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Severity, ValidationLevel
from .validators.base import ValidatorReport


@dataclass
class QualityReport:
    """Reporte completo de calidad de datos."""

    # Metadata
    ticker: str
    theta: float
    year: int
    month: int
    execution_start: datetime
    silver_path: str

    # Resultados
    validator_reports: list[ValidatorReport] = field(default_factory=list)
    execution_end: datetime | None = None

    def add_validator_report(self, report: ValidatorReport) -> None:
        """Agrega un reporte de validador."""
        self.validator_reports.append(report)

    def finalize(self) -> None:
        """Marca el reporte como finalizado."""
        self.execution_end = datetime.now()

    @property
    def execution_time_ms(self) -> float:
        """Tiempo de ejecución en milisegundos."""
        if self.execution_end is None:
            return 0.0
        delta = self.execution_end - self.execution_start
        return delta.total_seconds() * 1000

    @property
    def total_tests(self) -> int:
        return sum(len(vr.results) for vr in self.validator_reports)

    @property
    def total_passed(self) -> int:
        return sum(vr.n_passed for vr in self.validator_reports)

    @property
    def total_failed(self) -> int:
        return sum(vr.n_failed for vr in self.validator_reports)

    @property
    def has_critical_failures(self) -> bool:
        return any(vr.has_critical_failures for vr in self.validator_reports)

    @property
    def total_assertions(self) -> int:
        """Total de tests que son aserciones (excluye INFO)."""
        return self.total_passed + self.total_failed

    @property
    def total_statistics(self) -> int:
        """Total de tests estadísticos (INFO)."""
        return sum(vr.n_statistics for vr in self.validator_reports)

    @property
    def quality_score(self) -> float:
        """Score de calidad (0.0 - 1.0) basado solo en aserciones."""
        if self.total_assertions == 0:
            return 1.0
        return self.total_passed / self.total_assertions

    def get_statistics(self) -> list[dict]:
        """Obtiene todas las observaciones estadísticas (INFO)."""
        stats = []
        for vr in self.validator_reports:
            for r in vr.statistics:
                stats.append(r.to_dict())
        return stats

    def get_failures_by_severity(self, severity: Severity) -> list[dict]:
        """Obtiene fallos filtrados por severidad (solo aserciones)."""
        failures = []
        for vr in self.validator_reports:
            for r in vr.results:
                if r.is_assertion and not r.passed and r.severity == severity:
                    failures.append(r.to_dict())
        return failures

    def should_fail(self, level: ValidationLevel) -> bool:
        """Determina si el test suite debe fallar según el nivel."""
        if level == ValidationLevel.PERMISSIVE:
            return False

        if level == ValidationLevel.STRICT:
            return self.total_failed > 0

        # NORMAL: falla solo en CRITICAL o HIGH
        for vr in self.validator_reports:
            for r in vr.results:
                if not r.passed and r.severity in (Severity.CRITICAL, Severity.HIGH):
                    return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serializa el reporte completo."""
        return {
            "metadata": {
                "ticker": self.ticker,
                "theta": self.theta,
                "year": self.year,
                "month": self.month,
                "execution_time_ms": self.execution_time_ms,
                "silver_path": self.silver_path,
                "generated_at": datetime.now().isoformat(),
            },
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.total_passed,
                "failed": self.total_failed,
                "quality_score": round(self.quality_score, 4),
                "has_critical_failures": self.has_critical_failures,
            },
            "validators": [vr.to_dict() for vr in self.validator_reports],
            "failures": {
                "critical": self.get_failures_by_severity(Severity.CRITICAL),
                "high": self.get_failures_by_severity(Severity.HIGH),
                "medium": self.get_failures_by_severity(Severity.MEDIUM),
                "low": self.get_failures_by_severity(Severity.LOW),
            },
        }

    def save_json(self, path: Path) -> None:
        """Guarda el reporte como JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def get_statistics_summary(self) -> dict[str, Any]:
        """Agrega estadísticas mensuales para modo compacto.

        Returns:
            Diccionario con estadísticas agregadas por tipo
        """
        import re

        stats = self.get_statistics()
        summary: dict[str, Any] = {
            "zero_os": {"count": 0, "total": 0},
            "flash_events": {"count": 0, "total": 0},
            "trend_reversals": {"count": 0, "total": 0},
            "provisional_extremes": {"count": 0, "total": 0},
            "scale_law_deviations": {"outside": 0, "total": 0},
            "dc_durations": [],
            "determinism": {"skipped": 0, "passed": 0, "failed": 0},
        }

        for s in stats:
            name = s["test_name"]
            msg = s["message"]

            if "zero_os_rate" in name:
                # Extraer: "Zero OS rate: 2.7% (1/37)"
                match = re.search(r"\((\d+)/(\d+)\)", msg)
                if match:
                    summary["zero_os"]["count"] += int(match.group(1))
                    summary["zero_os"]["total"] += int(match.group(2))

            elif "flash_event_rate" in name:
                match = re.search(r"\((\d+)/(\d+)\)", msg)
                if match:
                    summary["flash_events"]["count"] += int(match.group(1))
                    summary["flash_events"]["total"] += int(match.group(2))

            elif "trend_continuity" in name:
                summary["trend_reversals"]["count"] += 1
                summary["trend_reversals"]["total"] += 1

            elif "extreme_provisional" in name:
                summary["provisional_extremes"]["count"] += 1
                summary["provisional_extremes"]["total"] += 1

            elif "avg_os_magnitude" in name or "avg_tmv" in name:
                summary["scale_law_deviations"]["total"] += 1
                if "outside" in msg:
                    summary["scale_law_deviations"]["outside"] += 1

            elif "dc_duration" in name:
                # Extraer avg de: "DC duration: min=..., avg=1847.3s, ..."
                match = re.search(r"avg=(\d+\.?\d*)s", msg)
                if match:
                    summary["dc_durations"].append(float(match.group(1)))

            elif "reprocess" in name:
                if "skipped" in msg.lower():
                    summary["determinism"]["skipped"] += 1
                elif "passed" in msg.lower() or "identical" in msg.lower():
                    summary["determinism"]["passed"] += 1
                else:
                    summary["determinism"]["failed"] += 1

        return summary

    def print_console(self, console: Console | None = None, verbose_stats: bool = False) -> None:
        """Imprime el reporte en consola con Rich.

        Args:
            console: Consola Rich (opcional)
            verbose_stats: Si True, muestra estadísticas día por día
        """
        if console is None:
            console = Console()

        # Header
        status = "[green]PASSED[/green]" if not self.has_critical_failures else "[red]FAILED[/red]"
        console.print(
            Panel(
                f"[bold]Silver Data Quality Report[/bold]\n"
                f"Ticker: {self.ticker} | θ={self.theta} | {self.year}-{self.month:02d}\n"
                f"Status: {status} | Score: {self.quality_score:.1%}",
                title="Quality Framework",
                border_style="blue",
            )
        )

        # Summary table (solo aserciones)
        table = Table(title="Validator Summary (Assertions)", box=box.ROUNDED)
        table.add_column("Validator", style="cyan")
        table.add_column("Indica", style="dim")
        table.add_column("Level", justify="center")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Stats", justify="right", style="dim")
        table.add_column("Status", justify="center")

        # Descripciones de cada validador
        validator_descriptions = {
            "structural": "Estructura del esquema",
            "intra_event": "Coherencia intra-evento",
            "temporal": "Orden temporal",
            "price": "Consistencia de precios",
            "collision": "Eventos duplicados",
            "threshold": "Umbrales DC",
            "cross_day": "Continuidad entre días",
            "scale_laws": "Leyes de escala DC",
            "determinism": "Reproducibilidad",
        }

        for vr in self.validator_reports:
            row_status = "[green]✓[/green]" if vr.passed else "[red]✗[/red]"
            table.add_row(
                vr.validator_name,
                validator_descriptions.get(vr.validator_name, "-"),
                str(vr.level),
                str(vr.n_passed),
                str(vr.n_failed),
                str(vr.n_statistics) if vr.n_statistics > 0 else "-",
                row_status,
            )

        console.print(table)

        # Failures detail (solo aserciones que fallaron)
        if self.total_failed > 0:
            console.print("\n[bold red]Errors:[/bold red]")
            for vr in self.validator_reports:
                for r in vr.results:
                    if r.is_assertion and not r.passed:
                        sev_color = {
                            Severity.CRITICAL: "red",
                            Severity.HIGH: "yellow",
                            Severity.MEDIUM: "orange3",
                            Severity.LOW: "dim",
                        }.get(r.severity, "white")
                        console.print(
                            f"  [{sev_color}][{r.severity.value.upper()}][/{sev_color}] "
                            f"{r.test_name}: {r.message}"
                        )

        # Statistics section
        if self.total_statistics > 0:
            if verbose_stats:
                # Modo verboso: lista completa
                console.print("\n[bold blue]Statistics / Observations:[/bold blue]")
                for vr in self.validator_reports:
                    for r in vr.statistics:
                        console.print(f"  [dim][INFO][/dim] {r.test_name}: {r.message}")
            else:
                # Modo compacto: tabla resumen
                self._print_statistics_summary(console)

    def _print_statistics_summary(self, console: Console) -> None:
        """Imprime tabla resumen de estadísticas."""
        summary = self.get_statistics_summary()

        # Calcular valores para mostrar
        def fmt_rate(count: int, total: int) -> str:
            if total == 0:
                return "-"
            rate = count / total * 100
            if rate < 1 and count > 0:
                return f"{count}/{total} (<1%)"
            return f"{count}/{total} ({rate:.0f}%)"

        # Promedio de duraciones DC
        durations = summary["dc_durations"]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Estado determinism
        det = summary["determinism"]
        if det["skipped"] > 0:
            det_status = "skipped"
        elif det["failed"] > 0:
            det_status = f"failed ({det['failed']})"
        else:
            det_status = f"passed ({det['passed']})"

        # Crear tabla
        table = Table(
            title=f"Statistics Summary ({len(durations)} days)",
            box=box.ROUNDED,
        )
        table.add_column("Statistic", style="cyan")
        table.add_column("Indica", style="dim")
        table.add_column("Value", justify="right")

        table.add_row(
            "Zero OS Events",
            "Reversiones inmediatas",
            fmt_rate(summary["zero_os"]["count"], summary["zero_os"]["total"]),
        )
        table.add_row(
            "Flash Events",
            "DC instantáneos",
            fmt_rate(summary["flash_events"]["count"], summary["flash_events"]["total"]),
        )
        table.add_row(
            "Trend Reversals",
            "Cambios día-a-día",
            fmt_rate(
                summary["trend_reversals"]["count"],
                summary["trend_reversals"]["total"],
            ),
        )
        table.add_row(
            "Provisional Extremes",
            "Eventos incompletos",
            f"{summary['provisional_extremes']['count']}/{summary['provisional_extremes']['total']}",
        )
        table.add_row(
            "Scale Law Deviations",
            "Días fuera de rango teórico",
            fmt_rate(
                summary["scale_law_deviations"]["outside"],
                summary["scale_law_deviations"]["total"],
            ),
        )
        table.add_row(
            "Avg DC Duration",
            "Duración media de ciclos",
            f"{avg_duration:,.0f}s" if avg_duration > 0 else "-",
        )
        table.add_row(
            "Determinism Tests",
            "Tests de reproceso",
            det_status,
        )

        console.print()
        console.print(table)
