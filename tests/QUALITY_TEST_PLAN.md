# Plan de Implementación: Silver Data Quality Framework

**Versión:** 1.0.0
**Fecha de Creación:** 2026-02-02
**Última Actualización:** 2026-02-02
**Estado Global:** IMPLEMENTADO (98.6% score)

---

## INSTRUCCIONES PARA LLM QUE CONTINÚE ESTE TRABAJO

### Contexto Obligatorio

Este documento define la implementación de un framework de validación de calidad para datos Silver del motor Intrinseca. **ANTES de hacer cualquier cosa:**

1. Lee completamente este documento
2. Verifica el estado de cada tarea en la sección "Tracking de Progreso"
3. Continúa SOLO la siguiente tarea marcada como `[ ]` (pendiente)
4. Al completar una tarea, actualiza su estado a `[x]` y agrega fecha
5. Si encuentras un bloqueo, documéntalo en "Notas de Implementación"

### Reglas Inquebrantables

1. **NO usar datos dummy.** Todos los tests cargan datos Silver reales.
2. **NO inventar paths.** Usar SIEMPRE la configuración de `conftest.py`.
3. **NO simplificar tests.** Cada test debe ser exhaustivo según su especificación.
4. **NO agregar tests no especificados** sin documentarlos primero aquí.
5. **Actualizar este documento** después de cada sesión de trabajo.

### Paths del Proyecto

```
PROYECTO_ROOT = /Users/byroncampo/My Drive/repositorio/eafit/intrinsica
SILVER_PATH = /Users/byroncampo/My Drive/datalake/financial/02_silver
TESTS_PATH = {PROYECTO_ROOT}/tests/quality/
```

---

## 1. ESTRUCTURA DE ARCHIVOS A CREAR

### 1.1 Árbol de Directorios

```
tests/
├── QUALITY_TEST_PLAN.md          # Este documento (YA EXISTE)
├── quality/
│   ├── __init__.py               # [x] COMPLETADO
│   ├── __main__.py               # [x] COMPLETADO - CLI entry point
│   ├── runner.py                 # [x] COMPLETADO - Orquestador
│   ├── report.py                 # [x] COMPLETADO - Generador de reportes
│   ├── config.py                 # [x] COMPLETADO - Configuración
│   ├── conftest.py               # [x] COMPLETADO - Fixtures pytest
│   └── validators/
│       ├── __init__.py           # [x] COMPLETADO
│       ├── base.py               # [x] COMPLETADO - Clase base validador
│       ├── structural.py         # [x] COMPLETADO - Nivel 1
│       ├── intra_event.py        # [x] COMPLETADO - Nivel 2
│       ├── temporal.py           # [x] COMPLETADO - Nivel 3
│       ├── price.py              # [x] COMPLETADO - Nivel 4
│       ├── threshold.py          # [x] COMPLETADO - Nivel 5
│       ├── collision.py          # [x] COMPLETADO - Nivel 6
│       ├── cross_day.py          # [x] COMPLETADO - Nivel 7
│       ├── scale_laws.py         # [x] COMPLETADO - Nivel 8
│       └── determinism.py        # [x] COMPLETADO - Nivel 9
```

---

## 2. TRACKING DE PROGRESO

### Leyenda de Estados

- `[ ]` = Pendiente
- `[~]` = En desarrollo
- `[x]` = Completado
- `[!]` = Bloqueado (ver notas)

### Fase 0: Infraestructura Base

| ID   | Tarea                                        | Estado | Fecha      | Notas         |
| ---- | -------------------------------------------- | ------ | ---------- | ------------- |
| F0.1 | Crear directorio `tests/quality/`            | [x]    | 2026-02-02 | -             |
| F0.2 | Crear `tests/quality/__init__.py`            | [x]    | 2026-02-02 | -             |
| F0.3 | Crear `tests/quality/config.py`              | [x]    | 2026-02-02 | Ver spec §3.1 |
| F0.4 | Crear `tests/quality/conftest.py`            | [x]    | 2026-02-02 | Ver spec §3.2 |
| F0.5 | Crear `tests/quality/validators/__init__.py` | [x]    | 2026-02-02 | -             |
| F0.6 | Crear `tests/quality/validators/base.py`     | [x]    | 2026-02-02 | Ver spec §3.3 |
| F0.7 | Crear `tests/quality/report.py`              | [x]    | 2026-02-02 | Ver spec §3.4 |
| F0.8 | Crear `tests/quality/runner.py`              | [x]    | 2026-02-02 | Ver spec §3.5 |
| F0.9 | Crear `tests/quality/__main__.py`            | [x]    | 2026-02-02 | Ver spec §3.6 |

### Fase 1: Validadores Críticos (Niveles 1-4, 6)

| ID   | Tarea                             | Estado | Fecha      | Notas                         |
| ---- | --------------------------------- | ------ | ---------- | ----------------------------- |
| F1.1 | Crear `validators/structural.py`  | [x]    | 2026-02-02 | 4 tests, 120 passed           |
| F1.2 | Crear `validators/intra_event.py` | [x]    | 2026-02-02 | 6 tests, 180 passed           |
| F1.3 | Crear `validators/temporal.py`    | [x]    | 2026-02-02 | 7 tests, 10 fallos temporales |
| F1.4 | Crear `validators/price.py`       | [x]    | 2026-02-02 | 6 tests, 180 passed           |
| F1.5 | Crear `validators/collision.py`   | [x]    | 2026-02-02 | 4 tests, 120 passed           |

### Fase 2: Validadores de Robustez (Niveles 5, 7)

| ID   | Tarea                           | Estado | Fecha      | Notas              |
| ---- | ------------------------------- | ------ | ---------- | ------------------ |
| F2.1 | Crear `validators/threshold.py` | [x]    | 2026-02-02 | 3 tests, 90 passed |
| F2.2 | Crear `validators/cross_day.py` | [x]    | 2026-02-02 | 2 tests, 59 passed |

### Fase 3: Validadores Estadísticos (Niveles 8, 9)

| ID   | Tarea                             | Estado | Fecha      | Notas                    |
| ---- | --------------------------------- | ------ | ---------- | ------------------------ |
| F3.1 | Crear `validators/scale_laws.py`  | [x]    | 2026-02-02 | 4 tests, 6 INFO warnings |
| F3.2 | Crear `validators/determinism.py` | [x]    | 2026-02-02 | 1 test, 30 passed        |

### Fase 4: Integración y Documentación

| ID   | Tarea                          | Estado | Fecha      | Notas                   |
| ---- | ------------------------------ | ------ | ---------- | ----------------------- |
| F4.1 | Test de integración completo   | [x]    | 2026-02-02 | 1093 tests, 98.6% score |
| F4.2 | Documentación de uso en README | [x]    | 2026-02-02 | Agregado al README.md   |
| F4.3 | Ejemplo de ejecución validado  | [x]    | 2026-02-02 | CLI funcionando         |

---

## 3. ESPECIFICACIONES DE INFRAESTRUCTURA

### 3.1 config.py

**Archivo:** `tests/quality/config.py`

```python
"""Configuración del framework de calidad de datos Silver.

NO MODIFICAR los paths sin actualizar QUALITY_TEST_PLAN.md.
"""

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Literal


class Severity(Enum):
    """Severidad de las validaciones."""
    CRITICAL = "critical"  # Falla el test suite completo
    HIGH = "high"          # Falla el test pero continúa
    MEDIUM = "medium"      # Warning con impacto en score
    LOW = "low"            # Warning informativo
    INFO = "info"          # Solo para reporte


class ValidationLevel(Enum):
    """Nivel de exigencia del framework."""
    STRICT = "strict"        # Falla en cualquier warning
    NORMAL = "normal"        # Falla solo en CRITICAL/HIGH
    PERMISSIVE = "permissive"  # Solo reporta, nunca falla


@dataclass(frozen=True)
class QualityConfig:
    """Configuración inmutable del framework."""

    # Paths - NO CAMBIAR SIN ACTUALIZAR QUALITY_TEST_PLAN.md
    silver_base_path: Path = field(
        default_factory=lambda: Path("/Users/byroncampo/My Drive/datalake/financial/02_silver")
    )

    # Defaults
    default_ticker: str = "BTCUSDT"
    default_theta: float = 0.005

    # Tolerancias numéricas
    theta_tolerance: float = 1e-9  # Para comparaciones de umbral
    price_tolerance: float = 1e-12  # Para comparaciones de precio

    # Límites estadísticos
    max_zero_os_rate: float = 0.30  # 30% máximo de OS vacíos
    os_magnitude_factor_min: float = 0.5  # avg(OS) > 0.5 * theta
    os_magnitude_factor_max: float = 2.0  # avg(OS) < 2.0 * theta
    tmv_factor_min: float = 1.5  # avg(TMV) > 1.5 * theta
    tmv_factor_max: float = 3.0  # avg(TMV) < 3.0 * theta
    max_slippage_factor: float = 2.0  # slippage < 2 * theta

    # Columnas requeridas en Silver (orden exacto)
    required_columns: tuple[str, ...] = (
        "event_type",
        "reference_price",
        "reference_time",
        "extreme_price",
        "extreme_time",
        "confirm_price",
        "confirm_time",
        "price_dc",
        "price_os",
        "time_dc",
        "time_os",
        "qty_dc",
        "qty_os",
        "dir_dc",
        "dir_os",
    )

    # Tipos esperados (para validación)
    expected_types: dict[str, str] = field(default_factory=lambda: {
        "event_type": "Int8",
        "reference_price": "Float64",
        "reference_time": "Int64",
        "extreme_price": "Float64",
        "extreme_time": "Int64",
        "confirm_price": "Float64",
        "confirm_time": "Int64",
        "price_dc": "List(Float64)",
        "price_os": "List(Float64)",
        "time_dc": "List(Int64)",
        "time_os": "List(Int64)",
        "qty_dc": "List(Float64)",
        "qty_os": "List(Float64)",
        "dir_dc": "List(Int8)",
        "dir_os": "List(Int8)",
    })


# Singleton de configuración
CONFIG = QualityConfig()
```

**Criterios de Aceptación F0.3:**

- [ ] El archivo existe en la ruta especificada
- [ ] `QualityConfig` es inmutable (frozen=True)
- [ ] Todos los paths son `Path` objects, no strings
- [ ] `CONFIG` es accesible como singleton
- [ ] Los tipos en `expected_types` coinciden con el schema real de Silver

---

### 3.2 conftest.py

**Archivo:** `tests/quality/conftest.py`

```python
"""Fixtures de pytest para el framework de calidad.

IMPORTANTE: Este archivo define cómo se cargan los datos Silver.
NO usar datos dummy. SIEMPRE cargar datos reales.
"""

from datetime import date
from pathlib import Path
from typing import Iterator

import polars as pl
import pytest

from .config import CONFIG, QualityConfig


def build_silver_path(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    day: int,
    base_path: Path | None = None,
) -> Path:
    """Construye el path a un archivo Silver específico.

    Args:
        ticker: Símbolo del instrumento (e.g., "BTCUSDT")
        theta: Umbral DC
        year: Año
        month: Mes
        day: Día
        base_path: Path base (default: CONFIG.silver_base_path)

    Returns:
        Path al archivo data.parquet

    Example:
        >>> build_silver_path("BTCUSDT", 0.005, 2025, 11, 15)
        PosixPath('/Users/.../02_silver/BTCUSDT/theta=0.005/year=2025/month=11/day=15/data.parquet')
    """
    if base_path is None:
        base_path = CONFIG.silver_base_path

    theta_str = f"{theta:.6f}".rstrip("0").rstrip(".")

    return (
        base_path
        / ticker
        / f"theta={theta_str}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"day={day:02d}"
        / "data.parquet"
    )


def list_available_days(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    base_path: Path | None = None,
) -> list[date]:
    """Lista los días disponibles en Silver para un mes dado.

    Returns:
        Lista de fechas ordenadas ascendentemente
    """
    if base_path is None:
        base_path = CONFIG.silver_base_path

    theta_str = f"{theta:.6f}".rstrip("0").rstrip(".")
    month_path = base_path / ticker / f"theta={theta_str}" / f"year={year}" / f"month={month:02d}"

    if not month_path.exists():
        return []

    days = []
    for day_dir in month_path.iterdir():
        if day_dir.is_dir() and day_dir.name.startswith("day="):
            day_num = int(day_dir.name.split("=")[1])
            parquet_path = day_dir / "data.parquet"
            if parquet_path.exists():
                days.append(date(year, month, day_num))

    return sorted(days)


def load_silver_day(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    day: int,
    base_path: Path | None = None,
) -> pl.DataFrame | None:
    """Carga datos Silver de un día específico.

    Returns:
        DataFrame Polars o None si no existe
    """
    path = build_silver_path(ticker, theta, year, month, day, base_path)

    if not path.exists():
        return None

    return pl.read_parquet(path)


def load_silver_month(
    ticker: str,
    theta: float,
    year: int,
    month: int,
    base_path: Path | None = None,
) -> dict[date, pl.DataFrame]:
    """Carga todos los días Silver de un mes.

    Returns:
        Dict {date: DataFrame} ordenado por fecha
    """
    days = list_available_days(ticker, theta, year, month, base_path)
    result = {}

    for d in days:
        df = load_silver_day(ticker, theta, year, month, d.day, base_path)
        if df is not None:
            result[d] = df

    return result


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def quality_config() -> QualityConfig:
    """Configuración del framework."""
    return CONFIG


@pytest.fixture(scope="session")
def silver_base_path() -> Path:
    """Path base de Silver."""
    return CONFIG.silver_base_path


@pytest.fixture
def silver_month_loader():
    """Factory fixture para cargar meses Silver."""
    def _loader(year: int, month: int, ticker: str = "BTCUSDT", theta: float = 0.005):
        return load_silver_month(ticker, theta, year, month)
    return _loader


@pytest.fixture
def silver_day_loader():
    """Factory fixture para cargar días Silver."""
    def _loader(year: int, month: int, day: int, ticker: str = "BTCUSDT", theta: float = 0.005):
        return load_silver_day(ticker, theta, year, month, day)
    return _loader
```

**Criterios de Aceptación F0.4:**

- [ ] `build_silver_path` genera paths correctos (verificar con path real)
- [ ] `list_available_days` encuentra días existentes
- [ ] `load_silver_day` carga datos sin error
- [ ] `load_silver_month` carga todos los días de un mes
- [ ] Fixtures son accesibles desde tests

---

### 3.3 validators/base.py

**Archivo:** `tests/quality/validators/base.py`

```python
"""Clase base para todos los validadores.

Todos los validadores DEBEN heredar de BaseValidator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from ..config import CONFIG, Severity


@dataclass
class ValidationResult:
    """Resultado de una validación individual."""

    test_name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    affected_events: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "affected_events": self.affected_events[:100],  # Limitar para reporte
            "n_affected": len(self.affected_events),
        }


@dataclass
class ValidatorReport:
    """Reporte agregado de un validador."""

    validator_name: str
    level: int  # 1-9
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True si todos los tests pasaron."""
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def has_critical_failures(self) -> bool:
        return any(
            not r.passed and r.severity == Severity.CRITICAL
            for r in self.results
        )

    def to_dict(self) -> dict:
        return {
            "validator_name": self.validator_name,
            "level": self.level,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "has_critical_failures": self.has_critical_failures,
            "results": [r.to_dict() for r in self.results],
        }


class BaseValidator(ABC):
    """Clase base abstracta para validadores.

    Todos los validadores deben:
    1. Heredar de esta clase
    2. Implementar `validate_day()`
    3. Definir `name` y `level`
    """

    name: str = "base"
    level: int = 0

    def __init__(self, config: type = CONFIG):
        """Inicializa el validador con configuración."""
        self.config = config

    @abstractmethod
    def validate_day(self, df: pl.DataFrame, day: date, theta: float) -> list[ValidationResult]:
        """Valida un día de datos Silver.

        Args:
            df: DataFrame Silver del día
            day: Fecha del día
            theta: Umbral DC usado

        Returns:
            Lista de resultados de validación
        """
        pass

    def validate_month(
        self,
        data: dict[date, pl.DataFrame],
        theta: float
    ) -> ValidatorReport:
        """Valida un mes completo de datos.

        Args:
            data: Dict {date: DataFrame} con datos del mes
            theta: Umbral DC

        Returns:
            Reporte agregado del validador
        """
        all_results = []

        for day, df in sorted(data.items()):
            day_results = self.validate_day(df, day, theta)
            all_results.extend(day_results)

        return ValidatorReport(
            validator_name=self.name,
            level=self.level,
            results=all_results,
        )

    def _make_result(
        self,
        test_name: str,
        passed: bool,
        severity: Severity,
        message: str,
        details: dict | None = None,
        affected_events: list[int] | None = None,
    ) -> ValidationResult:
        """Helper para crear resultados de validación."""
        return ValidationResult(
            test_name=f"{self.name}.{test_name}",
            passed=passed,
            severity=severity,
            message=message,
            details=details or {},
            affected_events=affected_events or [],
        )
```

**Criterios de Aceptación F0.6:**

- [ ] `ValidationResult` serializa correctamente a dict
- [ ] `ValidatorReport` calcula propiedades correctamente
- [ ] `BaseValidator` es abstracta y no instanciable directamente
- [ ] `validate_month` itera sobre días ordenados

---

### 3.4 report.py

**Archivo:** `tests/quality/report.py`

```python
"""Generador de reportes de calidad.

Genera reportes en formato JSON y consola.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

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
    def quality_score(self) -> float:
        """Score de calidad (0.0 - 1.0)."""
        if self.total_tests == 0:
            return 1.0
        return self.total_passed / self.total_tests

    def get_failures_by_severity(self, severity: Severity) -> list[dict]:
        """Obtiene fallos filtrados por severidad."""
        failures = []
        for vr in self.validator_reports:
            for r in vr.results:
                if not r.passed and r.severity == severity:
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

    def print_console(self, console: Console | None = None) -> None:
        """Imprime el reporte en consola con Rich."""
        if console is None:
            console = Console()

        # Header
        status = "[green]PASSED[/green]" if not self.has_critical_failures else "[red]FAILED[/red]"
        console.print(Panel(
            f"[bold]Silver Data Quality Report[/bold]\n"
            f"Ticker: {self.ticker} | θ={self.theta} | {self.year}-{self.month:02d}\n"
            f"Status: {status} | Score: {self.quality_score:.1%}",
            title="Quality Framework",
            border_style="blue",
        ))

        # Summary table
        table = Table(title="Validator Summary", box=box.ROUNDED)
        table.add_column("Validator", style="cyan")
        table.add_column("Level", justify="center")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Status", justify="center")

        for vr in self.validator_reports:
            status = "[green]✓[/green]" if vr.passed else "[red]✗[/red]"
            table.add_row(
                vr.validator_name,
                str(vr.level),
                str(vr.n_passed),
                str(vr.n_failed),
                status,
            )

        console.print(table)

        # Failures detail
        if self.total_failed > 0:
            console.print("\n[bold red]Failures:[/bold red]")
            for vr in self.validator_reports:
                for r in vr.results:
                    if not r.passed:
                        sev_color = {
                            Severity.CRITICAL: "red",
                            Severity.HIGH: "yellow",
                            Severity.MEDIUM: "orange3",
                            Severity.LOW: "dim",
                            Severity.INFO: "dim",
                        }.get(r.severity, "white")
                        console.print(
                            f"  [{sev_color}][{r.severity.value.upper()}][/{sev_color}] "
                            f"{r.test_name}: {r.message}"
                        )
```

**Criterios de Aceptación F0.7:**

- [ ] `QualityReport` serializa correctamente a JSON
- [ ] `should_fail` respeta los niveles de validación
- [ ] `print_console` muestra tabla formateada con Rich
- [ ] `save_json` crea el archivo correctamente

---

### 3.5 runner.py

**Archivo:** `tests/quality/runner.py`

```python
"""Orquestador del framework de calidad.

Ejecuta todos los validadores y genera el reporte.
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from .config import CONFIG, ValidationLevel
from .report import QualityReport
from .conftest import load_silver_month
from .validators.base import BaseValidator

# Importar todos los validadores
from .validators.structural import StructuralValidator
from .validators.intra_event import IntraEventValidator
from .validators.temporal import TemporalValidator
from .validators.price import PriceValidator
from .validators.collision import CollisionValidator
from .validators.threshold import ThresholdValidator
from .validators.cross_day import CrossDayValidator
from .validators.scale_laws import ScaleLawsValidator
from .validators.determinism import DeterminismValidator


# Registro de validadores en orden de ejecución
VALIDATORS: list[type[BaseValidator]] = [
    StructuralValidator,   # Nivel 1
    IntraEventValidator,   # Nivel 2
    TemporalValidator,     # Nivel 3
    PriceValidator,        # Nivel 4
    CollisionValidator,    # Nivel 6 (crítico, antes de threshold)
    ThresholdValidator,    # Nivel 5
    CrossDayValidator,     # Nivel 7
    ScaleLawsValidator,    # Nivel 8
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
        report.print_console(console)

    # Verificar si debe fallar
    if report.should_fail(level):
        raise AssertionError(
            f"Quality check failed with {report.total_failed} failures "
            f"(level={level.value})"
        )

    return report
```

**Criterios de Aceptación F0.8:**

- [ ] `VALIDATORS` contiene todos los validadores en orden
- [ ] `run_quality_check` carga datos reales
- [ ] Muestra progreso con Rich
- [ ] Falla correctamente según nivel
- [ ] Genera reporte JSON si se especifica path

---

### 3.6 **main**.py

**Archivo:** `tests/quality/__main__.py`

```python
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
        "--year", "-y",
        type=int,
        required=True,
        help="Año a validar",
    )
    parser.add_argument(
        "--month", "-m",
        type=int,
        required=True,
        help="Mes a validar (1-12)",
    )
    parser.add_argument(
        "--ticker", "-t",
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
        "--level", "-l",
        type=str,
        choices=["strict", "normal", "permissive"],
        default="normal",
        help="Nivel de exigencia (default: normal)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path para guardar reporte JSON",
    )
    parser.add_argument(
        "--quiet", "-q",
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
```

**Criterios de Aceptación F0.9:**

- [ ] CLI parsea argumentos correctamente
- [ ] `--help` muestra documentación completa
- [ ] Exit codes: 0=pass, 1=fail, 2=error
- [ ] `--quiet` suprime output a consola
- [ ] `--output` genera JSON

---

## 4. ESPECIFICACIONES DE VALIDADORES

### 4.1 structural.py (Nivel 1)

**Archivo:** `tests/quality/validators/structural.py`

**Propósito:** Validar la estructura básica de los datos Silver.

**Tests a implementar:**

#### Test 1: `test_required_columns`

```python
def test_required_columns(self, df: pl.DataFrame, day: date, theta: float) -> ValidationResult:
    """Verifica que todas las columnas requeridas estén presentes.

    Columnas requeridas (CONFIG.required_columns):
    - event_type, reference_price, reference_time, extreme_price, extreme_time,
    - confirm_price, confirm_time, price_dc, price_os, time_dc, time_os,
    - qty_dc, qty_os, dir_dc, dir_os

    Severidad: CRITICAL
    """
    missing = set(self.config.required_columns) - set(df.columns)

    return self._make_result(
        test_name="required_columns",
        passed=len(missing) == 0,
        severity=Severity.CRITICAL,
        message=f"Missing columns: {missing}" if missing else "All columns present",
        details={"missing": list(missing), "found": df.columns},
    )
```

#### Test 2: `test_column_types`

```python
def test_column_types(self, df: pl.DataFrame, day: date, theta: float) -> ValidationResult:
    """Verifica los tipos de datos de cada columna.

    Tipos esperados (CONFIG.expected_types):
    - event_type: Int8
    - reference_price: Float64
    - etc.

    Severidad: CRITICAL
    """
    type_errors = {}

    for col, expected_type_str in self.config.expected_types.items():
        if col not in df.columns:
            continue

        actual_type = str(df.schema[col])

        # Normalizar para comparación
        expected_normalized = expected_type_str.replace("List(", "List(").replace(")", ")")
        actual_normalized = actual_type.replace("List(", "List(").replace(")", ")")

        if expected_normalized not in actual_normalized:
            type_errors[col] = {"expected": expected_type_str, "actual": actual_type}

    return self._make_result(
        test_name="column_types",
        passed=len(type_errors) == 0,
        severity=Severity.CRITICAL,
        message=f"Type mismatches: {list(type_errors.keys())}" if type_errors else "All types correct",
        details={"type_errors": type_errors},
    )
```

#### Test 3: `test_no_null_scalars`

```python
def test_no_null_scalars(self, df: pl.DataFrame, day: date, theta: float) -> ValidationResult:
    """Verifica que no hay nulls en columnas escalares.

    Columnas escalares: event_type, reference_price, reference_time,
    extreme_price, extreme_time, confirm_price, confirm_time

    Nota: extreme_price puede ser -1.0 (provisional) pero NO null.

    Severidad: CRITICAL
    """
    scalar_cols = [
        "event_type", "reference_price", "reference_time",
        "extreme_price", "extreme_time", "confirm_price", "confirm_time"
    ]

    null_counts = {}
    for col in scalar_cols:
        if col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                null_counts[col] = null_count

    return self._make_result(
        test_name="no_null_scalars",
        passed=len(null_counts) == 0,
        severity=Severity.CRITICAL,
        message=f"Null values found: {null_counts}" if null_counts else "No null values",
        details={"null_counts": null_counts},
    )
```

#### Test 4: `test_non_empty_dataframe`

```python
def test_non_empty_dataframe(self, df: pl.DataFrame, day: date, theta: float) -> ValidationResult:
    """Verifica que el DataFrame no está vacío.

    Un día Silver sin eventos es válido pero sospechoso.

    Severidad: HIGH (no CRITICAL porque es posible legítimamente)
    """
    n_events = len(df)

    return self._make_result(
        test_name="non_empty_dataframe",
        passed=n_events > 0,
        severity=Severity.HIGH,
        message=f"DataFrame has {n_events} events" if n_events > 0 else "Empty DataFrame",
        details={"n_events": n_events, "day": str(day)},
    )
```

**Criterios de Aceptación F1.1:**

- [ ] Clase `StructuralValidator` hereda de `BaseValidator`
- [ ] `name = "structural"`, `level = 1`
- [ ] Implementa los 4 tests especificados
- [ ] `validate_day` retorna lista de `ValidationResult`
- [ ] Cada test tiene severidad correcta

---

### 4.2 intra_event.py (Nivel 2)

**Archivo:** `tests/quality/validators/intra_event.py`

**Tests a implementar:**

#### Test 1: `test_dc_list_lengths_consistent`

```python
"""Verifica que las 4 listas DC tienen la misma longitud para cada evento.

Para cada fila: len(price_dc) == len(time_dc) == len(qty_dc) == len(dir_dc)

Severidad: CRITICAL
"""
```

#### Test 2: `test_os_list_lengths_consistent`

```python
"""Verifica que las 4 listas OS tienen la misma longitud para cada evento.

Para cada fila: len(price_os) == len(time_os) == len(qty_os) == len(dir_os)

Severidad: CRITICAL
"""
```

#### Test 3: `test_event_type_valid_values`

```python
"""Verifica que event_type solo contiene 1 (upturn) o -1 (downturn).

Valores inválidos: 0, 2, -2, null, etc.

Severidad: CRITICAL
"""
```

#### Test 4: `test_dc_not_empty`

```python
"""Verifica que la fase DC nunca está vacía.

Invariante: len(price_dc) >= 1 para todo evento.
El DC siempre debe tener al menos el tick de confirmación.

Severidad: CRITICAL
"""
```

#### Test 5: `test_directions_valid_values`

```python
"""Verifica que direction solo contiene 1 (buy) o -1 (sell).

Aplica a: dir_dc y dir_os

Severidad: CRITICAL
"""
```

#### Test 6: `test_quantities_positive`

```python
"""Verifica que todas las cantidades son positivas.

Invariante: qty > 0 para todo tick en qty_dc y qty_os

Severidad: HIGH
"""
```

**Criterios de Aceptación F1.2:**

- [ ] Clase `IntraEventValidator` hereda de `BaseValidator`
- [ ] `name = "intra_event"`, `level = 2`
- [ ] Implementa los 6 tests especificados
- [ ] Para tests de listas, itera sobre CADA evento
- [ ] Reporta índices de eventos afectados

---

### 4.3 temporal.py (Nivel 3)

**Archivo:** `tests/quality/validators/temporal.py`

**Tests a implementar:**

#### Test 1: `test_dc_timestamps_monotonic`

```python
"""Verifica que timestamps dentro de time_dc son monotónicamente crecientes.

Para cada evento: time_dc[i] <= time_dc[i+1]
Permite timestamps iguales (ticks simultáneos) pero no decrecientes.

Severidad: CRITICAL
"""
```

#### Test 2: `test_os_timestamps_monotonic`

```python
"""Verifica que timestamps dentro de time_os son monotónicamente crecientes.

Para cada evento: time_os[i] <= time_os[i+1]

Severidad: CRITICAL
"""
```

#### Test 3: `test_dc_before_os`

```python
"""Verifica que la fase DC termina antes de que comience la fase OS.

Invariante: max(time_dc) <= min(time_os) si OS no está vacío.
Si OS está vacío, este test no aplica.

Severidad: CRITICAL
"""
```

#### Test 4: `test_reference_before_confirm`

```python
"""Verifica que reference_time < confirm_time.

El extremo de referencia siempre ocurre antes de la confirmación.

Severidad: CRITICAL
"""
```

#### Test 5: `test_confirm_before_extreme`

```python
"""Verifica que confirm_time < extreme_time (si extreme != -1).

La confirmación ocurre antes del siguiente extremo.
Excepción: extreme_time puede ser -1 (provisional) para el último evento.

Severidad: CRITICAL
"""
```

#### Test 6: `test_confirm_time_equals_last_dc_time`

```python
"""Verifica que confirm_time == time_dc[-1].

El timestamp de confirmación debe coincidir con el último tick del DC.

Severidad: CRITICAL
"""
```

#### Test 7: `test_timestamps_positive`

```python
"""Verifica que todos los timestamps son positivos (nanosegundos desde epoch).

Severidad: CRITICAL
"""
```

**Criterios de Aceptación F1.3:**

- [ ] Clase `TemporalValidator` hereda de `BaseValidator`
- [ ] `name = "temporal"`, `level = 3`
- [ ] Implementa los 7 tests especificados
- [ ] Maneja correctamente OS vacíos
- [ ] Maneja correctamente extreme_time = -1

---

### 4.4 price.py (Nivel 4)

**Archivo:** `tests/quality/validators/price.py`

**Tests a implementar:**

#### Test 1: `test_prices_positive`

```python
"""Verifica que todos los precios son positivos.

Aplica a: reference_price, confirm_price, price_dc, price_os
Excepción: extreme_price puede ser -1.0 (provisional)

Severidad: CRITICAL
"""
```

#### Test 2: `test_prices_finite`

```python
"""Verifica que no hay NaN ni Inf en precios.

Severidad: CRITICAL
"""
```

#### Test 3: `test_confirm_price_in_dc_range`

```python
"""Verifica que confirm_price está dentro del rango de price_dc.

Invariante: min(price_dc) <= confirm_price <= max(price_dc)

Severidad: HIGH
"""
```

#### Test 4: `test_confirm_price_matches_dc`

```python
"""Verifica que confirm_price aparece en price_dc.

Por la regla conservadora, el confirm_price puede no ser el último precio,
pero DEBE estar presente en la lista.

Severidad: CRITICAL
"""
```

#### Test 5: `test_reference_price_chain`

```python
"""Verifica la cadena de precios de referencia intra-día.

Para eventos N > 0 dentro del mismo día:
reference_price[N] == extreme_price[N-1] (si extreme_price[N-1] != -1)

Severidad: HIGH
"""
```

#### Test 6: `test_extreme_price_in_os_range`

```python
"""Verifica que extreme_price está dentro del rango de price_os.

Aplica solo si OS no está vacío y extreme_price != -1.

Para upturn: extreme_price == max(price_os)
Para downturn: extreme_price == min(price_os)

Severidad: HIGH
"""
```

**Criterios de Aceptación F1.4:**

- [ ] Clase `PriceValidator` hereda de `BaseValidator`
- [ ] `name = "price"`, `level = 4`
- [ ] Implementa los 6 tests especificados
- [ ] Maneja correctamente extreme_price = -1.0
- [ ] Considera event_type para validar extremos

---

### 4.5 threshold.py (Nivel 5)

**Archivo:** `tests/quality/validators/threshold.py`

**Tests a implementar:**

#### Test 1: `test_dc_magnitude_meets_theta`

```python
"""Verifica que la magnitud DC cumple el umbral θ.

Invariante: |confirm_price - reference_price| / reference_price >= θ - ε

Donde ε es CONFIG.theta_tolerance (para errores de redondeo).

Severidad: HIGH
"""
```

#### Test 2: `test_slippage_bounded`

```python
"""Verifica que el slippage está dentro de límites razonables.

slippage = confirm_price - reference_price * (1 + event_type * θ)

Límite típico: |slippage| < 2θ * reference_price

Severidad: MEDIUM
"""
```

#### Test 3: `test_direction_matches_event_type`

```python
"""Verifica coherencia entre event_type y movimiento de precio.

Para upturn (+1): confirm_price > reference_price
Para downturn (-1): confirm_price < reference_price

Severidad: CRITICAL
"""
```

**Criterios de Aceptación F2.1:**

- [ ] Clase `ThresholdValidator` hereda de `BaseValidator`
- [ ] `name = "threshold"`, `level = 5`
- [ ] Implementa los 3 tests especificados
- [ ] Usa CONFIG.theta_tolerance correctamente
- [ ] Calcula slippage según fórmula documentada

---

### 4.6 collision.py (Nivel 6)

**Archivo:** `tests/quality/validators/collision.py`

**Propósito:** Validar el manejo correcto de timestamps simultáneos (bug crítico corregido).

**Tests a implementar:**

#### Test 1: `test_no_os_ticks_at_confirm_time`

```python
"""CRÍTICO: Verifica que ningún tick del OS tiene el mismo timestamp que confirm_time.

Este test valida la corrección del bug donde ticks del mismo instante que el DCC
podían quedar en la fase OS en lugar de la fase DC.

Invariante: Para cada tick en time_os: tick_time > confirm_time

Severidad: CRITICAL
"""
```

#### Test 2: `test_all_confirm_instant_in_dc`

```python
"""CRÍTICO: Verifica que TODOS los ticks con timestamp == confirm_time están en DC.

Si hay N ticks con el mismo timestamp que la confirmación,
todos deben estar en time_dc, ninguno en time_os.

Severidad: CRITICAL
"""
```

#### Test 3: `test_dc_os_no_timestamp_overlap`

```python
"""Verifica que no hay solapamiento de timestamps entre DC y OS.

Invariante: max(time_dc) < min(time_os) OR time_os está vacío

Nota: Este test es más estricto que test_dc_before_os porque
requiere estrictamente menor, no menor-igual.

Severidad: HIGH
"""
```

#### Test 4: `test_conservative_price_selection`

```python
"""Verifica que se aplicó la regla conservadora para ticks simultáneos.

Si hay múltiples ticks con timestamp == confirm_time:
- Para upturn: confirm_price == min(precios que cruzan θ)
- Para downturn: confirm_price == max(precios que cruzan θ)

Severidad: HIGH
"""
```

**Criterios de Aceptación F1.5:**

- [ ] Clase `CollisionValidator` hereda de `BaseValidator`
- [ ] `name = "collision"`, `level = 6`
- [ ] Implementa los 4 tests especificados
- [ ] Reporta TODOS los eventos que violan invariantes
- [ ] Incluye detalles de timestamps conflictivos

---

### 4.7 cross_day.py (Nivel 7)

**Archivo:** `tests/quality/validators/cross_day.py`

**Tests a implementar:**

#### Test 1: `test_reference_chain_across_days`

```python
"""Verifica continuidad de precios de referencia entre días.

Para el primer evento del día N:
reference_price[día N, evento 0] == extreme_price[día N-1, último evento]

Excepción: Si el día anterior no tiene datos o extreme_price es provisional.

Severidad: MEDIUM
"""
```

#### Test 2: `test_trend_continuity`

```python
"""Verifica coherencia de tendencia entre días.

Si el último evento del día N-1 es upturn, el primer evento del día N
debería ser downturn (o upturn si hubo reversión intra-día).

Este test es informativo, no bloquea.

Severidad: INFO
"""
```

#### Test 3: `test_last_event_extreme_provisional`

```python
"""Verifica que el último evento del día puede tener extreme_price provisional.

El último evento de un día puede tener extreme_price == -1.0 porque
el verdadero extremo no se conoce hasta que lleguen datos del día siguiente.

Severidad: INFO (solo documenta, no falla)
"""
```

**Criterios de Aceptación F2.2:**

- [ ] Clase `CrossDayValidator` hereda de `BaseValidator`
- [ ] `name = "cross_day"`, `level = 7`
- [ ] `validate_month` tiene acceso a todos los días
- [ ] Implementa los 3 tests especificados
- [ ] Maneja correctamente días faltantes

---

### 4.8 scale_laws.py (Nivel 8)

**Archivo:** `tests/quality/validators/scale_laws.py`

**Tests a implementar:**

#### Test 1: `test_avg_os_magnitude_near_theta`

```python
"""Verifica la ley del factor 2: avg(|OS|) ≈ θ.

Límites configurables:
- CONFIG.os_magnitude_factor_min * θ < avg(|OS|)
- avg(|OS|) < CONFIG.os_magnitude_factor_max * θ

Excluye eventos con OS vacío.

Severidad: INFO
"""
```

#### Test 2: `test_avg_tmv_near_2theta`

```python
"""Verifica que TMV promedio ≈ 2θ.

TMV = |DC magnitude| + |OS magnitude|

Límites:
- CONFIG.tmv_factor_min * θ < avg(TMV)
- avg(TMV) < CONFIG.tmv_factor_max * θ

Severidad: INFO
"""
```

#### Test 3: `test_zero_os_rate_bounded`

```python
"""Verifica que la tasa de OS vacíos no es excesiva.

Límite: rate(OS vacíos) < CONFIG.max_zero_os_rate

Una tasa muy alta sugiere problemas de umbral o datos.

Severidad: MEDIUM
"""
```

#### Test 4: `test_dc_duration_distribution`

```python
"""Reporta estadísticas de duración DC.

No falla, solo genera estadísticas para el reporte.

Severidad: INFO
"""
```

**Criterios de Aceptación F3.1:**

- [ ] Clase `ScaleLawsValidator` hereda de `BaseValidator`
- [ ] `name = "scale_laws"`, `level = 8`
- [ ] Implementa los 4 tests especificados
- [ ] Calcula magnitudes correctamente
- [ ] Usa CONFIG para límites

---

### 4.9 determinism.py (Nivel 9)

**Archivo:** `tests/quality/validators/determinism.py`

**Propósito:** Validar que el reprocesamiento produce resultados idénticos.

**Tests a implementar:**

#### Test 1: `test_reprocess_identical`

```python
"""GOLD STANDARD: Reprocesar datos Bronze debe producir Silver idéntico.

Este test:
1. Carga datos Bronze del mes
2. Reprocesa con el Engine
3. Compara con datos Silver existentes
4. Falla si hay CUALQUIER diferencia

NOTA: Este test es costoso en tiempo. Se puede omitir con flag.

Severidad: CRITICAL
"""
```

**Criterios de Aceptación F3.2:**

- [ ] Clase `DeterminismValidator` hereda de `BaseValidator`
- [ ] `name = "determinism"`, `level = 9`
- [ ] Puede importar Engine y datos Bronze
- [ ] Comparación byte-a-byte o estructural
- [ ] Flag para omitir en ejecuciones rápidas

---

## 5. NOTAS DE IMPLEMENTACIÓN

### Decisiones de Diseño

1. **No usar fixtures de pytest para datos** - Los validadores cargan datos directamente para flexibilidad.

2. **Validadores stateless** - Cada validador es independiente y no guarda estado entre llamadas.

3. **Orden de ejecución** - Los niveles 1-4 y 6 son críticos y se ejecutan primero.

### Bloqueos Conocidos

(Completar durante implementación)

### Cambios al Plan

(Documentar cualquier desviación del plan original)

---

## 6. COMANDOS DE VERIFICACIÓN

### Verificar estructura de archivos

```bash
ls -la tests/quality/
ls -la tests/quality/validators/
```

### Ejecutar tests de infraestructura

```bash
python -c "from tests.quality.config import CONFIG; print(CONFIG)"
python -c "from tests.quality.conftest import list_available_days; print(list_available_days('BTCUSDT', 0.005, 2025, 11))"
```

### Ejecutar framework completo

```bash
python -m tests.quality --year 2025 --month 11 --level permissive
```

---

## 7. CHECKLIST FINAL

- [ ] Todos los archivos creados según estructura
- [ ] Todos los tests implementados según spec
- [ ] CLI funciona con todos los flags
- [ ] Reporte JSON se genera correctamente
- [ ] Reporte de consola es legible
- [ ] Tests pasan con datos reales de 2025-11
- [ ] Documentación de uso actualizada

---

**Última actualización de este documento:** 2026-02-02 por Claude Code
