"""Calendario de publicación de Binance y estado de un archivo que da 404 (ITSC-295).

Binance publica el diario de D el día D+1 y el mensual de M el primer lunes de
M+1 (README de binance-public-data; TRD-L1 §1.1 y §8.4). No publica hora ni SLA,
así que el día de publicación completo cuenta como "a tiempo".
"""

from datetime import date, timedelta

from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.pipeline import Unit

NOT_PUBLISHED = "source_not_published"
DELAYED = "source_delayed"
MONDAY = 0


def publication_date(unit: Unit) -> date:
    """Primer día en que Binance publica la unidad."""
    if unit.day is not None:
        return date(unit.year, unit.month, unit.day) + timedelta(days=1)
    first = (
        date(unit.year + 1, 1, 1)
        if unit.month == 12
        else date(unit.year, unit.month + 1, 1)
    )
    return first + timedelta(days=(MONDAY - first.weekday()) % 7)


def missing_source_check(unit: Unit, today: date, url: str) -> CheckResult:
    """Clasifica el 404 de `url` según la fecha UTC de la corrida (`today`).

    - antes de la publicación: `source_not_published`, info/pass;
    - el día de publicación: `source_not_published`, warning/fail;
    - después: `source_delayed`, error/fail, con los días de retraso.

    `metric_value` es, en `source_not_published`, los días que faltan para la
    publicación (0 el propio día) y, en `source_delayed`, los días de retraso.
    """
    expected = publication_date(unit)
    details = {
        "unit": str(unit),  # trae el día, que el hallazgo (año, mes) no tiene
        "expected_publication": expected.isoformat(),
        "source_url": url,
    }
    if today > expected:
        late = (today - expected).days
        details["reason"] = (
            f"{late} días de retraso sobre el calendario publicado de Binance"
        )
        return CheckResult(DELAYED, Severity.ERROR, Status.FAIL, late, details)
    remaining = (expected - today).days
    if today == expected:
        details["reason"] = "día de publicación, aún sin archivo"
        return CheckResult(NOT_PUBLISHED, Severity.WARNING, Status.FAIL, 0, details)
    details["reason"] = (
        "dentro del calendario de Binance; "
        f"publicación esperada el {expected.isoformat()}"
    )
    return CheckResult(NOT_PUBLISHED, Severity.INFO, Status.PASS, remaining, details)
