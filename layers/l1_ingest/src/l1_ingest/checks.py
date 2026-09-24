"""Resultado de un chequeo de un paso de L1, sin conocer el contexto del run."""

from dataclasses import dataclass, field
from typing import Any

from dq import Severity, Status


@dataclass
class CheckResult:
    check_type: str
    severity: Severity
    status: Status
    metric_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # El constructor del enum lanza ValueError si el valor no es válido.
        self.severity = Severity(self.severity)
        self.status = Status(self.status)
