"""Tipo Python del hallazgo de DQ y su conversión a tabla Arrow."""

import json
import time
import uuid
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

import pyarrow as pa

from dq.schema import FINDING_SCHEMA


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Stage(StrEnum):
    PROVISIONAL = "provisional"
    CANONICAL = "canonical"


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    CORRECTED = "corrected"


def _now_us() -> int:
    return time.time_ns() // 1000


# kw_only permite declarar los campos en el orden de FINDING_SCHEMA aunque
# los primeros tengan valor por defecto.
@dataclass(kw_only=True)
class Finding:
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Microsegundos desde la época, UTC.
    detected_at: int = field(default_factory=_now_us)
    layer: str
    # mode y check_type son texto libre: los fija cada capa.
    mode: str
    check_type: str
    severity: Severity
    stage: Stage
    status: Status
    provider: str
    market: str
    asset: str
    year: int
    month: int
    metric_value: float | None = None
    details: dict[str, Any]
    run_id: str
    image_version: str

    def __post_init__(self) -> None:
        # El constructor del enum lanza ValueError si el valor no es válido.
        self.severity = Severity(self.severity)
        self.stage = Stage(self.stage)
        self.status = Status(self.status)


def to_table(findings: list[Finding]) -> pa.Table:
    """Convierte hallazgos a una tabla con FINDING_SCHEMA."""
    columns = {f.name: [getattr(x, f.name) for x in findings] for f in fields(Finding)}
    columns["details"] = [json.dumps(d) for d in columns["details"]]
    return pa.table(columns, schema=FINDING_SCHEMA)
