"""Paso 6 del núcleo: RAW_SCHEMA a OUTPUT_SCHEMA (decimales exactos y µs)."""

import pyarrow as pa
import pyarrow.compute as pc
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.schema import OUTPUT_SCHEMA

CHECK_TYPE = "timestamp_unit_corrected"
MS_BAND = (10**12, 10**13)  # 13 dígitos
US_BAND = (10**15, 10**16)  # 16 dígitos
MS_TO_US = 1000


class TimestampUnitError(ValueError):
    """transact_time no está en ms ni en µs; lleva el hallazgo para emitirlo."""

    def __init__(self, message: str, check: CheckResult) -> None:
        super().__init__(message)
        self.check = check


def _in_band(value: int, band: tuple[int, int]) -> bool:
    return band[0] <= value < band[1]


def _normalize_time(times: pa.ChunkedArray) -> tuple[pa.ChunkedArray, CheckResult]:
    """La unidad se decide por la magnitud de mín y máx, nunca por la fecha."""
    bounds = pc.min_max(times)
    lo, hi = bounds["min"].as_py(), bounds["max"].as_py()

    if lo is not None and hi is not None:
        if _in_band(lo, MS_BAND) and _in_band(hi, MS_BAND):
            check = CheckResult(
                CHECK_TYPE,
                Severity.INFO,
                Status.CORRECTED,
                details={"unit": "ms", "min": lo, "max": hi},
            )
            return pc.multiply(times, MS_TO_US), check
        if _in_band(lo, US_BAND) and _in_band(hi, US_BAND):
            check = CheckResult(
                CHECK_TYPE,
                Severity.INFO,
                Status.PASS,
                details={"unit": "us", "min": lo, "max": hi},
            )
            return times, check

    check = CheckResult(
        CHECK_TYPE,
        Severity.ERROR,
        Status.FAIL,
        details={"min": lo, "max": hi},
    )
    raise TimestampUnitError(
        f"transact_time fuera de las bandas de 13 (ms) o 16 (µs) dígitos: "
        f"min={lo}, max={hi}",
        check,
    )


def conform(table: pa.Table) -> tuple[pa.Table, CheckResult]:
    """Devuelve la tabla con OUTPUT_SCHEMA y el hallazgo de la unidad de tiempo.

    El cast a decimal128(18,8) es seguro: un valor que no cabe o con más de
    8 decimales lanza pa.ArrowInvalid, nunca se redondea.
    """
    times, check = _normalize_time(table.column("transact_time"))
    out = table.set_column(
        table.schema.get_field_index("transact_time"),
        table.schema.field("transact_time"),
        times,
    )
    return out.cast(OUTPUT_SCHEMA), check
