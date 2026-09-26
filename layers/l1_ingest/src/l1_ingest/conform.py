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


def classify_unit(lo: int | None, hi: int | None) -> str | None:
    """`"ms"` o `"us"` si mín y máx caen en la misma banda; si no, None."""
    if lo is None or hi is None:
        return None
    if _in_band(lo, MS_BAND) and _in_band(hi, MS_BAND):
        return "ms"
    if _in_band(lo, US_BAND) and _in_band(hi, US_BAND):
        return "us"
    return None


def unit_check(unit: str, lo: int, hi: int) -> CheckResult:
    """Hallazgo de una unidad válida: ms se corrige a µs, µs pasa tal cual."""
    status = Status.CORRECTED if unit == "ms" else Status.PASS
    return CheckResult(
        CHECK_TYPE,
        Severity.INFO,
        status,
        details={"unit": unit, "min": lo, "max": hi},
    )


def unit_error(lo: int | None, hi: int | None) -> TimestampUnitError:
    check = CheckResult(
        CHECK_TYPE,
        Severity.ERROR,
        Status.FAIL,
        details={"min": lo, "max": hi},
    )
    return TimestampUnitError(
        f"transact_time fuera de las bandas de 13 (ms) o 16 (µs) dígitos: "
        f"min={lo}, max={hi}",
        check,
    )


def _normalize_time(times: pa.ChunkedArray) -> tuple[pa.ChunkedArray, CheckResult]:
    """La unidad se decide por la magnitud de mín y máx, nunca por la fecha."""
    bounds = pc.min_max(times)
    lo, hi = bounds["min"].as_py(), bounds["max"].as_py()
    unit = classify_unit(lo, hi)
    if unit is None:
        raise unit_error(lo, hi)
    check = unit_check(unit, lo, hi)
    return (pc.multiply(times, MS_TO_US) if unit == "ms" else times), check


def conform_batch(batch: pa.RecordBatch, unit: str) -> pa.RecordBatch:
    """Lote con OUTPUT_SCHEMA; la unidad ya se decidió sobre toda la unidad."""
    if unit == "ms":
        times = pc.multiply(batch.column("transact_time"), MS_TO_US)
        batch = batch.set_column(
            batch.schema.get_field_index("transact_time"),
            batch.schema.field("transact_time"),
            times,
        )
    return batch.cast(OUTPUT_SCHEMA)


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
