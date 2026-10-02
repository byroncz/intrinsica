"""`carry_over.parquet`: el estado de un θ al cierre del mes (§7.4, ADR-L2-07/08).

Una fila por `(theta, year, month)`. `dc_pyo3.CarryOver` trae los campos del
detector; aquí se le agregan las coordenadas del dato y se baja a Parquet, y al
revés. Cargar es fail-closed (ADR-L2-08): si falta o no es de la versión que
esta imagen sabe leer, levanta `CarryOverError` y el mes no avanza.
"""

from dataclasses import dataclass
from decimal import Decimal

import dc_pyo3
import pyarrow as pa
import pyarrow.parquet as pq
from pyutils import ContentHasher, PartitionWriter, resolve_fs

from l2_dc_events.schema import CARRY_OVER_SCHEMA

POINTS = ("ext_high", "ext_low", "pending_reference", "pending_confirm")


class CarryOverError(Exception):
    """El carry-over que el mes necesita no está o no se puede usar.

    `check_type` es el del hallazgo de TRD-L2 §9.3 que lo describe y `details`
    su payload.
    """

    def __init__(self, check_type: str, message: str, details: dict) -> None:
        super().__init__(message)
        self.check_type = check_type
        self.details = details


@dataclass(frozen=True)
class Coordinates:
    """Dónde vive un carry-over: el dato al que pertenece (columnas y ruta)."""

    provider: str
    market: str
    asset: str
    year: int
    month: int


def _scaled(value: int) -> Decimal:
    return Decimal(value).scaleb(-8)


def to_batch(carry: dc_pyo3.CarryOver, where: Coordinates) -> pa.RecordBatch:
    """La fila del carry-over de un θ, con el esquema de §7.4."""
    pending = carry.pending
    points = {
        "ext_high": carry.ext_high,
        "ext_low": carry.ext_low,
        "pending_reference": pending[0] if pending else None,
        "pending_confirm": pending[1] if pending else None,
    }
    row: dict = {
        "provider": where.provider,
        "market": where.market,
        "asset": where.asset,
        "theta": _scaled(carry.theta),
        "year": where.year,
        "month": where.month,
        "state_version": carry.state_version,
        "direction": carry.direction,
    }
    for name in POINTS[:2]:
        row |= _columns(name, points[name])
    row["has_pending_event"] = pending is not None
    for name in POINTS[2:]:
        row |= _columns(name, points[name])
    return pa.RecordBatch.from_pylist([row], schema=CARRY_OVER_SCHEMA)


def _columns(name: str, point: tuple[int, int, int] | None) -> dict:
    if point is None:
        return {
            f"{name}_price": None,
            f"{name}_time": None,
            f"{name}_agg_trade_id": None,
        }
    price, time, agg_trade_id = point
    return {
        f"{name}_price": _scaled(price),
        f"{name}_time": time,
        f"{name}_agg_trade_id": agg_trade_id,
    }


def write_carry_over(
    carry: dc_pyo3.CarryOver, where: Coordinates, path: str
) -> tuple[str, str]:
    """Publica el carry-over atómicamente. Devuelve `(path, content_hash)`."""
    batch = to_batch(carry, where)
    hasher = ContentHasher(CARRY_OVER_SCHEMA)
    hasher.update(batch)
    with PartitionWriter(path, CARRY_OVER_SCHEMA, compact_encoding=True) as writer:
        writer.write_batch(batch)
        return writer.commit(), hasher.hexdigest()


def _point(row: dict, name: str) -> tuple[int, int, int] | None:
    price = row[f"{name}_price"]
    if price is None:
        return None
    return (
        int(price.scaleb(8)),
        row[f"{name}_time"],
        row[f"{name}_agg_trade_id"],
    )


def read_carry_over(path: str, theta: int) -> dc_pyo3.CarryOver:
    """Carga el carry-over de `theta` desde `path`.

    Falla con `CarryOverError`: `carry_over_missing` si el archivo no existe,
    `carry_over_version_mismatch` si su `state_version` no es la de
    `dc_pyo3.STATE_VERSION` (se lee primero, sola, porque otra versión puede
    tener otras columnas) y `theta_config_drift` si su columna `theta` no es
    la de la partición en la que está.
    """
    fs, resolved = resolve_fs(path)
    if not fs.get_file_info(resolved).is_file:
        raise CarryOverError(
            "carry_over_missing",
            f"{path} no existe: el mes anterior no dejó su carry-over",
            {"theta": theta, "expected_path": path},
        )
    try:
        row = _read_row(path, resolved, fs, theta)
    except (pa.ArrowException, IndexError, ValueError) as exc:
        raise CarryOverError(
            "carry_over_version_mismatch",
            f"{path}: no se pudo leer el carry-over ({type(exc).__name__}: {exc})",
            {"theta": theta, "path": path, "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    if int(row["theta"].scaleb(8)) != theta:
        raise CarryOverError(
            "theta_config_drift",
            f"{path}: theta={row['theta']} no es el de la partición ({theta})",
            {"theta": theta, "found": str(row["theta"]), "path": path},
        )
    pending = None
    if row["has_pending_event"]:
        pending = (_point(row, "pending_reference"), _point(row, "pending_confirm"))
    return dc_pyo3.CarryOver(
        theta,
        row["state_version"],
        row["direction"],
        _point(row, "ext_high"),
        _point(row, "ext_low"),
        pending,
    )


def _read_row(path: str, resolved: str, fs, theta: int) -> dict:
    """La única fila del archivo. Los errores de `CarryOverError` salen tal cual;
    un archivo corrupto, vacío o con más de una fila levanta el error de Arrow
    o `ValueError`, que `read_carry_over` convierte."""
    with pq.ParquetFile(resolved, filesystem=fs) as parquet:
        schema = parquet.schema_arrow
        found = None
        if "state_version" in schema.names:
            found = parquet.read(columns=["state_version"]).column(0)[0].as_py()
        if found != dc_pyo3.STATE_VERSION:
            raise CarryOverError(
                "carry_over_version_mismatch",
                f"{path}: state_version={found!r}, la imagen lee "
                f"{dc_pyo3.STATE_VERSION!r}",
                {
                    "theta": theta,
                    "found": found,
                    "expected": dc_pyo3.STATE_VERSION,
                    "path": path,
                },
            )
        if not schema.equals(CARRY_OVER_SCHEMA):
            raise CarryOverError(
                "carry_over_version_mismatch",
                f"{path}: el esquema no es el de state_version={found!r}",
                {"theta": theta, "found": str(schema), "path": path},
            )
        (row,) = parquet.read().to_pylist()
    return row
