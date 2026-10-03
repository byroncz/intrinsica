"""Marcas de registros inválidos de Binance y filas con valores corruptos.

En abril de 2022 Binance auditó su histórico Spot y marcó los agg trades
duplicados con `p = 0, q = 0, f = -1, l = -1`, conservando `agg_trade_id` y
`transact_time` para no romper la secuencia
(https://github.com/binance/binance-spot-api-docs/blob/master/CHANGELOG_CN.md,
entrada 2022-04-12). No son transacciones: L1 las descarta antes de escribir.

Una fila es marca si y solo si cumple las cuatro igualdades. Cualquier otra
fila con `price <= 0` o `quantity <= 0` es dato corrupto y la unidad falla.
"""

from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc
from dq import Severity, Status

from l1_ingest.checks import CheckResult
from l1_ingest.integrity import DETAILS_MAX

MARKER_CHECK = "provider_invalid_marker"
RANGE_CHECK = "price_out_of_range"
ZERO = Decimal(0)


class ValueRangeError(ValueError):
    """Hay filas con price o quantity <= 0 que no son marca; lleva el hallazgo."""

    def __init__(self, message: str, check: CheckResult) -> None:
        super().__init__(message)
        self.check = check


class MarkFilter:
    """Descarta las marcas de cada lote y cuenta las filas corruptas.

    Opera sobre datos ya conformados (`OUTPUT_SCHEMA`), `RecordBatch` o `Table`.
    Solo retiene el conteo y los primeros `DETAILS_MAX` ids de cada clase.
    """

    def __init__(self) -> None:
        self.n_marks = 0
        self.mark_ids: list[int] = []
        self.n_invalid = 0
        self.invalid_ids: list[int] = []

    def feed(self, data):
        """Devuelve `data` sin sus marcas; si no tiene, la misma instancia."""
        price, quantity = data.column("price"), data.column("quantity")
        is_mark = pc.and_(
            pc.and_(pc.equal(price, ZERO), pc.equal(quantity, ZERO)),
            pc.and_(
                pc.equal(data.column("first_trade_id"), -1),
                pc.equal(data.column("last_trade_id"), -1),
            ),
        )
        non_positive = pc.or_(pc.less_equal(price, ZERO), pc.less_equal(quantity, ZERO))
        is_invalid = pc.and_(non_positive, pc.invert(is_mark))

        ids = data.column("agg_trade_id")
        self.n_invalid += _count(is_invalid)
        self.invalid_ids += _head(ids, is_invalid, DETAILS_MAX - len(self.invalid_ids))
        n_marks = _count(is_mark)
        if n_marks == 0:
            return data
        self.n_marks += n_marks
        self.mark_ids += _head(ids, is_mark, DETAILS_MAX - len(self.mark_ids))
        return data.filter(pc.invert(is_mark))

    def marker_result(self) -> CheckResult:
        if self.n_marks == 0:
            return CheckResult(MARKER_CHECK, Severity.INFO, Status.PASS, 0.0)
        return CheckResult(
            MARKER_CHECK,
            Severity.WARNING,
            Status.CORRECTED,
            float(self.n_marks),
            {"ids": self.mark_ids},
        )

    def raise_if_invalid(self) -> None:
        if self.n_invalid == 0:
            return
        check = CheckResult(
            RANGE_CHECK,
            Severity.ERROR,
            Status.FAIL,
            float(self.n_invalid),
            {"ids": self.invalid_ids},
        )
        raise ValueRangeError(
            f"{self.n_invalid} filas con price <= 0 o quantity <= 0 que no son "
            f"marca del proveedor; primeros agg_trade_id: {self.invalid_ids}",
            check,
        )


def _count(mask: pa.Array | pa.ChunkedArray) -> int:
    return pc.sum(mask).as_py() or 0


def _head(ids: pa.Array | pa.ChunkedArray, mask, room: int) -> list[int]:
    if room <= 0:
        return []
    return pc.filter(ids, mask).slice(0, room).to_pylist()
