"""Capa L3 de intrinsica: descriptores por evento DC y θ (TRD-L3).

El núcleo de la familia S: `summarize_month` escribe `summaries.parquet` de un mes y una
lista de θ, y sus errores son excepciones tipadas (`errors`). La imagen (`cli`,
`pipeline`, `frontier`, `findings`, `timing`) recorre los meses por θ con ese núcleo y
convierte las excepciones en hallazgos de DQ.
"""

from l3_dc_descriptors.errors import (
    L2InputMissing,
    L2StateInconsistent,
    PhaseQuantityMismatch,
    PhaseTicksMismatch,
    PhaseValueError,
    SkeletonMismatch,
    SummarizeError,
)
from l3_dc_descriptors.reconcile import Reconciliation
from l3_dc_descriptors.schema import FAMILY, STATE_VERSION, SUMMARIES_SCHEMA
from l3_dc_descriptors.summarize import ThetaResult, summarize_month

__all__ = [
    "FAMILY",
    "STATE_VERSION",
    "SUMMARIES_SCHEMA",
    "L2InputMissing",
    "L2StateInconsistent",
    "PhaseQuantityMismatch",
    "PhaseTicksMismatch",
    "PhaseValueError",
    "Reconciliation",
    "SkeletonMismatch",
    "SummarizeError",
    "ThetaResult",
    "summarize_month",
]
