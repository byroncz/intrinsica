"""Análisis de Convergencia para Silver Layer.

Proporciona herramientas para comparar series de eventos DC antes y después
de un reprocesamiento, detectando discrepancias y puntos de convergencia.

Conceptos Clave:
```
Serie Previa:    [E0] [E1] [E2] [E3] [E4] [E5] [E6]
Serie Nueva:     [E0] [E1] [E2*] [E3*] [E4] [E5] [E6]
                        ↑           ↑
                  first_discrepancy  convergence_idx

Discrepancia: Eventos donde timestamp o tipo difieren
Convergencia: Punto donde las series vuelven a coincidir
```

Uso típico:
```python
result = compare_dc_events(df_prev, df_new, "BTCUSDT", 0.005, date(2025, 11, 28))
if result.converged:
    print(f"Convergió en evento {result.convergence_idx}")
else:
    print(f"No convergió, {result.n_discrepant_events} discrepancias")
```
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

# =============================================================================
# TYPE ALIASES
# =============================================================================

ArrayI64 = NDArray[np.int64]
ArrayI8 = NDArray[np.int8]


# =============================================================================
# CONSTANTES
# =============================================================================

# Máximo de detalles de discrepancia a almacenar (evita memoria excesiva)
MAX_DISCREPANCY_DETAILS = 10

# Tolerancia por defecto en nanosegundos para comparación no-estricta
DEFAULT_TOLERANCE_NS = 1_000_000  # 1ms


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================


class DiscrepancyDetail(NamedTuple):
    """Detalle de una discrepancia entre eventos."""

    index: int
    prev_time: int
    new_time: int
    prev_type: int
    new_type: int
    time_diff_ns: int

    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            "index": self.index,
            "prev_time": self.prev_time,
            "new_time": self.new_time,
            "prev_type": self.prev_type,
            "new_type": self.new_type,
            "time_diff_ns": self.time_diff_ns,
        }


@dataclass
class ConvergenceResult:
    """Resultado del análisis de convergencia entre dos series DC para un día.

    Attributes:
    ----------
        ticker: Símbolo del instrumento
        theta: Umbral DC
        day: Fecha analizada
        n_events_prev: Número de eventos en serie previa
        n_events_new: Número de eventos en serie nueva
        n_discrepant_events: Cantidad de eventos que difieren
        first_discrepancy_idx: Índice del primer evento diferente (-1 si no hay)
        convergence_idx: Índice donde las series convergen (None si no convergen)
        converged: True si las series convergen en algún punto
        requires_forward_processing: True si se necesita procesar días posteriores
        analysis_applicable: False si no había datos previos para comparar
        discrepancy_details: Lista de detalles de las primeras discrepancias
        analysis_time_ms: Tiempo de análisis en milisegundos

    """

    # Identificadores
    ticker: str
    theta: float
    day: date

    # Métricas de discrepancia
    n_events_prev: int
    n_events_new: int
    n_discrepant_events: int
    first_discrepancy_idx: int  # -1 si no hay discrepancia
    convergence_idx: int | None  # None si no convergió

    # Flags
    converged: bool
    requires_forward_processing: bool
    analysis_applicable: bool = True

    # Detalles opcionales
    discrepancy_details: list[DiscrepancyDetail] = field(default_factory=list)
    analysis_time_ms: float = 0.0

    @property
    def n_events_diff(self) -> int:
        """Diferencia en número de eventos."""
        return abs(self.n_events_new - self.n_events_prev)

    @property
    def discrepancy_rate(self) -> float:
        """Tasa de discrepancia (0-1)."""
        total = max(self.n_events_prev, self.n_events_new)
        return self.n_discrepant_events / total if total > 0 else 0.0

    @property
    def is_perfect_match(self) -> bool:
        """True si ambas series son idénticas."""
        return (
            self.converged
            and self.n_discrepant_events == 0
            and self.n_events_prev == self.n_events_new
        )

    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            "ticker": self.ticker,
            "theta": self.theta,
            "day": self.day.isoformat(),
            "n_events_prev": self.n_events_prev,
            "n_events_new": self.n_events_new,
            "n_discrepant_events": self.n_discrepant_events,
            "first_discrepancy_idx": self.first_discrepancy_idx,
            "convergence_idx": self.convergence_idx,
            "converged": self.converged,
            "requires_forward_processing": self.requires_forward_processing,
            "analysis_applicable": self.analysis_applicable,
            "discrepancy_details": [d.to_dict() for d in self.discrepancy_details],
            "analysis_time_ms": self.analysis_time_ms,
            # Propiedades calculadas
            "n_events_diff": self.n_events_diff,
            "discrepancy_rate": self.discrepancy_rate,
            "is_perfect_match": self.is_perfect_match,
        }

    def summary(self) -> str:
        """Genera resumen de una línea."""
        if not self.analysis_applicable:
            return f"{self.day}: N/A (sin datos previos)"

        if self.is_perfect_match:
            return f"{self.day}: ✅ Match perfecto ({self.n_events_new} eventos)"

        if self.converged:
            return (
                f"{self.day}: ✅ Convergió en idx={self.convergence_idx} "
                f"({self.n_discrepant_events} discrepancias)"
            )

        return (
            f"{self.day}: ⚠️ No convergió "
            f"({self.n_discrepant_events} discrepancias, "
            f"prev={self.n_events_prev}, new={self.n_events_new})"
        )


@dataclass
class ConvergenceReport:
    """Reporte consolidado de convergencia para un rango de fechas.

    Attributes:
    ----------
        ticker: Símbolo del instrumento
        theta: Umbral DC
        results: Diccionario de resultados por fecha
        global_convergence_date: Primera fecha donde se alcanzó convergencia
        total_discrepant_events: Suma de discrepancias en todos los días

    """

    ticker: str
    theta: float
    results: dict[str, ConvergenceResult] = field(default_factory=dict)
    global_convergence_date: date | None = None
    total_discrepant_events: int = 0

    def add_result(self, result: ConvergenceResult) -> None:
        """Agrega un resultado diario al reporte.

        Args:
        ----
            result: Resultado de convergencia para un día

        """
        key = result.day.isoformat()
        self.results[key] = result
        self.total_discrepant_events += result.n_discrepant_events

        if result.converged and self.global_convergence_date is None:
            self.global_convergence_date = result.day

    @property
    def converged(self) -> bool:
        """Indica si se alcanzó convergencia global."""
        return self.global_convergence_date is not None

    @property
    def n_days_total(self) -> int:
        """Número total de días en el reporte."""
        return len(self.results)

    @property
    def n_days_without_prev_data(self) -> int:
        """Número de días donde no había datos previos."""
        return sum(1 for r in self.results.values() if not r.analysis_applicable)

    @property
    def n_days_analyzed(self) -> int:
        """Número de días donde sí se realizó análisis."""
        return sum(1 for r in self.results.values() if r.analysis_applicable)

    @property
    def n_days_with_discrepancies(self) -> int:
        """Número de días con al menos una discrepancia."""
        return sum(
            1 for r in self.results.values() if r.analysis_applicable and r.n_discrepant_events > 0
        )

    @property
    def n_perfect_matches(self) -> int:
        """Número de días con match perfecto."""
        return sum(1 for r in self.results.values() if r.is_perfect_match)

    @property
    def avg_discrepancy_rate(self) -> float:
        """Tasa de discrepancia promedio."""
        rates = [r.discrepancy_rate for r in self.results.values() if r.analysis_applicable]
        return sum(rates) / len(rates) if rates else 0.0

    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            "ticker": self.ticker,
            "theta": self.theta,
            "global_convergence_date": (
                self.global_convergence_date.isoformat() if self.global_convergence_date else None
            ),
            "total_discrepant_events": self.total_discrepant_events,
            "converged": self.converged,
            # Métricas agregadas
            "n_days_total": self.n_days_total,
            "n_days_analyzed": self.n_days_analyzed,
            "n_days_without_prev_data": self.n_days_without_prev_data,
            "n_days_with_discrepancies": self.n_days_with_discrepancies,
            "n_perfect_matches": self.n_perfect_matches,
            "avg_discrepancy_rate": self.avg_discrepancy_rate,
            # Resultados por día
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

    def save(self, path: Path) -> int:
        """Guarda el reporte como JSON.

        Args:
        ----
            path: Ruta de destino

        Returns:
        -------
            Tamaño del archivo en bytes

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path.stat().st_size

    def generate_summary(self) -> str:
        """Genera un resumen legible del reporte."""
        if not self.results:
            return "Sin resultados de convergencia."

        lines = [
            f"{'=' * 60}",
            f"📊 Reporte de Convergencia: {self.ticker} (θ={self.theta})",
            f"{'=' * 60}",
            "",
            f"📅 Días procesados: {self.n_days_total}",
        ]

        if self.n_days_without_prev_data > 0:
            lines.append(f"   🆕 Sin datos previos: {self.n_days_without_prev_data}")

        if self.n_days_analyzed > 0:
            lines.extend(
                [
                    f"   🔍 Analizados: {self.n_days_analyzed}",
                    f"   ✅ Match perfecto: {self.n_perfect_matches}",
                    f"   ⚠️ Con discrepancias: {self.n_days_with_discrepancies}",
                    "",
                    "📈 Métricas:",
                    f"   Total discrepancias: {self.total_discrepant_events}",
                    f"   Tasa promedio: {self.avg_discrepancy_rate:.2%}",
                ]
            )

            if self.converged:
                lines.append(f"   🎯 Convergencia global: {self.global_convergence_date}")
            else:
                lines.append("   ❌ No se alcanzó convergencia")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def get_worst_days(self, n: int = 5) -> list[ConvergenceResult]:
        """Retorna los N días con más discrepancias.

        Args:
        ----
            n: Número de días a retornar

        Returns:
        -------
            Lista de resultados ordenados por discrepancias (descendente)

        """
        applicable = [r for r in self.results.values() if r.analysis_applicable]
        sorted_results = sorted(applicable, key=lambda r: r.n_discrepant_events, reverse=True)
        return sorted_results[:n]


# =============================================================================
# FUNCIONES DE COMPARACIÓN
# =============================================================================


def compare_dc_events(
    df_prev: pl.DataFrame,
    df_new: pl.DataFrame,
    ticker: str,
    theta: float,
    day: date,
    strict_comparison: bool = True,
    tolerance_ns: int = 0,
) -> ConvergenceResult:
    """Compara dos series de eventos DC y detecta convergencia.

    Implementación vectorizada con numpy para máximo rendimiento.

    Args:
    ----
        df_prev: DataFrame con eventos del procesamiento anterior
        df_new: DataFrame con eventos del nuevo procesamiento
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC
        day: Fecha siendo analizada
        strict_comparison: Si True, comparación exacta (0 ns tolerancia)
        tolerance_ns: Tolerancia en nanosegundos si strict_comparison=False

    Returns:
    -------
        ConvergenceResult con métricas de discrepancia y convergencia

    Example:
    -------
        >>> result = compare_dc_events(df_prev, df_new, "BTCUSDT", 0.005, date(2025, 11, 28))
        >>> print(result.summary())

    """
    t_start = time.perf_counter()

    n_prev = len(df_prev)
    n_new = len(df_new)

    # Caso especial: sin datos previos
    if n_prev == 0:
        return ConvergenceResult(
            ticker=ticker,
            theta=theta,
            day=day,
            n_events_prev=0,
            n_events_new=n_new,
            n_discrepant_events=0,
            first_discrepancy_idx=-1,
            convergence_idx=0,
            converged=True,
            requires_forward_processing=False,
            analysis_time_ms=(time.perf_counter() - t_start) * 1000,
        )

    # Caso especial: nuevos datos vacíos
    if n_new == 0:
        return ConvergenceResult(
            ticker=ticker,
            theta=theta,
            day=day,
            n_events_prev=n_prev,
            n_events_new=0,
            n_discrepant_events=n_prev,
            first_discrepancy_idx=0,
            convergence_idx=None,
            converged=False,
            requires_forward_processing=True,
            analysis_time_ms=(time.perf_counter() - t_start) * 1000,
        )

    # Determinar tolerancia efectiva
    tol = 0 if strict_comparison else tolerance_ns

    # Extraer arrays una sola vez (evitar múltiples to_numpy)
    prev_times: ArrayI64 = df_prev.select(pl.col("time_dc").list.first().alias("t"))["t"].to_numpy()

    new_times: ArrayI64 = df_new.select(pl.col("time_dc").list.first().alias("t"))["t"].to_numpy()

    prev_types: ArrayI8 = df_prev["event_type"].to_numpy()
    new_types: ArrayI8 = df_new["event_type"].to_numpy()

    # Comparación vectorizada
    min_len = min(n_prev, n_new)

    # Calcular diferencias de tiempo y coincidencias de tipo
    time_diffs = np.abs(prev_times[:min_len] - new_times[:min_len])
    time_matches = time_diffs <= tol
    type_matches = prev_types[:min_len] == new_types[:min_len]
    events_equal = time_matches & type_matches

    # Encontrar discrepancias
    discrepancy_mask = ~events_equal
    discrepancy_indices = np.where(discrepancy_mask)[0]
    n_discrepant = len(discrepancy_indices)

    # Determinar first_discrepancy_idx y convergence_idx
    if n_discrepant == 0:
        # Sin discrepancias: match perfecto (hasta min_len)
        first_discrepancy_idx = -1
        convergence_idx = 0
        converged = True
    else:
        first_discrepancy_idx = int(discrepancy_indices[0])

        # Buscar convergencia: primer True después del primer False
        # Usar numpy para encontrar transición False -> True
        converged = False
        convergence_idx = None

        # Buscar después de la primera discrepancia
        search_start = first_discrepancy_idx + 1
        if search_start < min_len:
            # Buscar primer True después de search_start
            remaining = events_equal[search_start:]
            if np.any(remaining):
                # Encontrar primera posición True
                first_true = np.argmax(remaining)
                if remaining[first_true]:  # Verificar que realmente es True
                    convergence_idx = search_start + int(first_true)
                    converged = True

    # Construir detalles de discrepancia (limitados)
    discrepancy_details: list[DiscrepancyDetail] = []
    for idx in discrepancy_indices[:MAX_DISCREPANCY_DETAILS]:
        i = int(idx)
        discrepancy_details.append(
            DiscrepancyDetail(
                index=i,
                prev_time=int(prev_times[i]),
                new_time=int(new_times[i]),
                prev_type=int(prev_types[i]),
                new_type=int(new_types[i]),
                time_diff_ns=int(time_diffs[i]),
            )
        )

    return ConvergenceResult(
        ticker=ticker,
        theta=theta,
        day=day,
        n_events_prev=n_prev,
        n_events_new=n_new,
        n_discrepant_events=n_discrepant,
        first_discrepancy_idx=first_discrepancy_idx,
        convergence_idx=convergence_idx,
        converged=converged,
        requires_forward_processing=not converged,
        discrepancy_details=discrepancy_details,
        analysis_time_ms=(time.perf_counter() - t_start) * 1000,
    )


# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================


def load_report(path: Path) -> ConvergenceReport | None:
    """Carga un reporte de convergencia desde JSON.

    Args:
    ----
        path: Ruta al archivo JSON

    Returns:
    -------
        ConvergenceReport si existe, None en caso contrario

    Raises:
    ------
        ValueError: Si el archivo tiene formato inválido

    """
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Validar campos requeridos
    required = {"ticker", "theta", "results"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Campos faltantes en JSON: {missing}")

    # Reconstruir reporte
    report = ConvergenceReport(
        ticker=data["ticker"],
        theta=data["theta"],
    )

    # Reconstruir resultados
    for _, result_dict in data.get("results", {}).items():
        # Reconstruir discrepancy_details
        details = [DiscrepancyDetail(**d) for d in result_dict.get("discrepancy_details", [])]

        result = ConvergenceResult(
            ticker=result_dict["ticker"],
            theta=result_dict["theta"],
            day=date.fromisoformat(result_dict["day"]),
            n_events_prev=result_dict["n_events_prev"],
            n_events_new=result_dict["n_events_new"],
            n_discrepant_events=result_dict["n_discrepant_events"],
            first_discrepancy_idx=result_dict["first_discrepancy_idx"],
            convergence_idx=result_dict["convergence_idx"],
            converged=result_dict["converged"],
            requires_forward_processing=result_dict["requires_forward_processing"],
            analysis_applicable=result_dict.get("analysis_applicable", True),
            discrepancy_details=details,
            analysis_time_ms=result_dict.get("analysis_time_ms", 0.0),
        )
        report.add_result(result)

    # Restaurar convergence_date si estaba en el JSON
    if data.get("global_convergence_date"):
        report.global_convergence_date = date.fromisoformat(data["global_convergence_date"])

    return report


def compare_reports(
    report_a: ConvergenceReport,
    report_b: ConvergenceReport,
) -> dict:
    """Compara dos reportes de convergencia.

    Útil para comparar resultados con diferentes thetas o configuraciones.

    Args:
    ----
        report_a: Primer reporte
        report_b: Segundo reporte

    Returns:
    -------
        Dict con métricas de comparación

    """
    days_a = set(report_a.results.keys())
    days_b = set(report_b.results.keys())

    common_days = days_a & days_b
    only_a = days_a - days_b
    only_b = days_b - days_a

    # Comparar métricas en días comunes
    discrepancy_diff = []
    for day in common_days:
        diff = report_a.results[day].n_discrepant_events - report_b.results[day].n_discrepant_events
        discrepancy_diff.append(diff)

    return {
        "report_a": {"ticker": report_a.ticker, "theta": report_a.theta},
        "report_b": {"ticker": report_b.ticker, "theta": report_b.theta},
        "n_common_days": len(common_days),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "total_discrepancies_a": report_a.total_discrepant_events,
        "total_discrepancies_b": report_b.total_discrepant_events,
        "avg_discrepancy_diff": (
            sum(discrepancy_diff) / len(discrepancy_diff) if discrepancy_diff else 0.0
        ),
        "converged_a": report_a.converged,
        "converged_b": report_b.converged,
    }
