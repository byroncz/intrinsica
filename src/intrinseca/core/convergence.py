"""
Análisis de Convergencia para Silver Layer.

Proporciona herramientas para comparar series de eventos DC antes y después
de un reprocesamiento, detectando discrepancias y puntos de convergencia.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl


@dataclass
class ConvergenceResult:
    """Resultado del análisis de convergencia entre dos series DC para un día."""
    
    # Identificadores
    ticker: str
    theta: float
    day: date
    
    # Métricas de discrepancia
    n_events_prev: int
    n_events_new: int
    n_discrepant_events: int
    first_discrepancy_idx: int  # -1 si no hay discrepancia
    convergence_idx: Optional[int]  # None si no convergió
    
    # Flags
    converged: bool
    requires_forward_processing: bool
    
    # Detalles opcionales
    discrepancy_details: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d


@dataclass
class ConvergenceReport:
    """Reporte consolidado de convergencia para un rango de fechas."""
    
    ticker: str
    theta: float
    results: dict[str, ConvergenceResult] = field(default_factory=dict)  # date.isoformat -> result
    global_convergence_date: Optional[date] = None
    total_discrepant_events: int = 0
    
    def add_result(self, result: ConvergenceResult) -> None:
        """Agrega un resultado diario al reporte."""
        self.results[result.day.isoformat()] = result
        self.total_discrepant_events += result.n_discrepant_events
        
        if result.converged and self.global_convergence_date is None:
            self.global_convergence_date = result.day
    
    @property
    def converged(self) -> bool:
        """Indica si se alcanzó convergencia global."""
        return self.global_convergence_date is not None
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            "ticker": self.ticker,
            "theta": self.theta,
            "global_convergence_date": (
                self.global_convergence_date.isoformat() 
                if self.global_convergence_date else None
            ),
            "total_discrepant_events": self.total_discrepant_events,
            "converged": self.converged,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }
    
    def save(self, path: Path) -> None:
        """Guarda el reporte como JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def generate_summary(self) -> str:
        """Genera un resumen legible del reporte."""
        if not self.results:
            return "Sin resultados de convergencia."
        
        lines = [
            f"📊 Reporte de Convergencia: {self.ticker} (θ={self.theta})",
            f"   Total días analizados: {len(self.results)}",
            f"   Total eventos discrepantes: {self.total_discrepant_events}",
        ]
        
        if self.converged:
            lines.append(f"   ✅ Convergencia alcanzada: {self.global_convergence_date}")
        else:
            lines.append("   ⚠️ No se alcanzó convergencia en el período")
        
        return "\n".join(lines)


def compare_dc_events(
    df_prev: pl.DataFrame,
    df_new: pl.DataFrame,
    ticker: str,
    theta: float,
    day: date,
    strict_comparison: bool = True,
    tolerance_ns: int = 0,
) -> ConvergenceResult:
    """
    Compara dos series de eventos DC y detecta convergencia.
    
    Args:
        df_prev: DataFrame con eventos del procesamiento anterior
        df_new: DataFrame con eventos del nuevo procesamiento
        ticker: Símbolo del instrumento
        theta: Umbral del algoritmo DC
        day: Fecha siendo analizada
        strict_comparison: Si True, comparación exacta (0 ns tolerancia)
        tolerance_ns: Tolerancia en nanosegundos si strict_comparison=False
    
    Returns:
        ConvergenceResult con métricas de discrepancia y convergencia
    """
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
        )
    
    # Determinar tolerancia efectiva
    tol = 0 if strict_comparison else tolerance_ns
    
    # Extraer timestamps del primer tick DC de cada evento
    prev_times = df_prev.select(
        pl.col("time_dc").list.first().alias("t")
    )["t"].to_numpy()
    
    new_times = df_new.select(
        pl.col("time_dc").list.first().alias("t")
    )["t"].to_numpy()
    
    prev_types = df_prev["event_type"].to_numpy()
    new_types = df_new["event_type"].to_numpy()
    
    # Buscar discrepancias y convergencia
    first_discrepancy_idx = -1
    convergence_idx = None
    n_discrepant = 0
    discrepancy_details = []
    
    min_len = min(n_prev, n_new)
    in_discrepancy_zone = False
    
    for i in range(min_len):
        time_match = abs(prev_times[i] - new_times[i]) <= tol
        type_match = prev_types[i] == new_types[i]
        events_equal = time_match and type_match
        
        if not events_equal:
            n_discrepant += 1
            in_discrepancy_zone = True
            
            if first_discrepancy_idx == -1:
                first_discrepancy_idx = i
            
            discrepancy_details.append({
                "index": i,
                "prev_time": int(prev_times[i]),
                "new_time": int(new_times[i]),
                "prev_type": int(prev_types[i]),
                "new_type": int(new_types[i]),
            })
        
        elif in_discrepancy_zone and events_equal:
            # Encontramos convergencia
            convergence_idx = i
            break
    
    # Si no hubo discrepancias, convergencia desde el inicio
    if first_discrepancy_idx == -1:
        convergence_idx = 0
    
    converged = convergence_idx is not None
    
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
        discrepancy_details=discrepancy_details[:10],  # Limitar a 10 para no sobrecargar
    )
