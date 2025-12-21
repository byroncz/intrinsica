"""
Indicadores derivados del análisis Directional Change.

Métricas cuantitativas para caracterizar la dinámica del mercado
basadas en los eventos DC detectados.

Indicadores principales:
    - TMV (Total Movement Variable): Suma de retornos absolutos DC
    - T (Time): Duración promedio entre eventos
    - OS (Overshoot): Extensión del movimiento más allá de theta
    - R (Return): Retorno promedio por evento DC
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from intrinseca.core.event_detector import DCResult


@dataclass(frozen=True, slots=True)
class DCMetrics:
    """
    Métricas agregadas de un análisis DC.
    
    Attributes:
        tmv: Total Movement Variable (suma de |retornos DC|).
        avg_duration: Duración promedio entre eventos (en observaciones).
        avg_return: Retorno promedio por evento DC.
        avg_overshoot: Overshoot promedio.
        n_events: Número total de eventos.
        upturn_ratio: Proporción de upturns vs total.
        volatility_dc: Volatilidad medida en unidades DC.
    """
    
    tmv: float
    avg_duration: float
    avg_return: float
    avg_overshoot: float
    n_events: int
    upturn_ratio: float
    volatility_dc: float
    
    def to_dict(self) -> dict:
        """Convierte métricas a diccionario."""
        return {
            "tmv": self.tmv,
            "avg_duration": self.avg_duration,
            "avg_return": self.avg_return,
            "avg_overshoot": self.avg_overshoot,
            "n_events": self.n_events,
            "upturn_ratio": self.upturn_ratio,
            "volatility_dc": self.volatility_dc
        }


@njit(cache=True, fastmath=True)
def _compute_rolling_metrics(
    event_indices: NDArray[np.int64],
    event_returns: NDArray[np.float64],
    overshoots: NDArray[np.float64],
    window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calcula métricas DC en ventana móvil.
    
    Returns:
        rolling_tmv, rolling_avg_duration, rolling_avg_os
    """
    n = len(event_indices)
    
    rolling_tmv = np.zeros(n, dtype=np.float64)
    rolling_duration = np.zeros(n, dtype=np.float64)
    rolling_os = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        start = max(0, i - window + 1)
        
        # TMV: suma de retornos absolutos
        # En la ventana [start, i]
        rolling_tmv[i] = np.sum(np.abs(event_returns[start : i + 1]))
        
        # Duración promedio
        # Diferencia de índices/tiempos entre eventos en la ventana
        if i > start:
            # La duración ya viene calculada en event_durations, pero si queremos recalcular
            # basado en indices puros (frecuencia de sampling):
            # Aquí asumimos que queremos la media de la distancia entre eventos OBSERVADOS.
            # Pero event_indices[j] - event_indices[j-1] es una aproximación de T.
            # Mejor usar diffs de indices directos.
            
            # Sin embargo, para mantener coherencia con la implementación anterior:
            count = 0
            dur_sum = 0.0
            for j in range(start + 1, i + 1):
                dur_sum += event_indices[j] - event_indices[j - 1]
                count += 1
            rolling_duration[i] = dur_sum / count if count > 0 else 0.0
        
        # Overshoot promedio
        rolling_os[i] = np.mean(overshoots[start : i + 1])
    
    return rolling_tmv, rolling_duration, rolling_os


class DCIndicators:
    """
    Calculadora de indicadores Directional Change.
    
    Transforma eventos DC en métricas cuantitativas para análisis
    y construcción de features para modelos.
    
    Example:
        >>> from intrinseca import DCDetector, DCIndicators
        >>> detector = DCDetector(theta=0.01)
        >>> result = detector.detect(prices)
        >>> indicators = DCIndicators()
        >>> metrics = indicators.compute_metrics(result)
        >>> print(f"TMV: {metrics.tmv:.4f}")
    """
    
    def compute_metrics(self, result: DCResult) -> DCMetrics:
        """
        Calcula métricas agregadas de un resultado DC.
        
        Args:
            result: Resultado de DCDetector.detect()
        
        Returns:
            DCMetrics con indicadores calculados.
        """
        if result.n_events == 0:
            return DCMetrics(
                tmv=0.0,
                avg_duration=0.0,
                avg_return=0.0,
                avg_overshoot=0.0,
                n_events=0,
                upturn_ratio=0.0,
                volatility_dc=0.0
            )
        
        # Extraer arrays directamente del resultado (Zero-Copy)
        returns = result.returns_price
        durations = result.event_durations
        overshoots = result.overshoots
        types = result.event_types
        
        # TMV (Total Movement Variable)
        tmv = np.sum(np.abs(returns))
        
        # Promedios
        avg_duration = np.mean(durations) if len(durations) > 0 else 0.0
        avg_return = np.mean(returns) if len(returns) > 0 else 0.0
        avg_overshoot = np.mean(overshoots) if len(overshoots) > 0 else 0.0
        
        # Ratio de upturns (1 = upturn, -1 = downturn)
        n_upturns = np.sum(types == 1)
        upturn_ratio = n_upturns / result.n_events
        
        # Volatilidad DC (desviación estándar de retornos DC)
        volatility_dc = np.std(returns) if len(returns) > 1 else 0.0
        
        return DCMetrics(
            tmv=float(tmv),
            avg_duration=float(avg_duration),
            avg_return=float(avg_return),
            avg_overshoot=float(avg_overshoot),
            n_events=result.n_events,
            upturn_ratio=float(upturn_ratio),
            volatility_dc=float(volatility_dc)
        )
    
    def compute_rolling_metrics(
        self,
        result: DCResult,
        window: int = 20
    ) -> dict[str, NDArray[np.float64]]:
        """
        Calcula métricas DC en ventana móvil sobre eventos.
        
        Útil para detectar cambios en la dinámica del mercado.
        
        Args:
            result: Resultado de DCDetector.detect()
            window: Número de eventos en la ventana.
        
        Returns:
            Diccionario con arrays de métricas rolling.
        """
        if result.n_events < window:
            return {
                "rolling_tmv": np.array([]),
                "rolling_duration": np.array([]),
                "rolling_overshoot": np.array([])
            }
        
        # Pasar arrays numba-friendly
        rolling_tmv, rolling_duration, rolling_os = _compute_rolling_metrics(
            result.event_indices, 
            result.returns_price, 
            result.overshoots, 
            window
        )
        
        return {
            "rolling_tmv": rolling_tmv,
            "rolling_duration": rolling_duration,
            "rolling_overshoot": rolling_os
        }
    
    def extract_features(
        self,
        result: DCResult,
        n_last_events: int = 10
    ) -> NDArray[np.float64]:
        """
        Extrae vector de features para modelos ML.
        
        Genera un vector numérico a partir de los últimos N eventos DC,
        listo para alimentar a un modelo de machine learning.
        
        Args:
            result: Resultado de DCDetector.detect()
            n_last_events: Número de eventos recientes a considerar.
        
        Returns:
            Array 1D con features concatenadas.
        """
        features = []
        
        # Métricas globales
        metrics = self.compute_metrics(result)
        features.extend([
            metrics.tmv,
            metrics.avg_duration,
            metrics.avg_return,
            metrics.avg_overshoot,
            metrics.upturn_ratio,
            metrics.volatility_dc
        ])
        
        # Features de últimos N eventos
        if result.n_events >= n_last_events:
            # Slicing directo de numpy arrays
            last_returns = result.returns_price[-n_last_events:]
            features.extend(last_returns)
            
            # Duraciones normalizadas
            # Usamos event_durations del result
            last_dur_raw = result.event_durations[-n_last_events:]
            if metrics.avg_duration > 0:
                last_durations = last_dur_raw / metrics.avg_duration
            else:
                last_durations = np.zeros_like(last_dur_raw, dtype=np.float64)
            features.extend(last_durations)
            
            # Tipo de evento (1=upturn, -1=downturn)
            # Ya viene como 1/-1 en result.event_types, solo casteamos a float si es necesario
            last_types = result.event_types[-n_last_events:]
            features.extend(last_types)
        else:
            # Padding con ceros si no hay suficientes eventos
            padding_size = n_last_events * 3  # returns, durations, types
            features.extend([0.0] * padding_size)
        
        return np.array(features, dtype=np.float64)
    
    def compute_coastline(
        self,
        prices: NDArray[np.float64],
        thetas: list[float]
    ) -> dict[float, float]:
        """
        Calcula la 'coastline' del mercado a múltiples escalas.
        
        La coastline es el TMV total para diferentes valores de theta.
        Revela la estructura fractal del mercado.
        
        Args:
            prices: Array de precios.
            thetas: Lista de umbrales a evaluar.
        
        Returns:
            Diccionario {theta: tmv}.
        """
        # Importación tardía para evitar ciclos, aunque con el refactor podría no ser necesario
        # si DCDetector ya no depende de indicators.
        from intrinseca.core.event_detector import _dc_core_algorithm_optimized
        
        coastline = {}
        # Timestamps dummy para cumplir firma (no afectan TMV)
        # Asumimos que prices es solo array de precios, creamos dummy timestamps
        dummy_ts = np.arange(len(prices), dtype=np.int64)
        
        for theta in thetas:
            # Ejecución kernel
            (
                _, _, _, _, _,
                _, _, _,
                _, ret_pr, _, _,
                _, _,
                _
            ) = _dc_core_algorithm_optimized(prices, dummy_ts, float(theta))
            
            # Calculo TMV manual rápido
            tmv = np.sum(np.abs(ret_pr))
            coastline[theta] = float(tmv)
        
        return coastline