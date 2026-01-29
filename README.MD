# Intrinseca

> Motor de alto rendimiento para análisis **Directional Change (DC)** en mercados financieros.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)]()

---

## Qué es Directional Change

El paradigma **Directional Change (DC)** transforma series temporales de precios en secuencias de eventos discretos. A diferencia del análisis tradicional basado en intervalos fijos (velas de 1m, 1h, 1d), DC detecta cambios de tendencia cuando el precio se mueve un porcentaje significativo (**θ**).

```
Precio tradicional (tiempo fijo):     ████████████████████████████
Eventos DC (movimiento fijo):         ▲────▼────▲──▼──────▲───────▼
```

**Ventajas:**
- Captura la microestructura real del mercado
- Invariante al tiempo (elimina ruido de baja volatilidad)
- Base teórica sólida para indicadores intrínsecos

---

## Arquitectura

Intrinseca implementa una arquitectura **Medallion** (Bronze → Silver → Gold):

```
┌─────────────────────────────────────────────────────────────────┐
│                         BRONZE LAYER                            │
│              Ticks crudos (price, time, qty, dir)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENGINE + KERNEL                             │
│           Numba JIT · Zero-Copy · Regla Conservadora            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SILVER LAYER                            │
│     Eventos DC anidados (Parquet + Arrow ListArrays)            │
│     • price_dc[], time_dc[] (fase DC)                           │
│     • price_os[], time_os[] (fase Overshoot)                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                          GOLD LAYER                             │
│              Indicadores calculados (Polars Expressions)        │
│     Duration · Velocity · Overshoot · Return · TMV · Runs       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Instalación

```bash
# Solo núcleo (para pipelines de datos)
pip install intrinseca

# Con visualización estática (matplotlib)
pip install intrinseca[plot]

# Con dashboards interactivos (Panel + HoloViews)
pip install intrinseca[interactive]

# Todo incluido
pip install intrinseca[full]
```

---

## Uso Rápido

### Procesar datos Bronze → Silver

```python
from intrinseca.core import Engine
from pathlib import Path

# Configurar engine
engine = Engine(
    theta=0.005,  # Umbral DC del 0.5%
    silver_base_path=Path("./data/silver")
)

# Procesar un mes de datos
results, convergence = engine.process_date_range(
    df_bronze=df_ticks,  # DataFrame Polars con columnas: time, price, quantity, direction
    ticker="BTCUSDT",
    analyze_convergence=True  # Opcional: verificar consistencia con ejecuciones previas
)

print(f"Días procesados: {len(results)}")
```

### Calcular indicadores (Gold Layer)

```python
from intrinseca.indicators import IndicatorRegistry
from intrinseca.indicators.metrics import register_all
import polars as pl

# Cargar datos Silver
df_silver = pl.read_parquet("./data/silver/BTCUSDT/theta=0.005/year=2025/month=11/day=28/data.parquet")

# Crear registry y registrar métricas
registry = IndicatorRegistry()
register_all(registry)

# Calcular indicadores
df_events = registry.compute(df_silver, indicators=[
    "duration_ns",
    "velocity",
    "overshoot",
    "return",
    "runs_count"
])

print(df_events)
```

### Indicadores de agregación (resumen global)

```python
from intrinseca.indicators.metrics import TMV, AvgDuration, VolatilityDC

# Métricas globales
tmv = df_events.select(TMV().get_expression().alias("tmv")).item()
avg_duration = df_events.select(AvgDuration().get_expression()).item()
volatility = df_events.select(VolatilityDC().get_expression()).item()

print(f"TMV: {tmv:.4f}")
print(f"Duración promedio: {avg_duration / 1e9:.2f} segundos")
print(f"Volatilidad DC: {volatility:.6f}")
```

---

## Indicadores Disponibles

### Event-Level (por evento)

| Indicador | Descripción |
|-----------|-------------|
| `duration_ns` | Duración de la fase DC en nanosegundos |
| `velocity` | Velocidad del cambio de precio (price/time) |
| `overshoot` | Magnitud del movimiento DC (Extreme → Confirmation) |
| `return` | Retorno de la fase DC |
| `os_return` | Retorno de la fase Overshoot |
| `runs_count` | Número de cruces de grilla durante el evento |

### Aggregation-Level (globales)

| Indicador | Descripción |
|-----------|-------------|
| `tmv` | Total Movement Value (suma de retornos absolutos) |
| `avg_duration` | Duración promedio de eventos |
| `avg_return` | Retorno promedio |
| `avg_overshoot` | Overshoot promedio |
| `volatility_dc` | Volatilidad medida como std de retornos DC |
| `upturn_ratio` | Proporción de eventos upturn vs total |

---

## Features Técnicas

### Kernel Numba JIT
- Compilación ahead-of-time para máximo rendimiento
- Zero-copy entre NumPy y Arrow
- Procesamiento de millones de ticks por segundo

### Regla Conservadora
Cuando múltiples ticks tienen el mismo timestamp:
- **Upturn**: selecciona el precio MÍNIMO que cruza el threshold
- **Downturn**: selecciona el precio MÁXIMO que cruza el threshold

Esto garantiza resultados determinísticos y conservadores.

### State Stitching
Continuidad matemática entre días de procesamiento:
- Estado persistido en Arrow
- Ticks huérfanos se integran al siguiente día
- Garantiza detección correcta de eventos cross-day

### Análisis de Convergencia
Verifica consistencia entre ejecuciones:
```python
results, report = engine.process_date_range(
    df_bronze, ticker,
    analyze_convergence=True,
    strict_comparison=True,  # Tolerancia 0 ns
)

if report.converged:
    print("Datos consistentes con ejecución anterior")
```

---

## Estructura del Proyecto

```
intrinseca/
├── src/intrinseca/
│   ├── core/                    # Motor de alto rendimiento
│   │   ├── engine.py            # Orquestador Bronze → Silver
│   │   ├── kernel.py            # Kernel Numba JIT
│   │   ├── state.py             # Manejo de estado (stitching)
│   │   └── convergence.py       # Análisis de convergencia
│   │
│   ├── indicators/              # Sistema de indicadores modular
│   │   ├── base.py              # Clase base BaseIndicator
│   │   ├── registry.py          # Registry para composición
│   │   └── metrics/
│   │       ├── core.py          # EventTime, EventPrice, etc.
│   │       ├── dynamics.py      # Duration, Velocity
│   │       ├── geometry.py      # Overshoot, Return
│   │       ├── microstructure.py # RunsCount
│   │       └── aggregation.py   # TMV, AvgDuration, etc.
│   │
│   └── visualization/           # Visualización (opcional)
│       ├── static_plots.py      # Gráficos Matplotlib
│       └── interactive.py       # Dashboards Panel/HoloViews
│
├── playground_silver.py         # Script de procesamiento Silver
├── playground_gold.py           # Script de análisis Gold
└── pyproject.toml
```

---

## Dependencias Principales

| Paquete | Uso |
|---------|-----|
| `polars` | DataFrames de alto rendimiento |
| `pyarrow` | Formato columnar y ListArrays |
| `numba` | Compilación JIT del kernel |
| `numpy` | Arrays numéricos |
| `rich` | Progress bars y logging |

---

## Roadmap

- [ ] Streaming detector para live trading
- [ ] Integración con InfluxDB/TimescaleDB
- [ ] Backtesting framework
- [ ] REST API para consultas
- [ ] Soporte multi-asset

---

## Licencia

MIT

---

## Referencias

- Guillaume, D. M., et al. (1997). "From the bird's eye to the microscope: A survey of new stylized facts of the intra-daily foreign exchange markets."
- Glattfelder, J. B., et al. (2011). "Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws."
