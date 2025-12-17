# Intrinseca

> Motor de Transformación Directional Change (DC) de alto rendimiento para trading algorítmico.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ¿Qué es Directional Change?

El paradigma **Directional Change (DC)** transforma series temporales de precios en secuencias de eventos discretos, capturando la microestructura del mercado de manera invariante al tiempo. A diferencia del análisis tradicional basado en intervalos fijos (velas de 1 minuto, 1 hora, etc.), DC detecta cambios de tendencia cuando el precio se mueve un porcentaje significativo (theta).

## Instalación

### Solo núcleo (para live-trading)

```bash
pip install intrinseca
```

### Con visualización

```bash
pip install intrinseca[plot]
```

### Con gráficos interactivos

```bash
pip install intrinseca[interactive]
```

### Todo incluido

```bash
pip install intrinseca[full]
```

## Uso Rápido

### Detección de eventos DC

```python
import numpy as np
from intrinseca import DCDetector, DCIndicators

# Datos de ejemplo
prices = np.random.randn(1000).cumsum() + 100

# Crear detector con umbral del 1%
detector = DCDetector(theta=0.01)

# Detectar eventos
result = detector.detect(prices)

print(f"Eventos detectados: {result.n_events}")
print(f"Upturns: {result.n_upturns}")
print(f"Downturns: {result.n_downturns}")

# Calcular métricas
indicators = DCIndicators()
metrics = indicators.compute_metrics(result)
print(f"TMV: {metrics.tmv:.4f}")
```

### Detección en tiempo real (streaming)

```python
detector = DCDetector(theta=0.01)
state = None

for price in live_price_feed:
    event, state = detector.detect_streaming(price, state)
    if event:
        print(f"Evento: {event.event_type} en precio {event.price}")
```

### Visualización

```python
from intrinseca.visualization import plot_dc_events

fig = plot_dc_events(prices, result)
fig.savefig("dc_analysis.png", dpi=150)
```

## Estructura del Proyecto

```
intrinseca/
├── pyproject.toml
├── src/
│   └── intrinseca/
│       ├── __init__.py
│       ├── core/                   # Núcleo de cálculo (High Performance)
│       │   ├── event_detector.py   # Algoritmo DC optimizado con Numba
│       │   ├── indicators.py       # Cálculo de TMV, T, OS, R
│       │   └── bridging.py         # Puente Polars <-> Numpy
│       └── visualization/          # Módulo opcional
│           ├── static_plots.py     # Gráficos Matplotlib
│           └── interactive.py      # Gráficos Plotly
└── tests/
```

## Licencia

MIT

```

---

## Archivo adicional: `src/intrinseca/py.typed`

Crea este archivo vacío para indicar que la librería tiene type hints:
```

# Este archivo indica que intrinseca soporta type hints (PEP 561)
