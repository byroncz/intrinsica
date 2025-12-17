lib-dc-analytics/
├── pyproject.toml
├── src/
│   └── dc_analytics/
│       ├── __init__.py
│       ├── core/                   # Núcleo de cálculo (High Performance)
│       │   ├── __init__.py
│       │   ├── event_detector.py   # Algoritmo DC optimizado con Numba
│       │   ├── indicators.py       # Cálculo de TMV, T, OS, R
│       │   └── bridging.py         # Puente Polars <-> Numpy
│       └── visualization/          # Módulo opcional
│           ├── __init__.py
│           ├── static_plots.py     # Gráficos Matplotlib/Seaborn
│           └── interactive.py      # Gráficos Plotly (opcional)
└── tests/
    ├── test_core_numba.py          # Tests de la lógica matemática
    └── test_visualization.py       # Tests de generación de figuras