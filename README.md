# intrinsica

Intrinsica es un proyecto para el análisis de series de tiempo financieras
bajo el paradigma **Directional Change (DC)**, que detecta eventos de
tendencia por movimiento de precio en vez de por intervalos fijos de tiempo.
Este repositorio arranca desde cero para su rediseño; por ahora solo trae la
huella mínima del template [devkit](https://github.com/byroncz/intrinsica)
y este README.

La implementación académica v0, con el motor DC, los indicadores y el
framework de calidad, quedó congelada como referencia en la rama
[`legacy/v0-local`](https://github.com/byroncz/intrinsica/tree/legacy/v0-local)
y en el tag [`v0.2.0-legacy`](https://github.com/byroncz/intrinsica/releases/tag/v0.2.0-legacy).
No es la base de código del rediseño: se conserva solo para consultar
decisiones y resultados previos.

## Estructura del monorepo

Diseño fijado en el [TRD maestro §8.1](docs/TRD/plataforma_directional_change.md)
y en el [TRD de la capa 1](docs/TRD/l1.md). Cada carpeta trae un README que
dice qué va y qué no.

```
intrinsica/
├── shared/            # código compartido (dc_core, dc_pyo3, pyutils, dq)
├── layers/            # una carpeta por capa, una imagen por capa
├── infra/             # Terraform: modules/ y stacks/
├── scaffold/          # plantilla para capas nuevas
└── docs/              # TRD, data-contracts.md
```
