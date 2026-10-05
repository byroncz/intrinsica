# Imagen única de viz (ADR-VZ-05). Se construye desde la raíz del workspace:
#   docker build -f layers/viz_tiles/Dockerfile --build-arg VERSION=... --build-arg GIT_SHA=... .
FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY shared ./shared
COPY layers ./layers

# --no-editable copia viz_tiles, dq y pyutils dentro del entorno: la etapa final
# no necesita el árbol fuente. Sin Rust: viz_tiles no depende de dc_pyo3 (lee los
# Parquet de L1 y L2 por nombre de columna, sin importar esas capas).
RUN uv sync --locked --package viz_tiles --no-dev --no-editable

FROM python:3.14-slim

# VERSION viene del archivo layers/viz_tiles/VERSION; lo pasa el CI.
# ARROW_DEFAULT_MEMORY_POOL y MALLOC_MMAP_THRESHOLD_ mantienen el RSS de la
# unidad O(lote): ver viz_tiles/memory.py.
ARG VERSION=unknown
ARG GIT_SHA=unknown
ENV IMAGE_VERSION=${VERSION}+${GIT_SHA} \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ARROW_DEFAULT_MEMORY_POOL=system \
    MALLOC_MMAP_THRESHOLD_=16384

COPY --from=build /app/.venv /app/.venv

ENTRYPOINT ["python", "-m", "viz_tiles"]
