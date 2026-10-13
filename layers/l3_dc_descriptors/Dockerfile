# Imagen única de L3 (TRD-L3 §11). Se construye desde la raíz del workspace:
#   docker build -f layers/l3_dc_descriptors/Dockerfile --build-arg VERSION=... --build-arg GIT_SHA=... .
FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY shared ./shared
COPY layers ./layers

# --no-editable copia l3_dc_descriptors, dc_frames, dq y pyutils dentro del entorno: la
# etapa final no necesita el árbol fuente. Sin Rust: L3 no depende de dc_pyo3 (lee los
# Parquet de L1 y L2 por nombre de columna, sin importar esas capas; dc_pyo3 solo lo usan
# las pruebas, para generar el lago).
RUN uv sync --locked --package l3_dc_descriptors --no-dev --no-editable

FROM python:3.14-slim

# VERSION viene del archivo layers/l3_dc_descriptors/VERSION; lo pasa el CI.
# ARROW_DEFAULT_MEMORY_POOL y MALLOC_MMAP_THRESHOLD_ mantienen el RSS de la unidad O(lote),
# como en L2 (medido en ITSC-275; ver l3_dc_descriptors/memory.py). L3 no lleva el catálogo
# de θ: L3_THETAS_URI apunta al mismo l2/thetas.yaml que usa L2.
ARG VERSION=unknown
ARG GIT_SHA=unknown
ENV IMAGE_VERSION=${VERSION}+${GIT_SHA} \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ARROW_DEFAULT_MEMORY_POOL=system \
    MALLOC_MMAP_THRESHOLD_=16384

COPY --from=build /app/.venv /app/.venv

ENTRYPOINT ["python", "-m", "l3_dc_descriptors"]
