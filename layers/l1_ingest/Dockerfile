# Imagen única de L1 (ADR-L1-10). Se construye desde la raíz del workspace:
#   docker build -f layers/l1_ingest/Dockerfile --build-arg VERSION=... --build-arg GIT_SHA=... .
FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY shared ./shared
COPY layers ./layers

# --no-editable copia el código de l1_ingest y dq dentro del entorno, así la
# etapa final no necesita el árbol fuente.
RUN uv sync --locked --package l1_ingest --no-dev --no-editable

FROM python:3.14-slim

# VERSION viene del archivo layers/l1_ingest/VERSION; lo pasa el CI.
ARG VERSION=unknown
ARG GIT_SHA=unknown
ENV IMAGE_VERSION=${VERSION}+${GIT_SHA} \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY --from=build /app/.venv /app/.venv

ENTRYPOINT ["python", "-m", "l1_ingest"]
