# Imagen única de L2 (ADR-07). Se construye desde la raíz del workspace:
#   docker build -f layers/l2_dc_events/Dockerfile --build-arg VERSION=... --build-arg GIT_SHA=... .
FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

# dc_pyo3 es una extensión Rust: maturin la compila durante uv sync, y necesita
# cargo y un enlazador (gcc). Nada de esto llega a la etapa final.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

# La versión de Rust la fija rust-toolchain.toml (una sola fuente con el
# desarrollo local y el CI). Va antes que el código para que esta capa solo se
# reconstruya si cambia la toolchain.
COPY rust-toolchain.toml ./
RUN curl -fsSL -o /tmp/rustup-init "https://static.rust-lang.org/rustup/dist/$(uname -m)-unknown-linux-gnu/rustup-init" \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --default-toolchain none --no-modify-path \
    && rm /tmp/rustup-init \
    && rustup toolchain install \
    && cargo --version

COPY Cargo.toml Cargo.lock pyproject.toml uv.lock ./
COPY shared ./shared
COPY layers ./layers

# --no-editable copia el código de l2_dc_events, dq y pyutils, y la rueda de
# dc_pyo3 compilada, dentro del entorno: la etapa final no necesita el árbol
# fuente ni Rust.
RUN uv sync --locked --package l2_dc_events --no-dev --no-editable

FROM python:3.14-slim

# VERSION viene del archivo layers/l2_dc_events/VERSION; lo pasa el CI.
# ARROW_DEFAULT_MEMORY_POOL y MALLOC_MMAP_THRESHOLD_ mantienen el RSS de la
# unidad O(lote) con los escritores en varios hilos (medido en ITSC-275; ver
# el docstring de l2_dc_events/memory.py).
ARG VERSION=unknown
ARG GIT_SHA=unknown
ENV IMAGE_VERSION=${VERSION}+${GIT_SHA} \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ARROW_DEFAULT_MEMORY_POOL=system \
    MALLOC_MMAP_THRESHOLD_=16384

COPY --from=build /app/.venv /app/.venv

ENTRYPOINT ["python", "-m", "l2_dc_events"]
