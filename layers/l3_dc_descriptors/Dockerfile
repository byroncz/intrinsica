# Imagen única de L3 (TRD-L3 §11). Se construye desde la raíz del workspace:
#   docker build -f layers/l3_dc_descriptors/Dockerfile --build-arg VERSION=... --build-arg GIT_SHA=... .
FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

# dc_pyo3 es una extensión Rust (trae el motor de agregación, dc_descriptors): maturin la
# compila durante uv sync, y necesita cargo y un enlazador (gcc). Nada de esto llega a la
# etapa final. Misma etapa de build que L2.
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
# desarrollo local, el CI y L2). Va antes que el código para que esta capa solo se
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

# --no-editable copia l3_dc_descriptors, dc_frames, dq y pyutils, y la rueda de dc_pyo3
# compilada, dentro del entorno: la etapa final no necesita el árbol fuente ni Rust.
RUN uv sync --locked --package l3_dc_descriptors --no-dev --no-editable

FROM python:3.14-slim

# VERSION viene del archivo layers/l3_dc_descriptors/VERSION; lo pasa el CI.
# ARROW_DEFAULT_MEMORY_POOL mantiene el RSS de la unidad O(lote), como en L2 (ITSC-275).
# MALLOC_MMAP_THRESHOLD_ en 32 MiB (el máximo de glibc) y MALLOC_TRIM_THRESHOLD_ en 1 GiB
# (sin recorte del heap) no son los de L2: miden −31 % de CPU a cambio de un RSS de 3,1 GiB
# (ITSC-366; decisión en TRD-L3 ADR-L3-16 y l3_dc_descriptors/memory.py). L3 no lleva el
# catálogo de θ: L3_THETAS_URI apunta al mismo l2/thetas.yaml que usa L2.
ARG VERSION=unknown
ARG GIT_SHA=unknown
ENV IMAGE_VERSION=${VERSION}+${GIT_SHA} \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ARROW_DEFAULT_MEMORY_POOL=system \
    MALLOC_MMAP_THRESHOLD_=33554432 \
    MALLOC_TRIM_THRESHOLD_=1073741824

COPY --from=build /app/.venv /app/.venv

ENTRYPOINT ["python", "-m", "l3_dc_descriptors"]
