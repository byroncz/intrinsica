"""Capa viz de intrinsica: la página de un día con sus ticks y los eventos DC de sus θ."""

import os

# Antes de que alguien importe pyarrow (Arrow lo lee al cargarse): ver `memory.py`.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
