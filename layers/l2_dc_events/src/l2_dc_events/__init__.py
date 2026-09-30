"""Capa L2 de intrinsica: eventos Directional Change de 50 theta."""

import os

# Antes de que alguien importe pyarrow (Arrow lo lee al cargarse): ver `memory.py`.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
