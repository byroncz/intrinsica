"""Capa viz de intrinsica: reduce un día de ticks y eventos DC a tiles M4."""

import os

# Antes de que alguien importe pyarrow (Arrow lo lee al cargarse): ver `memory.py`.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
