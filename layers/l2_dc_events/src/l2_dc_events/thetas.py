"""Los 50 θ congelados en `config/thetas.yaml`, dentro del paquete (TRD-L2 §7.3, ADR-L2-10)."""

from itertools import pairwise
from pathlib import Path

import yaml

SCALE = 10**8
THETAS_CONFIG = Path(__file__).resolve().parent / "config" / "thetas.yaml"


class ThetasError(ValueError):
    """El archivo de θ no cumple el contrato de §7.3."""


def load_thetas(path: Path = THETAS_CONFIG) -> list[int]:
    """Los `theta_int` (`round(θ × 10⁸)`) del archivo, en su orden.

    Valida solo la forma: escala `10⁸`, enteros con `0 < θ < 1` y estrictamente
    crecientes. Que sean 50 y sigan la regla de ADR-L2-10 lo comprueba una
    prueba, no la carga: el archivo es la fuente de verdad, no la fórmula.
    """
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("scale") != SCALE:
        raise ThetasError(f"{path}: `scale` debe ser {SCALE}")
    thetas = config.get("thetas")
    if not isinstance(thetas, list) or not thetas:
        raise ThetasError(f"{path}: `thetas` debe ser una lista no vacía")
    if not all(type(t) is int and 0 < t < SCALE for t in thetas):
        raise ThetasError(f"{path}: cada θ debe ser un entero con 0 < θ < {SCALE}")
    if any(a >= b for a, b in pairwise(thetas)):
        raise ThetasError(f"{path}: los θ deben ser estrictamente crecientes")
    return thetas
