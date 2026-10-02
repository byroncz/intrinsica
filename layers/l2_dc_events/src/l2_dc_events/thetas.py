"""El catálogo de θ: qué θ calcula L2 (TRD-L2 §7.3, ADR-L2-10).

La fuente de verdad en la nube es un objeto en GCS (`L2_THETAS_URI`); el
`config/thetas.yaml` del paquete es la semilla para local y pruebas. Los dos
tienen el mismo formato y pasan por la misma validación.
"""

from collections import Counter
from pathlib import Path

import pyarrow as pa
import yaml
from pyutils import resolve_fs

SCALE = 10**8
# Rango del experimento: [10⁻⁴, 5·10⁻²] en unidades de 10⁻⁸.
THETA_MIN = 10_000
THETA_MAX = 5_000_000
THETAS_CONFIG = Path(__file__).resolve().parent / "config" / "thetas.yaml"


class ThetasError(ValueError):
    """El catálogo no existe o no cumple el contrato de §7.3.

    `problems` lista cada incumplimiento, para el payload del hallazgo
    `theta_catalog_invalid`.
    """

    def __init__(self, source: str | Path, problems: list[str]) -> None:
        super().__init__(f"{source}: {'; '.join(problems)}")
        self.source = str(source)
        self.problems = problems


def _read(source: str | Path) -> str:
    """El texto del catálogo, de un archivo local o de un objeto `gs://`."""
    try:
        if isinstance(source, str) and "://" in source:
            fs, resolved = resolve_fs(source)
            with fs.open_input_stream(resolved) as stream:
                return stream.read().decode()
        return Path(source).read_text()
    except (OSError, pa.ArrowException, UnicodeDecodeError) as exc:
        raise ThetasError(source, [f"no se pudo leer ({exc})"]) from exc


def parse_thetas(text: str, source: str | Path = "<catálogo>") -> list[int]:
    """Los `theta_int` (`round(θ × 10⁸)`) del catálogo, de menor a mayor.

    Valida: `scale` exacta de 10⁸, una lista no vacía de enteros (la escala
    exacta: ningún valor es un decimal que haya que redondear), únicos y dentro
    de [`THETA_MIN`, `THETA_MAX`]. El orden del archivo es libre: agregar un θ
    es escribir una línea donde sea. Que sean 50 y sigan la regla de ADR-L2-10
    lo comprueba una prueba sobre la semilla, no la carga.
    """
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ThetasError(source, [f"no es YAML válido ({exc})"]) from exc
    if not isinstance(config, dict):
        raise ThetasError(source, ["debe ser un mapa con `scale` y `thetas`"])
    problems = []
    if config.get("scale") != SCALE:
        problems.append(f"`scale` debe ser {SCALE}")
    thetas = config.get("thetas")
    if not isinstance(thetas, list) or not thetas:
        raise ThetasError(source, [*problems, "`thetas` debe ser una lista no vacía"])
    bad = [t for t in thetas if type(t) is not int]
    if bad:
        problems.append(f"cada θ debe ser un entero (round(θ × 10⁸)): {bad}")
    ints = [t for t in thetas if type(t) is int]
    outside = [t for t in ints if not THETA_MIN <= t <= THETA_MAX]
    if outside:
        problems.append(f"fuera de [{THETA_MIN}, {THETA_MAX}]: {outside}")
    repeated = sorted(t for t, n in Counter(ints).items() if n > 1)
    if repeated:
        problems.append(f"repetidos: {repeated}")
    if problems:
        raise ThetasError(source, problems)
    return sorted(ints)


def load_thetas(source: str | Path = THETAS_CONFIG) -> list[int]:
    """Lee y valida el catálogo de `source` (ruta local, `gs://` o la semilla)."""
    return parse_thetas(_read(source), source)
