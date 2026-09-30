"""Rutas de la salida de L2: particiones `events` y `carry_over` (§7.2 y §7.4).

El escritor atómico y el hash de contenido son de `shared/pyutils`, los mismos
que usa L1; aquí solo queda lo propio de L2.
"""

from pathlib import Path

EVENTS = "events.parquet"
CARRY_OVER = "carry_over.parquet"
THETA_DIGITS = 8


def format_theta(theta: int) -> str:
    """`theta = round(θ × 10⁸)` como el `theta=<t>` de la ruta: `0.00010000`.

    Ancho fijo, sin recortar ceros (§7.2): ordena bien como texto y deja
    a la vista la escala que comparte con el precio.
    """
    if not 0 < theta < 10**THETA_DIGITS:
        raise ValueError(f"theta={theta} fuera de (0, 10^{THETA_DIGITS})")
    return f"0.{theta:0{THETA_DIGITS}d}"


def partition_path(
    root: str | Path,
    provider: str,
    market: str,
    asset: str,
    theta: int,
    year: int,
    month: int,
    filename: str,
) -> str:
    """Ruta hive del archivo bajo `root` (local o `gs://<bucket>/<prefijo>`)."""
    return (
        f"{str(root).rstrip('/')}/provider={provider}/market={market}/asset={asset}"
        f"/theta={format_theta(theta)}/year={year:04d}/month={month:02d}/{filename}"
    )
