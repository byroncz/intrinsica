"""Ayudas de las pruebas de la CLI de L3: catálogo temporal, entorno y hallazgos emitidos."""

import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow.dataset as ds
from l3_lake import Lake


def write_catalog(path: Path, thetas: Iterable[int]) -> str:
    """Un catálogo de θ con el formato de L2 (`scale` y `thetas`); devuelve su ruta.

    Es del tamaño de la prueba y nunca un catálogo del repo: L3 no lleva copia del de L2.
    """
    lines = ["scale: 100000000", "thetas:", *(f"  - {t}" for t in thetas)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def env_for(
    lake: Lake, base: Path, catalog: str | None, **extra: str
) -> dict[str, str]:
    """El entorno de la imagen sobre un lago de prueba (`catalog` `None`: sin URI)."""
    env = {
        "L3_LANDING_ROOT": str(lake.l1),
        "L3_EVENTS_ROOT": str(lake.l2),
        "L3_SUMMARIES_ROOT": str(lake.l3),
        "L3_DQ_ROOT": str(base / "dq"),
        "IMAGE_VERSION": "0.4.0+test",
        # En proceso por defecto: un hijo por unidad (ITSC-372) cuesta ≈ 1 s de arranque y las
        # pruebas lo piden donde lo prueban (`test_l3_unit_process.py`).
        "L3_UNIT_PROCESS": "0",
        **extra,
    }
    if catalog is not None:
        env["L3_THETAS_URI"] = catalog
    return env


def emitted(base: Path, check_type: str | None = None) -> list[dict]:
    """Los hallazgos del lago de DQ, con `details` ya decodificado, en orden de emisión."""
    root = base / "dq"
    if not root.exists():
        return []
    rows = ds.dataset(root, partitioning="hive").to_table().to_pylist()
    rows.sort(key=lambda r: r["detected_at"])
    for row in rows:
        row["details"] = json.loads(row["details"])
    if check_type is not None:
        rows = [r for r in rows if r["check_type"] == check_type]
    return rows


def l3_files(lake: Lake) -> list[Path]:
    return sorted(p for p in lake.l3.rglob("*.parquet") if p.is_file())
