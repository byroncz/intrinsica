"""Construye el lago chico del humo de la imagen (`smoke.sh`): L1, L2 y un catálogo temporal.

Usa el mismo generador que las pruebas (`l3_lake.build_lake`: ticks reales, eventos del
detector real de L2, mes a mes con su carry-over) y deja en `<destino>`: `l1/`, `l2/` y
`thetas.yaml`, el catálogo de este lago. El catálogo es del humo, generado aquí y no
versionado: L3 no lleva ninguno propio (TRD-L3 §11), la imagen lo recibe por
`L3_THETAS_URI`.

Uso: python smoke_lake.py <destino>
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "shared/dc_frames/tests")]

from cli_support import write_catalog
from l3_lake import build_lake, fixture_ticks, month_cuts


def main(destination: str) -> None:
    base = Path(destination)
    ticks = fixture_ticks()
    # Tres meses, de dos a cuatro row groups de L1 cada uno, y eventos que cruzan el borde.
    lake = build_lake(base, ticks, month_cuts(ticks, 1500, 3000))
    write_catalog(base / "thetas.yaml", lake.thetas)
    events = sum(len(rows) for rows in lake.events.values())
    print(
        f"lago: {len(lake.months)} meses, {len(lake.thetas)} θ, {events} eventos "
        f"en {base}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
