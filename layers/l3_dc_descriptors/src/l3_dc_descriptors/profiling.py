"""Perfil opcional de una unidad (`L3_PROFILE=1`), para el hallazgo `unit_profile` (TRD-L3 §9.3).

Responde en qué funciones se gasta la pared de una unidad. Es para **proporciones, no para
absolutos**: `cProfile` instrumenta cada llamada y añade entre 30 % y 100 % de pared, más
a las funciones pequeñas y muy llamadas. Sin la variable, nada de este módulo se usa: ni se
crea un `Profile` ni cambia `unit_timing`.

`cProfile` solo ve el hilo que lo activa. La unidad corre hoy en un solo hilo; si un día
agrega θ en varios, el perfil verá solo lo del hilo principal.
"""

import cProfile
import pstats

TOP = 40


def _top(stats: pstats.Stats, key: str) -> list[dict]:
    """Las `TOP` funciones más altas por `key`, en el orden de `pstats`."""
    stats.sort_stats(key)
    rows = []
    for func in stats.fcn_list[:TOP]:
        file, line, name = func
        _, calls, tottime, cumtime, _ = stats.stats[func]
        rows.append(
            {
                "function": name,
                "file": file,
                "line": line,
                "ncalls": calls,
                "tottime_s": round(tottime, 3),
                "cumtime_s": round(cumtime, 3),
            }
        )
    return rows


def top_functions(profiler: cProfile.Profile) -> dict[str, list[dict]]:
    """`top_tottime` (tiempo propio) y `top_cumtime` (con lo que llama) de un perfil ya detenido."""
    stats = pstats.Stats(profiler)
    return {
        "top_tottime": _top(stats, "tottime"),
        "top_cumtime": _top(stats, "cumulative"),
    }
