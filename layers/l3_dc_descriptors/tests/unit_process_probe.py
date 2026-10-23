"""Cuerpos de hijo que fallan, para probar `unit_process` sin esperar a un OOM de verdad.

Viven en un módulo aparte porque el hijo (`spawn`) importa el `target` por su nombre.
"""

import os
import signal

from l3_dc_descriptors.unit_process import _child as real_child

KILLED_MONTH = 9


def kill_in_september(sender, unit, ctx, thetas, started):
    """Como el OOM del cgroup: el hijo de septiembre muere con SIGKILL; los demás corren."""
    if unit.month == KILLED_MONTH:
        os.kill(os.getpid(), signal.SIGKILL)
    real_child(sender, unit, ctx, thetas, started)


def fail_with_an_exception(sender, unit, ctx, thetas, started):
    """Una excepción que la unidad no tipó (ni `SummarizeError` ni `FramesError`): un θ que no
    es un entero.
    """
    real_child(sender, unit, ctx, [None], started)
