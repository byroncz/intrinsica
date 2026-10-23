"""Una unidad en un proceso hijo: su memoria muere con ella (ITSC-372, ADR-L3-16).

Dentro de un proceso, el RSS que una unidad deja al cerrar sube de mes en mes aunque se
devuelva lo libre al kernel (`memory.release_to_kernel`): lo que queda es fragmentación de las
arenas de glibc, y en 110 meses mata el backfill. Ajustar el asignador es medir y suponer; un
proceso por unidad lo resuelve por construcción: el siguiente mes arranca con el RSS de un
intérprete recién iniciado. Cuesta `spawn_s` (≈ 1 s: importar pyarrow y el motor) por mes.

El hijo se lanza con `spawn`, nunca `fork`: el padre tiene hilos vivos (E/S de DQ, el pool de
Arrow) y un `fork` los copiaría a medias. El padre solo espera: el hijo corre `process_unit` y
le devuelve el `Result` por una tubería, sin el `Timing` (tiene candados y no se serializa; sus
medidas viajan en el hallazgo `unit_timing`). Si el hijo muere sin devolverlo (el OOM del cgroup
lo mata con la señal 9), la unidad queda fallida con el hallazgo `unit_process_died`, que lleva
`exit_signal`, y el padre sigue vivo para emitirlo.
"""

import logging
import multiprocessing
import os
import time
import traceback
from dataclasses import replace
from multiprocessing.connection import Connection

from dq import configure_logging

from l3_dc_descriptors import findings
from l3_dc_descriptors.context import RunContext, Unit
from l3_dc_descriptors.memory import tune_allocators
from l3_dc_descriptors.pipeline import Result, process_unit

logger = logging.getLogger(__name__)


def _child(
    sender: Connection,
    unit: Unit,
    ctx: RunContext,
    thetas: list[int],
    started: float,
) -> None:
    """El cuerpo del hijo: corre la unidad y devuelve su `Result` (o el motivo de un fallo)."""
    spawn_s = time.time() - started
    try:
        configure_logging(os.environ)
        tune_allocators()
        result = process_unit(unit, ctx, thetas, spawn_s=spawn_s)
        sender.send(("result", replace(result, timing=None)))
    except BaseException:  # noqa: BLE001 - el padre lo registra como unidad fallida
        text = traceback.format_exc()
        print(text, flush=True)
        sender.send(("error", text))
        raise SystemExit(1) from None
    finally:
        sender.close()


def run_unit(unit: Unit, ctx: RunContext, thetas: list[int]) -> Result:
    """`process_unit` en un proceso hijo nuevo; el `Result` vuelve sin `timing`.

    Un hijo que muere por una señal o por una excepción inesperada deja un `Result` fallido
    con el hallazgo `unit_process_died`; no lanza.
    """
    spawned = multiprocessing.get_context("spawn")
    receiver, sender = spawned.Pipe(duplex=False)
    process = spawned.Process(
        target=_child,
        args=(sender, unit, ctx, thetas, time.time()),
        name=f"l3-unit-{unit.year:04d}-{unit.month:02d}",
    )
    process.start()
    sender.close()  # sin esta copia, la muerte del hijo no cierra la tubería
    message = None
    try:
        message = receiver.recv()
    except EOFError:  # el hijo murió sin enviar nada
        pass
    except BaseException:  # la interrupción del padre no deja un hijo huérfano
        process.kill()
        raise
    finally:
        receiver.close()
        process.join()
    if message is not None and message[0] == "result":
        return message[1]
    code = process.exitcode
    signal_number = -code if code is not None and code < 0 else None
    reason = message[1] if message is not None else None
    logger.error(
        "unidad %s: el proceso hijo terminó sin resultado (señal %s, código %s)",
        unit,
        signal_number,
        code,
    )
    died = findings.unit_process_died(
        ctx, unit, exit_signal=signal_number, exit_code=code, error=reason
    )
    how = f"la señal {signal_number}" if signal_number else f"el código {code}"
    return Result(
        (died,), error=f"unit_process_died: el proceso hijo terminó con {how}"
    )
