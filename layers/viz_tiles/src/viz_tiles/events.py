"""Eventos exactos de un θ y confirmaciones multiescala de un día.

TRD-viz §7.4 y §7.5. El estado por columna de `dir-<w>.u8` no representa varios
eventos en una columna (ADR-VZ-09); el navegador dibuja las franjas desde los
tiempos exactos de cada evento, que aquí se bajan a milisegundos desde el inicio
del día. Las confirmaciones multiescala (cuántos θ confirman en una columna y
cuántos lo hacen en el mismo instante) se acumulan en arreglos de tamaño fijo por
nivel, como M4: lo único que crece con los eventos es la lista de tiempos de
confirmación, 8 bytes por evento del día, hasta `finish`.
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from viz_tiles.contract import (
    DAY_MS,
    DAY_US,
    FLAG_CONFIRM_CLIPPED,
    FLAG_EXTREME_CLIPPED,
    FLAG_PROVISIONAL,
    FLAG_REF_CLIPPED,
    FLAG_UP,
    LEVELS,
    MAX_THETAS,
)
from viz_tiles.direction import PendingEvent


@dataclass(frozen=True)
class EventRows:
    """Los eventos de un θ que tocan el día, en orden de referencia.

    `reference`, `confirm` y `extreme` son milisegundos desde el inicio del día
    (`int32`, `⌊µs / 1000⌋`), recortados a `[0, 86 400 000]` si el punto cae fuera
    del día; `flags` marca el sentido, la cola provisional y cada recorte
    (`contract.FLAGS`).
    """

    reference: np.ndarray
    confirm: np.ndarray
    extreme: np.ndarray
    flags: np.ndarray

    def __len__(self) -> int:
        return len(self.reference)

    @classmethod
    def empty(cls) -> EventRows:
        return cls(
            np.empty(0, "<i4"),
            np.empty(0, "<i4"),
            np.empty(0, "<i4"),
            np.empty(0, "u1"),
        )


def _clip(times_us: np.ndarray, flag: int) -> tuple[np.ndarray, np.ndarray]:
    """`(ms recortados al día, banderas)` de tiempos en µs relativos al inicio del día."""
    outside = (times_us < 0) | (times_us >= DAY_US)
    ms = np.clip(times_us // 1000, 0, DAY_MS).astype("<i4")
    return ms, np.where(outside, flag, 0).astype("u1")


def event_rows(
    events: pa.Table | pa.RecordBatch,
    pending: PendingEvent | None,
    provisional: bool,
    day_start_us: int,
) -> tuple[EventRows, np.ndarray]:
    """Las filas de eventos de un θ y los tiempos (µs) en que confirman dentro del día.

    `events` son las filas de `events.parquet` que tocan el día, ordenadas por
    referencia, y `pending` la cola del carry-over si el día la incluye (con sus
    tres tiempos); `provisional` marca esa cola como candidata, no definitiva. El
    segundo valor son las confirmaciones del día, relativas a su inicio, para
    `ConfirmAccumulator.add`: nunca se guarda junto a las filas.
    """
    reference = events.column("reference_time").to_numpy()
    confirm = events.column("confirm_time").to_numpy()
    extreme = events.column("extreme_time").to_numpy()
    up = events.column("direction").to_numpy() == 1
    flags = np.where(up, FLAG_UP, 0).astype("u1")
    if pending is not None:
        if None in (pending.reference_time, pending.confirm_time, pending.extreme_time):
            raise ValueError("el pendiente no trae los tiempos de sus tres puntos")
        reference = np.append(reference, pending.reference_time)
        confirm = np.append(confirm, pending.confirm_time)
        extreme = np.append(extreme, pending.extreme_time)
        flags = np.append(
            flags,
            (FLAG_UP if pending.direction == 1 else 0)
            | (FLAG_PROVISIONAL if provisional else 0),
        ).astype("u1")
    reference, confirm, extreme = (
        np.asarray(t, dtype=np.int64) - day_start_us
        for t in (reference, confirm, extreme)
    )
    ref_ms, ref_flag = _clip(reference, FLAG_REF_CLIPPED)
    confirm_ms, confirm_flag = _clip(confirm, FLAG_CONFIRM_CLIPPED)
    extreme_ms, extreme_flag = _clip(extreme, FLAG_EXTREME_CLIPPED)
    rows = EventRows(
        ref_ms, confirm_ms, extreme_ms, flags | ref_flag | confirm_flag | extreme_flag
    )
    inside = (confirm >= 0) & (confirm < DAY_US)
    return rows, confirm[inside]


def pack_events(rows: list[EventRows]) -> memoryview:
    """Los bytes de `events.bin`: cuatro secciones con los eventos de todos los θ en orden."""
    parts = [
        np.concatenate([getattr(r, name) for r in rows]) if rows else np.empty(0, dt)
        for name, dt in (
            ("reference", "<i4"),
            ("confirm", "<i4"),
            ("extreme", "<i4"),
            ("flags", "u1"),
        )
    ]
    out = np.concatenate([p.view("u1") for p in parts]) if rows else np.empty(0, "u1")
    return memoryview(out)


@dataclass(frozen=True)
class Confirmations:
    """Confirmaciones multiescala de un día: un arreglo `uint8` de `w` valores por nivel.

    `confirms[w][c]` es el número de θ con al menos una confirmación en la columna
    `c`; `simul[w][c]`, el mayor número de θ que comparten un mismo `confirm_time`
    dentro de la columna. 0 si la columna no tiene confirmaciones.
    """

    confirms: dict[int, np.ndarray]
    simul: dict[int, np.ndarray]

    @classmethod
    def empty(cls) -> Confirmations:
        return cls(
            {w: np.zeros(w, "u1") for w in LEVELS},
            {w: np.zeros(w, "u1") for w in LEVELS},
        )


class ConfirmAccumulator:
    """Acumula θ por θ las confirmaciones del día: no retiene ningún evento.

    `confirms` se acumula por nivel directamente (un θ cuenta una vez por columna
    aunque confirme varias veces en ella: no se puede derivar del nivel fino).
    `simul` necesita agrupar los θ por instante exacto de confirmación: se guardan
    los tiempos de confirmación (µs, un arreglo por θ) y se agrupan en `finish`.
    """

    def __init__(self) -> None:
        self._confirms = {w: np.zeros(w, np.int64) for w in LEVELS}
        self._times: list[np.ndarray] = []

    def add(self, confirm_us: np.ndarray) -> None:
        """Suma un θ: `confirm_us` son sus confirmaciones dentro del día (µs desde su inicio)."""
        if len(self._times) >= MAX_THETAS:
            raise ValueError(f"más de {MAX_THETAS} θ no caben en un uint8")
        once = np.unique(confirm_us)  # un θ cuenta una vez por instante
        for w, level in self._confirms.items():
            level[np.unique(once * w // DAY_US)] += 1
        self._times.append(once)

    def finish(self) -> Confirmations:
        """Los tiles del día. Suelta los tiempos acumulados."""
        times, self._times = self._times, []
        confirms = {w: a.astype("u1") for w, a in self._confirms.items()}
        simul = {w: np.zeros(w, "u1") for w in LEVELS}
        if times:
            instants, thetas = np.unique(np.concatenate(times), return_counts=True)
            del times
            for w, level in simul.items():
                np.maximum.at(level, instants * w // DAY_US, thetas.astype("u1"))
        return Confirmations(confirms, simul)
