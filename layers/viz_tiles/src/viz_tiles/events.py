"""Eventos exactos de los θ de un día: las filas de `events.bin`.

TRD-viz §7.4. El navegador dibuja las franjas desde los tiempos exactos de cada
evento, que aquí se bajan a milisegundos desde el inicio del día, y deriva de ellos
las confirmaciones por píxel (cuántos θ confirman y cuántos en el mismo instante).
"""

from dataclasses import dataclass

import numpy as np

from viz_tiles.chain import PendingEvent
from viz_tiles.contract import (
    DAY_MS,
    DAY_US,
    FLAG_CONFIRM_CLIPPED,
    FLAG_EXTREME_CLIPPED,
    FLAG_PROVISIONAL,
    FLAG_REF_CLIPPED,
    FLAG_UP,
)
from viz_tiles.lake import MonthEvents


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
    events: MonthEvents,
    pending: PendingEvent | None,
    provisional: bool,
    day_start_us: int,
) -> EventRows:
    """Las filas de eventos de un θ para el día que arranca en `day_start_us` (µs).

    `events` son los eventos del θ que tocan el día, ordenados por referencia, y
    `pending` la cola del carry-over si el día la incluye (con sus tres tiempos);
    `provisional` marca esa cola como candidata, no definitiva.
    """
    reference = events.reference_time
    confirm = events.confirm_time
    extreme = events.extreme_time
    flags = np.where(events.direction == 1, FLAG_UP, 0).astype("u1")
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
    return EventRows(
        ref_ms, confirm_ms, extreme_ms, flags | ref_flag | confirm_flag | extreme_flag
    )


class EventsBuffer:
    """Los eventos del día de todos los θ en un solo buffer: los bytes de `events.bin`.

    Cuatro secciones (`reference`, `confirm` y `extreme` en `int32`, y `flags` en
    `uint8`), cada una con los eventos de todos los θ en orden. Es la única
    representación de los eventos del día en RAM (13 bytes por evento): `add` copia
    las filas de un θ y quien llama las suelta; `packed` junta las secciones dentro
    del mismo buffer, sin concatenar. El buffer crece al doble; mientras crece
    conviven el viejo y el nuevo, una vez por duplicación.
    """

    SECTIONS = (
        ("reference", "<i4"),
        ("confirm", "<i4"),
        ("extreme", "<i4"),
        ("flags", "u1"),
    )
    _WIDTHS = tuple(np.dtype(dt).itemsize for _, dt in SECTIONS)
    _CHUNK = 1 << 20  # bytes por copia al juntar las secciones

    def __init__(self) -> None:
        self._buf = np.empty(0, "u1")
        self._capacity = 0  # eventos que caben; las secciones se separan por ella
        self._count = 0
        self._packed = False

    def __len__(self) -> int:
        return self._count

    def _section(self, i: int, count: int) -> np.ndarray:
        """Los primeros `count` valores de la sección `i` del buffer actual."""
        start = sum(self._WIDTHS[:i]) * self._capacity
        raw = self._buf[start : start + self._WIDTHS[i] * count]
        return raw.view(self.SECTIONS[i][1])

    def add(self, rows: EventRows) -> None:
        """Agrega los eventos de un θ al final de cada sección."""
        if self._packed:
            raise ValueError("el buffer ya se empaquetó: no admite más eventos")
        n = len(rows)
        if self._count + n > self._capacity:
            self._grow(max(2 * self._capacity, self._count + n, 1024))
        end = self._count + n
        for i, (name, _) in enumerate(self.SECTIONS):
            self._section(i, end)[self._count : end] = getattr(rows, name)
        self._count = end

    def _grow(self, capacity: int) -> None:
        sections = [self._section(i, self._count) for i in range(len(self.SECTIONS))]
        old = self._buf
        self._buf = np.empty(capacity * sum(self._WIDTHS), "u1")
        previous, self._capacity = self._capacity, capacity
        for i, section in enumerate(sections):
            self._section(i, self._count)[:] = section
        del sections, old, previous

    def packed(self) -> memoryview:
        """Los bytes de `events.bin`: las secciones contiguas, sin otro buffer.

        Mueve cada sección hacia el inicio del mismo buffer, por tramos: el destino
        siempre queda antes que el origen, así que copiar hacia delante es seguro.
        Se puede llamar más de una vez: las siguientes devuelven la misma vista.
        """
        n = self._count
        if not self._packed:
            dst = 0
            for i, width in enumerate(self._WIDTHS):
                src = sum(self._WIDTHS[:i]) * self._capacity
                size = width * n
                for off in range(0, size, self._CHUNK):
                    stop = min(off + self._CHUNK, size)
                    self._buf[dst + off : dst + stop] = self._buf[
                        src + off : src + stop
                    ]
                dst += size
            self._packed = True
        return memoryview(self._buf[: sum(self._WIDTHS) * n])
