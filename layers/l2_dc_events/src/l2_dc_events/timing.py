"""Tiempo por fase de una unidad de L2 (ITSC-289): dónde se va la pared.

El hilo principal encadena fases en serie (carga del carry-over, espera de la
lectura, detección y espera de los escritores); la lectura anticipada y los
escritores corren en otros hilos. Por eso hay dos clases de número:

- **Pared del hilo principal** (`carry_s`, `open_s`, `read_s`, `detect_s`,
  `wait_s`): son disjuntas, suman a lo más `wall_s` y lo que falta es `other_s`.
- **CPU o tiempo acumulado entre hilos** (`decode_s`, `detect_cpu_s`,
  `write_s`): `decode_s` es la CPU del hilo lector al decodificar (ITSC-290: ya
  no es pared del principal); `detect_cpu_s`, la CPU del hilo principal durante
  el fan-out; `write_s`, el tiempo que los θ del mes pasan codificando y subiendo
  Parquet, sumado. Con k hilos `write_s` puede llegar a k veces la pared, así
  que no entra en `other_s`; se compara con `wait_s`, lo que el hilo principal
  esperó por ellos. `detect_cpu_s` contra `detect_s` distingue un detector lento
  (CPU alta) de uno desalojado (CPU baja).

`wait_s` se desglosa (ITSC-293) en `backpressure_s` (el principal bloqueado en
`submit` mientras lee y detecta: los escritores van atrasados), `drain_s` (tras
el último tramo, esperar a que cada θ escriba lo que le queda en cola) y
`publish_s` (cerrar y mover los 100 archivos). La suma de las tres es `wait_s`.
Los 50 `ThetaTiming` dicen, por θ, cuánto tardó cada fase y cuándo terminó: el
θ con `events_done_s` mayor cerró su último tramo al final y es el que retrasa
la unidad; `done_s` solo refleja el orden de la cola de publicación.
"""

from dataclasses import dataclass


class Phases:
    """Acumuladores de una unidad. Los escribe un solo hilo cada uno (el
    principal; `write_s` se llena al final con lo que sumó cada escritor)."""

    __slots__ = (
        "backpressure_s",
        "bytes_in",
        "carry_s",
        "decode_s",
        "detect_cpu_s",
        "detect_s",
        "drain_s",
        "open_s",
        "publish_s",
        "read_s",
        "row_groups",
        "write_s",
    )

    def __init__(self) -> None:
        self.read_s = 0.0
        self.decode_s = 0.0
        self.detect_s = 0.0
        self.detect_cpu_s = 0.0
        self.backpressure_s = 0.0
        self.drain_s = 0.0
        self.publish_s = 0.0
        self.open_s = 0.0
        self.carry_s = 0.0
        self.write_s = 0.0
        self.row_groups = 0
        self.bytes_in = 0

    @property
    def wait_s(self) -> float:
        return self.backpressure_s + self.drain_s + self.publish_s


@dataclass(frozen=True)
class ThetaTiming:
    """Lo que un θ gastó en escribir su mes (ITSC-293), en segundos.

    `encode_s` arma el lote, hashea y codifica Parquet (en GCS incluye la
    subida en streaming); `close_s` y `move_s` cierran el archivo y lo mueven
    al destino; `carry_write_s` escribe su `carry_over.parquet` entero.
    `blocked_s` es lo que el hilo principal esperó a tramos que este θ cerró
    último, y `queued_s`, lo que sus bloques esperaron en cola antes de
    escribirse. `events_done_s` y `done_s` son offsets desde el inicio de la
    unidad: cuándo el escritor terminó su último tramo y cuándo terminó de
    publicar eventos y carry-over.
    """

    theta: int
    events: int
    open_s: float
    encode_s: float
    close_s: float
    move_s: float
    carry_write_s: float
    events_done_s: float
    done_s: float
    blocked_s: float = 0.0
    queued_s: float = 0.0

    @property
    def write_s(self) -> float:
        return self.encode_s + self.close_s + self.move_s + self.carry_write_s

    def details(self) -> dict:
        return {
            "theta": self.theta,
            "events": self.events,
            **{
                name: round(getattr(self, name), 3)
                for name in (
                    "open_s",
                    "encode_s",
                    "close_s",
                    "move_s",
                    "carry_write_s",
                    "events_done_s",
                    "done_s",
                    "blocked_s",
                    "queued_s",
                )
            },
        }


@dataclass(frozen=True)
class Timing:
    """Lo que midió la unidad. `wall_s` corre desde que `process_unit` empieza
    hasta justo antes de emitir los hallazgos."""

    wall_s: float
    read_s: float
    decode_s: float
    detect_s: float
    write_s: float
    carry_s: float
    wait_s: float
    row_groups: int
    bytes_in: int
    cores: float
    cores_visible: int
    cores_source: str
    write_workers: int
    cpu_throttled_s: float | None
    detect_cpu_s: float = 0.0
    fanout_threads: int = 1
    open_s: float = 0.0
    backpressure_s: float = 0.0
    drain_s: float = 0.0
    publish_s: float = 0.0
    thetas: tuple[ThetaTiming, ...] = ()

    @property
    def other_s(self) -> float:
        """La pared que ninguna fase explica (troceo de lotes, reloj entre
        fases); las fases del hilo principal no se solapan."""
        serial = self.carry_s + self.open_s + self.read_s + self.detect_s
        return max(0.0, self.wall_s - serial - self.wait_s)

    @property
    def encode_s(self) -> float:
        return sum((t.encode_s for t in self.thetas), 0.0)

    @property
    def close_s(self) -> float:
        return sum((t.close_s for t in self.thetas), 0.0)

    @property
    def move_s(self) -> float:
        return sum((t.move_s for t in self.thetas), 0.0)

    @property
    def carry_write_s(self) -> float:
        return sum((t.carry_write_s for t in self.thetas), 0.0)

    @property
    def heaviest(self) -> ThetaTiming | None:
        """El θ que más tiempo de escritura sumó."""
        return max(self.thetas, key=lambda t: t.write_s, default=None)

    @property
    def last(self) -> ThetaTiming | None:
        """El θ cuyo último tramo se escribió al final: el que más retrasa el cierre.

        Se ordena por `events_done_s` y no por `done_s`: la publicación recorre
        los θ en orden de cola, así que `done_s` mide ese orden y no quién
        tardó.
        """
        return max(self.thetas, key=lambda t: t.events_done_s, default=None)

    def details(self) -> dict:
        """El payload del hallazgo `unit_timing` (segundos con ms de resolución)."""
        seconds = {
            "wall_s": self.wall_s,
            "read_s": self.read_s,
            "decode_s": self.decode_s,
            "detect_s": self.detect_s,
            "detect_cpu_s": self.detect_cpu_s,
            "write_s": self.write_s,
            "carry_s": self.carry_s,
            "wait_s": self.wait_s,
            "other_s": self.other_s,
            "open_s": self.open_s,
            "backpressure_s": self.backpressure_s,
            "drain_s": self.drain_s,
            "publish_s": self.publish_s,
            "encode_s": self.encode_s,
            "close_s": self.close_s,
            "move_s": self.move_s,
            "carry_write_s": self.carry_write_s,
        }
        return {
            **{name: round(value, 3) for name, value in seconds.items()},
            "row_groups": self.row_groups,
            "bytes_in": self.bytes_in,
            "cores": self.cores,
            "cores_visible": self.cores_visible,
            "cores_source": self.cores_source,
            "fanout_threads": self.fanout_threads,
            "write_workers": self.write_workers,
            "cpu_throttled_s": (
                None if self.cpu_throttled_s is None else round(self.cpu_throttled_s, 3)
            ),
            "theta_timing": [t.details() for t in self.thetas],
        }
