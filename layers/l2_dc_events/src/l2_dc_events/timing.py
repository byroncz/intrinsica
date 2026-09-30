"""Tiempo por fase de una unidad de L2 (ITSC-289): dónde se va la pared.

El hilo principal encadena fases en serie (carga del carry-over, lectura,
decodificación, detección y espera de los escritores); los escritores corren en
otros hilos. Por eso hay dos clases de número:

- **Pared del hilo principal** (`carry_s`, `read_s`, `decode_s`, `detect_s`,
  `wait_s`): son disjuntas, suman a lo más `wall_s` y lo que falta es `other_s`.
- **Acumulado entre hilos** (`write_s`): el tiempo que los 50 θ pasan
  codificando y subiendo Parquet, sumado. Con k hilos puede llegar a k veces la
  pared, así que no entra en `other_s`; se compara con `wait_s`, lo que el hilo
  principal esperó por ellos.
"""

from dataclasses import dataclass


class Phases:
    """Acumuladores de una unidad. Los escribe un solo hilo cada uno (el
    principal; `write_s` se llena al final con lo que sumó cada escritor)."""

    __slots__ = (
        "bytes_in",
        "carry_s",
        "decode_s",
        "detect_s",
        "read_s",
        "row_groups",
        "wait_s",
        "write_s",
    )

    def __init__(self) -> None:
        self.read_s = 0.0
        self.decode_s = 0.0
        self.detect_s = 0.0
        self.wait_s = 0.0
        self.carry_s = 0.0
        self.write_s = 0.0
        self.row_groups = 0
        self.bytes_in = 0


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

    @property
    def other_s(self) -> float:
        """La pared que ninguna fase explica (apertura de escritores, troceo de
        lotes, reloj entre fases); las fases del hilo principal no se solapan."""
        serial = self.carry_s + self.read_s + self.decode_s + self.detect_s
        return max(0.0, self.wall_s - serial - self.wait_s)

    def details(self) -> dict:
        """El payload del hallazgo `unit_timing` (segundos con ms de resolución)."""
        seconds = {
            "wall_s": self.wall_s,
            "read_s": self.read_s,
            "decode_s": self.decode_s,
            "detect_s": self.detect_s,
            "write_s": self.write_s,
            "carry_s": self.carry_s,
            "wait_s": self.wait_s,
            "other_s": self.other_s,
        }
        return {
            **{name: round(value, 3) for name, value in seconds.items()},
            "row_groups": self.row_groups,
            "bytes_in": self.bytes_in,
            "cores": self.cores,
            "cores_visible": self.cores_visible,
            "cores_source": self.cores_source,
            "write_workers": self.write_workers,
            "cpu_throttled_s": (
                None if self.cpu_throttled_s is None else round(self.cpu_throttled_s, 3)
            ),
        }
