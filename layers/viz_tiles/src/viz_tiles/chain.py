"""La cola de un θ al cierre de un mes y su cadena de carry-overs (TRD-viz §7.6).

Un evento DC se escribe en el mes que confirma el siguiente, así que al cierre
del mes `M` puede quedar un evento confirmado sin extremo: el pendiente. Sus
ticks se dibujan con el extremo vigente (un candidato que un tick posterior puede
superar) hasta que L2 procesa un mes que lo cierra. `theta_month` sigue esa
cadena: los `carry_over.parquet` de `M+1`, `M+2`… mientras traigan el mismo
pendiente, y el `events.parquet` del primer mes que ya no lo trae.
"""

from dataclasses import dataclass
from datetime import date

from viz_tiles.contract import DAY_US
from viz_tiles.direction import PendingEvent
from viz_tiles.lake import (
    CARRY_OVER,
    EVENTS,
    EventsIndex,
    InputError,
    Month,
    events_rel,
    find_extreme,
    join,
    month_of,
    ordinal,
    read_carry_over,
)
from viz_tiles.reduce import day_start_us


@dataclass(frozen=True)
class ThetaMonth:
    """Lo que un θ aporta a los días de un mes, además de su `events.parquet`.

    `tail` es el evento pendiente al cierre del mes (`None` si no hay): con el
    extremo vigente de la cadena mientras está abierta, o con el definitivo si ya
    cerró (`resolved`). Dos tiempos (µs) de candidato:

    - `own_candidate_time`, el del `carry_over.parquet` del propio mes: desde su
      día en adelante la cadena puede cambiar los tiles, y los archivos de `chain`
      (rutas relativas a la raíz de L2) entran en el `input_hash`.
    - `last_candidate_time`, el de la última carry-over de la cadena con ese
      pendiente: el extremo más reciente que se conoce, desde el que el estado es
      provisional.
    """

    theta: str
    tail: PendingEvent | None = None
    own_candidate_time: int | None = None
    last_candidate_time: int | None = None
    resolved: bool = False
    chain: tuple[str, ...] = ()

    def affects(self, day: date) -> bool:
        """¿La cadena puede cambiar los tiles de este día? (criterio del `input_hash`)

        Verdadero desde el día del candidato del propio mes. Se mide con ese
        candidato y no con el de la última carry-over (TRD-viz §7.8): si la
        primera carry-over de la cadena ya mueve el candidato a un mes posterior,
        el día pasa a ser todo overshoot certero y sus tiles cambian; con el
        candidato último el hash seguiría igual y el día quedaría con la cola
        provisional vieja.
        """
        return (
            self.tail is not None
            and self.own_candidate_time is not None
            and self.own_candidate_time < day_start_us(day) + DAY_US
        )

    def provisional_from_s(self, day: date) -> float | None:
        """Segundos desde el inicio del día desde los que su estado es provisional.

        `None` si el día es definitivo: la cadena cerró, o el candidato más
        reciente es posterior al día. Acotado a 0 si cae en un día anterior.
        """
        if self.resolved or self.last_candidate_time is None:
            return None
        if self.last_candidate_time >= day_start_us(day) + DAY_US:
            return None
        return max(0.0, (self.last_candidate_time - day_start_us(day)) / 1_000_000)


def _candidate(row: dict) -> tuple[PendingEvent, int]:
    """El pendiente de una fila de carry-over y el tiempo (µs) de su extremo vigente."""
    pending = PendingEvent.from_carry_over(row)
    side = "ext_high" if int(row["direction"]) == 1 else "ext_low"
    return pending, int(row[f"{side}_time"])


def theta_month(
    events_root: str,
    index: EventsIndex,
    provider: str,
    market: str,
    asset: str,
    theta: str,
    month: Month,
) -> ThetaMonth:
    """El `ThetaMonth` de `theta` en `month`. Lanza `InputError` si la cadena se rompe.

    Con la cadena abierta la cola usa el candidato de la última carry-over
    existente (el extremo más reciente que se conoce). Con la cadena cerrada en
    `M+k`, el extremo definitivo sale del `events.parquet` de `M+k`; si ese
    archivo no trae el evento, es una entrada rota (`input_missing`).
    """

    def rel(m: Month, filename: str) -> str:
        return events_rel(provider, market, asset, theta, m, filename)

    row = read_carry_over(join(events_root, rel(month, CARRY_OVER)), theta)
    if not row["has_pending_event"]:
        return ThetaMonth(theta)
    tail, own_time = _candidate(row)
    last_time = own_time
    chain: list[str] = []
    n = ordinal(month)
    while True:
        n += 1
        later = month_of(n)
        if not index.has(later, theta, CARRY_OVER):
            # L2 escribe primero los eventos y luego el carry-over: la cadena
            # sigue abierta hasta que este exista.
            return ThetaMonth(theta, tail, own_time, last_time, False, tuple(chain))
        carry_rel = rel(later, CARRY_OVER)
        row = read_carry_over(join(events_root, carry_rel), theta)
        chain.append(carry_rel)
        if (
            row["has_pending_event"]
            and row["pending_confirm_agg_trade_id"] == tail.confirm_agg_trade_id
        ):
            tail, last_time = _candidate(row)
            continue
        break
    # La cadena cerró en `later`: el evento ya está en su `events.parquet`.
    events_file = rel(later, EVENTS)
    chain.append(events_file)
    path = join(events_root, events_file)
    found = (
        find_extreme(path, tail.reference_agg_trade_id)
        if index.has(later, theta, EVENTS)
        else None
    )
    if found is None:
        raise InputError(
            "events",
            f"{path}: no trae el evento pendiente de {theta} (referencia "
            f"{tail.reference_agg_trade_id}), que cerró en {later[0]:04d}-{later[1]:02d}",
            theta=theta,
            path=path,
            reason="falta el evento pendiente cerrado",
        )
    resolved = PendingEvent(
        tail.reference_agg_trade_id,
        tail.confirm_agg_trade_id,
        found[0],
        tail.direction,
        tail.reference_time,
        tail.confirm_time,
        found[1],
    )
    return ThetaMonth(theta, resolved, own_time, last_time, True, tuple(chain))
