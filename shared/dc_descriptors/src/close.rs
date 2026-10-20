//! El cierre de un evento: de sus agregados a la fila de `summaries.parquet`.
//!
//! Hace lo mismo que `build_row` y `close_phase` de L3 en Python: las cantidades de las
//! fases, el VWAP half-even y los invariantes 1 a 7 de TRD-L3 §7.2 contra la fila de L2,
//! en el mismo orden, así que el primer fallo de un evento es el mismo. Cada fallo trae
//! el campo y los valores que el hallazgo `l2_state_inconsistent` lleva en `details`.
//!
//! El invariante 8 (`buy + sell` es la suma de `quantity`) vale por construcción: el
//! motor no guarda la suma aparte, la deriva de las dos.

use crate::arith::divide_half_even;
use crate::input::EventIn;
use crate::layout::{col, Row, LIMIT, WIDTH};
use crate::phase::{EventAcc, PhaseAcc};
use std::fmt;

/// La fase de un evento.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseName {
    Confirmation,
    Overshoot,
}

impl PhaseName {
    /// El nombre que usan los mensajes de L3.
    pub fn text(self) -> &'static str {
        match self {
            Self::Confirmation => "confirmación",
            Self::Overshoot => "overshoot",
        }
    }
}

/// Un evento que no se pudo cerrar: cuál es (θ por posición, índice en `events.parquet`
/// y `confirm_agg_trade_id`) y por qué.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventError {
    pub theta: usize,
    pub event: u64,
    pub confirm_agg_trade_id: i64,
    pub kind: EventErrorKind,
}

impl fmt::Display for EventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "θ en la posición {}, evento {} (confirmación {}): {}",
            self.theta, self.event, self.confirm_agg_trade_id, self.kind
        )
    }
}

impl std::error::Error for EventError {}

/// Por qué un evento no se puede cerrar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventErrorKind {
    /// `direction` de L2 no es 1 ni -1 (`l2_state_inconsistent`, campo `direction`).
    Direction { got: i64 },
    /// Una cantidad de la fase no cabe en `DECIMAL(18,8)`.
    QuantityTooLarge { phase: PhaseName },
    /// La fase tiene ticks y ninguna cantidad.
    NoQuantity { phase: PhaseName },
    /// `Σ price·quantity` de la fase no cupo en 128 bits.
    NotionalOverflow { phase: PhaseName },
    /// Un invariante de §7.2 contra L2 (`l2_state_inconsistent`).
    Invariant {
        field: &'static str,
        got: i128,
        wanted: String,
    },
}

impl fmt::Display for EventErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direction { got } => {
                write!(f, "direction vale {got} y debería valer 1 o -1")
            }
            Self::QuantityTooLarge { phase } => write!(
                f,
                "la cantidad de la fase {} no cabe en DECIMAL(18,8)",
                phase.text()
            ),
            Self::NoQuantity { phase } => {
                write!(f, "la fase {} no tiene cantidad", phase.text())
            }
            Self::NotionalOverflow { phase } => write!(
                f,
                "la suma de price·quantity de la fase {} no cabe en 128 bits",
                phase.text()
            ),
            Self::Invariant { field, got, wanted } => {
                write!(f, "{field} vale {got} y debería valer {wanted}")
            }
        }
    }
}

/// Una fase cerrada. Con `ticks == 0` los demás campos son 0.
struct Closed {
    ticks: u64,
    buy: i64,
    sell: i64,
    high: i64,
    low: i64,
    duration: i64,
    vwap: u128,
}

fn close_phase(acc: &PhaseAcc, phase: PhaseName) -> Result<Closed, EventErrorKind> {
    if acc.ticks == 0 {
        return Ok(Closed {
            ticks: 0,
            buy: 0,
            sell: 0,
            high: 0,
            low: 0,
            duration: 0,
            vwap: 0,
        });
    }
    let limit = LIMIT as u128;
    if acc.buy >= limit || acc.sell >= limit {
        return Err(EventErrorKind::QuantityTooLarge { phase });
    }
    let total = acc.buy + acc.sell;
    if total == 0 {
        return Err(EventErrorKind::NoQuantity { phase });
    }
    if acc.overflow {
        return Err(EventErrorKind::NotionalOverflow { phase });
    }
    Ok(Closed {
        ticks: acc.ticks,
        buy: acc.buy as i64,
        sell: acc.sell as i64,
        high: acc.high,
        low: acc.low,
        duration: acc.last_time - acc.first_time,
        // Σ price·quantity (10⁻¹⁶) / Σ quantity (10⁻⁸) queda en 10⁻⁸.
        vwap: divide_half_even(acc.notional, total),
    })
}

fn expect(
    ok: bool,
    field: &'static str,
    got: impl Into<i128>,
    wanted: impl FnOnce() -> String,
) -> Result<(), EventErrorKind> {
    if ok {
        Ok(())
    } else {
        Err(EventErrorKind::Invariant {
            field,
            got: got.into(),
            wanted: wanted(),
        })
    }
}

fn zero_or_more(zero: bool) -> String {
    if zero { "0" } else { "más de 0" }.to_owned()
}

/// Cierra un evento: comprueba sus fases y los invariantes de §7.2 y arma su fila.
pub fn close_event(acc: &EventAcc, event: &EventIn) -> Result<Row, EventErrorKind> {
    if event.direction != 1 && event.direction != -1 {
        return Err(EventErrorKind::Direction {
            got: event.direction,
        });
    }
    let c = close_phase(&acc.confirmation, PhaseName::Confirmation)?;
    let o = close_phase(&acc.overshoot, PhaseName::Overshoot)?;

    let up = event.direction == 1;
    let instant = &acc.instant;
    let (best, worst) = if up {
        (instant.high, instant.low)
    } else {
        (instant.low, instant.high)
    };
    let price = event.confirm_price;
    let (r, cc, e) = (event.reference, event.confirm, event.extreme);

    // 1. c_ticks ≥ confirm_instant_ticks ≥ 1.
    expect(
        (1..=c.ticks).contains(&instant.ticks),
        "confirm_instant_ticks",
        instant.ticks,
        || format!("1..{}", c.ticks),
    )?;
    // 2. peor ≤ confirm_price ≤ mejor en un upturn; al revés en un downturn.
    let within = if up {
        worst <= price && price <= best
    } else {
        best <= price && price <= worst
    };
    expect(within, "confirm_price", price, || {
        format!("entre {worst} y {best}")
    })?;
    // 3. El extremo de la confirmación está en el instante de confirmación.
    let top = if up { c.high } else { c.low };
    expect(
        top == best,
        if up { "c_price_max" } else { "c_price_min" },
        top,
        || best.to_string(),
    )?;
    // 4. El extremo del overshoot es `extreme_price`; sin overshoot, `confirm_price`.
    if o.ticks > 0 {
        let edge = if up { o.high } else { o.low };
        expect(
            edge == event.extreme_price,
            if up { "os_price_max" } else { "os_price_min" },
            edge,
            || event.extreme_price.to_string(),
        )?;
    } else {
        expect(
            event.extreme_price == price,
            "extreme_price",
            event.extreme_price,
            || price.to_string(),
        )?;
    }
    // 5. os_ticks = 0 si y solo si E = C.
    expect((o.ticks == 0) == (e == cc), "os_ticks", o.ticks, || {
        zero_or_more(e == cc)
    })?;
    // 6. Sin huecos del proveedor las fases miden C − R y E − C; con ellos, menos.
    let (before, after) = (
        i128::from(cc) - i128::from(r),
        i128::from(e) - i128::from(cc),
    );
    expect(i128::from(c.ticks) <= before, "c_ticks", c.ticks, || {
        format!("≤ {before}")
    })?;
    expect(i128::from(o.ticks) <= after, "os_ticks", o.ticks, || {
        format!("≤ {after}")
    })?;
    // 7. Duración 0 sii todos los ticks son del instante; VWAP entre mínimo y máximo.
    expect(
        (c.duration == 0) == (c.ticks == instant.ticks),
        "c_duration",
        c.duration,
        || zero_or_more(c.ticks == instant.ticks),
    )?;
    let between = |vwap: u128, low: i64, high: i64| {
        i128::try_from(vwap).is_ok_and(|v| i128::from(low) <= v && v <= i128::from(high))
    };
    expect(
        between(c.vwap, c.low, c.high),
        "c_vwap",
        c.vwap as i128,
        || format!("entre {} y {}", c.low, c.high),
    )?;
    if o.ticks > 0 {
        expect(
            between(o.vwap, o.low, o.high),
            "os_vwap",
            o.vwap as i128,
            || format!("entre {} y {}", o.low, o.high),
        )?;
    }

    let mut row = [0i64; WIDTH];
    row[col::CONFIRM_AGG_TRADE_ID] = cc;
    row[col::C_TICKS] = c.ticks as i64;
    row[col::C_QUANTITY_BUY] = c.buy;
    row[col::C_QUANTITY_SELL] = c.sell;
    row[col::C_PRICE_MAX] = c.high;
    row[col::C_PRICE_MIN] = c.low;
    row[col::C_DURATION] = c.duration;
    row[col::C_VWAP] = c.vwap as i64;
    row[col::OS_TICKS] = o.ticks as i64;
    row[col::OS_QUANTITY_BUY] = o.buy;
    row[col::OS_QUANTITY_SELL] = o.sell;
    if o.ticks > 0 {
        row[col::OS_PRICE_MAX] = o.high;
        row[col::OS_PRICE_MIN] = o.low;
        row[col::OS_DURATION] = o.duration;
        row[col::OS_VWAP] = o.vwap as i64;
    }
    row[col::CONFIRM_INSTANT_TICKS] = instant.ticks as i64;
    row[col::CONFIRM_INSTANT_PRICE_BEST] = best;
    row[col::CONFIRM_INSTANT_PRICE_WORST] = worst;
    Ok(row)
}
