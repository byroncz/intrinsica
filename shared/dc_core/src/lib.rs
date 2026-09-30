//! Núcleo de Directional Change (DC): detección de puntos de inflexión en
//! series de precios.
//!
//! Expone el detector de un solo θ ([`Detector`]), el fan-out de N θ
//! multihilo sobre lotes ([`FanOut`]) y el carry-over que ambos entregan y
//! retoman entre meses ([`CarryOver`]). Las reglas salen del TRD-L2
//! (`docs/TRD/l2.md`): aritmética entera (ADR-L2-01/02), regla
//! conservadora de empates y validación de DC no vacío (ADR-L2-03), forma
//! del evento (§7.2) y carry-over O(1) por θ (§7.4, ADR-L2-07). Los bindings
//! se construyen encima sin tocarlos.

mod detector;
mod fanout;

pub use detector::{
    CarryOver, CarryOverError, CarryPending, Detector, Direction, Event, Extremes, Group,
    PendingEvent, Point, State, ThetaError, Tick, PRICE_LIMIT, SCALE, STATE_VERSION,
};
pub use fanout::{FanOut, FanOutCarryError};
