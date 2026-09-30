//! Núcleo de Directional Change (DC): detección de puntos de inflexión en
//! series de precios.
//!
//! Expone el detector de un solo θ ([`Detector`]) y el fan-out de N θ
//! multihilo sobre lotes ([`FanOut`]). Las reglas salen del TRD-L2
//! (`docs/TRD/l2.md`): aritmética entera (ADR-L2-01/02), regla
//! conservadora de empates y validación de DC no vacío (ADR-L2-03) y forma
//! del evento (§7.2). El carry-over y los bindings se construyen encima sin
//! tocarlos.

mod detector;
mod fanout;

pub use detector::{
    Detector, Direction, Event, Extremes, Group, PendingEvent, Point, State, ThetaError, Tick,
    PRICE_LIMIT, SCALE,
};
pub use fanout::FanOut;
