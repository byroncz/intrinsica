//! Núcleo de Directional Change (DC): detección de puntos de inflexión en
//! series de precios.
//!
//! Hoy expone el detector de un solo θ ([`Detector`]). Las reglas salen del
//! TRD-L2 (`docs/TRD/l2.md`): aritmética entera (ADR-L2-01/02), regla
//! conservadora de empates y validación de DC no vacío (ADR-L2-03) y forma
//! del evento (§7.2). El fan-out de 50 θ, el carry-over y los bindings se
//! construyen encima sin tocarlo.

mod detector;

pub use detector::{
    Detector, Direction, Event, Extremes, Group, PendingEvent, Point, State, ThetaError, Tick,
    PRICE_LIMIT, SCALE,
};
