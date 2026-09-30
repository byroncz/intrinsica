//! Bindings PyO3 sobre `dc_core`: el fan-out de N θ y el carry-over para
//! Python. Aquí no hay lógica de detección: solo conversión de tipos y de
//! errores; el detector es `dc_core` sin tocarlo.
//!
//! El módulo Python se llama `dc_pyo3`. `extension-module` no está activo por
//! defecto (ver `Cargo.toml`): lo activa maturin al construir el wheel.

use pyo3::prelude::*;

mod carry;
mod columns;
mod convert;
mod event;
mod fanout;

#[cfg(target_endian = "big")]
compile_error!("dc_pyo3 lee los buffers de Arrow como little-endian");

#[pymodule]
fn dc_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fanout::FanOut>()?;
    m.add_class::<carry::CarryOver>()?;
    m.add_class::<event::Event>()?;
    m.add_class::<columns::EventColumns>()?;
    m.add("SCALE", dc_core::SCALE)?;
    m.add("PRICE_LIMIT", dc_core::PRICE_LIMIT)?;
    m.add("STATE_VERSION", dc_core::STATE_VERSION)?;
    Ok(())
}
