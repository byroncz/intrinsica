//! El evento DC cerrado, de solo lectura.

use crate::convert::{to_py_point, PyPoint};
use dc_core::Event as CoreEvent;
use pyo3::prelude::*;

/// Evento DC cerrado (TRD-L2 §7.2): tres puntos `(price, time,
/// agg_trade_id)` con el precio entero en escala `SCALE`, y la dirección
/// (`1` upturn, `-1` downturn). `extreme` es siempre conocido: el detector
/// solo entrega un evento cuando confirma el siguiente.
#[pyclass(frozen, eq, module = "dc_pyo3")]
#[derive(Clone, PartialEq, Eq)]
pub struct Event(pub CoreEvent);

#[pymethods]
impl Event {
    #[getter]
    fn direction(&self) -> i8 {
        self.0.direction.as_i8()
    }

    #[getter]
    fn reference(&self) -> PyPoint {
        to_py_point(self.0.reference)
    }

    #[getter]
    fn confirm(&self) -> PyPoint {
        to_py_point(self.0.confirm)
    }

    #[getter]
    fn extreme(&self) -> PyPoint {
        to_py_point(self.0.extreme)
    }

    fn __repr__(&self) -> String {
        format!(
            "Event(direction={}, reference={:?}, confirm={:?}, extreme={:?})",
            self.direction(),
            self.reference(),
            self.confirm(),
            self.extreme()
        )
    }
}
