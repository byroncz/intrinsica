//! El carry-over de un θ (TRD-L2 §7.4), de solo lectura.

use crate::convert::{from_py_point, to_py_point, value_error, PyPoint};
use dc_core::{CarryOver as CoreCarry, CarryPending};
use pyo3::prelude::*;

/// Una fila del carry-over de un θ: los mismos campos que `dc_core::CarryOver`.
/// Los precios van enteros en escala `SCALE`; `pending` es `None` (sin
/// `has_pending_event`) o `(reference, confirm)`, cada uno `(price, time,
/// agg_trade_id)`. Quien lo escribe a Parquet agrega las coordenadas del dato.
#[pyclass(frozen, eq, module = "dc_pyo3")]
#[derive(Clone, PartialEq, Eq)]
pub struct CarryOver(pub CoreCarry);

#[pymethods]
impl CarryOver {
    #[new]
    #[pyo3(signature = (theta, state_version, direction, ext_high, ext_low, pending=None))]
    fn new(
        theta: i64,
        state_version: String,
        direction: i8,
        ext_high: PyPoint,
        ext_low: PyPoint,
        pending: Option<(PyPoint, PyPoint)>,
    ) -> Self {
        Self(CoreCarry {
            theta,
            state_version,
            direction,
            ext_high: from_py_point(ext_high),
            ext_low: from_py_point(ext_low),
            pending: pending.map(|(reference, confirm)| CarryPending {
                reference: from_py_point(reference),
                confirm: from_py_point(confirm),
            }),
        })
    }

    #[getter]
    fn theta(&self) -> i64 {
        self.0.theta
    }

    #[getter]
    fn state_version(&self) -> &str {
        &self.0.state_version
    }

    #[getter]
    fn direction(&self) -> i8 {
        self.0.direction
    }

    #[getter]
    fn ext_high(&self) -> PyPoint {
        to_py_point(self.0.ext_high)
    }

    #[getter]
    fn ext_low(&self) -> PyPoint {
        to_py_point(self.0.ext_low)
    }

    #[getter]
    fn pending(&self) -> Option<(PyPoint, PyPoint)> {
        self.0
            .pending
            .map(|p| (to_py_point(p.reference), to_py_point(p.confirm)))
    }

    /// Bytes canónicos de `dc_core` (`CarryOver::to_bytes`): transporte
    /// interno, no el formato del TRD.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        self.0.to_bytes().map_err(value_error)
    }

    /// Inversa de `to_bytes`. Falla si la versión no es `STATE_VERSION`.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        CoreCarry::from_bytes(data).map(Self).map_err(value_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "CarryOver(theta={}, state_version={:?}, direction={}, ext_high={:?}, ext_low={:?}, pending={:?})",
            self.0.theta,
            self.0.state_version,
            self.0.direction,
            self.ext_high(),
            self.ext_low(),
            self.pending()
        )
    }
}
