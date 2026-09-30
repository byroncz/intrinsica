//! Conversiones compartidas entre los tipos de `dc_core` y los de Python.

use dc_core::Point;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt::Display;

/// Un punto en Python: `(price, time, agg_trade_id)`, precio entero en escala
/// `SCALE`.
pub type PyPoint = (i64, i64, i64);

pub fn to_py_point(p: Point) -> PyPoint {
    (p.price, p.time, p.agg_trade_id)
}

pub fn from_py_point((price, time, agg_trade_id): PyPoint) -> Point {
    Point {
        price,
        time,
        agg_trade_id,
    }
}

/// Todo error de `dc_core` llega a Python como `ValueError` con el mensaje
/// del crate.
pub fn value_error(e: impl Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
