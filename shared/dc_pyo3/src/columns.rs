//! Los eventos de un θ como columnas, sin un objeto de Python por evento.

use dc_core::{Event as CoreEvent, Point};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Bytes de un `decimal128` de Arrow.
const DECIMAL_LEN: usize = 16;

/// Columnas de `events.parquet` que trae un bloque, en el orden del esquema
/// (TRD-L2 §7.2): `reference_*`, `confirm_*`, `extreme_*` (precio, tiempo,
/// `agg_trade_id`) y `direction`. `theta` no va: es constante por bloque.
const N_COLUMNS: usize = 10;

/// Eventos cerrados de un θ, una columna por buffer de bytes.
///
/// Cada buffer ya está en el layout de Arrow, listo para
/// `pyarrow.py_buffer(...)` sin copia: los precios, `decimal128` de 16 bytes
/// little-endian (el entero sin escalar); los tiempos y los ids, `int64`
/// little-endian; `direction`, `int8`. Se construye directo en la memoria del
/// `bytes` de Python, así que nunca conviven los eventos y una segunda copia
/// más allá de la propia construcción.
#[pyclass(frozen, module = "dc_pyo3")]
pub struct EventColumns {
    len: usize,
    buffers: Vec<Py<PyBytes>>,
}

#[pymethods]
impl EventColumns {
    /// Los 10 buffers en el orden del esquema (ver arriba).
    fn buffers<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.buffers.iter().map(|b| b.bind(py).clone()).collect()
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn __repr__(&self) -> String {
        format!("EventColumns(len={})", self.len)
    }
}

impl EventColumns {
    pub fn new(py: Python<'_>, events: &[CoreEvent]) -> PyResult<Self> {
        let point = |get: fn(&CoreEvent) -> &Point| -> PyResult<[Py<PyBytes>; 3]> {
            Ok([
                decimal_column(py, events, |e| get(e).price)?,
                int64_column(py, events, |e| get(e).time)?,
                int64_column(py, events, |e| get(e).agg_trade_id)?,
            ])
        };
        let mut buffers = Vec::with_capacity(N_COLUMNS);
        buffers.extend(point(|e| &e.reference)?);
        buffers.extend(point(|e| &e.confirm)?);
        buffers.extend(point(|e| &e.extreme)?);
        buffers.push(
            PyBytes::new_with(py, events.len(), |out| {
                for (dst, e) in out.iter_mut().zip(events) {
                    *dst = e.direction.as_i8() as u8;
                }
                Ok(())
            })?
            .unbind(),
        );
        Ok(Self {
            len: events.len(),
            buffers,
        })
    }
}

fn decimal_column(
    py: Python<'_>,
    events: &[CoreEvent],
    get: impl Fn(&CoreEvent) -> i64,
) -> PyResult<Py<PyBytes>> {
    Ok(PyBytes::new_with(py, events.len() * DECIMAL_LEN, |out| {
        for (dst, e) in out.chunks_exact_mut(DECIMAL_LEN).zip(events) {
            // Extensión de signo a 128 bits: el entero sin escalar del decimal.
            dst.copy_from_slice(&i128::from(get(e)).to_le_bytes());
        }
        Ok(())
    })?
    .unbind())
}

fn int64_column(
    py: Python<'_>,
    events: &[CoreEvent],
    get: impl Fn(&CoreEvent) -> i64,
) -> PyResult<Py<PyBytes>> {
    Ok(PyBytes::new_with(py, events.len() * 8, |out| {
        for (dst, e) in out.chunks_exact_mut(8).zip(events) {
            dst.copy_from_slice(&get(e).to_le_bytes());
        }
        Ok(())
    })?
    .unbind())
}
