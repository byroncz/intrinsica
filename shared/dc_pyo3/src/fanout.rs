//! El fan-out de N θ de `dc_core` como clase Python.

use crate::carry::CarryOver;
use crate::convert::value_error;
use crate::event::Event;
use dc_core::{FanOut as CoreFanOut, PRICE_LIMIT};
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Ticks que se decodifican y se alimentan por vez: el precio entero de un
/// tramo (512 KB) es lo único que se materializa además del buffer de Arrow.
/// Con 50 θ un tramo son ~23 ms de cómputo, así que crear los hilos del
/// fan-out en cada tramo no pesa (`PARALLEL_MIN_WORK` de `dc_core`).
const CHUNK: usize = 65_536;

/// Bytes de un `decimal128` de Arrow.
const DECIMAL_LEN: usize = 16;

/// N detectores DC, uno por θ, que evalúan cada lote de ticks en paralelo.
///
/// `thetas` son `round(θ × 10⁸)`, en el orden en que salen los resultados.
/// `threads` es el máximo de hilos (por defecto, los núcleos disponibles).
#[pyclass(module = "dc_pyo3")]
pub struct FanOut {
    inner: CoreFanOut,
    thetas: Vec<i64>,
}

#[pymethods]
impl FanOut {
    #[new]
    #[pyo3(signature = (thetas, threads=None))]
    fn new(thetas: Vec<i64>, threads: Option<usize>) -> PyResult<Self> {
        let inner = match threads {
            Some(threads) => CoreFanOut::with_threads(&thetas, threads),
            None => CoreFanOut::new(&thetas),
        }
        .map_err(value_error)?;
        Ok(Self { inner, thetas })
    }

    /// Fan-out que retoma el carry-over de cada θ (uno por `thetas`, mismo
    /// orden). Falla si la cuenta no coincide o algún estado no lo acepta el
    /// detector (otro θ, otra versión, invariantes rotas).
    #[staticmethod]
    #[pyo3(signature = (thetas, carry, threads=None))]
    fn from_carry_over(
        thetas: Vec<i64>,
        carry: Vec<PyRef<'_, CarryOver>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let carry: Vec<_> = carry.iter().map(|c| c.0.clone()).collect();
        let inner = match threads {
            Some(threads) => CoreFanOut::from_carry_over_with_threads(&thetas, &carry, threads),
            None => CoreFanOut::from_carry_over(&thetas, &carry),
        }
        .map_err(value_error)?;
        Ok(Self { inner, thetas })
    }

    #[getter]
    fn thetas(&self) -> Vec<i64> {
        self.thetas.clone()
    }

    /// Alimenta un lote a todos los θ y devuelve, por θ, los eventos que dejó
    /// cerrados. Las tres entradas son objetos con protocolo de buffer, sin
    /// copia (los `Buffer` de un `RecordBatch` de Arrow, ver `README.md`):
    ///
    /// - `prices`: `decimal128` de Arrow (16 bytes little-endian por tick), el
    ///   entero sin escalar de `DECIMAL(18,8)`; cada precio cumple
    ///   `0 < price < PRICE_LIMIT`;
    /// - `times`: `transact_time`, `int64`;
    /// - `ids`: `agg_trade_id`, `int64`.
    ///
    /// Se valida todo el lote antes de alimentar: si falla con `ValueError`, el
    /// fan-out no se tocó. El GIL se mantiene durante el lote, porque los
    /// buffers son de Python.
    fn feed_batch(
        &mut self,
        py: Python<'_>,
        prices: PyBuffer<u8>,
        times: PyBuffer<i64>,
        ids: PyBuffer<i64>,
    ) -> PyResult<Vec<Vec<Event>>> {
        let (prices, times, ids) = (view(py, &prices)?, view(py, &times)?, view(py, &ids)?);
        let n = times.len();
        if ids.len() != n || prices.len() != n * DECIMAL_LEN {
            return Err(PyValueError::new_err(format!(
                "columnas de distinto largo: prices {} bytes ({} ticks), times {n}, ids {}",
                prices.len(),
                prices.len() / DECIMAL_LEN,
                ids.len()
            )));
        }
        for (i, raw) in prices.chunks_exact(DECIMAL_LEN).enumerate() {
            decode_price(raw).map_err(|why| {
                PyValueError::new_err(format!(
                    "prices[{i}] no cumple 0 < price < {PRICE_LIMIT}: {why}"
                ))
            })?;
        }

        let mut out = vec![Vec::new(); self.thetas.len()];
        let mut scratch: Vec<i64> = Vec::with_capacity(n.min(CHUNK));
        for start in (0..n).step_by(CHUNK) {
            let end = (start + CHUNK).min(n);
            scratch.clear();
            scratch.extend(
                prices[start * DECIMAL_LEN..end * DECIMAL_LEN]
                    .chunks_exact(DECIMAL_LEN)
                    .map(|raw| decode_price(raw).expect("validado arriba")),
            );
            let closed = self
                .inner
                .feed_batch(&scratch, &times[start..end], &ids[start..end]);
            for (acc, events) in out.iter_mut().zip(closed) {
                acc.extend(events.into_iter().map(Event));
            }
        }
        Ok(out)
    }

    /// Fin de entrada (RF-L2-12): cierra el grupo de empate abierto de cada θ.
    /// Devuelve, por θ, el evento que cierra (o `None`).
    fn finish(&mut self) -> Vec<Option<Event>> {
        self.inner
            .finish()
            .into_iter()
            .map(|e| e.map(Event))
            .collect()
    }

    /// Eventos descartados por la validación "un DC tiene al menos un tick"
    /// (§9.1), por θ. Distinto de cero señala un defecto del detector.
    fn discarded(&self) -> Vec<u64> {
        self.inner
            .detectors()
            .iter()
            .map(dc_core::Detector::discarded)
            .collect()
    }

    /// El carry-over de cada θ, en orden, para entregarlo al cerrar la unidad.
    /// Exige haber llamado a `finish()` y haber visto al menos un tick.
    fn carry_overs(&self) -> PyResult<Vec<CarryOver>> {
        let carry = self.inner.carry_overs().map_err(value_error)?;
        Ok(carry.into_iter().map(CarryOver).collect())
    }

    fn __len__(&self) -> usize {
        self.thetas.len()
    }
}

/// Precio entero de un `decimal128` little-endian: el valor completo cabe en
/// `i64` y cumple el contrato de L1.
fn decode_price(raw: &[u8]) -> Result<i64, &'static str> {
    let value = i128::from_le_bytes(raw.try_into().expect("16 bytes"));
    let price = i64::try_from(value).map_err(|_| "no cabe en i64")?;
    if price > 0 && price < PRICE_LIMIT {
        Ok(price)
    } else {
        Err("fuera de rango")
    }
}

/// Un buffer contiguo de Python como slice, sin copia.
fn view<'a, T: Element>(py: Python<'a>, buffer: &'a PyBuffer<T>) -> PyResult<&'a [T]> {
    if buffer.item_count() == 0 {
        // Un buffer vacío puede tener puntero nulo: no hay nada que mirar.
        return Ok(&[]);
    }
    // `as_slice` es `None` si el buffer no es contiguo, de un eje y alineado.
    let cells = buffer.as_slice(py).ok_or_else(|| {
        PyValueError::new_err("el buffer debe ser contiguo, alineado y de un solo eje")
    })?;
    // SAFETY: `ReadOnlyCell<T>` es `repr(transparent)` sobre `T`, así que
    // `&[ReadOnlyCell<T>]` y `&[T]` tienen el mismo layout. Los buffers de
    // Arrow son inmutables y el GIL se mantiene mientras dure el préstamo,
    // así que nadie los escribe desde Python.
    Ok(unsafe { std::slice::from_raw_parts(cells.as_ptr().cast::<T>(), cells.len()) })
}
