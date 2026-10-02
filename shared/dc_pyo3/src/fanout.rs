//! El fan-out de N θ de `dc_core` como clase Python.

use crate::carry::CarryOver;
use crate::columns::EventColumns;
use crate::convert::value_error;
use crate::event::Event;
use dc_core::{Event as CoreEvent, FanOut as CoreFanOut, PRICE_LIMIT};
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
    /// fan-out no se tocó. El GIL se suelta mientras se valida y se detecta
    /// (ITSC-290): otros hilos de Python, como los escritores de Parquet,
    /// siguen corriendo. Los `PyBuffer` retienen los buffers hasta que la
    /// llamada vuelve, así que las vistas que usa el detector no se liberan.
    ///
    /// Un objeto `Event` por evento: para pruebas y usos pequeños. La capa usa
    /// `feed_batch_columns`.
    fn feed_batch(
        &mut self,
        py: Python<'_>,
        prices: PyBuffer<u8>,
        times: PyBuffer<i64>,
        ids: PyBuffer<i64>,
    ) -> PyResult<Vec<Vec<Event>>> {
        let closed = self.feed_core(py, &prices, &times, &ids)?;
        Ok(closed
            .into_iter()
            .map(|events| events.into_iter().map(Event).collect())
            .collect())
    }

    /// Igual que `feed_batch`, pero los eventos de cada θ salen como un
    /// `EventColumns` (buffers por columna, ya en el layout de Arrow) y no
    /// como objetos `Event`: no se crea un objeto de Python por evento.
    fn feed_batch_columns(
        &mut self,
        py: Python<'_>,
        prices: PyBuffer<u8>,
        times: PyBuffer<i64>,
        ids: PyBuffer<i64>,
    ) -> PyResult<Vec<EventColumns>> {
        let closed = self.feed_core(py, &prices, &times, &ids)?;
        // Un θ a la vez: los eventos de `dc_core` de un θ se sueltan al
        // terminar sus columnas, no todos juntos al final.
        closed
            .into_iter()
            .map(|events| EventColumns::new(py, &events))
            .collect()
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

    /// `finish` con los eventos como columnas: un `EventColumns` por θ, de 0
    /// o 1 evento.
    fn finish_columns(&mut self, py: Python<'_>) -> PyResult<Vec<EventColumns>> {
        let inner = &mut self.inner;
        let closed = py.detach(|| inner.finish());
        closed
            .into_iter()
            .map(|e| EventColumns::new(py, e.as_slice()))
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

impl FanOut {
    /// Valida el lote y lo alimenta por tramos; devuelve los eventos cerrados
    /// por θ, en el tipo de `dc_core`.
    ///
    /// Las vistas se toman con el GIL y el cómputo corre sin él: los
    /// `PyBuffer` que las respaldan siguen vivos hasta el final de la llamada.
    fn feed_core(
        &mut self,
        py: Python<'_>,
        prices: &PyBuffer<u8>,
        times: &PyBuffer<i64>,
        ids: &PyBuffer<i64>,
    ) -> PyResult<Vec<Vec<CoreEvent>>> {
        let (prices, times, ids) = (view(prices)?, view(times)?, view(ids)?);
        let inner = &mut self.inner;
        let n_thetas = self.thetas.len();
        py.detach(|| feed_slices(inner, n_thetas, prices, times, ids))
            .map_err(PyValueError::new_err)
    }
}

/// El cuerpo de `feed_core`, sin tocar Python: valida y alimenta por tramos.
/// El error es el mensaje de un `ValueError`.
fn feed_slices(
    inner: &mut CoreFanOut,
    n_thetas: usize,
    prices: &[u8],
    times: &[i64],
    ids: &[i64],
) -> Result<Vec<Vec<CoreEvent>>, String> {
    let n = times.len();
    if ids.len() != n || prices.len() != n * DECIMAL_LEN {
        return Err(format!(
            "columnas de distinto largo: prices {} bytes ({} ticks), times {n}, ids {}",
            prices.len(),
            prices.len() / DECIMAL_LEN,
            ids.len()
        ));
    }
    for (i, raw) in prices.chunks_exact(DECIMAL_LEN).enumerate() {
        decode_price(raw)
            .map_err(|why| format!("prices[{i}] no cumple 0 < price < {PRICE_LIMIT}: {why}"))?;
    }

    let mut out = vec![Vec::new(); n_thetas];
    let mut scratch: Vec<i64> = Vec::with_capacity(n.min(CHUNK));
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        scratch.clear();
        scratch.extend(
            prices[start * DECIMAL_LEN..end * DECIMAL_LEN]
                .chunks_exact(DECIMAL_LEN)
                .map(|raw| decode_price(raw).expect("validado arriba")),
        );
        let closed = inner.feed_batch(&scratch, &times[start..end], &ids[start..end]);
        for (acc, events) in out.iter_mut().zip(closed) {
            acc.extend(events);
        }
    }
    Ok(out)
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
///
/// El slice vive lo que el `PyBuffer`, no el token de Python: así se puede
/// usar con el GIL suelto.
fn view<T: Element>(buffer: &PyBuffer<T>) -> PyResult<&[T]> {
    if buffer.item_count() == 0 {
        // Un buffer vacío puede tener puntero nulo: no hay nada que mirar.
        return Ok(&[]);
    }
    let ptr = buffer.buf_ptr().cast::<T>();
    if buffer.dimensions() != 1
        || !buffer.is_c_contiguous()
        || !ptr.is_aligned()
        || buffer.item_size() != std::mem::size_of::<T>()
    {
        return Err(PyValueError::new_err(
            "el buffer debe ser contiguo, alineado y de un solo eje",
        ));
    }
    // SAFETY: el `PyBuffer` mantiene exportado el buffer (y a su dueño vivo)
    // hasta que se suelta, y el préstamo no sobrevive a `buffer`. Los buffers
    // de Arrow son inmutables, así que nadie los escribe desde Python mientras
    // el GIL está suelto. El puntero es no nulo, está alineado y cubre
    // `item_count` elementos contiguos de un eje (comprobado arriba).
    Ok(unsafe { std::slice::from_raw_parts(ptr, buffer.item_count()) })
}
