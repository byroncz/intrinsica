//! El motor de agregación de L3 (`dc_descriptors`) como clase Python.
//!
//! Aquí no hay lógica de agregación: solo conversión de buffers y de errores. Los ticks
//! del row group entran sin copia (el motor retiene el dueño del buffer hasta terminar
//! el row group); las filas de L2 de los eventos, que son pocas, se copian a la
//! estructura del motor; los buffers de salida los posee el motor y salen a Python por
//! el protocolo de buffer, sin copia.

use crate::buffers::{view, view_nd};
use dc_descriptors::{
    Config, Engine as CoreEngine, EngineError, EventErrorKind, EventIn, Region, RowGroupInput,
    SealedRowGroup, ThetaPiece, TickColumns, COLUMNS,
};
use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::os::raw::{c_int, c_void};
use std::thread;

create_exception!(
    dc_pyo3,
    DescriptorError,
    PyValueError,
    "Un fallo del motor de L3. `kind` dice cuál; el resto de los atributos, dónde."
);

/// Un buffer de Python que el motor retiene mientras procesa un row group.
///
/// `owner` es el objeto que posee la memoria (un `memoryview` de un `Buffer` de Arrow) y
/// `ptr`/`len` su contenido. Soltarlo sin el GIL (desde un hilo del motor) es seguro:
/// PyO3 difiere el `decref` hasta la siguiente vez que Python corre.
struct Pinned {
    _owner: Py<PyAny>,
    ptr: *const u8,
    len: usize,
}

// SAFETY: el puntero apunta a la memoria de `_owner`, que sigue viva mientras `Pinned`
// exista, y la memoria es inmutable (los buffers de Arrow lo son): leerla desde varios
// hilos es seguro. `Py<PyAny>` es `Send + Sync`.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl AsRef<[u8]> for Pinned {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ver `Send for Pinned`; `ptr` es no nulo y cubre `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// `(buffer, offset)` de una columna de ticks: el `memoryview` de bytes del `Buffer` de
/// Arrow y el `offset` del `Array`, en elementos (en bits para `is_buyer_maker`).
fn pin(column: &Bound<'_, PyAny>, offset: usize) -> PyResult<Region<Pinned>> {
    let buffer = PyBuffer::<u8>::get(column)?;
    let bytes = view(&buffer)?;
    let pinned = Pinned {
        _owner: column.clone().unbind(),
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    };
    Ok(Region {
        buffer: pinned,
        offset,
    })
}

/// El entero sin escalar de un `decimal128` de Arrow que cabe en `i64` (contrato de L2).
fn decimal(bytes: &[u8], offset: usize, i: usize, what: &str) -> PyResult<i64> {
    let at = (offset + i) * 16;
    let raw = bytes.get(at..at + 16).ok_or_else(|| {
        PyValueError::new_err(format!("el buffer de {what} es más corto que sus eventos"))
    })?;
    i64::try_from(i128::from_le_bytes(raw.try_into().expect("16 bytes")))
        .map_err(|_| PyValueError::new_err(format!("{what} no cabe en i64")))
}

/// Las columnas de L2 de los eventos de un θ en un row group, como las entrega el lector
/// (`ThetaEvents`): arreglos NumPy y buffers de Arrow, todos del mismo largo `n`. Es
/// cualquier objeto con estos atributos (la capa usa un `NamedTuple`).
#[derive(FromPyObject)]
struct PyPiece {
    theta: usize,
    first_event: u64,
    /// `(n, 3)` `int64`: posiciones de `R`, `C` y `E`.
    positions: PyBuffer<i64>,
    /// `n` bytes 0/1.
    starts_before: PyBuffer<u8>,
    ends_after: PyBuffer<u8>,
    ticks_before_month: PyBuffer<i64>,
    reference: PyBuffer<i64>,
    confirm: PyBuffer<i64>,
    confirm_time: PyBuffer<i64>,
    extreme: PyBuffer<i64>,
    direction: PyBuffer<i8>,
    confirm_price: (PyBuffer<u8>, usize),
    extreme_price: (PyBuffer<u8>, usize),
}

impl PyPiece {
    fn to_core(&self) -> PyResult<ThetaPiece> {
        let positions = view_nd(&self.positions)?;
        let starts_before = view(&self.starts_before)?;
        let ends_after = view(&self.ends_after)?;
        let before_month = view(&self.ticks_before_month)?;
        let reference = view(&self.reference)?;
        let confirm = view(&self.confirm)?;
        let confirm_time = view(&self.confirm_time)?;
        let extreme = view(&self.extreme)?;
        let direction = view(&self.direction)?;
        let confirm_price = (view(&self.confirm_price.0)?, self.confirm_price.1);
        let extreme_price = (view(&self.extreme_price.0)?, self.extreme_price.1);
        let n = direction.len();
        let same = [
            positions.len() == 3 * n,
            starts_before.len() == n,
            ends_after.len() == n,
            before_month.len() == n,
            reference.len() == n,
            confirm.len() == n,
            confirm_time.len() == n,
            extreme.len() == n,
        ];
        if same.contains(&false) {
            return Err(PyValueError::new_err(format!(
                "las columnas del θ {} no miden lo mismo ({n} eventos)",
                self.theta
            )));
        }
        let mut events = Vec::with_capacity(n);
        for i in 0..n {
            let at = |p: usize| {
                usize::try_from(positions[3 * i + p]).map_err(|_| {
                    PyValueError::new_err(format!("posición negativa en el θ {}", self.theta))
                })
            };
            events.push(EventIn {
                positions: [at(0)?, at(1)?, at(2)?],
                starts_before: starts_before[i] != 0,
                ends_after: ends_after[i] != 0,
                ticks_before_month: before_month[i],
                reference: reference[i],
                confirm: confirm[i],
                confirm_time: confirm_time[i],
                extreme: extreme[i],
                direction: i64::from(direction[i]),
                confirm_price: decimal(confirm_price.0, confirm_price.1, i, "confirm_price")?,
                extreme_price: decimal(extreme_price.0, extreme_price.1, i, "extreme_price")?,
            });
        }
        Ok(ThetaPiece {
            theta: self.theta,
            first_event: self.first_event,
            events,
        })
    }
}

/// Una columna de un row group de salida sellado: bytes en el layout de Arrow que el
/// motor entrega con su dueño. Soporta el protocolo de buffer (`pyarrow.py_buffer`,
/// `memoryview`) sin copiar.
#[pyclass(frozen, module = "dc_pyo3")]
pub struct SealedColumn {
    data: Vec<u8>,
}

#[pymethods]
impl SealedColumn {
    /// # Safety
    /// Lo llama el intérprete con un `view` válido (el contrato de `bf_getbuffer`).
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view es nulo"));
        }
        let data = &slf.get().data;
        // `PyBuffer_FillInfo` toma una referencia a `slf`, que queda viva hasta que el
        // consumidor suelta el buffer, y rechaza los pedidos de escritura (readonly = 1).
        let status = ffi::PyBuffer_FillInfo(
            view,
            slf.as_ptr(),
            data.as_ptr().cast_mut().cast::<c_void>(),
            data.len() as ffi::Py_ssize_t,
            1,
            flags,
        );
        if status == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    fn __len__(&self) -> usize {
        self.data.len()
    }
}

/// Las cuentas de un θ: `(rows, closed_ticks, before_month, first_reference, last_extreme)`.
type PyStats = (u64, u64, u64, Option<i64>, Option<i64>);

/// El motor de agregación de L3: hilos nativos que consumen row groups de L1.
///
/// `thetas` son `round(θ × 10⁸)` en el orden en que se reportan los errores; `threads`
/// son los hilos nativos (por defecto, los núcleos disponibles); `schema` son las
/// columnas de datos de salida como `(nombre, bytes por valor, nulable)`, que el motor
/// comprueba contra las suyas; `row_groups[t]` son las filas de cada row group de salida
/// del θ `t` (el esqueleto de L2).
#[pyclass(module = "dc_pyo3")]
pub struct Engine {
    inner: Option<CoreEngine<Pinned>>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (thetas, schema, row_groups, threads=None))]
    fn new(
        thetas: Vec<i64>,
        schema: Vec<(String, usize, bool)>,
        row_groups: Vec<Vec<usize>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let wanted: Vec<_> = COLUMNS
            .iter()
            .map(|c| (c.name.to_owned(), c.bytes(), c.nullable))
            .collect();
        if schema != wanted {
            return Err(PyValueError::new_err(format!(
                "el esquema de salida no es el del motor: se recibió {schema:?} y el motor \
                 escribe {wanted:?}"
            )));
        }
        let threads = threads
            .or_else(|| thread::available_parallelism().ok().map(Into::into))
            .unwrap_or(1);
        let inner = CoreEngine::new(Config {
            thetas,
            threads,
            row_groups,
        })
        .map_err(to_py_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Encola un row group de L1 y vuelve: el trabajo corre en los hilos del motor.
    ///
    /// `row_group_id` es el ordinal del row group entre los que entrega el lector
    /// (0, 1, 2, ...; pueden llegar en cualquier orden). Cada columna de ticks es un
    /// `(memoryview de bytes del Buffer de Arrow, offset del Array)`: `time` `int64`,
    /// `price` y `quantity` `decimal128` e `is_buyer_maker` booleano (offset en bits). El
    /// motor retiene los buffers hasta terminar el row group (ver `inflight`). `pieces`
    /// son las columnas de L2 de cada θ que toca el row group (ver el README del crate).
    #[pyo3(signature = (row_group_id, rows, time, price, quantity, sold, pieces))]
    #[allow(clippy::too_many_arguments)]
    fn push(
        &self,
        row_group_id: u64,
        rows: usize,
        time: (Bound<'_, PyAny>, usize),
        price: (Bound<'_, PyAny>, usize),
        quantity: (Bound<'_, PyAny>, usize),
        sold: (Bound<'_, PyAny>, usize),
        pieces: Vec<PyPiece>,
    ) -> PyResult<()> {
        let engine = self.engine()?;
        let ticks = TickColumns::new(
            rows,
            pin(&time.0, time.1)?,
            pin(&price.0, price.1)?,
            pin(&quantity.0, quantity.1)?,
            pin(&sold.0, sold.1)?,
        )
        .map_err(|e| to_py_error(e.into()))?;
        let pieces = pieces
            .iter()
            .map(PyPiece::to_core)
            .collect::<PyResult<Vec<_>>>()?;
        engine
            .push(RowGroupInput {
                id: row_group_id,
                ticks,
                pieces,
            })
            .map_err(to_py_error)
    }

    /// El siguiente row group de salida sellado, o `None`: `(theta, index, rows,
    /// columns, validity, null_count)`. `theta` es la posición del θ en `thetas` e
    /// `index` la del row group de salida; cada θ sale en orden de `index`. `columns` son
    /// las 18 columnas de datos (`SealedColumn`, en el orden del esquema) y `validity` el
    /// mapa de bits de las filas con overshoot, o `None` si todas lo tienen.
    fn pop_sealed<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(sealed) = self.engine()?.pop_sealed().map_err(to_py_error)? else {
            return Ok(None);
        };
        let SealedRowGroup {
            theta,
            index,
            rows,
            columns,
            validity,
            null_count,
        } = sealed;
        let columns = columns
            .into_iter()
            .map(|data| Py::new(py, SealedColumn { data }))
            .collect::<PyResult<Vec<_>>>()?;
        let validity = validity
            .map(|data| Py::new(py, SealedColumn { data }))
            .transpose()?;
        let tuple = (
            theta,
            index,
            rows,
            PyList::new(py, columns)?,
            validity,
            null_count,
        );
        Ok(Some(tuple.into_pyobject(py)?.into_any()))
    }

    /// Row groups empujados cuyos ítems no terminaron: los que aún anclan sus buffers.
    fn inflight(&self) -> PyResult<usize> {
        Ok(self.engine()?.inflight())
    }

    /// Fracción del tiempo, desde el primer `push`, con ítems esperando en la cola.
    fn busy(&self) -> PyResult<f64> {
        Ok(self.engine()?.busy())
    }

    /// Fin de la entrada: espera los ítems pendientes (con el GIL suelto) y devuelve, por
    /// θ, `(rows, closed_ticks, before_month, first_reference, last_extreme)`. Lanza
    /// `DescriptorError` si algo falló o falta alguna fila.
    fn finish(&self, py: Python<'_>) -> PyResult<Vec<PyStats>> {
        let engine = self.engine()?;
        let stats = py.detach(|| engine.finish()).map_err(to_py_error)?;
        Ok(stats
            .into_iter()
            .map(|s| {
                (
                    s.rows,
                    s.closed_ticks,
                    s.before_month,
                    s.first_reference,
                    s.last_extreme,
                )
            })
            .collect())
    }

    /// Apaga los hilos del motor. Idempotente; el motor no se puede usar después.
    fn close(&mut self, py: Python<'_>) {
        if let Some(engine) = self.inner.take() {
            py.detach(|| drop(engine));
        }
    }
}

impl Engine {
    fn engine(&self) -> PyResult<&CoreEngine<Pinned>> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("el motor está cerrado"))
    }
}

/// El error del motor como `DescriptorError` con atributos: `kind` y, según el caso,
/// `theta` (posición), `event`, `confirm_agg_trade_id`, `phase`, `field`, `got`,
/// `wanted`, `missing`, `expected`.
fn to_py_error(error: EngineError) -> PyErr {
    let message = error.to_string();
    let err = DescriptorError::new_err(message);
    Python::attach(|py| {
        let value = err.value(py);
        let set = |name: &str, v: Bound<'_, PyAny>| {
            let _ = value.setattr(name, v);
        };
        let kind = |text: &str| set("kind", text.into_pyobject(py).unwrap().into_any());
        match &error {
            EngineError::Input(_) => kind("input"),
            EngineError::Ticks(_) => kind("ticks"),
            EngineError::Internal(_) => kind("internal"),
            EngineError::Skeleton { theta, event, rows } => {
                kind("skeleton");
                set("theta", theta.into_pyobject(py).unwrap().into_any());
                set("event", event.into_pyobject(py).unwrap().into_any());
                set("rows", rows.into_pyobject(py).unwrap().into_any());
            }
            EngineError::Incomplete {
                theta,
                missing,
                expected,
            } => {
                kind("incomplete");
                set("theta", theta.into_pyobject(py).unwrap().into_any());
                set("missing", missing.into_pyobject(py).unwrap().into_any());
                set("expected", expected.into_pyobject(py).unwrap().into_any());
            }
            EngineError::Event(event) => {
                set("theta", event.theta.into_pyobject(py).unwrap().into_any());
                set("event", event.event.into_pyobject(py).unwrap().into_any());
                set(
                    "confirm_agg_trade_id",
                    event
                        .confirm_agg_trade_id
                        .into_pyobject(py)
                        .unwrap()
                        .into_any(),
                );
                match &event.kind {
                    EventErrorKind::Direction { got } => {
                        kind("direction");
                        set("got", got.into_pyobject(py).unwrap().into_any());
                    }
                    EventErrorKind::QuantityTooLarge { phase } => {
                        kind("quantity_too_large");
                        set("phase", phase.text().into_pyobject(py).unwrap().into_any());
                    }
                    EventErrorKind::NoQuantity { phase } => {
                        kind("no_quantity");
                        set("phase", phase.text().into_pyobject(py).unwrap().into_any());
                    }
                    EventErrorKind::NotionalOverflow { phase } => {
                        kind("notional_overflow");
                        set("phase", phase.text().into_pyobject(py).unwrap().into_any());
                    }
                    EventErrorKind::Invariant { field, got, wanted } => {
                        kind("invariant");
                        set("field", field.into_pyobject(py).unwrap().into_any());
                        set("got", got.to_string().into_pyobject(py).unwrap().into_any());
                        set("wanted", wanted.into_pyobject(py).unwrap().into_any());
                    }
                }
            }
        }
    });
    err
}
