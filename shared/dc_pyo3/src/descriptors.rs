//! El motor de agregación de L3 (`dc_descriptors`) como clase Python.
//!
//! Aquí no hay lógica de agregación: solo conversión de buffers y de errores. Los ticks
//! del row group entran sin copia (el motor retiene el dueño del buffer hasta terminar
//! el row group), igual que las columnas de L2 de los eventos (`set_boundaries`, una vez
//! por θ; el motor las retiene hasta cerrarse); los buffers de salida los posee el motor y
//! salen a Python por el protocolo de buffer, sin copia.

use crate::buffers::view;
use dc_descriptors::{
    BoundaryKind, Config, Engine as CoreEngine, EngineError, EventBatch, EventErrorKind,
    KeyColumns, Region, RowGroupInput, SealedRowGroup, TickColumns, COLUMNS,
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

/// Un row group de `events.parquet` de L2 de un θ: sus siete columnas como
/// `(memoryview de bytes del Buffer de Arrow, offset del Array)`, del mismo largo `rows`.
/// Es cualquier objeto con estos atributos (la capa usa un `NamedTuple`).
#[derive(FromPyObject)]
struct PyBatch<'py> {
    rows: usize,
    reference: (Bound<'py, PyAny>, usize),
    confirm: (Bound<'py, PyAny>, usize),
    confirm_time: (Bound<'py, PyAny>, usize),
    extreme: (Bound<'py, PyAny>, usize),
    /// `int8`.
    direction: (Bound<'py, PyAny>, usize),
    /// `decimal128`.
    confirm_price: (Bound<'py, PyAny>, usize),
    extreme_price: (Bound<'py, PyAny>, usize),
}

impl PyBatch<'_> {
    fn to_core(&self) -> PyResult<EventBatch<Pinned>> {
        EventBatch::new(
            self.rows,
            pin(&self.reference.0, self.reference.1)?,
            pin(&self.confirm.0, self.confirm.1)?,
            pin(&self.confirm_time.0, self.confirm_time.1)?,
            pin(&self.extreme.0, self.extreme.1)?,
            pin(&self.direction.0, self.direction.1)?,
            pin(&self.confirm_price.0, self.confirm_price.1)?,
            pin(&self.extreme_price.0, self.extreme_price.1)?,
        )
        .map_err(|e| to_py_error(e.into()))
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

const BOUNDARY_COLUMNS: [&str; 3] = [
    "reference_agg_trade_id",
    "confirm_agg_trade_id",
    "extreme_agg_trade_id",
];

/// Un evento ubicado en un row group: `(índice, (R, C, E), starts_before, ends_after)`.
type PyLocated = (u64, (usize, usize, usize), bool, bool);

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

    /// Fija una vez las filas de L2 de los eventos del θ en la posición `theta`: sus row
    /// groups de `events.parquet`, en orden, sin copia (el motor retiene los buffers hasta
    /// cerrarse). Hay que llamarla para cada θ antes del primer `push`; un θ sin eventos
    /// lleva una lista vacía. El motor calcula solo dónde cae cada frontera en cada row
    /// group, con las comprobaciones del lector.
    fn set_boundaries(&self, theta: usize, batches: Vec<PyBatch<'_>>) -> PyResult<()> {
        let batches = batches
            .iter()
            .map(PyBatch::to_core)
            .collect::<PyResult<Vec<_>>>()?;
        self.engine()?
            .set_boundaries(theta, batches)
            .map_err(to_py_error)
    }

    /// Encola un row group de L1 y vuelve: el trabajo corre en los hilos del motor.
    ///
    /// `row_group_id` es el ordinal del row group entre los que entrega el lector
    /// (0, 1, 2, ...; pueden llegar en cualquier orden, pero un row group espera a que
    /// llegue el anterior). Cada columna de ticks es un `(memoryview de bytes del Buffer
    /// de Arrow, offset del Array)`: `ids` y `time` `int64`, `price` y `quantity`
    /// `decimal128` e `is_buyer_maker` booleano (offset en bits). El motor retiene los
    /// buffers hasta terminar el row group (ver `inflight`). `before_month` dice que el
    /// row group es de un mes anterior al de los eventos.
    #[pyo3(signature = (row_group_id, rows, ids, time, price, quantity, sold, before_month))]
    #[allow(clippy::too_many_arguments)]
    fn push(
        &self,
        row_group_id: u64,
        rows: usize,
        ids: (Bound<'_, PyAny>, usize),
        time: (Bound<'_, PyAny>, usize),
        price: (Bound<'_, PyAny>, usize),
        quantity: (Bound<'_, PyAny>, usize),
        sold: (Bound<'_, PyAny>, usize),
        before_month: bool,
    ) -> PyResult<()> {
        let engine = self.engine()?;
        let ticks = TickColumns::new(
            rows,
            pin(&ids.0, ids.1)?,
            pin(&time.0, time.1)?,
            pin(&price.0, price.1)?,
            pin(&quantity.0, quantity.1)?,
            pin(&sold.0, sold.1)?,
        )
        .map_err(|e| to_py_error(e.into()))?;
        engine
            .push(RowGroupInput {
                id: row_group_id,
                ticks,
                before_month,
            })
            .map_err(to_py_error)
    }

    /// Dónde caen en un row group las fronteras de los eventos del θ que lo tocan, sin
    /// empujarlo: `(índice del evento, (R, C, E), starts_before, ends_after)` por evento.
    /// Es lo que calcula un ítem de `push` (para probar y diagnosticar). `prev_high` es el
    /// último id del row group anterior, o `None` en el primero. Lanza `DescriptorError`
    /// con `kind` `boundary` o `confirm_time` si el row group no cuadra con L2.
    #[pyo3(signature = (theta, rows, ids, time, prev_high=None))]
    fn locate(
        &self,
        theta: usize,
        rows: usize,
        ids: (Bound<'_, PyAny>, usize),
        time: (Bound<'_, PyAny>, usize),
        prev_high: Option<i64>,
    ) -> PyResult<Vec<PyLocated>> {
        let keys = KeyColumns::new(rows, pin(&ids.0, ids.1)?, pin(&time.0, time.1)?)
            .map_err(|e| to_py_error(e.into()))?;
        let located = self
            .engine()?
            .locate(theta, &keys, prev_high)
            .map_err(to_py_error)?;
        Ok(located
            .into_iter()
            .map(|at| {
                let [r, c, e] = at.positions;
                (at.index, (r, c, e), at.starts_before, at.ends_after)
            })
            .collect())
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
/// `wanted`, `missing`, `expected`, y en `boundary` también `reason`, `bounds`, `low` y
/// `high`.
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
            EngineError::Boundary(error) => {
                kind("boundary");
                set("theta", error.theta.into_pyobject(py).unwrap().into_any());
                set("event", error.event.into_pyobject(py).unwrap().into_any());
                set(
                    "field",
                    BOUNDARY_COLUMNS[error.field]
                        .into_pyobject(py)
                        .unwrap()
                        .into_any(),
                );
                let reason = match error.kind {
                    BoundaryKind::NotATick => "not_a_tick",
                    BoundaryKind::Unseen => "unseen",
                };
                set("reason", reason.into_pyobject(py).unwrap().into_any());
                set("bounds", error.bounds.into_pyobject(py).unwrap().into_any());
                set(
                    "got",
                    error.bounds[error.field]
                        .into_pyobject(py)
                        .unwrap()
                        .into_any(),
                );
                set("low", error.low.into_pyobject(py).unwrap().into_any());
                set("high", error.high.into_pyobject(py).unwrap().into_any());
            }
            EngineError::ConfirmTime(error) => {
                kind("confirm_time");
                set("theta", error.theta.into_pyobject(py).unwrap().into_any());
                set("event", error.event.into_pyobject(py).unwrap().into_any());
                set("bounds", error.bounds.into_pyobject(py).unwrap().into_any());
                set("got", error.got.into_pyobject(py).unwrap().into_any());
                set("wanted", error.wanted.into_pyobject(py).unwrap().into_any());
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
