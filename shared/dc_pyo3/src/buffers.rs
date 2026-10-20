//! Vistas de buffers de Python como slices de Rust, sin copia.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Un buffer contiguo de Python de un solo eje como slice, sin copia.
///
/// El slice vive lo que el `PyBuffer`, no el token de Python: así se puede
/// usar con el GIL suelto.
pub fn view<T: Element>(buffer: &PyBuffer<T>) -> PyResult<&[T]> {
    if buffer.dimensions() > 1 {
        return Err(PyValueError::new_err(
            "el buffer debe ser contiguo, alineado y de un solo eje",
        ));
    }
    view_nd(buffer)
}

/// Igual que [`view`] para un buffer C-contiguo de cualquier número de ejes: el slice
/// recorre los elementos en orden de fila (una matriz `(n, 3)` son `3 n` elementos).
pub fn view_nd<T: Element>(buffer: &PyBuffer<T>) -> PyResult<&[T]> {
    if buffer.item_count() == 0 {
        // Un buffer vacío puede tener puntero nulo: no hay nada que mirar.
        return Ok(&[]);
    }
    let ptr = buffer.buf_ptr().cast::<T>();
    if !buffer.is_c_contiguous()
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
    // `item_count` elementos contiguos (comprobado arriba).
    Ok(unsafe { std::slice::from_raw_parts(ptr, buffer.item_count()) })
}
