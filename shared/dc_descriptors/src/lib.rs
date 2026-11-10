//! Motor de agregación de L3: los descriptores de `summaries.parquet` por evento DC, sin
//! pasar por el intérprete de Python.
//!
//! Recibe, por θ y en lotes, las filas de L2 de sus eventos (suelta cada lote al cerrarse
//! sus eventos) y, por row group, los ticks de L1 (buffers de Arrow, sin copia). Ubica solo
//! las fronteras de cada evento en cada row group, con los ids y las mismas comprobaciones que el lector. Reparte
//! el trabajo en ítems `(row group, θ)` que hilos nativos toman de una cola, cierra cada
//! evento en su casilla (la fila del evento en el esqueleto) y entrega row groups de
//! salida sellados. Las sumas son `u128` comprobadas (en la familia A, `i128` de punto fijo) y los parciales de un evento que
//! cruza row groups se fusionan en cualquier orden, así que el archivo no depende de qué
//! hilo tomó qué ítem. La familia F no suma: guarda los retornos de cada fase abierta y calcula
//! una FFT exacta al cerrarla. El modelo está en el `README.md` del crate y en ADR-L3-16 de
//! `docs/TRD/l3.md`.
//!
//! Sin libpython: los bindings viven en `dc_pyo3`.

pub mod arith;
pub mod close;
pub mod engine;
pub mod fit;
pub mod input;
pub mod layout;
pub mod output;
pub mod path;
pub mod phase;
pub mod spectrum;

pub use arith::{divide_half_even, Wide};
pub use close::{close_event, EventError, EventErrorKind, PhaseName};
pub use engine::{
    locate_events, BoundaryError, BoundaryKind, Config, ConfirmTimeError, Engine, EngineError,
    Located, ThetaStats,
};
pub use fit::{close_fit, FitAcc, Frame, Moments};
pub use input::{
    Boundaries, EventBatch, EventIn, InputError, KeyColumns, Region, RowGroupInput, TickColumns,
};
pub use layout::{
    col, fit_col, path_col, spectrum_col, ColumnSpec, Family, FitRow, PathRow, Row, SpectrumRow,
    COLUMNS, FIT_COLUMNS, FIT_WIDTH, LIMIT, PATH_COLUMNS, PATH_WIDTH, SPECTRUM_COLUMNS,
    SPECTRUM_WIDTH, WIDTH,
};
pub use output::{ColumnMask, OverBudget, SealedRowGroup};
pub use path::{close_path, PathAcc, PhasePath};
pub use phase::{EventAcc, InstantAcc, PhaseAcc, TickError};
pub use spectrum::{
    close_spectrum, PhaseReturns, SpectrumAcc, BANDS as SPECTRUM_BANDS, DEFAULT_BUDGET_TICKS,
    N_MIN as SPECTRUM_N_MIN,
};
