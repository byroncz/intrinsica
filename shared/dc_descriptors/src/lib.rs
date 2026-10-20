//! Motor de agregación de L3: los descriptores de `summaries.parquet` por evento DC, sin
//! pasar por el intérprete de Python.
//!
//! Recibe, por θ y en lotes, las filas de L2 de sus eventos (suelta cada lote al cerrarse
//! sus eventos) y, por row group, los ticks de L1 (buffers de Arrow, sin copia). Ubica solo
//! las fronteras de cada evento en cada row group, con los ids y las mismas comprobaciones que el lector. Reparte
//! el trabajo en ítems `(row group, θ)` que hilos nativos toman de una cola, cierra cada
//! evento en su casilla (la fila del evento en el esqueleto) y entrega row groups de
//! salida sellados. Las sumas son `u128` comprobadas y los parciales de un evento que
//! cruza row groups se fusionan en cualquier orden, así que el archivo no depende de qué
//! hilo tomó qué ítem. El modelo está en el `README.md` del crate y en ADR-L3-16 de
//! `docs/TRD/l3.md`.
//!
//! Sin libpython: los bindings viven en `dc_pyo3`.

pub mod arith;
pub mod close;
pub mod engine;
pub mod input;
pub mod layout;
pub mod output;
pub mod phase;

pub use arith::divide_half_even;
pub use close::{close_event, EventError, EventErrorKind, PhaseName};
pub use engine::{
    locate_events, BoundaryError, BoundaryKind, Config, ConfirmTimeError, Engine, EngineError,
    Located, ThetaStats,
};
pub use input::{
    Boundaries, EventBatch, EventIn, InputError, KeyColumns, Region, RowGroupInput, TickColumns,
};
pub use layout::{col, ColumnSpec, Row, COLUMNS, LIMIT, WIDTH};
pub use output::SealedRowGroup;
pub use phase::{EventAcc, InstantAcc, PhaseAcc, TickError};
